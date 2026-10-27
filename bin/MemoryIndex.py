"""Grammar-derived, inverted retrieval index over completed LTM fields.

Postings are derived in memory from the retained fields after their owner
loads. No forward word list or saved posting table is a second authority.
"""
from __future__ import annotations

import heapq
import itertools
import math

import torch
from Occurrence import AddressIndex
from torch.nn import functional as F


from Generative import preserve_operator_activations


@torch.no_grad()
@preserve_operator_activations
def unfold_idea(language, basis, idea, limit, *, work=None, activation=None,
                order=0, order_of=None, sigma_inverse=None, stream=None,
                constituents=None, constituent_valid=None, constituent_families=None, candidate_limit=16,
                binding=None):
    """Unfold a detached idea with the current generate MLP and tied faces.

    Neither recorded actions nor expected leaf codes are inputs. Stops count
    as recovered codes only within numerical tolerance of a dictionary row;
    stopping on an off-codebook root is an explicit failure, not root snapping.
    """
    if type(order) is not int or order < 0:
        raise ValueError('unfolding requires the stored nonnegative order stamp')
    limit = min(int(limit), 128)
    weights = activation() if callable(activation) else activation
    if not torch.is_tensor(weights):
        return dict(codes=(), operations=(), spent=0, complete=False)
    if weights.ndim == 2:
        if stream is None:
            if weights.shape[0] != 1:
                return dict(codes=(), operations=(), spent=0, complete=False)
            stream = 0
        if type(stream) is not int or not 0 <= stream < weights.shape[0]:
            raise ValueError('unfold priming has no current stream row')
        weights = weights[stream]
    if weights.ndim != 1 or weights.numel() < len(basis):
        raise ValueError('unfold priming differs from its vocabulary')
    # Priming owns physical rows; unrestricted reads see the active prefix.
    # These are the same row addresses, with dormant rows excluded.
    weights = weights[:len(basis)]
    candidates = (weights > 1.).nonzero().flatten()
    if candidates.numel() > 32:
        candidates = candidates[torch.argsort(weights[candidates], descending=True, stable=True)[:32]]
    if candidates.numel() == 0:
        return dict(codes=(), operations=(), spent=0, complete=False)
    candidate_basis = basis[candidates.to(basis.device)]
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    candidate_orders = (None if order_of is None else
                        candidates.new_tensor([order_of(int(code)) for code in candidates]))
    pending, codes, operations, spent, complete = [(idea.detach(), order, binding)], [], [], 0, True
    while pending and spent < limit:
        if work is not None and not work.consume('unfold'):
            break
        value, current_order, current_binding = pending.pop()
        top = value.reshape(1, -1)
        if not bool(torch.isfinite(top).all()):
            raise FloatingPointError('nonfinite idea during grammar unfolding')
        spent += 1
        gate = torch.ones(1, dtype=torch.bool, device=top.device)
        from Generative import inverse_menu
        lefts, rights, legal = inverse_menu(language, top, gate,
            basis=candidate_basis[None], basis_valid=gate[:, None].expand(1, len(candidates)),
            priming=weights[candidates][None].to(top), limit=candidate_limit,
            constituents=constituents, constituent_valid=constituent_valid,
            constituent_families=constituent_families,
            terminal_valid=(None if candidate_orders is None else
                            ((candidate_orders >= 0) & (candidate_orders <= current_order))[None]))
        determined = {}
        if candidate_orders is not None:
            # A higher-order noun may share its point with a particular.
            # Its stamp, not an invented numerical difference, tells the
            # inverse that a determiner is still needed.
            from Generative import determiner_expansions
            determined = determiner_expansions(language, top[0], current_order,
                candidate_basis, candidate_orders, binding=current_binding)
            distance = (candidate_basis - top).norm(dim=-1)
            terminal = ((distance <= 1e-4 * max(1., float(top.norm())))
                        & ((candidate_orders >= 0) & (candidate_orders <= current_order))).any()
            legal[:, -1] &= terminal
            if determined and not bool(terminal):
                legal.zero_()
                for index in determined:
                    legal[:, index] = True
        if not bool(legal.any()):
            complete = False
            break
        from Language import LanguageSpace
        action = int(LanguageSpace.choose_generate(language,
            language.generate_policy_logits(top).masked_fill(~legal, -torch.inf)))
        if action < len(binary):
            operations.append(('binary', action))
            if action in determined:
                left, right, left_order, right_order = determined[action]
                pending.extend(((right, right_order, None), (left, left_order, None)))
            else:
                pending.extend(((rights[0, action], current_order, None),
                                (lefts[0, action], current_order, None)))
        elif action < len(binary) + len(unary):
            operations.append(('unary', action - len(binary)))
            pending.append((lefts[0, action], current_order, current_binding))
        else:
            # Terminal words can only name currently active symbols.
            distance = (candidate_basis.detach().to(top) - top).norm(dim=-1)
            if order_of is not None:
                eligible = distance.new_tensor([0 <= order_of(int(code)) <= current_order
                                                 for code in candidates], dtype=torch.bool)
                distance = distance.masked_fill(~eligible, torch.inf)
            value, row = distance.min(0)
            if float(value) > 1e-4 * max(1., float(top.norm())):
                complete = False
                break
            code = int(candidates[row])
            # Named abstractions are themselves valid retrieval terms. Their
            # lower-order witnesses are unfolded under the same work budget.
            codes.append(code)
            current_order = current_order if order_of is None else order_of(code)
            if current_order:
                children = () if sigma_inverse is None else sigma_inverse(code, current_order)
                if not children:
                    complete = False
                    break
                operations.append(('sigma', current_order, code))
                pending.extend((point, current_order - 1, None) for point in reversed(children))
            else:
                operations.append(('code', code))
    return {'codes': tuple(codes), 'operations': tuple(operations), 'spent': spent,
            'complete': complete and not pending}


def stored_binding(meaning, role):
    """A filled identity licenses definite reference; provenance is not input.

    Other records leave the choice to the learned grammar. In particular,
    an identity alone cannot establish that its original wording was an
    indefinite: its familiarity depends on the generation context.
    """
    if meaning is None or role is None or meaning.role_refs[role] is None:
        return None
    from ThoughtReferences import bindings
    return 'bind' if ('referent', role) in bindings(meaning).get('_bound_roles', ()) else None


def configure_model_index(model, store):
    """Bind the existing CS and generate owners before a closing or reader call."""
    if store is None:
        return
    from Queries import _existing_row, _basis
    registry = getattr(model, 'grammatical_thoughts', None) or getattr(getattr(model, 'symbolSpace', None), 'grammatical_thoughts', None)
    space = getattr(registry, 'space', None) or getattr(model, 'conceptualSpace', None)
    language = getattr(model, 'languageSpace', None)
    attached = store.__dict__.get('_concept_owner')
    if (getattr(store, '_index_owner_bound', False) and callable(attached)
            and attached() is space):
        return
    if space is None or getattr(space, 'similarity_codebook', None) is None:
        return
    from ClauseRow import attach_clause_index
    attach_clause_index(model, store, space)
    store._index_owner_bound = True
