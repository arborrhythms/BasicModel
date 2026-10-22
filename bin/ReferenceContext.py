"""Bounded reference choices carried by the predictor's existing context.

This module has no memory reader or identity table. A caller may supply only
live constituents and frames already admitted to the predictor's situation.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch.nn import functional as F


def exclude_replaced_occurrences(store, identities, valid, document, position):
    """A re-reading cannot use the occurrence it is about to replace.

    These are structural eligibility checks, independent of relevance or
    attention. The store has no versioned old interpretation to bind: such
    a reference would become a self-edge when the selected reading commits.
    The same applies to its embedded rows, whose addresses are reused.
    """
    from Occurrence import document_digest
    current = document_digest(document)
    embedded = document_digest(('embedded', current, position))
    result = valid.clone()
    for column, identity in enumerate(identities.detach().cpu().tolist()):
        row = store.index_of_row(identity)
        if row is None or not bool(valid[column]):
            continue
        source = bytes(store.document_keys[row].detach().cpu().tolist())
        if source == embedded or (source == current and int(store.sentence_index[row]) == position):
            result[column] = False
    return result



@dataclass(frozen=True)
class SituationAddress:
    """STM retains an occurrence address; its current region lives in LTM."""
    occurrence: tuple
    symbols: tuple = ()
    activations: tuple = ()

    def detached(self):
        return self

    def read(self, store):
        index = None if store is None else store.location(self.occurrence)
        return None if index is None else replace(SituationFrame.from_row(store.row(index)),
            symbol_ids=self.symbols, activations=self.activations)


@dataclass(frozen=True)
class SituationFrame:
    depth: int
    roles: torch.Tensor
    occupied: object
    row_id: int
    point: object
    order: int = 1
    when: object = None
    symbol_ids: tuple = ()
    activations: tuple = ()

    def __iter__(self):
        # The existing predictor reads its three-role context unchanged.
        return iter((self.depth, self.roles, self.occupied))

    def detached(self):
        return replace(self, roles=self.roles.detach().clone(),
                       point=None if self.point is None else self.point.detach().clone(),
                       when=None if self.when is None else self.when.detach().clone())

    @classmethod
    def from_row(cls, row):
        meaning = row['meaning']
        if meaning is None or row.get('row_id', -1) in (-1, 0):
            return None
        return cls(int(meaning.role_mask.sum()), meaning.roles.detach().clone(),
                   meaning.role_mask, row['row_id'],
                   meaning.roles[0].detach().clone() if meaning.sentence_kind == 'idea' else None,
                   order=int(row.get('order', -1)), when=row.get('when'))


def shifted_order(stamp, delta):
    """Apply an operation's order change; unknown stays unknown, field is zero."""
    if torch.is_tensor(stamp):
        return torch.where(stamp >= 0, (stamp + delta).clamp_min(0), -1)
    return max(0, stamp + delta) if stamp >= 0 else -1


def operation_order(rule, stamp):
    """A determiner constructs a member or extension from its description.

    The description can be a field type or a raised kind. Its stamp does
    not determine the member's order. Other operations retain their delta;
    these fixed outputs impose no ceiling on their operands or results.
    """
    mode = getattr(rule, 'determiner_mode', None)
    if mode in ('mint', 'bind', 'kind'):
        order = 2 if mode == 'kind' else 1
        return torch.full_like(stamp, order) if torch.is_tensor(stamp) else order
    return shifted_order(stamp, getattr(rule, 'order_delta', 0))


def result_order(rule, operands):
    """The head's stamp, or the joined operands' stamp, followed by the operation."""
    head = getattr(rule, 'head_role', 0)
    if head:
        stamp = operands[head - 1]
    elif torch.is_tensor(operands[0]):
        stamps = torch.stack(operands, -1)
        stamp = torch.where((stamps >= 0).all(-1), stamps.amax(-1), -1)
    else:
        stamp = max(operands) if min(operands) >= 0 else -1
    return operation_order(rule, stamp)


def operation_modes(rule):
    """Temporary reference behaviour belongs to operations, never role names."""
    modes = dict(getattr(rule, 'reference_modes', ()))
    determiner = getattr(rule, 'determiner_mode', None)
    if determiner is not None:
        modes['argin' + str(rule.head_role)] = determiner
    return modes


def reference_requests(language, actions, *, leaf_orders=None, operation_refs=None):
    """Resolve a selected role at its head; the enclosing scope owns its kind.

    A kept numerical journal optionally adds the selected native reference
    to each request. No identity is inferred from a word or its coordinates.
    """
    requests, stack = {}, []
    stamps = None if leaf_orders is None else leaf_orders.detach().cpu().tolist()
    references = None if operation_refs is None else operation_refs.detach().cpu().tolist()
    for action, (kind, local, word) in enumerate(actions.detach().cpu().tolist()):
        if kind < 0:
            break
        if kind == 0:
            stack.append(((word,), -1 if stamps is None else int(stamps[word])))
            continue
        if kind == 3:
            stack[-2:] = reversed(stack[-2:])
            continue
        unary_position = max(0, word) if kind == 2 else 0
        if unary_position:
            stack.append(stack.pop(-1-unary_position))
        arity = 2 if kind == 1 else 1
        rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        if kind not in (1, 2) or not 0 <= local < len(rules) or len(stack) < arity:
            raise ValueError('invalid selected reference rule')
        operands = stack[-arity:]
        del stack[-arity:]
        rule = rules[local]
        for role in getattr(rule, 'reference_roles', ()):
            index = int(role[5:]) - 1
            order, mode = binding_spec(rule, role, operands[index][1])
            for leaf in operands[index][0]:
                request = order, mode
                if references is not None:
                    request = (*request, int(references[action][index]))
                requests[leaf] = request
        head = getattr(rule, 'head_role', 0)
        leaves = (operands[head - 1][0] if head else
                  tuple(leaf for operand, _ in operands for leaf in operand))
        stack.append((leaves, result_order(rule, tuple(stamp for _, stamp in operands))))
        if unary_position:
            stack.insert(len(stack)-1-unary_position, stack.pop())
    return requests


class ReferenceTypes(NamedTuple):
    """Existing dictionary interpretations of the current sentence's words."""
    sources: torch.Tensor
    orders: torch.Tensor
    identities: torch.Tensor
    values: torch.Tensor


class ReferenceBank(NamedTuple):
    ids: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    relations: torch.Tensor
    query: torch.Tensor
    predicted: torch.Tensor
    types: object = None
    cases: object = None
    column_ids: object = None
    interpreter: object = None
    associations: object = None
    orders: object = None
    close_references: bool = False
    lexical_capacity: int = 0
    literal_ids: object = None

    def detached(self):
        """Freeze the rebuilt situation at a compose fork, retaining its API."""
        def freeze(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            if isinstance(value, tuple) and hasattr(value, '_fields'):
                return type(value)(*(freeze(item) for item in value))
            if isinstance(value, tuple):
                return tuple(freeze(item) for item in value)
            # The interpreter is an operation owner, not a captured value.
            return value
        return freeze(self)


def binding_spec(rule, role, stamp, *, implicit=False):
    """The grammatical operand's reference contract, independent of its word."""
    requested = role in getattr(rule, 'reference_roles', ())
    mode = operation_modes(rule).get(role)
    if implicit and not requested:
        clause = getattr(rule, 'clause_form', None)
        relation = getattr(rule, 'relation_kind', None)
        if ((clause == 'S' and role == 'argin1') or
                (clause == 'VP' and role == 'argin2') or
                (relation in ('part', 'whole', 'equal') and role in ('argin1', 'argin2'))):
            requested = True
    return (operation_order(rule, stamp) if requested else None), mode


def resolve_order(value, identity, order, types):
    """Select an existing grammar-requested type before its numerical use.

    A newly requested singleton keeps its source point during the trial;
    the winning closing admits its identity. This read never allocates.
    """
    if types is None:
        return value, identity
    batch, positions, width = value.shape
    words, orders = types.identities.shape[1:]
    source = identity[:, :, None] == types.sources[:, None, :]
    associated = (identity[:, :, None, None] == types.identities[:, None]).any(-1)
    matching = ((source | associated) & (identity > 0)[:, :, None])[:, :, :, None]
    requested = torch.as_tensor(order, device=value.device).expand(batch, positions)
    matching = matching & (types.orders[None, None, None, :] == requested[:, :, None, None])
    matching = matching & (types.identities[:, None] > 0)
    matching = matching.reshape(batch, positions, words * orders)
    found = matching.any(-1)
    selected = matching.to(torch.long).argmax(-1)
    ids = types.identities.reshape(batch, 1, -1).expand(batch, positions, -1)
    ids = ids.gather(2, selected[:, :, None]).squeeze(2)
    points = types.values.reshape(batch, 1, words * orders, width)
    points = points.expand(batch, positions, -1, width).gather(
        2, selected[:, :, None, None].expand(batch, positions, 1, width)).squeeze(2)
    # The existing identity already has its live activation and modifiers.
    changed = found & (ids != identity)
    return (torch.where(changed[:, :, None], points, value),
            torch.where(found, ids, identity))


def resolve_operand(value, identity, position, *, mode, bank, live, active, order=None, forced_reference=None,
                    local=None):
    """Numerical adapter for explicit, already bounded reading fixtures.

    Production resolves order-one references through ReferenceClose. This
    adapter retains the numerical fixture contract, with residual evidence
    instead of a cosine rule; it has no mode-dependent prediction mask or
    positional candidate filter.
    """
    live_values, live_ids, live_orders, live_relations, live_positions = live[:5]
    live_local = live[5] if len(live) > 5 else torch.zeros_like(live_ids, dtype=torch.bool)
    local = torch.zeros_like(identity, dtype=torch.bool) if local is None else local
    B, P, D = value.shape
    requested = (torch.full_like(identity, -1) if order is None else
                 torch.as_tensor(order, device=identity.device).expand_as(identity))
    C = bank.ids.shape[1]
    K = live_ids.shape[1]
    if mode == 'mint':
        # Admission belongs to the kept closing. A trial names no durable row.
        return value, torch.full_like(identity, -1), torch.zeros_like(active), torch.ones_like(active)
    own_valid = ((identity != -1) & (identity != 0)) & ~local & (mode not in ('pronoun', 'bind'))
    if bank.orders is not None:
        own_known = ((identity[:, :, None] == bank.ids[:, None])
                     & (bank.orders[:, None] >= 0))
        own_valid &= (~own_known.any(-1) |
                      (own_known & (bank.orders[:, None] == requested[:, :, None])).any(-1))
    held = bank.valid
    earlier = live_ids[:, None, :] != identity[:, :, None]
    earlier &= ((live_ids[:, None, :] != -1) & (live_ids[:, None, :] != 0))
    earlier &= live_orders[:, None, :] == requested[:, :, None]
    earlier &= ~live_local[:, None, :]
    ids = torch.cat((identity[:, :, None], bank.ids[:, None].expand(B, P, C),
                     live_ids[:, None].expand(B, P, K)), -1)
    values = torch.cat((value[:, :, None], bank.values[:, None].expand(B, P, C, D),
                        live_values[:, None].expand(B, P, K, D)), -2)
    held = held[:, None].expand(B, P, C)
    if bank.orders is not None:
        held = held & ((bank.orders[:, None] < 0) | (bank.orders[:, None] == requested[:, :, None]))
    valid = torch.cat((own_valid[:, :, None], held, earlier), -1)
    relations = torch.cat((torch.zeros_like(own_valid[:, :, None]),
                          bank.relations[:, None].expand(B, P, C),
                          live_relations[:, None].expand(B, P, K)), -1)
    # The first occurrence of an address owns its candidate mass.
    size = ids.shape[-1]
    preceding = torch.arange(size, device=ids.device)[
        None, :] < torch.arange(size, device=ids.device)[:, None]
    duplicate = ((ids[..., :, None] == ids[..., None, :]) & preceding & valid[..., None, :]).any(-1)
    valid = valid & ~duplicate
    available = valid.any(-1)
    # An unavailable proposal receives a harmless local value but is masked
    # out of the operation chooser. Keep the softmax finite for all rows.
    safe_valid = valid | (~available[:, :, None] & (torch.arange(size, device=ids.device) == 0))
    query = torch.where(bank.predicted[:, None, None], bank.query[:, None, :], value)
    residual = (values - query[:, :, None, :]).square().sum(-1)
    logits = (-residual / query.square().sum(-1, keepdim=True).clamp_min(1e-8)).masked_fill(~safe_valid, -torch.inf)
    probabilities = logits.softmax(-1)
    selected = probabilities.argmax(-1)
    if forced_reference is not None:
        forced = torch.as_tensor(forced_reference, device=ids.device)
        permitted = valid & (ids == forced[..., None])
        if not torch.compiler.is_compiling() and not bool(permitted.any(-1).all()):
            raise ValueError('forced identity is outside the bounded candidate set')
        torch._assert_async(permitted.any(-1).all(),
                            'forced identity is outside the bounded candidate set')
        selected = permitted.to(torch.long).argmax(-1)
    point = values.gather(2, selected[:, :, None, None].expand(B, P, 1, D)).squeeze(2)
    probability = probabilities.gather(2, selected[:, :, None]).squeeze(2)
    resolved = point * (1 + (probability-probability.detach()))[:, :, None]
    refs = ids.gather(2, selected[:, :, None]).squeeze(2)
    relative = relations.gather(2, selected[:, :, None]).squeeze(2)
    used = active & available
    return (torch.where(used[:, :, None], resolved, value),
            torch.where(used, refs, identity), relative & used, available | ~active)


def prepare_operands(window, identities, flags, positions, *, orders, rules, unary_rules,
                     bank, live, active):
    """Enumerate reference variants of each operation for one global softmax.

    A variant retains the grammar operation's index and learned scoring row.
    No local score, argmax, allocation or dictionary choice occurs here.
    """
    B, N, D = window.shape
    interpreter = getattr(bank, 'interpreter', None)

    def propose(value, ids, scope, pos, stamp, rule, role, valid):
        order, mode = binding_spec(rule, role, stamp, implicit=interpreter is not None)
        if order is None:
            if interpreter is not None and bank.associations is not None and bank.associations.numel():
                from ClauseScope import ClauseScope
                # At a non-reference argument, disambiguate only the word's
                # already-admitted definitions in this bounded situation.
                # Do not manufacture the full bind/mint menu at every role.
                associated = ((ids[..., None, None] == bank.associations[:, 0]) &
                    (bank.ids[:, None, :, None] == bank.associations[:, 1])).any(-1)
                eligible = (scope.bitwise_and(ClauseScope.LOCAL | ClauseScope.DETERMINED) == 0)
                available = associated & bank.valid[:, None] & (eligible & valid)[..., None]
                count = min(bank.ids.shape[1], bank.lexical_capacity or bank.ids.shape[1])
                priority = available * torch.arange(bank.ids.shape[1], 0, -1, device=ids.device)
                selected = priority.topk(count, dim=-1).indices
                choices = []
                for c in range(count):
                    at = selected[..., c]
                    choices.append((bank.values.gather(1, at[..., None].expand(*at.shape, D)),
                        bank.ids.gather(1, at), bank.relations.gather(1, at),
                        available.gather(-1, at[..., None]).squeeze(-1)))
                choices.append((value, ids, scope.bitwise_and(1) != 0, valid & ~available.any(-1)))
                return choices
            return [(value, ids, (scope.bitwise_and(1) != 0), torch.ones_like(valid))]
        source_ids = ids
        typed_value, typed_ids = resolve_order(value, ids, order, getattr(bank, 'types', None))
        at_close = bank.close_references
        if at_close:
            value = torch.where((order == 1)[..., None], value, typed_value)
            ids = torch.where(order == 1, ids, typed_ids)
        else:
            value, ids = typed_value, typed_ids
        if mode in ('kind', 'generic'):
            return [(value, ids, (scope.bitwise_and(1) != 0), torch.ones_like(valid))]
        closing = None
        if at_close:
            # Compose selects the operation, not an individual. Its operands
            # retain their content until the completed predicate is visible.
            deferred = order == 1
            if mode is None and bank.associations is not None and bank.associations.numel():
                # An ambiguous WORD has no object at the requested order.
                # Its already-retained definitions are ordinary candidates,
                # even when their conceptual order differs from this use.
                # Deferring this case would carry the WORD itself through
                # the implicit reference, with no later binding request.
                defined = (source_ids[..., None] == bank.associations[:, 0]).any(-1)
                deferred = deferred & ~defined
            refs = (torch.full_like(ids, -1 if mode == 'mint' else 0)
                    if mode in ('mint', 'bind', 'pronoun') else ids)
            closing = (value, refs, scope.bitwise_and(1) != 0, valid & deferred)
            # Orders other than one retain their existing operation contracts.
            # Neither the field nor higher relations are individual columns.
            valid = valid & ~deferred
        # The operation offers references at the operand's actual order.
        # Only dictionary admission is specific to order-one individuals.
        binding = order >= 0
        if mode is None and bank.literal_ids is not None:
            # An admitted objectless marker is a literal symbol. An
            # implicit participant read cannot silently replace it with
            # an object. Explicit mint/bind operations keep their contracts.
            binding = binding & ~torch.isin(source_ids, bank.literal_ids)
        original_valid = valid
        def alternatives(options):
            # The first variant carries non-binding typed operands. Keep
            # the existing candidate menu and its learned scoring rows.
            point, ref, relative, available, *extra = options[0]
            options[0] = (torch.where(binding[..., None], point, value),
                torch.where(binding, ref, ids),
                torch.where(binding, relative, scope.bitwise_and(1) != 0),
                available | (~binding & original_valid),
                *(item & binding for item in extra))
            if closing is not None:
                # Reuse the first variant for deferred rows. Appending a
                # masked variant would change the unrelated field/higher-
                # order menu and its sampled departures.
                point, ref, relative, available, *extra = options[0]
                options[0] = (torch.where(deferred[..., None], closing[0], point),
                    torch.where(deferred, closing[1], ref),
                    torch.where(deferred, closing[2], relative),
                    available | closing[3], *(item & ~deferred for item in extra))
            return options
        valid = valid & binding
        if mode == 'mint' and interpreter is None:
            return alternatives([(value, torch.full_like(ids, -1), torch.zeros_like(valid), valid)])
        inventory = getattr(bank, 'column_ids', None)
        if interpreter is None and inventory is not None and inventory.numel() == 0:
            # A binder with no referent leaves an open slot. The sentence
            # can still close; its closing will ask on this same reference.
            return alternatives([(value, torch.full_like(ids, -1), torch.zeros_like(valid), valid)])
        from ClauseScope import ClauseScope
        P, C = value.shape[1], bank.ids.shape[1]
        live_values, live_ids, live_orders, live_relations, live_positions = live[:5]
        live_local = live[5] if len(live) > 5 else torch.zeros_like(live_ids, dtype=torch.bool)
        # A referent is an individual use, not a conceptual-order stamp.
        # LTM0 and live constituents retain their actual, unrestricted order.
        member = order == 1
        earlier = (live_positions[:, None] < pos[:, :, None]) & (
            member[:, :, None] | (live_orders[:, None] == order[:, :, None]))
        earlier = earlier & ~live_local[:, None]
        if inventory is not None:
            earlier = earlier & torch.isin(live_ids, inventory)[:, None]
        earlier = earlier & (live_ids[:, None] != -1) & (live_ids[:, None] != 0)
        own_inventory = bank.ids.flatten() if inventory is None else inventory
        own = torch.isin(ids, own_inventory) & (scope.bitwise_and(ClauseScope.LOCAL) == 0)
        # A lowering cannot reuse the higher-order source as its own
        # target. A changed association was selected at the requested order.
        own = own & (member | (stamp == order) | (ids != source_ids))
        # Only a native column can be its own candidate. A word's dictionary
        # type is never silently promoted to an individual.
        candidate_ids = torch.cat((ids[:, :, None], bank.ids.flip(1)[:, None].expand(B, P, C),
                                   live_ids[:, None].expand(B, P, -1)), -1)
        candidate_values = torch.cat((value[:, :, None], bank.values.flip(1)[:, None].expand(B, P, C, D),
                                      live_values[:, None].expand(B, P, -1, D)), -2)
        candidate_relations = torch.cat(((scope.bitwise_and(1) != 0)[:, :, None],
                                         bank.relations.flip(1)[:, None].expand(B, P, C),
                                         live_relations[:, None].expand(B, P, -1)), -1)
        held = bank.valid.flip(1)[:, None].expand(B, P, C)
        if bank.orders is not None:
            held_orders = bank.orders.flip(1)[:, None]
            held = held & (member[:, :, None] | (held_orders < 0) | (held_orders == order[:, :, None]))
        available = torch.cat((own[:, :, None], held, earlier), -1)
        size = candidate_ids.shape[-1]
        preceding = torch.arange(size, device=ids.device)[None] < torch.arange(size, device=ids.device)[:, None]
        duplicate = ((candidate_ids[..., :, None] == candidate_ids[..., None, :])
                     & preceding & available[..., None, :]).any(-1)
        available = available & ~duplicate & valid[..., None]
        # Enumerate no more than the existing situation/STM capacity, even
        # when a column appears in both. This is recency enumeration, not a
        # content score or a referent selection.
        count = min(size, max(C, live_ids.shape[1]))
        if inventory is not None:
            count = min(count, inventory.numel())
        priority = torch.arange(size, 0, -1, device=ids.device) * available
        retained = priority.topk(count, dim=-1).indices
        candidate_ids = candidate_ids.gather(-1, retained)
        candidate_values = candidate_values.gather(-2, retained[..., None].expand(B, P, count, D))
        candidate_relations = candidate_relations.gather(-1, retained)
        available = available.gather(-1, retained)
        if interpreter is not None:
            # Retain by each row's recency before interpret enumerates its
            # fixed masked variants. Another row cannot alter this layout.
            options = interpreter(value, binding=dict(identity=ids, scope=scope,
                candidate_ids=candidate_ids, candidate_values=candidate_values,
                candidate_relations=candidate_relations, available=available,
                associations=bank.associations, mode=mode, active=valid))
        else:
            options = [(candidate_values[:, :, c], candidate_ids[:, :, c],
                        candidate_relations[:, :, c], available[:, :, c]) for c in range(count)]
            if mode not in ('pronoun', 'bind'):
                options.append((value, torch.full_like(ids, -1), torch.zeros_like(valid), valid))
            else:
                options.append((value, torch.full_like(ids, -1), torch.zeros_like(valid),
                                valid & ~available.any(-1)))
        return alternatives(options)

    def choice_tag(refs, rule, role, stamp, source):
        order, mode = binding_spec(rule, role, stamp, implicit=interpreter is not None)
        if order is None:
            if interpreter is not None and bank.associations is not None and bank.associations.numel():
                defined = (source[..., None] == bank.associations[:, 0]).any(-1)
                return torch.where(defined, refs, -1)
            return torch.full_like(refs, -1)
        if mode in ('kind', 'generic'):
            return torch.full_like(refs, -1)
        # -2 is a comparison-only tag for mint, never a native reference.
        binding = order >= 0
        if mode is None and bank.literal_ids is not None:
            binding = binding & ~torch.isin(source, bank.literal_ids)
        if bank.close_references and mode not in ('mint', 'bind', 'pronoun'):
            binding = binding & (order != 1)
        return torch.where(binding, torch.where(refs == -1, -2, refs), -1)

    lefts, rights, brefs, brels, bvalid, case_weights, binary_ops = [], [], [], [], [], [], []
    bchoices, uchoices, bfound, ufound = [], [], [], []

    def found(option):
        return option[4] if len(option) > 4 else torch.zeros_like(option[3])
    case_bank = getattr(bank, 'cases', None)
    case_count = 1 if case_bank is None else len(case_bank.ids)
    for operation, rule in enumerate(rules):
        left_options = propose(window[:, :-1], identities[:, :-1], flags[:, :-1], positions[:, :-1],
                       orders[:, :-1], rule, 'argin1', active[:, :-1] & active[:, 1:])
        right_options = propose(window[:, 1:], identities[:, 1:], flags[:, 1:], positions[:, 1:],
                        orders[:, 1:], rule, 'argin2', active[:, :-1] & active[:, 1:])
        for left in left_options:
            for right in right_options:
                binary_ops.append(operation)
                lefts.append(left[0])
                rights.append(right[0])
                brefs.append(torch.stack((left[1], right[1]), -1))
                bchoices.append(torch.stack((choice_tag(left[1], rule, 'argin1', orders[:, :-1], identities[:, :-1]),
                                             choice_tag(right[1], rule, 'argin2', orders[:, 1:], identities[:, 1:])), -1))
                bfound.append(torch.stack((found(left), found(right)), -1))
                brels.append(torch.stack((left[2], right[2]), -1))
                valid = left[3] & right[3]
                weights = window.new_zeros(B, N-1, case_count, 2)
                case_head = getattr(rule, 'case_head_role', 0)
                if case_head:
                    if case_bank is None:
                        valid = torch.zeros_like(valid)
                    else:
                        from CaseSelection import select_cases
                        modifier, head = (left[1], right[1]) if case_head == 2 else (right[1], left[1])
                        selection = select_cases(case_bank, modifier, head)
                        weights = selection.weights
                        valid = valid & selection.available
                case_weights.append(weights)
                bvalid.append(valid)
    values, urefs, urels, uvalid, unary_ops = [], [], [], [], []
    for operation, rule in enumerate(unary_rules):
        for operand in propose(window, identities, flags, positions, orders, rule, 'argin1', active):
            unary_ops.append(operation)
            values.append(operand[0])
            urefs.append(torch.stack((operand[1], torch.full_like(operand[1], -1)), -1))
            uchoices.append(torch.stack((choice_tag(operand[1], rule, 'argin1', orders, identities),
                                         torch.full_like(operand[1], -1)), -1))
            ufound.append(torch.stack((found(operand), torch.zeros_like(operand[3])), -1))
            urels.append(torch.stack((operand[2], torch.zeros_like(operand[2])), -1))
            uvalid.append(operand[3])

    def stack(items, width, tail=(), *, dtype=None):
        return torch.stack(items, 2) if items else torch.zeros((B, width, 0, *tail), device=window.device, dtype=dtype or window.dtype)
    return dict(binary_ops=tuple(binary_ops), unary_ops=tuple(unary_ops),
                binding_choices=interpreter is not None,
                case_bank=case_bank, case_weights=stack(case_weights, N-1, (case_count, 2)), left=stack(lefts, N-1, (D,)), right=stack(rights, N-1, (D,)), unary=stack(values, N, (D,)),
                binary_refs=stack(brefs, N-1, (2,), dtype=torch.long), unary_refs=stack(urefs, N, (2,), dtype=torch.long),
                binary_choices=stack(bchoices, N-1, (2,), dtype=torch.long), unary_choices=stack(uchoices, N, (2,), dtype=torch.long),
                binary_found=stack(bfound, N-1, (2,), dtype=torch.bool), unary_found=stack(ufound, N, (2,), dtype=torch.bool),
                binary_relations=stack(brels, N-1, (2,), dtype=torch.bool), unary_relations=stack(urels, N, (2,), dtype=torch.bool),
                binary_valid=stack(bvalid, N-1, dtype=torch.bool), unary_valid=stack(uvalid, N, dtype=torch.bool))
