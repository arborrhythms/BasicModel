"""Bounded reference choices carried by the predictor's existing context.

This module has no memory reader or identity table. A caller may supply only
live constituents and frames already admitted to the predictor's situation.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SituationFrame:
    depth: int
    roles: torch.Tensor
    occupied: object
    row_id: int
    point: object
    order: int = 1
    where: object = None
    when: object = None

    def __iter__(self):
        # The existing predictor reads its three-role context unchanged.
        return iter((self.depth, self.roles, self.occupied))

    def detached(self):
        return replace(self, roles=self.roles.detach().clone(),
                       point=None if self.point is None else self.point.detach().clone(),
                       where=None if self.where is None else self.where.detach().clone(),
                       when=None if self.when is None else self.when.detach().clone())

    @classmethod
    def from_row(cls, row):
        meaning = row['meaning']
        if meaning is None or row.get('row_id', -1) in (-1, 0):
            return None
        return cls(int(meaning.role_mask.sum()), meaning.roles.detach().clone(),
                   meaning.role_mask, row['row_id'],
                   row['np1'].detach().clone() if row['rel_type'] == 0 else None,
                   where=row.get('where'), when=row.get('when'))


def reference_requests(language, actions, *, operation_refs=None):
    """Resolve a selected role at its head; the enclosing scope owns its kind.

    A kept numerical journal optionally adds the selected native reference
    to each request. No identity is inferred from a word or its coordinates.
    """
    requests, stack = {}, []
    references = None if operation_refs is None else operation_refs.detach().cpu().tolist()
    for action, (kind, local, word) in enumerate(actions.detach().cpu().tolist()):
        if kind < 0:
            break
        if kind == 0:
            stack.append((word,))
            continue
        arity = 2 if kind == 1 else 1
        rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        if kind not in (1, 2) or not 0 <= local < len(rules) or len(stack) < arity:
            raise ValueError('invalid selected reference rule')
        operands = stack[-arity:]
        del stack[-arity:]
        modes = dict(getattr(rules[local], 'reference_kinds', ()))
        determiner = getattr(rules[local], 'determiner_mode', None)
        if determiner is not None:
            modes['I' + str(rules[local].head_role)] = determiner
        for role, order in getattr(rules[local], 'reference_orders', ()):
            for leaf in operands[int(role[1:]) - 1]:
                request = order, modes.get(role)
                if references is not None:
                    request = (*request, int(references[action][int(role[1:]) - 1]))
                requests[leaf] = request
        head = getattr(rules[local], 'head_role', 0)
        stack.append(operands[head - 1]
                     if head else tuple(leaf for operand in operands for leaf in operand))
    return requests


def selected_individual_references(language, program):
    """The kept choice, including -1 for mint, after enclosing kind scopes."""
    if program is None or program.operation_refs is None:
        return ()
    requests = reference_requests(language, program.actions,
                                  operation_refs=program.operation_refs)
    return tuple(dict.fromkeys(reference for order, mode, reference in requests.values()
                               if order == 1 and mode != 'kind'))


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
    matching = matching & (types.orders == order)[None, None, None, :]
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


def resolve_operand(value, identity, position, *, mode, bank, live, active, forced_reference=None,
                    local=None):
    """A hard reference proposal before its numerical grammar operation.

    Only the existing predictor anchors and earlier occupied constituents are
    candidates. The caller masks unavailable proposals before the global
    operation softmax; an unselected pronoun proposal cannot reject a reading.
    """
    live_values, live_ids, live_orders, live_relations, live_positions = live[:5]
    live_local = live[5] if len(live) > 5 else torch.zeros_like(live_ids, dtype=torch.bool)
    local = torch.zeros_like(identity, dtype=torch.bool) if local is None else local
    B, P, D = value.shape
    C = bank.ids.shape[1]
    K = live_ids.shape[1]
    if mode == 'mint':
        # Admission belongs to the kept closing. A trial names no durable row.
        return value, torch.full_like(identity, -1), torch.zeros_like(active), torch.ones_like(active)
    own_valid = ((identity != -1) & (identity != 0)) & ~local & (mode not in ('pronoun', 'bind'))
    held = bank.valid if mode == 'bind' else bank.valid & bank.predicted[:, None]
    earlier = (live_positions[:, None, :] < position[:, :, None])
    earlier &= ((live_ids[:, None, :] != -1) & (live_ids[:, None, :] != 0)) & (live_orders[:, None, :] == 1)
    earlier &= ~live_local[:, None, :]
    ids = torch.cat((identity[:, :, None], bank.ids[:, None].expand(B, P, C),
                     live_ids[:, None].expand(B, P, K)), -1)
    values = torch.cat((value[:, :, None], bank.values[:, None].expand(B, P, C, D),
                        live_values[:, None].expand(B, P, K, D)), -2)
    valid = torch.cat((own_valid[:, :, None], held[:, None].expand(B, P, C), earlier), -1)
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
    logits = F.cosine_similarity(
        values, query[:, :, None, :], dim=-1).masked_fill(~safe_valid, -torch.inf)
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


def prepare_operands(window, identities, flags, positions, *, rules, unary_rules,
                     bank, live, active):
    """Enumerate reference variants of each operation for one global softmax.

    A variant retains the grammar operation's index and learned scoring row.
    No local score, argmax, allocation or dictionary choice occurs here.
    """
    B, N, D = window.shape

    def propose(value, ids, scope, pos, rule, role, valid):
        order = dict(getattr(rule, 'reference_orders', ())).get(role)
        mode = dict(getattr(rule, 'reference_kinds', ())).get(role)
        determiner = getattr(rule, 'determiner_mode', None)
        if determiner is not None and role == 'I' + str(rule.head_role):
            mode = determiner
            if mode == 'kind':
                return [(value, ids, (scope.bitwise_and(1) != 0), torch.ones_like(valid))]
            order = 1
        if mode == 'mint':
            return [(value, torch.full_like(ids, -1), torch.zeros_like(valid), torch.ones_like(valid))]
        if order is not None and order != 1:
            value, ids = resolve_order(value, ids, order, getattr(bank, 'types', None))
        if order != 1:
            return [(value, ids, (scope.bitwise_and(1) != 0), torch.ones_like(valid))]
        inventory = getattr(bank, 'column_ids', None)
        if inventory is not None and inventory.numel() == 0:
            # A cold inventory cannot bind. Keep one masked proposal for a
            # pronoun/definite, or the single mint proposal, without expanding
            # a compiled graph into many copies of unavailable candidates.
            permitted = torch.zeros_like(valid) if mode in ('pronoun', 'bind') else valid
            return [(value, torch.full_like(ids, -1), torch.zeros_like(valid), permitted)]
        from ClauseScope import ClauseScope
        P, C = value.shape[1], bank.ids.shape[1]
        live_values, live_ids, live_orders, live_relations, live_positions = live[:5]
        live_local = live[5] if len(live) > 5 else torch.zeros_like(live_ids, dtype=torch.bool)
        earlier = (live_positions[:, None] < pos[:, :, None]) & (live_orders[:, None] == 1)
        earlier = earlier & ~live_local[:, None]
        if inventory is not None:
            earlier = earlier & torch.isin(live_ids, inventory)[:, None]
        earlier = earlier & (live_ids[:, None] != -1) & (live_ids[:, None] != 0)
        own_inventory = bank.ids.flatten() if inventory is None else inventory
        own = torch.isin(ids, own_inventory) & (scope.bitwise_and(ClauseScope.LOCAL) == 0)
        # Only a native column can be its own candidate. A word's dictionary
        # type is never silently promoted to an individual.
        candidate_ids = torch.cat((ids[:, :, None], bank.ids.flip(1)[:, None].expand(B, P, C),
                                   live_ids[:, None].expand(B, P, -1)), -1)
        candidate_values = torch.cat((value[:, :, None], bank.values.flip(1)[:, None].expand(B, P, C, D),
                                      live_values[:, None].expand(B, P, -1, D)), -2)
        candidate_relations = torch.cat(((scope.bitwise_and(1) != 0)[:, :, None],
                                         bank.relations.flip(1)[:, None].expand(B, P, C),
                                         live_relations[:, None].expand(B, P, -1)), -1)
        available = torch.cat((own[:, :, None], bank.valid.flip(1)[:, None].expand(B, P, C), earlier), -1)
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
        options = [(candidate_values[:, :, c], candidate_ids[:, :, c],
                    candidate_relations[:, :, c], available[:, :, c]) for c in range(count)]
        if mode not in ('pronoun', 'bind'):
            options.append((value, torch.full_like(ids, -1), torch.zeros_like(valid), valid))
        return options

    def choice_tag(refs, rule, role):
        order = dict(getattr(rule, 'reference_orders', ())).get(role)
        mode = dict(getattr(rule, 'reference_kinds', ())).get(role)
        determiner = getattr(rule, 'determiner_mode', None)
        if determiner is not None and role == 'I' + str(rule.head_role):
            order, mode = 1, determiner
        if order != 1 or mode == 'kind':
            return torch.full_like(refs, -1)
        # -2 is a comparison-only tag for mint, never a native reference.
        return torch.where(refs == -1, -2, refs)

    lefts, rights, brefs, brels, bvalid, case_weights, binary_ops = [], [], [], [], [], [], []
    bchoices, uchoices = [], []
    case_bank = getattr(bank, 'cases', None)
    case_count = 1 if case_bank is None else len(case_bank.ids)
    for operation, rule in enumerate(rules):
        left_options = propose(window[:, :-1], identities[:, :-1], flags[:, :-1], positions[:, :-1],
                       rule, 'I1', active[:, :-1] & active[:, 1:])
        right_options = propose(window[:, 1:], identities[:, 1:], flags[:, 1:], positions[:, 1:],
                        rule, 'I2', active[:, :-1] & active[:, 1:])
        for left in left_options:
            for right in right_options:
                binary_ops.append(operation)
                lefts.append(left[0])
                rights.append(right[0])
                brefs.append(torch.stack((left[1], right[1]), -1))
                bchoices.append(torch.stack((choice_tag(left[1], rule, 'I1'),
                                             choice_tag(right[1], rule, 'I2')), -1))
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
        for operand in propose(window, identities, flags, positions, rule, 'I1', active):
            unary_ops.append(operation)
            values.append(operand[0])
            urefs.append(torch.stack((operand[1], torch.full_like(operand[1], -1)), -1))
            uchoices.append(torch.stack((choice_tag(operand[1], rule, 'I1'),
                                         torch.full_like(operand[1], -1)), -1))
            urels.append(torch.stack((operand[2], torch.zeros_like(operand[2])), -1))
            uvalid.append(operand[3])

    def stack(items, width, tail=(), *, dtype=None):
        return torch.stack(items, 2) if items else torch.zeros((B, width, 0, *tail), device=window.device, dtype=dtype or window.dtype)
    return dict(binary_ops=tuple(binary_ops), unary_ops=tuple(unary_ops),
                case_bank=case_bank, case_weights=stack(case_weights, N-1, (case_count, 2)), left=stack(lefts, N-1, (D,)), right=stack(rights, N-1, (D,)), unary=stack(values, N, (D,)),
                binary_refs=stack(brefs, N-1, (2,), dtype=torch.long), unary_refs=stack(urefs, N, (2,), dtype=torch.long),
                binary_choices=stack(bchoices, N-1, (2,), dtype=torch.long), unary_choices=stack(uchoices, N, (2,), dtype=torch.long),
                binary_relations=stack(brels, N-1, (2,), dtype=torch.bool), unary_relations=stack(urels, N, (2,), dtype=torch.bool),
                binary_valid=stack(bvalid, N-1, dtype=torch.bool), unary_valid=stack(uvalid, N, dtype=torch.bool))
