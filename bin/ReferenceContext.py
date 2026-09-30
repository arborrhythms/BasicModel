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

    def __iter__(self):
        # The existing predictor reads its three-role context unchanged.
        return iter((self.depth, self.roles, self.occupied))

    def detached(self):
        return replace(self, roles=self.roles.detach().clone(),
                       point=None if self.point is None else self.point.detach().clone())

    @classmethod
    def from_row(cls, row):
        meaning = row['meaning']
        if meaning is None or row.get('row_id', -1) <= 0:
            return None
        return cls(int(meaning.role_mask.sum()), meaning.roles.detach().clone(),
                   meaning.role_mask, row['row_id'],
                   row['np1'].detach().clone() if row['rel_type'] == 0 else None)


def reference_requests(language, actions):
    """Resolve a selected role at its head; the enclosing scope owns its kind."""
    requests, stack = {}, []
    for kind, local, word in actions.detach().cpu().tolist():
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
        for role, order in getattr(rules[local], 'reference_orders', ()):
            for leaf in operands[int(role[1:]) - 1]:
                requests[leaf] = order, modes.get(role)
        head = getattr(rules[local], 'head_role', 0)
        if not head and getattr(rules[local], 'method_name', None) in ('lower', 'bind', 'surface', 'preposition'):
            head = 2
        stack.append(operands[head - 1]
                     if head else tuple(leaf for operand in operands for leaf in operand))
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


def resolve_operand(value, identity, position, *, mode, bank, live, active, forced_reference=None):
    """A hard reference proposal before its numerical grammar operation.

    Only the existing predictor anchors and earlier occupied constituents are
    candidates. The caller masks unavailable proposals before the global
    operation softmax; an unselected pronoun proposal cannot reject a reading.
    """
    live_values, live_ids, live_orders, live_relations, live_positions = live
    B, P, D = value.shape
    C = bank.ids.shape[1]
    K = live_ids.shape[1]
    own_valid = (identity > 0) & (mode != 'pronoun')
    held = bank.valid & bank.predicted[:, None]
    earlier = (live_positions[:, None, :] < position[:, :, None])
    earlier &= (live_ids[:, None, :] > 0) & (live_orders[:, None, :] == 1)
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
    """Resolve each candidate's declared noun roles before evaluating it."""
    B, N, D = window.shape

    def propose(value, ids, scope, pos, rule, role, valid):
        order = dict(getattr(rule, 'reference_orders', ())).get(role)
        mode = dict(getattr(rule, 'reference_kinds', ())).get(role)
        if order is not None:
            value, ids = resolve_order(value, ids, order, getattr(bank, 'types', None))
        if order != 1:
            return value, ids, (scope.bitwise_and(1) != 0), torch.ones_like(valid)
        return resolve_operand(value, ids, pos, mode=mode, bank=bank, live=live, active=valid)
    lefts, rights, brefs, brels, bvalid = [], [], [], [], []
    for rule in rules:
        left = propose(window[:, :-1], identities[:, :-1], flags[:, :-1], positions[:, :-1],
                       rule, 'I1', active[:, :-1] & active[:, 1:])
        right = propose(window[:, 1:], identities[:, 1:], flags[:, 1:], positions[:, 1:],
                        rule, 'I2', active[:, :-1] & active[:, 1:])
        lefts.append(left[0])
        rights.append(right[0])
        brefs.append(torch.stack((left[1], right[1]), -1))
        brels.append(torch.stack((left[2], right[2]), -1))
        bvalid.append(left[3] & right[3])
    values, urefs, urels, uvalid = [], [], [], []
    for rule in unary_rules:
        operand = propose(window, identities, flags, positions, rule, 'I1', active)
        values.append(operand[0])
        urefs.append(torch.stack((operand[1], torch.full_like(operand[1], -1)), -1))
        urels.append(torch.stack((operand[2], torch.zeros_like(operand[2])), -1))
        uvalid.append(operand[3])

    def stack(items, width, tail=(), *, dtype=None):
        return torch.stack(items, 2) if items else torch.zeros((B, width, 0, *tail), device=window.device, dtype=dtype or window.dtype)
    return dict(left=stack(lefts, N-1, (D,)), right=stack(rights, N-1, (D,)), unary=stack(values, N, (D,)),
                binary_refs=stack(brefs, N-1, (2,), dtype=torch.long), unary_refs=stack(urefs, N, (2,), dtype=torch.long),
                binary_relations=stack(brels, N-1, (2,), dtype=torch.bool), unary_relations=stack(urels, N, (2,), dtype=torch.bool),
                binary_valid=stack(bvalid, N-1, dtype=torch.bool), unary_valid=stack(uvalid, N, dtype=torch.bool))
