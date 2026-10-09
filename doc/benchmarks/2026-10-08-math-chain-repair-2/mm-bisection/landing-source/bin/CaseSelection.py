"""Bounded native case selection for grammatical subtyping.

The owner stages existing sigma definitions once per reading. Selection uses
native references and paired evidence; codes are only the content refolded
from the retained cases. There are no parameters, allocations or readers in
the tensor step.
"""
from typing import NamedTuple

import torch


class CaseBank(NamedTuple):
    ids: torch.Tensor
    codes: torch.Tensor
    cases: torch.Tensor
    observed: torch.Tensor
    eligible: torch.Tensor | None = None

    def detached(self):
        """Answer reads freeze all staged dictionary keys and memberships."""
        return CaseBank(*(value.detach() if torch.is_tensor(value) else value for value in self))


class SelectedCases(NamedTuple):
    codes: torch.Tensor
    weights: torch.Tensor
    available: torch.Tensor


def select_cases(bank, modifier, head):
    """Reverse the named head's sigma, conditioned on the modifier's field."""
    mod_match = (modifier[..., None] == bank.ids) & (modifier[..., None] > 0)
    head_match = (head[..., None] == bank.ids) & (head[..., None] > 0)
    mod = bank.observed[mod_match.long().argmax(-1)]
    requested = bank.cases[head_match.long().argmax(-1)]
    known = mod_match.any(-1) & head_match.any(-1)
    weights = torch.minimum(requested, mod) * known[..., None, None]
    available = known & (weights > 0).any(-1).any(-1)
    return SelectedCases(bank.codes, weights, available)


def fold_cases(selected):
    """Read the retained sigma cases as one bounded conceptual point.

    Opposite poles share a code with opposite signs. A single weak witness
    keeps its magnitude; multiple cases pool without leaving the code chart.
    """
    positive, negative = selected.weights.unbind(-1)
    signed = positive - negative
    mass = (positive + negative).sum(-1, keepdim=True).clamp_min(1.)
    return (signed @ selected.codes) / mass


def stage_case_bank(space, required_ids, *, limit):
    """Snapshot a bounded reachable sigma view, with no concept admission.

    This is eager owner-side staging, like the primed symbol bank. Only the
    selected native definitions and their existing children can enter it.
    The compiled grammar body receives the four tensors, never the owner.
    """
    if int(limit) < 1:
        raise ValueError('case staging requires a positive existing candidate limit')
    if required_ids is None:
        required_ids=torch.empty(0,dtype=torch.long,device=space.similarity_codebook.getW().device)
    allocator = space._concept_allocator
    store = allocator.layer(0)
    basis = space.similarity_codebook.getW()
    roots = sorted(set(int(v) for v in required_ids.detach().reshape(-1).cpu().tolist() if int(v) > 0))
    rows = []
    for identity in roots:
        row = space._csw_row_of(identity)
        if row is not None and row not in rows:
            rows.append(row)
    rows = rows[:int(limit)]
    written = [] if store.values is None else (store.values.detach() > 0).cpu().tolist()
    edges = [(target, source % (store.nOutput + 1))
             for (target, source), present in zip(zip(store._rows, store._cols), written)
             if present and source % (store.nOutput + 1) < store.nOutput]
    for _ in range(int(limit)):
        additions = sorted({source for target, source in edges if target in rows and source not in rows})
        if not additions or len(rows) == int(limit):
            break
        rows.extend(additions[:int(limit) - len(rows)])
    if not rows:
        return CaseBank(torch.full((1,), -1, device=basis.device, dtype=torch.long),
                        basis.new_zeros(1, basis.shape[-1]),
                        basis.new_zeros(1, 1, 2), basis.new_zeros(1, 1, 2))
    addresses = torch.tensor(rows, dtype=torch.long, device=basis.device)
    identities = torch.tensor([space.concept_id_at_row(row) or -1 for row in rows],
                              dtype=torch.long, device=basis.device)
    codes = space.similarity_codebook.lookup_rows(addresses).clone()
    count = len(rows)
    identity = torch.eye(count, device=basis.device, dtype=basis.dtype)
    matrix = store.bind(rows)
    # Supplying an observed field makes attribution retain every written
    # case. No unconditioned strongest-case choice runs during staging.
    source = matrix.attribute_presence(identity, observed=basis.new_ones(2 * (count + 1), count))
    cases = torch.stack((source[:count].t(), source[count + 1:2 * count + 1].t()), -1)
    request = torch.stack((identity, torch.zeros_like(identity)), -1).unsqueeze(2)
    observed = space.cs_reverse_presence(request, observed=torch.ones_like(request), inventory_rows=addresses)
    required = required_ids.reshape(1, -1) if required_ids.ndim < 2 else required_ids.reshape(required_ids.shape[0], -1)
    eligible = ((identities[None, :, None] == required[:, None].to(identities.device))
                & (identities[None, :, None] > 0)).any(-1)
    return CaseBank(identities, codes, cases, observed[:, :, 0].permute(1, 0, 2), eligible)


def search_cases(bank, parent, *, allowed_ids=None):
    """Search only primed operand pairs, using the same forward case fold.

    Neither a compose action nor an occurrence witness enters this inverse.
    An unavailable pair stays unavailable rather than returning a pseudo-split.
    """
    count = len(bank.ids)
    selection = select_cases(bank, bank.ids[:, None], bank.ids[None, :])
    candidates = fold_cases(selection).reshape(count * count, -1)
    allowed = bank.eligible
    if allowed_ids is not None:
        allowed = (allowed_ids[:, :, None] == bank.ids[None, None]).any(1) & (bank.ids[None] > 0)
    if allowed is None:
        allowed = (bank.ids > 0)[None].expand(parent.shape[0], count)
    valid = (allowed[:, :, None] & allowed[:, None, :]
             & selection.available[None]).reshape(parent.shape[0], -1)
    available = valid.any(-1)
    costs = (parent[:, None] - candidates[None]).square().mean(-1)
    safe = valid | (~available[:, None] & (torch.arange(count * count, device=parent.device)[None] == 0))
    scores = (-costs).masked_fill(~safe, -torch.inf)
    probability = scores.softmax(-1)
    best = scores.argmax(-1)
    weight = probability.gather(1, best[:, None])
    credit = 1 + weight - weight.detach()
    modifier = bank.codes[best // count] * credit
    head = bank.codes[best % count] * credit
    return (torch.where(available[:, None], modifier, 0.),
            torch.where(available[:, None], head, 0.), available)
