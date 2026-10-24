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
    reference: torch.Tensor | None = None

    def view(self, head):
        """Read the supplied head's cases without selecting a phase."""
        match = (head[..., None] == self.ids) & (head[..., None] > 0)
        index = match.long().argmax(-1)
        if self.cases.ndim == 4:
            batch = torch.arange(head.shape[0], device=head.device)
            batch = batch.reshape(-1, *([1] * (head.ndim-1)))
            weights = self.cases[batch, index]
        else:
            weights = self.cases[index]
        weights = weights * match.any(-1)[..., None, None]
        shape = (*head.shape, self.codes.shape[-1])
        reference = self.codes if self.reference is None else self.reference
        if reference.ndim == 3:
            batch = torch.arange(head.shape[0], device=head.device).reshape(-1, *([1]*(head.ndim-1)))
            reference = reference[batch, index]
        else:
            reference = reference[index]
        reference = reference * match.any(-1)[..., None]
        return CaseView(self.codes, weights, self.codes.new_ones(shape),
                        torch.zeros_like(head, dtype=torch.bool), reference)

    def for_values(self, values, *, form_width):
        """The free inverse identifies a unique primed form, without a trace.

        Ambiguous or unspelled candidates have no named case definition.
        This correspondence supplies cases only; it never establishes an
        individual's identity or adds a binding candidate.
        """
        codes = self.codes if self.codes.ndim == 3 else self.codes[None]
        codes = codes.reshape(codes.shape[0], *([1]*(values.ndim-2)), *codes.shape[-2:])
        error = (values[..., None, :form_width]-codes[..., :form_width]).square().sum(-1)
        tolerance = 8*torch.finfo(values.dtype).eps*values[..., :form_width].square().sum(-1).clamp_min(1.)
        match = (error <= tolerance[..., None]) & (self.ids > 0)
        head = self.ids[match.long().argmax(-1)]
        return self.view(torch.where(match.sum(-1) == 1, head, -1))

    def detached(self):
        """Answer reads freeze all staged dictionary keys and memberships."""
        return CaseBank(*(value.detach() if torch.is_tensor(value) else value for value in self))

    def for_stream(self, b):
        """The same named cases with one stream's occurrence-context codes."""
        return self._replace(codes=self.codes[b] if self.codes.ndim == 3 else self.codes,
            cases=self.cases[b] if self.cases.ndim == 4 else self.cases,
            eligible=None if self.eligible is None else self.eligible[b:b+1],
            reference=self.reference[b] if self.reference is not None and self.reference.ndim == 3 else self.reference)


class SelectedCases(NamedTuple):
    codes: torch.Tensor
    weights: torch.Tensor
    available: torch.Tensor


class CaseView(NamedTuple):
    """An open head's cases and its accumulated content test.

    The bank owns case values; the parse carries only membership and a
    coordinatewise test. Reapplying a test is an intersection, not another
    gain. Neither a row clock nor a lexical classification enters this view.
    """
    codes: torch.Tensor
    weights: torch.Tensor
    test: torch.Tensor
    selected: torch.Tensor
    reference: torch.Tensor | None = None

    def select(self, test):
        return self._replace(test=torch.minimum(self.test, test),
                             selected=torch.ones_like(self.selected))

    def origin(self, *, meaning_start):
        """The head's unfiltered field is stable across repeated selections."""
        return self._replace(test=torch.ones_like(self.test)).fold(meaning_start=meaning_start)

    def fold(self, *, meaning_start):
        """Refold two evidence lanes without subtracting one from the other."""
        values = self.codes[..., meaning_start:]
        test = self.test[..., meaning_start:]
        # Codes may be shared by a batch or by every candidate position.
        while values.ndim < test.ndim + 1:
            values = values.unsqueeze(-3)
        values = torch.minimum(values, test[..., None, :]).unflatten(-1, (2, -1))
        weights = self.weights * values.ne(0).any(-1)
        weight = weights[..., None]
        mass = weights.sum(-2)[..., None]
        return ((values * weight).sum(-3) /
                mass.clamp_min(torch.finfo(values.dtype).tiny)).flatten(-2)

    def coverage(self, request, *, meaning_start):
        """An unselected head supplies no answer about a particular case."""
        from IdentityRows import meet_ratios
        value = self.fold(meaning_start=meaning_start).unflatten(-1, (2, -1))
        wanted = request[..., meaning_start:].unflatten(-1, (2, -1))
        _, fraction = meet_ratios(wanted, value)
        return torch.where(self.selected[..., None], fraction, 0.)


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
    folded = (torch.einsum('b...c,bcd->b...d', signed, selected.codes)
              if selected.codes.ndim == 3 else signed @ selected.codes)
    return folded / mass


def stage_case_bank(space, required_ids, *, limit):
    """Snapshot a bounded reachable sigma view, with no concept admission.

    This is eager owner-side staging, like the primed symbol bank. Only the
    selected native definitions and their existing children can enter it.
    The compiled grammar body receives the four tensors, never the owner.
    """
    if int(limit) < 1:
        raise ValueError('case staging requires a positive existing candidate limit')
    if required_ids is None:
        required_ids=torch.empty(0,dtype=torch.long,device=space.similarity_codebook.W.device)
    allocator = space._concept_allocator
    store = allocator.layer(0)
    basis = space.similarity_codebook.W
    batch = required_ids.shape[0] if required_ids.ndim >= 2 else None
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
                        basis.new_zeros((1,basis.shape[-1]) if batch is None else (batch,1,basis.shape[-1])),
                        basis.new_zeros(1, 1, 2), basis.new_zeros(1, 1, 2))
    addresses = torch.tensor(rows, dtype=torch.long, device=basis.device)
    identities = torch.tensor([space.concept_id_at_row(row) or -1 for row in rows],
                              dtype=torch.long, device=basis.device)
    codes = space.similarity_codebook.lookup_rows(addresses if batch is None else
        addresses[None].expand(batch,-1), streams=batch is not None).clone()
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


def stage_event_cases(bank, resolver, frames, *, limit):
    """Add accessible before/change/after fields within the same case bound.

    A change is the retained verb's conceptual operator, whose diagonal the
    action reads. It is not a subtraction of states. The snapshots belong
    to MTM events; this view writes neither definitions nor durable rows.
    """
    from IdentityRows import point
    owner, store = resolver.components.owner, resolver.store
    heads = {index: resolver._event_symbol(int(identity)) for index, identity in enumerate(bank.ids)
             if int(identity) > 0}
    batch, size = len(frames), len(bank.ids)
    remaining = max(0, int(limit)-size)
    additions, seen, current_values = [], set(), {}
    for b, references in enumerate(frames):
        inspected = 0
        accessible = resolver.individuals(resolver.accessible(references))
        for identity, record in accessible.items():
            if remaining < 3 or inspected >= limit:
                break
            token = owner.mtm.entries.get(record.get('token') or identity)
            if token is None:
                continue
            for event in reversed(token.events):
                if remaining < 3 or inspected >= limit:
                    break
                inspected += 1
                targets = tuple(index for index, symbol in heads.items() if symbol == event.verb)
                key = (b, event.occurrence, event.verb)
                if (remaining < 3 or not targets or key in seen or event.role != 0
                        or event.before is None or event.after is None):
                    continue
                change = point(owner, event.verb)
                row = store.location(event.occurrence)
                if change is None or row is None:
                    continue
                if store.KINDS[int(store.record_kind[row])] in ('cover', 'estimate', 'question'):
                    continue
                seen.add(key)
                for target in targets:
                    current_values[b, identity, target] = token.value
                pair = bank.codes.new_tensor((store.c_plus[row], store.c_minus[row]))
                for value in (event.before, change, event.after):
                    additions.append((b, targets, value, pair))
                remaining -= 3
    if not additions:
        return bank
    count, width = size+len(additions), bank.codes.shape[-1]
    codes = bank.codes.new_zeros(batch, count, width)
    codes[:, :size] = bank.codes
    reference = codes.clone()
    for b, target in {(b, target) for b, _, target in current_values}:
        current = [value for (row, _, head), value in current_values.items() if row == b and head == target]
        reference[b, target] = torch.stack(current).mean(0).to(reference)
    cases = bank.codes.new_zeros(batch, count, count, 2)
    cases[:, :size, :size] = bank.cases
    observed = bank.codes.new_zeros(count, count, 2)
    observed[:size, :size] = bank.observed
    eligible = torch.zeros(batch, count, dtype=torch.bool, device=bank.ids.device)
    eligible[:, :size] = bank.ids > 0 if bank.eligible is None else bank.eligible
    for column, (b, targets, value, pair) in enumerate(additions, size):
        codes[b, column, :min(width, value.numel())] = value[:width].to(codes)
        for target in targets:
            cases[b, target, column] = pair
    # Negative keys are local bank indices, never native symbols or binding
    # candidates. Only the existing primed native heads/modifiers are eligible.
    ids = torch.cat((bank.ids, -2-torch.arange(len(additions), device=bank.ids.device)))
    return CaseBank(ids, codes, cases, observed, eligible, reference)


def search_cases(bank, parent, *, allowed_ids=None):
    """Search only primed operand pairs, using the same forward case fold.

    Neither a compose action nor an occurrence witness enters this inverse.
    An unavailable pair stays unavailable rather than returning a pseudo-split.
    """
    count = len(bank.ids)
    batched = bank.codes.ndim == 3
    selection = select_cases(bank, bank.ids[None,:,None] if batched else bank.ids[:,None],
                             bank.ids[None,None,:] if batched else bank.ids[None,:])
    candidates = fold_cases(selection).reshape(-1,count*count,parent.shape[-1])
    allowed = bank.eligible
    if allowed_ids is not None:
        allowed = (allowed_ids[:, :, None] == bank.ids[None, None]).any(1) & (bank.ids[None] > 0)
    if allowed is None:
        allowed = (bank.ids > 0)[None].expand(parent.shape[0], count)
    valid = (allowed[:, :, None] & allowed[:, None, :]
             & (selection.available if batched else selection.available[None])).reshape(parent.shape[0], -1)
    available = valid.any(-1)
    costs = (parent[:, None] - candidates).square().mean(-1)
    safe = valid | (~available[:, None] & (torch.arange(count * count, device=parent.device)[None] == 0))
    scores = (-costs).masked_fill(~safe, -torch.inf)
    probability = scores.softmax(-1)
    best = scores.argmax(-1)
    weight = probability.gather(1, best[:, None])
    credit = 1 + weight - weight.detach()
    codes = bank.codes if batched else bank.codes[None].expand(parent.shape[0],-1,-1)
    rows = torch.arange(parent.shape[0],device=parent.device)
    modifier = codes[rows,best // count] * credit
    head = codes[rows,best % count] * credit
    return (torch.where(available[:, None], modifier, 0.),
            torch.where(available[:, None], head, 0.), available)


def inverse_unary(bank, parent, operation, *, whole_bank=None):
    """Search a one-operand face without a saved marker or compose witness."""
    candidates = bank.codes if bank.codes.ndim == 3 else bank.codes[None].expand(parent.shape[0], -1, -1)
    ids = bank.ids[None].expand(parent.shape[0], -1)
    view = bank.view(ids)
    valid = (ids > 0) & (view.weights > 0).any(-1).any(-1)
    if bank.eligible is not None:
        valid = valid & bank.eligible
    folded = operation.compose(candidates, case_view=view)
    source = parent
    if whole_bank is not None:
        from FormCentroids import unary
        source = unary(source, whole_bank)
    error = ((folded - source[:, None]).square().mean(-1) /
             source.square().mean(-1).clamp_min(torch.finfo(source.dtype).tiny)[:, None])
    index = error.masked_fill(~valid, torch.inf).argmin(-1)
    result = candidates.gather(1, index[:, None, None].expand(-1, 1, parent.shape[-1]))[:, 0]
    available = valid.any(-1)
    return torch.where(available[:, None], result, 0.), available
