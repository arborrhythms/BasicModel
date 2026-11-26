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
        # An empty lane contributes zero. Its zero numerator must not be
        # divided by the smallest float: the backward can overflow before
        # multiplication by its zero weights. Nonempty lanes keep their
        # exact weighted mean, with no lower bound on positive mass.
        return ((values * weight).sum(-3) /
                torch.where(mass > 0, mass, 1.)).flatten(-2)

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


def case_addresses(roots, parents, children, written, *, size, limit, rounds):
    """Root order follows requested identities. New rows enter by breadth, then seat."""
    width = min(limit, size)
    valid = roots >= 0
    place = valid.long().cumsum(0) - 1
    ranks = roots.new_full((size + 1,), width)
    ranks.scatter_reduce_(0, torch.where(valid, roots, size), place.clamp(0, width), 'amin')
    ranks = ranks[:size]
    for _ in range(rounds):
        live = (ranks[parents] < width) & written
        reached = ranks.new_zeros(size, dtype=torch.int32)
        reached.scatter_reduce_(0, children, live.int(), 'amax')
        new = (reached > 0) & (ranks == width)
        places = (ranks < width).sum() + new.long().cumsum(0) - 1
        ranks = torch.where(new, places.clamp_max(width), ranks)
    addresses = roots.new_full((width + 1,), -1)
    addresses.scatter_reduce_(0, ranks, torch.arange(size, device=roots.device), 'amax')
    return addresses[:width]


def case_edges(store, device):
    """Structural edges from each definition to the seats held by its symbols."""
    indices, _ = store._indices(device)
    stamp = (id(store.symbol_table), store.symbol_table.revision, id(store._tensor_rows), len(store._tensor_rows))
    cached = store.__dict__.get('_case_edges')
    if cached is None or cached[0] is not indices or cached[1] != stamp:
        edges, complete = [], []
        for edge, (parent, column) in enumerate(zip(store._rows, store._cols, strict=True)):
            _, children, whole = store._thing_spec(column)
            complete.append(whole)
            edges.extend((edge, parent, child) for child in children)
        table = indices.new_tensor(edges).reshape(-1, 3).T.contiguous()
        cached = indices, stamp, table, indices.new_tensor(complete, dtype=torch.bool), len(set(store._rows))
        store.__dict__['_case_edges'] = cached
    return cached[2:]


def case_membership(addresses, topology, weights, *, size, order_zero):
    """Direct signed cases and all cases reached by reversing higher definitions."""
    (edges, parents, children), complete, depth = topology
    count = len(addresses)
    seat = addresses.new_full((size,), count)
    seat.scatter_(0, addresses, torch.arange(count, device=addresses.device))
    parent, child = seat[parents], seat[children]
    # A symbol with even one absent member cannot be read at this binding.
    missing = (~complete).int().scatter_reduce(0, edges, (child == count).int(), 'amax')
    value = weights.detach()[edges]
    live = (parent < count) & (child < count) & (missing[edges] == 0) & (value != 0)
    active = torch.arange(len(edges), device=addresses.device)[live]
    parent, child = parent.index_select(0, active), child.index_select(0, active)
    negative = value.index_select(0, active) < 0
    destinations = (parent * count + child) * 2 + negative
    flat = weights.new_zeros(2 * count * count)
    cases = flat.index_fill(0, destinations, 1.).reshape(count, count, 2)
    identity = torch.eye(count, dtype=weights.dtype, device=addresses.device)
    observed = torch.stack((identity, torch.zeros_like(identity)), -1)
    # A negated entry exchanges the two lanes. Each reached child takes
    # their maximum across incoming entries. The standing bias has no child.
    target = child[:, None, None].expand(-1, count, 2)
    negative = negative[:, None, None]
    higher = (addresses >= order_zero)[None, :, None]
    for _ in range(min(count, 2 * depth + 1) if len(edges) else 0):
        request = (observed * higher).index_select(1, parent).transpose(0, 1)
        incoming = torch.where(negative, request.flip(-1), request)
        observed = observed.transpose(0, 1).scatter_reduce(0, target, incoming, 'amax').transpose(0, 1)
    return cases, observed


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
    required_names = required_ids.to(basis.device).flatten().unique(sorted=True)
    roots = store.concept_rows(torch.where(required_names > 0, required_names, -1))
    topology = case_edges(store, basis.device)
    (edges, parents, children), _, depth = topology
    written = torch.zeros_like(edges, dtype=torch.bool) if store.values is None else store.values.detach()[edges] != 0
    packed = case_addresses(roots, parents, children, written, size=store.nOutput,
                            limit=int(limit), rounds=min(int(limit) - 1, depth))
    addresses = packed[packed >= 0]
    if not len(addresses):
        return CaseBank(torch.full((1,), -1, device=basis.device, dtype=torch.long),
                        basis.new_zeros((1,basis.shape[-1]) if batch is None else (batch,1,basis.shape[-1])),
                        basis.new_zeros(1, 1, 2), basis.new_zeros(1, 1, 2))
    from FieldProjection import seated_names
    names, _ = seated_names(allocator, store.nOutput, addresses)
    identities = names[addresses]
    identities = torch.where(identities != 0, identities, -1)
    codes = space.similarity_codebook.lookup_rows(addresses if batch is None else
        addresses[None].expand(batch,-1), streams=batch is not None).clone()
    from CandidateGrowth import visible_store
    projected = visible_store(store)
    if projected is not store:
        topology = case_edges(projected, basis.device)
    weights = basis.new_empty(0) if projected.values is None else projected.values.to(basis)
    cases, observed = case_membership(addresses, topology, weights, size=store.nOutput,
                                      order_zero=space._order_caps()[0])
    required = required_ids.reshape(1, -1) if required_ids.ndim < 2 else required_ids.reshape(required_ids.shape[0], -1)
    eligible = ((identities[None, :, None] == required[:, None].to(identities.device))
                & (identities[None, :, None] > 0)).any(-1)
    return CaseBank(identities, codes, cases, observed, eligible)


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
        accessible = resolver.individuals(resolver.context_records(references))
        for identity, record in accessible.items():
            if inspected >= limit:
                break
            token = owner.mtm.entries.get(record.get('token') or identity)
            if token is None:
                continue
            for event in reversed(token.events):
                if inspected >= limit:
                    break
                inspected += 1
                targets = tuple(index for index, symbol in heads.items() if symbol == event.verb)
                key = (b, event.occurrence, event.verb)
                if (not targets or key in seen or event.role != 0
                        or event.before is None):
                    continue
                change = point(owner, event.verb)
                source = owner.mtm.idea_store(event.occurrence, store)
                row = source.location(event.occurrence)
                if change is None or row is None:
                    continue
                record = source.row(row)
                from ThoughtReferences import open_slots
                if open_slots(record['meaning']):
                    continue
                values = (event.before, change, owner.mtm.event_value(event))
                # The two signed rows of one conclusion are witnesses of
                # the same cases. Case membership joins their lanes, without
                # spending another three seats on identical case values.
                previous = next((start for start in range(0, len(additions), 3)
                    if all(additions[start+i][0] == b and additions[start+i][1] == targets
                           and torch.equal(additions[start+i][2], value)
                           for i, value in enumerate(values))), None)
                if previous is None and remaining < 3:
                    continue
                seen.add(key)
                for target in targets:
                    # The accessible record selected this collection. The
                    # event row has no token, and its symbol may name several.
                    current_values[b, identity, target] = owner.mtm.value_of(token)
                pair = bank.codes.new_tensor(record['evidence'])
                if previous is None:
                    for value in values:
                        additions.append((b, targets, value, pair.clone()))
                    remaining -= 3
                else:
                    for index in range(previous, previous+3):
                        additions[index][3].copy_(torch.maximum(additions[index][3], pair))
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
