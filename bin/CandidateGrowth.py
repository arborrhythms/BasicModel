"""Implicit growth of the Pi above order zero and of the store's Sigma, by
candidate pools kept with MTM (:class:`Memory.MTMPairRows`,
:class:`Memory.MTM1`).

At each look, in training:

* the Pi above order zero: concepts of one order present together at the
  look offer a candidate row over the pair (a symbol of the pair); a
  candidate held above the penalty becomes a row of the pair
  (:func:`pair_rows`);
* the Sigma: a kind expected at the look by the spread and not read there,
  with a thing present at the look, offers a candidate entry of the kind on
  that thing; held above the penalty it is written as the kind's entry. A
  thing read at another look is never offered.

A look's offers are read on the device over what is present at it: the
bound concepts and the things all of whose members they are, each at a
fixed width. What holds between looks (a pair already a row, an entry
already in the store, an order, a symbol holding both) is read for those
from the promoted rows, the store's indices and the symbols' members.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import copy

import torch
from torch.nn import functional as F


#: Presence at or above which a concept or a thing is present at a look.
PRESENT = .5


_current_trial = ContextVar('candidate_growth_trial', default=None)


class GrowthTrial:
    """Provisional observations and sparse promotions of one batched reading."""

    def __init__(self, owner, *, path=None, stream=None):
        self.owner = owner
        self.path, self.stream = path, stream
        self.store = owner._concept_allocator.layer(0)
        self.candidates = None
        self.edges = {}
        self.streams = {}
        self.looks = []
        self.expected = None
        self.costs = {'_expectation_looks': [], '_expectation_percept_looks': []}

    def from_reading(self, *, path=None):
        """An output-only departure inherits the completed reading by value."""
        other = GrowthTrial(self.owner, path=path, stream=self.stream)
        other.streams = {b: stream.from_reading(path=path) for b, stream in self.streams.items()}
        other.expected = self.expected
        other.looks = self.looks.copy()
        other.edges = self.edges.copy()
        other.costs = {name: entries.copy() for name, entries in self.costs.items()}
        if self.candidates is not None:
            other.candidates = other.fork_matrices(self.candidates)
        return other

    def fork_matrices(self, held):
        return dict(pi=self.owner.mtm.candidate_relations(self.path, self.stream, source=held['pi'], pairs=True),
                    sigma=self.owner.mtm.candidate_relations(self.path, self.stream, source=held['sigma']))

    @contextmanager
    def active(self, model=None, batch=None):
        token = _current_trial.set(self)
        if model is None:
            try:
                yield self
            finally:
                _current_trial.reset(token)
            return
        field = self.owner._expected_knowing[:batch]
        opening = field.detach().clone()
        self.opening_forecast = opening if self.expected is None else self.expected
        prior = {name: getattr(model, name, None) for name in self.costs}
        try:
            if self.expected is not None:
                field.copy_(self.expected)
            for name, terms in self.costs.items():
                setattr(model, name, terms)
            yield self
            self.expected = field.detach().clone()
        finally:
            field.copy_(opening)
            self.opening_forecast = None
            for name, original in prior.items():
                setattr(model, name, original)
            _current_trial.reset(token)

    @staticmethod
    def keep_forecasts(model, trials, selected, indices=None):
        # An output-only departure uses the greedy reading's same trial.
        streams = torch.arange(len(selected), device=selected.device)
        owner = trials[0].owner
        owner._expected_knowing[:len(selected)].copy_(
            torch.stack([trial.expected for trial in trials], 1)[streams, selected])
        indices = selected.tolist() if indices is None else indices
        retained = {trials[index] for index in indices}
        for name in trials[0].costs:
            terms = []
            for trial in dict.fromkeys(trials):
                if trial not in retained:
                    continue
                winners = torch.zeros_like(selected, dtype=torch.bool)
                for index, candidate in enumerate(trials):
                    if candidate is trial:
                        winners |= selected == index
                terms.extend((torch.where(winners, cost, 0.), mask & winners)
                             for cost, mask in trial.costs[name])
            setattr(model, name, terms)

    def projected_store(self):
        # Keep the optimizer's registered Parameter and sparse metadata intact.
        # Earlier weights retain their gradient through the concatenation. New
        # edges become parameters only after the retained sentence's update.
        store = self.store
        added = [edge for edge in self.edges if edge not in store._index]
        if not added:
            return store
        view = copy(store)
        view._rows = store._rows + [r for r, _ in added]
        view._cols = store._cols + [c for _, c in added]
        view._index = {**store._index, **{edge: store.nnz + i for i, edge in enumerate(added)}}
        view._dev_cache = None
        weight = store.values
        additions = torch.ones(len(added), device=store._device(),
                               dtype=weight.dtype if weight is not None else torch.get_default_dtype())
        object.__setattr__(view, 'values', additions if weight is None else torch.cat((weight, additions)))
        return view

    def promote(self, kind, member):
        owner = self.owner
        row, source = owner._concept_row_of(kind), owner._concept_row_of(member)
        if row is None or source is None or owner._row_order(source) >= owner._row_order(row):
            return None
        if owner.reads_concept(row, source) or (getattr(owner, '_frozen_concepts', None) and row in owner._frozen_rows()):
            return row
        # The candidate came from an existing symbol over this member.
        for thing in owner.things_reading(row, source, make=False):
            owner._check_thing_order(row, thing)
            self.edges[row, self.store.column_of_thing(thing)] = None
        return row

    @staticmethod
    @torch.no_grad()
    def commit(trials, selected):
        """Replay observations in look/stream order, including mixed winners."""
        indices = selected.tolist() if torch.is_tensor(selected) else selected
        winners = [trials[index] for index in indices]
        turns = max((len(trial.looks) for trial in winners), default=0)
        if not turns:
            return
        owner = winners[0].owner
        pool = getattr(owner.mtm, '_candidate_pairs', None)
        together = pool is not None and all(trial.owner is owner for trial in winners)
        releases = pool.reserve_publication(turns * len(winners)) if together else None
        written = 0
        try:
            for turn in range(turns):
                events = [(batch, trial.looks[turn][batch]) for batch, trial in enumerate(winners)
                          if turn < len(trial.looks) and batch in trial.looks[turn]]
                offers = retained_pair_offers(owner, pool, events) if together else {}
                things = retained_things(owner, events) if together else {}
                if offers or things:
                    alloc = owner._concept_allocator
                    table, store = alloc.symbol_table(), alloc.layer(0)
                    stamp = table.revision, len(store._tensor_row_keys)
                for batch, (present, sentence, expected) in events:
                    trial = winners[batch]
                    if releases is None:
                        pi_step(trial.owner, sentence)
                    else:
                        pending = []
                        # A preceding Sigma promotion can create a new symbol.
                        # Re-read the layout if its membership has changed.
                        if (offers or things) and stamp != (table.revision, len(store._tensor_row_keys)):
                            offers.clear()
                            things.clear()
                        pi_step(owner, sentence, collect=pending, prior=releases[:written], offers=offers.get(batch))
                        if pending:
                            releases[written].copy_(pending[0][1])
                            written += 1
                    if batch in things:
                        sigma_step(trial.owner, present, expected, things_read=things[batch])
                    else:
                        sigma_step(trial.owner, present, expected)
        finally:
            if written:
                # Only the spread reads these pairs. During replay its
                # forecasts are already fixed. Later offers in the replay
                # still see each earlier promotion in the device slab.
                promoted = releases[:written].flatten(0, 1)
                count, number = torch.stack((pool.count, (promoted[:, 0] >= 0).sum())).tolist()
                pool._remember_count(count)
                pool.append_pairs(promoted, number)


def row_scope(owner, batch):
    trial = _current_trial.get()
    if trial is None or trial.owner is not owner or not trial.streams:
        return nullcontext()
    stream = trial.streams.get(batch)
    return (stream if stream is not None else GrowthTrial(owner)).active()


def read_streams(owner, read, *, axis, live=None, seats=None):
    """Use each stream's own provisional topology, sharing equal topologies.

    The callback returns independent batch rows. A compacted forecast supplies
    its physical stream mask so its first live row keeps its original owner.
    """
    trial = _current_trial.get()
    if trial is None or trial.owner is not owner or not trial.streams:
        return read()
    base = getattr(getattr(owner, 'mtm', None), '_candidate_pairs', None)
    base = 0 if base is None else base.pair_count
    groups = {}
    for b, stream in trial.streams.items():
        # A stream with its own promoted pairs reads alone.
        held = (stream.candidates or {}).get('pi')
        key = (tuple(stream.edges), b if held is not None and held.pair_count > base else None)
        if key != ((), None):
            groups.setdefault(key, []).append(b)
    result = read()
    for members in groups.values():
        with trial.streams[members[0]].active():
            local = read()
        indices = torch.tensor(members, device=result.device)
        if live is None:
            mask = torch.zeros(result.shape[axis], dtype=torch.bool, device=result.device)
            mask[indices] = True
        else:
            mask = torch.zeros_like(live)
            mask[indices] = True
            mask = mask[live] if seats is None else mask.index_select(0, seats)
        shape = [1] * result.ndim
        shape[axis] = -1
        result = torch.where(mask.view(shape), local, result)
    return result


def speculative(owner):
    trial = _current_trial.get()
    return trial is not None and trial.owner is owner


def visible_store(store):
    trial = _current_trial.get()
    return trial.projected_store() if trial is not None and trial.store is store and trial.edges else store


def pair_pool(owner):
    """The Pi candidates whose promoted pairs the current trial reads."""
    trial = _current_trial.get()
    if trial is not None and trial.owner is owner and trial.candidates is not None:
        return trial.candidates['pi']
    return getattr(getattr(owner, 'mtm', None), '_candidate_pairs', None)


def visible_pairs(owner):
    """The promoted pairs ``[P, 2]`` on the device, or ``None``."""
    pool = pair_pool(owner)
    return None if pool is None else pool.promoted_pairs()


def matrices(owner):
    """The candidate matrices of ``owner``'s MTM, made on first use, each
    sized by ``architecture.candidateCapacity``."""
    pool = owner.mtm
    held = getattr(pool, 'candidates', None)
    trial = _current_trial.get()
    if trial is not None and trial.owner is owner:
        if trial.candidates is not None:
            return trial.candidates
        if held is not None:
            trial.candidates = trial.fork_matrices(held)
            return trial.candidates
    else:
        trial = None
    if held is None:
        if not hasattr(pool, '_candidate_relations'):
            pool.reserve_relation_trials(0, 0, owner.similarity_codebook.W.device)
        sigma = (pool.candidate_relations() if trial is None else pool.candidate_relations(
            trial.path, trial.stream, source=pool._candidate_relations))
        pi = (pool.candidate_relations(pairs=True) if trial is None else pool.candidate_relations(
            trial.path, trial.stream, source=pool._candidate_pairs, pairs=True))
        held = dict(pi=pi, sigma=sigma)
        if trial is None:
            pool.candidates = held
        else:
            trial.candidates = held
    return held


def presences(field):
    """Stage one sparse batch of positive readings for the host candidate tables."""
    ids = field.concept_ids
    if ids.ndim == 1:
        ids = ids[:, None].expand(-1, field.evidence.shape[1])
    lanes = field.evidence.detach().amax(2)
    values = (lanes[..., 0] - lanes[..., 1]).clamp_min(0)
    row, batch = ((ids >= 0) & (values > 0)).nonzero(as_tuple=True)
    places = torch.stack((batch, ids[row, batch]), -1).tolist()
    result = [{} for _ in range(field.evidence.shape[1])]
    for (b, cid), value in zip(places, values[row, batch].tolist(), strict=True):
        result[b][cid] = value
    return result


def field_levels(concepts, evidence, ids, active):
    """Gather each named concept's last positive observation and its first location."""
    if concepts.ndim == 1:
        concepts = concepts[:, None].expand(-1, evidence.shape[1])
    lanes = evidence.detach().amax(2)
    levels = (lanes[..., 0] - lanes[..., 1]).clamp_min(0).T
    return named_levels(concepts, levels, ids, active)


def named_levels(concepts, levels, ids, active):
    """Gather positive levels by name, retaining the first and last witness."""
    B, R = levels.shape
    keys = concepts.T.contiguous()
    index = torch.searchsorted(ids, keys)
    valid = (index < len(ids)) & (ids[index.clamp_max(len(ids) - 1)] == keys)
    valid &= (levels > 0) & active[:, None]
    index = torch.where(valid, index, len(ids))
    place = torch.arange(R, device=ids.device).expand(B, R)
    earliest = ids.new_full((B, len(ids) + 1), R).scatter_reduce(1, index, place, 'amin')[:, :-1]
    latest = ids.new_full((B, len(ids) + 1), -1).scatter_reduce(1, index, place, 'amax')[:, :-1]
    padded = torch.nn.functional.pad(levels, (0, 1))
    values = padded.gather(1, torch.where(latest >= 0, latest, R))
    return values, earliest


def binding_names(ids, width):
    """The bound concepts ``ids`` (``-1`` for none), each once in ascending
    order, padded to ``width`` with the largest id."""
    top = torch.iinfo(ids.dtype).max
    flat = torch.where(ids >= 0, ids, top).reshape(-1).sort().values
    first = torch.ones_like(flat, dtype=torch.bool)
    first[1:] = flat[1:] != flat[:-1]
    first &= flat != top
    place = torch.where(first, first.long().cumsum(0) - 1, width)
    return flat.new_full((width + 1,), top).scatter_(0, place, flat)[:width]


def field_readings(owner, field, active, binding):
    """Use the fixed candidate inventory for a small field.

    The field's binding supplies its possible names, each once: the width
    of every reading of the look. Evidence and page activity select the
    observations. The MTM tally capacity bounds the speculative pair grid.
    """
    if binding is None:
        return None
    if len(binding) > owner.mtm._candidate_pairs.binding_width:
        return None
    if torch.is_tensor(binding) and len(binding):
        names = binding
    else:
        names = binding_names(owner._cs_field_concept_ids, max(1, len(binding)))
    if active is None:
        active = torch.ones(field.evidence.shape[1], dtype=torch.bool, device=names.device)
    values, ranks = field_levels(field.concept_ids, field.evidence, names, active)
    return [(names, row, rank, len(binding)) for row, rank in zip(values.unbind(), ranks.unbind(), strict=True)]


def dict_readings(held, present, *, places=False):
    """A reading ``{cid: value}`` as a bound one: names ascending, each
    one's value and, with ``places``, the place of its first reading (its
    order in ``present``)."""
    names = sorted(present)
    read = held.refs.new_tensor(names), held.activity.new_tensor([present[cid] for cid in names])
    if not places:
        return read
    first = {cid: place for place, cid in enumerate(present)}
    return *read, held.refs.new_tensor([first[cid] for cid in names])


def presence(owner, field, batch):
    """The positive readings of one stream, for callers outside the look loop."""
    return presences(field)[batch]


def unpromoted_offers(offers, prior):
    """Remove earlier promotions from the same retained sentence's offers."""
    previous = prior.reshape(-1, 2)
    if not len(offers) or not len(previous):
        return offers
    block = max(1, (1 << 18) // len(previous))
    blocks = []
    for pairs in offers.split(block):
        used = (pairs[:, None] == previous).all(-1).any(-1)
        blocks.append(torch.where(used[:, None], -1, pairs))
    return torch.cat(blocks)


def seat_of(names, keys):
    """Each of ``keys``' place in ``names`` (ascending), ``len(names)`` where
    it is not there."""
    place = torch.searchsorted(names, keys.clamp_min(0))
    return torch.where((keys >= 0) & (F.pad(names, (0, 1), value=-1)[place] == keys), place, len(names))


def pair_layout(owner, held, names):
    """The unpromoted pairs of like order for one binding of native names."""
    symbols = owner.mtm._candidate_relations
    symbols.read_symbols(owner)
    # Pairs in ascending order of their two names, each read where both
    # concepts are present, of one order, held together by no symbol and
    # not yet a row.
    alloc = owner._concept_allocator
    store = alloc.layer(0)
    width = len(names)
    seats = torch.triu_indices(width, width, 1, device=names.device).T
    ends = store.concept_rows(names)
    order = torch.where((ends >= 0) & (names > 0), symbols.orders[ends.clamp_min(0)], -1)
    blocked = torch.zeros(width + 1, width + 1, dtype=torch.bool, device=names.device)
    member = seat_of(names, symbols.members[:len(alloc.symbol_table())])
    blocked[member[:, :, None].expand(-1, -1, member.shape[1]),
            member[:, None, :].expand(-1, member.shape[1], -1)] = True
    promoted = seat_of(names, held.promoted_pairs())
    blocked[promoted[:, 0], promoted[:, 1]] = True
    left, right = seats.unbind(-1)
    admitted = (order[left] == order[right]) & (order[left] >= 0) & ~blocked[left, right]
    return seats, torch.where(admitted[:, None], names[seats], -1)


def present_pairs(seats, pairs, levels):
    """Offer a pair to each retained stream in which both names are present."""
    both = (levels[..., seats] >= PRESENT).all(-1)
    return torch.where(both[..., None], pairs, -1)


def retained_pair_offers(owner, held, events):
    """Streams sharing one binding use its topology once for the whole look."""
    bindings = {}
    for batch, (_, sentence, _) in events:
        if isinstance(sentence, tuple):
            names, levels, _, count = sentence
            if count >= 2:
                bindings.setdefault(id(names), (names, []))[1].append((batch, levels))
    prepared = {}
    for names, streams in bindings.values():
        if len(streams) < 2:
            continue
        seats, pairs = pair_layout(owner, held, names)
        levels = torch.stack([value for _, value in streams])
        offers = present_pairs(seats, pairs, levels)
        prepared.update((batch, page) for (batch, _), page in zip(streams, offers.unbind(), strict=True))
    return prepared


def pi_step(owner, present, *, collect=None, prior=None, offers=None):
    """One step of the Pi's candidates from the concepts present at a look;
    the symbols made."""
    held = matrices(owner)['pi']
    if isinstance(present, tuple):
        names, levels, _, count = present
    else:
        count = sum(value >= PRESENT for value in present.values())
    occupied = held._known_count()
    if count < 2 and occupied == 0:
        return []
    if not isinstance(present, tuple):
        names, levels = dict_readings(held, present)
    if offers is None:
        seats, pairs = pair_layout(owner, held, names)
        offers = present_pairs(seats, pairs, levels)
    pairs = offers
    levels = levels.to(held.activity)
    if prior is not None:
        pairs = unpromoted_offers(pairs, prior)
    promoted = held.observe_pairs(pairs, names, levels)
    if occupied == 0 and held.hold > 1:
        return []
    if collect is not None:
        collect.append((held, promoted, True, None))
        return []
    written = held.publish(promoted)
    held.append_pairs(promoted, len(written))
    return written


def pair_rows(owner):
    """The rows of the Pi above order zero that the candidates promoted,
    each a pair of concepts of one order, as a list; the spread reads them
    as symbols (:func:`ConceptSpread.structure`). They are not the fit of a
    concept alone, so no kind reads them for a member."""
    pool = pair_pool(owner)
    return [] if pool is None else [tuple(pair) for pair in pool.promoted_pairs().tolist()]


def present_things(names, values, ranks, members, capacity):
    """Read the things at each look. First member orders offers, last member supplies presence."""
    padded, placed = F.pad(values, (0, 1)), F.pad(ranks, (0, 1))
    valid = members >= 0
    seats = seat_of(names, members)
    named = valid.any(-1) & ((seats < len(names)) | ~valid).all(-1)
    seen = named & ((padded[..., seats] >= PRESENT) | ~valid).all(-1)
    positions = placed[..., seats]
    first = positions.masked_fill(~valid, torch.iinfo(ranks.dtype).max).amin(-1)
    last = positions.masked_fill(~valid, -1).argmax(-1, keepdim=True)
    width = min(capacity, len(members))
    if len(members) > width:
        torch._assert_async((seen.sum(-1) <= width).all(), 'more things present at a look than architecture.candidateThings')
    order = first.masked_fill(~seen, torch.iinfo(first.dtype).max).argsort(dim=-1, stable=True)[..., :width]
    things = torch.where(seen.gather(-1, order), order, -1)
    last_seats = seats.expand_as(positions).gather(-1, last).squeeze(-1)
    return things, padded.gather(-1, last_seats.gather(-1, order))


def retained_things(owner, events):
    """Share the symbol membership read among retained streams with the same binding."""
    bindings = {}
    for batch, (present, _, _) in events:
        if isinstance(present, tuple):
            names, values, ranks, count = present
            if count:
                bindings.setdefault(id(names), (names, []))[1].append((batch, values, ranks))
    prepared = {}
    for names, streams in bindings.values():
        if len(streams) < 2:
            continue
        symbols = owner.mtm._candidate_relations
        symbols.read_symbols(owner)
        members = symbols.members[:len(owner._concept_allocator.symbol_table())]
        values = torch.stack([value for _, value, _ in streams])
        ranks = torch.stack([rank for _, _, rank in streams])
        things, presence = present_things(names, values, ranks, members, symbols.thing_width)
        for (batch, _, _), held, levels in zip(streams, things.unbind(), presence.unbind(), strict=True):
            prepared[batch] = held, levels
    return prepared


def sigma_step(owner, present, expected, batch=None, *, collect=None, things_read=None):
    """One step of the Sigma's candidates at a look: a kind expected there
    and not read, with a thing present there. The entries written."""
    alloc = owner._concept_allocator
    if isinstance(present, tuple):
        names, values, ranks, count = present
        if not count:
            return []
    elif not any(value >= PRESENT for value in present.values()):
        return []
    held = matrices(owner)['sigma']
    if not isinstance(present, tuple):
        names, values, ranks = dict_readings(held, present, places=True)
    store = visible_store(alloc.layer(0))
    symbols = owner.mtm._candidate_relations
    symbols.read_symbols(owner)
    if batch is not None and expected is not None:
        expected = expected[batch, :, 0]
    # The forecast is numerical. Only inventory rows above order zero can
    # teach a kind; kinds are read in ascending row order.
    if isinstance(expected, tuple) and len(expected) == 2 and torch.is_tensor(expected[1]):
        rows, levels = expected
    elif torch.is_tensor(expected):
        rows, levels = None, expected[:len(symbols.kinds)]
    else:
        entries = list(expected or ())
        rows = held.refs.new_tensor([row for row, _ in entries])
        levels = held.activity.new_tensor([value for _, value in entries])
    levels = levels.to(held.activity)
    if rows is None:
        # A row of order zero holds no kind.
        start = min(owner._order_caps()[0], len(levels))
        rows, levels = torch.arange(start, len(levels), device=levels.device), levels[start:]
    kinds = symbols.kinds[rows]
    observed = F.pad(values, (0, 1))[seat_of(names, kinds)].to(held.activity)
    member = symbols.members[:len(store.symbol_table)]
    if things_read is None:
        things_read = present_things(names, values, ranks, member, symbols.thing_width)
    things, presence = things_read
    presence = presence.to(held.activity)
    width = things.shape[-1]
    # A kind may be offered a thing of lower order on which it has no entry.
    sources, columns = store._indices(names.device)
    place = torch.full((len(member) + 1,), width, dtype=torch.long, device=names.device)
    place.scatter_(0, torch.where(things >= 0, things, len(member)), torch.arange(width, device=names.device))
    place[-1] = width
    seat = torch.full((len(symbols.kinds) + 1,), len(rows), dtype=torch.long, device=names.device)
    seat.scatter_(0, rows, torch.arange(len(rows), device=names.device))
    entry = torch.zeros(len(rows) + 1, width + 1, dtype=torch.bool, device=names.device)
    entry[seat[sources.clamp_max(len(symbols.kinds))], place[columns.clamp_max(len(member))]] = True
    below = store.concept_rows(F.pad(member[:, 0], (0, 1), value=-1)[torch.where(things >= 0, things, len(member))])
    below = torch.where(below >= 0, symbols.orders[below.clamp_min(0)], torch.iinfo(torch.long).max)
    legal = (kinds >= 0)[:, None] & (below < symbols.orders[rows][:, None]) & ~entry[:-1, :-1]
    wanted = (levels > observed) & (levels > 0)
    take = legal & wanted[:, None] & (things >= 0)
    ordered = things.expand(len(kinds), -1)
    pairs = torch.stack((kinds[:, None].expand_as(ordered), ordered), -1)
    pairs = torch.where(take[..., None], pairs, -1).reshape(-1, 2)
    occupied = held._known_count()
    promoted = held.observe(pairs, kinds, torch.where(wanted, levels, 0.), things, presence,
        observed=torch.where(wanted, observed, 0.), read_pairs=True)
    if occupied == 0 and held.hold > 1:
        return []
    trial = _current_trial.get()
    if collect is not None:
        collect.append((held, promoted, False, trial))
        return []
    return publish_relations(owner, held.publish(promoted), trial)


def publish_relations(owner, promoted, trial):
    table = owner._concept_allocator.symbol_table()
    promote = trial.promote if trial is not None and trial.owner is owner else owner.kind_entry
    written = []
    for kind, symbol in promoted:
        if promote(kind, table.brought[symbol]) is not None:
            written.append((kind, symbol))
    return written


def publish_look(owner, pending):
    """Expose this look's promotions to their own next fixation."""
    if not pending:
        return
    records = torch.stack([torch.cat((pool.count.expand(1, 2), pairs))
                           for pool, pairs, _, _ in pending]).tolist()
    for (count, *records), (pool, promoted, paired, trial) in zip(records, pending, strict=True):
        pool._remember_count(count[0])
        pairs = [(left, right) for left, right in records if left >= 0]
        if paired:
            pool.append_pairs(promoted, len(pairs))
        elif pairs:
            with trial.active():
                publish_relations(owner, pairs, trial)


def withheld_names(owner, targets, extent):
    """MTM names for omitted occurrences inside this look, with -1 for the rest."""
    if targets is None or extent is None or 'word_bytes' not in targets:
        return None
    forms = targets.get('withheld_form_rows')
    pool = getattr(getattr(owner, 'mtm', None), 'word_pool', None)
    if forms is None or pool is None:
        return None
    hidden = targets['draw'].forms.any(-1)
    width = min(extent.shape[1], hidden.shape[1], forms.shape[1])
    forms = forms[:, :width]
    return torch.where(extent[:, :width] & hidden[:, :width] & (forms >= 0),
                       pool.words[forms.clamp_min(0)], -1)


def omitted_field_readings(field, withheld, active, limit):
    """Read visible evidence for Sigma and restore withheld names at one for Pi."""
    evidence = field.evidence
    batch = evidence.shape[1]
    if active is None:
        active = torch.ones(batch, dtype=torch.bool, device=evidence.device)
    concepts = field.concept_ids
    if concepts.ndim == 1:
        concepts = concepts[:, None].expand(-1, batch)
    lanes = evidence.detach().amax(2)
    levels = (lanes[..., 0] - lanes[..., 1]).clamp_min(0).T
    positive = (levels.T > 0) & active[None, :] & (concepts >= 0)
    missing = (withheld >= 0) & active[:, None]
    names = torch.cat((concepts[positive], withheld[missing])).unique(sorted=True)
    if len(names) > limit:
        return None
    if not len(names):
        return [(page, (names, levels[page, :0], names, 0), (names, levels[page, :0], names, 0))
                for page, live in enumerate(active.tolist()) if live]
    values, ranks, sentence = omitted_levels(concepts, levels, names, active, withheld)
    counts = torch.stack((active.long(), (values >= PRESENT).sum(1),
                          (sentence >= PRESENT).sum(1)), 1).tolist()
    return [(page, (names, values[page], ranks[page], visible_count),
             (names, sentence[page], ranks[page], whole_count))
            for page, (live, visible_count, whole_count) in enumerate(counts) if live]


def omitted_levels(concepts, levels, names, active, withheld):
    """Restore withheld names at one on the page that contained them."""
    values, ranks = named_levels(concepts, levels, names, active)
    missing = (withheld >= 0) & active[:, None]
    seats = seat_of(names, withheld)
    restored = values.new_zeros((len(values), len(names) + 1))
    restored.scatter_reduce_(1, seats, missing.to(values), 'amax')
    sentence = torch.where(restored[:, :-1] > 0, 1., values)
    return values, ranks, sentence


@torch.no_grad()
def withheld_words(model, extent):
    """The concepts of the words the omission withheld inside a look's
    extent (``extent`` ``[B, W]``, the word places the look holds), each at
    one: ``[{cid: 1.}]`` for each stream. What the sentence held, though
    the stem the look read had them taken out."""
    targets = getattr(model, '_omission_targets', None)
    if extent is None or targets is None or 'word_bytes' not in targets:
        return [{} for _ in range(len(extent) if extent is not None else 0)]
    owner = model._concept_owner()
    words = withheld_names(owner, targets, extent)
    if words is not None:
        # Growth journals still carry sparse host dictionaries. Only named
        # hidden occurrences enter them, in their original encounter order.
        pages = torch.arange(len(extent), device=words.device)[:, None].expand_as(words)
        named = torch.stack((pages, words), -1)[words >= 0]
        held = [{} for _ in range(len(extent))]
        for page, word in named.tolist():
            held[page][word] = 1.
        return held
    hidden = targets['draw'].forms.any(-1)
    words = targets['word_bytes']
    held = [{} for _ in range(len(extent))]
    width = min(extent.shape[1], hidden.shape[1])
    for b, w in (extent[:, :width] & hidden[:, :width].to(extent.device)).nonzero().tolist():
        if b < len(words) and w < len(words[b]):
            raw = words[b][w]
            form = raw.decode('latin1') if isinstance(raw, bytes) else str(raw)
            cid = owner.interpret.word_for_form(form.strip())
            if cid is not None:
                held[b][int(cid)] = 1.
    return held


def at_opening(model, field, extent, active, prior=None):
    """All departures inherit this observation before their first fixation."""
    trial = _current_trial.get()
    if trial is None:
        at_look(model, field, extent, active)
        return None
    if prior is None:
        at_look(model, field, extent, active)
        # Paths zero through explore_paths own the writable trial seats.
        # The final reserved set holds the unchanged opening for siblings.
        return trial.from_reading(path=model.explore_paths + 1)
    trial.looks.extend(prior.looks)
    trial.streams = {b: stream.from_reading(path=trial.path) for b, stream in prior.streams.items()}
    return prior


def at_look(model, field, extent=None, active=None, *, forecast=None, binding=None):
    """The candidates' step at one look, in training. A pair is offered and
    weighed from what the sentence held at the look: its field, and the
    words the omission withheld inside the look's ``extent`` (``[B, W]``)."""
    if not (getattr(model, 'training', False) and getattr(model, '_sentence_training', True)) or field is None:
        return
    owner = model._concept_owner()
    if getattr(owner, 'mtm', None) is None:
        return
    omission = getattr(model, '_omission_targets', None)
    trial = _current_trial.get()
    if forecast is None:
        # The first look shares the opening already retained by its trial.
        expected = getattr(trial, 'opening_forecast', None)
        if expected is None:
            expected = getattr(owner, '_expected_knowing', None)
        kind_rows = None
        values = None if expected is None else expected[..., 0].detach()
    else:
        kind_rows, values = forecast
    events = {}
    if trial is not None:
        trial.looks.append(events)
    held = None
    observed = None
    activity = active if trial is not None else None
    pool = getattr(owner.mtm, '_candidate_pairs', None)
    withheld = withheld_names(owner, omission, extent)
    if withheld is not None:
        if bool((withheld >= 0).any()):
            if pool is not None:
                observed = omitted_field_readings(field, withheld, activity, pool.binding_width)
            if observed is None:
                held = withheld_words(model, extent)
    elif extent is not None and omission is not None and 'word_bytes' in omission:
        held = withheld_words(model, extent)
    if observed is None:
        readings = (field_readings(owner, field, activity, binding)
                    if pool is not None and not any(held or ()) else None)
        live = None
        if readings is None:
            readings = presences(field)
            live = activity.tolist() if activity is not None else None
        observed = [(b, present, {**present, **held[b]} if held and held[b] else present)
                    for b, present in enumerate(readings) if live is None or live[b]]
    pending = [] if trial is not None else None
    for batch, present, sentence in observed:
        forecast = None if values is None else values[batch]
        if kind_rows is not None:
            forecast = (kind_rows, forecast)
        if trial is not None:
            events[batch] = (present, sentence, forecast)
            if batch not in trial.streams:
                trial.streams[batch] = GrowthTrial(owner, path=trial.path, stream=batch)
        with row_scope(owner, batch):
            pi_step(owner, sentence, collect=pending)
            sigma_step(owner, present, forecast, collect=pending)
    publish_look(owner, pending)
