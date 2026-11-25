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
                for batch, trial in enumerate(winners):
                    if turn >= len(trial.looks) or batch not in trial.looks[turn]:
                        continue
                    present, sentence, expected = trial.looks[turn][batch]
                    if releases is None:
                        pi_step(trial.owner, sentence)
                    else:
                        pending = []
                        pi_step(owner, sentence, collect=pending, prior=releases[:written])
                        if pending:
                            releases[written].copy_(pending[0][1])
                            written += 1
                    sigma_step(trial.owner, present, expected)
        finally:
            if written:
                # Only the spread reads these pairs. During replay its
                # forecasts are already fixed. Later offers in the replay
                # still see each earlier promotion in the device slab.
                promoted = releases[:written].flatten(0, 1)
                pool.append_pairs(promoted, len(pool.publish(promoted)))


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


def pi_step(owner, present, *, collect=None, prior=None):
    """One step of the Pi's candidates from the concepts present at a look;
    the symbols made."""
    held = matrices(owner)['pi']
    alloc = owner._concept_allocator
    store = alloc.layer(0)
    if isinstance(present, tuple):
        names, levels, _, count = present
    else:
        count = sum(value >= PRESENT for value in present.values())
    occupied = held._known_count()
    if count < 2 and occupied == 0:
        return []
    if not isinstance(present, tuple):
        names, levels = dict_readings(held, present)
    symbols = owner.mtm._candidate_relations
    symbols.read_symbols(owner)
    # Pairs in ascending order of their two names, each read where both
    # concepts are present, of one order, held together by no symbol and
    # not yet a row.
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
    pairs = torch.where((admitted & (levels[seats] >= PRESENT).all(-1))[:, None], names[seats], -1)
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


def sigma_step(owner, present, expected, batch=None, *, collect=None):
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
    top = len(names)
    padded, placed = F.pad(values, (0, 1)), F.pad(ranks, (0, 1))
    observed = padded[seat_of(names, kinds)].to(held.activity)
    # A thing is a symbol whose members are all present. The first member
    # to occur sets its offer order; the last supplies its observed value.
    table = store.symbol_table
    member = symbols.members[:len(table)]
    valid = member >= 0
    seats = seat_of(names, member)
    named = valid.any(-1) & ((seats < top) | ~valid).all(-1)
    seen = named & ((padded[seats] >= PRESENT) | ~valid).all(-1)
    first = placed[seats].masked_fill(~valid, torch.iinfo(ranks.dtype).max).amin(-1)
    last = placed[seats].masked_fill(~valid, -1).argmax(-1, keepdim=True)
    # The things present, in offer order, at a fixed width: no more than
    # there are symbols.
    width = min(symbols.thing_width, len(member))
    if len(member) > width:
        torch._assert_async(seen.sum() <= width, 'more things present at a look than architecture.candidateThings')
    order = first.masked_fill(~seen, torch.iinfo(first.dtype).max).argsort(stable=True)[:width]
    things = F.pad(torch.where(seen[order], order, -1), (0, width - len(order)), value=-1)
    presence = F.pad(padded[seats.gather(1, last)[:, 0]][order], (0, width - len(order))).to(held.activity)
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
    held = withheld_words(model, extent) if extent is not None and omission is not None and 'word_bytes' in omission else None
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
    readings = (field_readings(owner, field, active if trial is not None else None, binding)
                if hasattr(owner.mtm, '_candidate_pairs') and not any(held or ()) else None)
    if readings is None:
        readings = presences(field)
        live = active.tolist() if trial is not None and active is not None else None
    else:
        live = None
    pending = [] if trial is not None else None
    for batch, present in enumerate(readings):
        if live is not None and not live[batch]:
            continue
        sentence = present if not held or not held[batch] else {**present, **held[batch]}
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
