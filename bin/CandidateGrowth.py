"""Implicit growth of the Pi above order zero and of the store's Sigma, by
candidate matrices kept with MTM (:mod:`CandidateMatrix`).

At each look, in training:

* the Pi above order zero: concepts of one order present together at the
  look offer a candidate row over the pair (a symbol of the pair); a
  candidate held above the penalty becomes a row of the pair
  (:func:`pair_rows`);
* the Sigma: a kind expected at the look by the spread and not read there,
  with a thing present at the look, offers a candidate entry of the kind on
  that thing; held above the penalty it is written as the kind's entry. A
  thing read at another look is never offered.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import copy
import itertools

import torch


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
        self.pairs = None
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
        other.pairs = self.pairs.copy() if self.pairs is not None else None
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
        for turn in range(max((len(trial.looks) for trial in winners), default=0)):
            for batch, trial in enumerate(winners):
                if turn >= len(trial.looks) or batch not in trial.looks[turn]:
                    continue
                present, sentence, expected = trial.looks[turn][batch]
                pi_step(trial.owner, sentence)
                sigma_step(trial.owner, present, expected)


def row_scope(owner, batch):
    trial = _current_trial.get()
    if trial is None or trial.owner is not owner or not trial.streams:
        return nullcontext()
    stream = trial.streams.get(batch)
    return (stream if stream is not None else GrowthTrial(owner)).active()


def read_streams(owner, read, *, axis, live=None):
    """Use each stream's own provisional topology, sharing equal topologies.

    The callback returns independent batch rows. A compacted forecast supplies
    its physical stream mask so its first live row keeps its original owner.
    """
    trial = _current_trial.get()
    if trial is None or trial.owner is not owner or not trial.streams:
        return read()
    base_pairs = getattr(getattr(owner, 'mtm', None), 'pair_rows', None) or ()
    groups = {}
    for b, stream in trial.streams.items():
        pairs = stream.pairs[len(base_pairs):] if stream.pairs else ()
        key = (tuple(stream.edges), tuple(pairs))
        if key != ((), ()):
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
            mask = mask[live]
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


def visible_pairs(owner):
    trial = _current_trial.get()
    if trial is not None and trial.owner is owner and trial.pairs is not None:
        return trial.pairs
    return getattr(getattr(owner, 'mtm', None), 'pair_rows', None) or ()


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


def bound_nominations(owner, names):
    """Derive possible pairs from the shared symbol topology once for this read."""
    allocator = owner._concept_allocator
    table = allocator.symbol_table()
    orders = tuple(allocator.order_of(name) for name in names)
    seats = tuple((i, j) for i, j in itertools.combinations(range(len(names)), 2)
                  if orders[i] == orders[j] >= 0 and table.fit(names[i], names[j]) is None)
    return table.revision, seats


def field_readings(owner, field, active, binding):
    """Use the fixed candidate inventory for a small field.

    The field's binding supplies its possible names. Evidence and page
    activity select the observations. The MTM tally capacity bounds the
    speculative pair grid.
    """
    if binding is None:
        return None
    names = binding
    if not names:
        empty = field.concept_ids.new_empty(0)
        return [(names, empty, field.evidence.new_empty(0), empty, bound_nominations(owner, names))] * field.evidence.shape[1]
    if len(names) * (len(names) - 1) // 2 > len(owner.mtm._candidate_pairs.tally_refs):
        return None
    ids = field.concept_ids.new_tensor(names)
    if active is None:
        active = torch.ones(field.evidence.shape[1], dtype=torch.bool, device=ids.device)
    values, ranks = field_levels(field.concept_ids, field.evidence, ids, active)
    nominations = bound_nominations(owner, names)
    return [(names, ids, row, rank, nominations) for row, rank in zip(values.unbind(), ranks.unbind(), strict=True)]


def presence(owner, field, batch):
    """The positive readings of one stream, for callers outside the look loop."""
    return presences(field)[batch]


def pi_step(owner, present, *, collect=None):
    """One step of the Pi's candidates from the concepts present at a look;
    the symbols made."""
    table = owner._concept_allocator.symbol_table()
    order = owner._concept_allocator.order_of
    held = matrices(owner)['pi']
    numerical = isinstance(present, tuple)
    rows = pair_rows(owner)
    if numerical:
        names, ids, levels, _, (revision, eligible) = present
        if table.revision != revision:
            _, eligible = bound_nominations(owner, names)
        nominations = [(i, j) for i, j in eligible if (names[i], names[j]) not in rows]
    else:
        names = sorted(cid for cid, value in present.items() if value >= PRESENT)
        nominations = [(i, j) for i, j in itertools.combinations(range(len(names)), 2)
                       if order(names[i]) == order(names[j]) >= 0 and (names[i], names[j]) not in rows
                       and table.fit(names[i], names[j]) is None]
    occupied = held._known_count()
    if not nominations and occupied == 0:
        return []
    if numerical:
        seats = held.refs.new_tensor(nominations).reshape(-1, 2)
        pairs = torch.where((levels[seats] >= PRESENT).all(-1, keepdim=True), ids[seats], -1)
        levels = levels.to(held.activity)
    else:
        pairs = held.refs.new_tensor([(names[i], names[j]) for i, j in nominations]).reshape(-1, 2)
        ids, levels = held.refs.new_tensor(list(present)), held.activity.new_tensor(list(present.values()))
    promoted = held.observe_pairs(pairs, ids, levels)
    if occupied == 0 and held.hold > 1:
        return []
    if collect is not None:
        collect.append((held, promoted, rows, None))
        return []
    written = held.publish(promoted)
    rows.extend(key for key in written if key not in rows)
    return written


def pair_rows(owner):
    """The rows of the Pi above order zero that the candidates promoted,
    each a pair of concepts of one order; the spread reads them as symbols
    (:func:`ConceptSpread.structure`). They are not the fit of a concept
    alone, so no kind reads them for a member."""
    pool = owner.mtm
    rows = getattr(pool, 'pair_rows', None)
    trial = _current_trial.get()
    if trial is not None and trial.owner is owner:
        if trial.pairs is None:
            trial.pairs = list(rows or ())
        return trial.pairs
    if rows is None:
        rows = pool.pair_rows = []
    return rows


def sigma_step(owner, present, expected, batch=None, *, collect=None):
    """One step of the Sigma's candidates at a look: a kind expected there
    and not read, with a thing present there. The entries written."""
    alloc = owner._concept_allocator
    table = alloc.symbol_table()
    numerical = isinstance(present, tuple)
    if numerical:
        names, ids, values, ranks, _ = present
        by_name = {name: i for i, name in enumerate(names)}
        members = [table.members(symbol) for symbol in range(len(table))]
        eligible = [(symbol, [by_name[cid] for cid in row]) for symbol, row in enumerate(members)
                    if row and all(cid in by_name for cid in row)]
        if not eligible:
            return []
        things = [symbol for symbol, _ in eligible]
    else:
        things = {}
        for cid, value in present.items():
            if value >= PRESENT:
                for symbol in table.reading(cid):
                    if all(present.get(member, 0.) >= PRESENT for member in table.members(symbol)):
                        things[int(symbol)] = value
        if not things:
            return []
    held = matrices(owner)['sigma']
    store = visible_store(alloc.layer(0))
    if batch is not None and expected is not None:
        expected = expected[batch, :, 0]
    # The forecast is numerical. Only inventory rows above order zero can
    # teach a kind. Read the same rows in nomination order on every device.
    if isinstance(expected, tuple) and len(expected) == 2 and torch.is_tensor(expected[1]):
        rows, levels = expected
        levels = levels.to(held.activity)
    elif torch.is_tensor(expected):
        rows = sorted(row for row in store._tensor_row_keys
                      if owner._order_caps()[0] <= row < len(expected))
        levels = expected.index_select(0, held.refs.new_tensor(rows)).to(held.activity)
    else:
        entries = list(expected or ())
        rows = [row for row, _ in entries]
        levels = held.activity.new_tensor([value for _, value in entries])
    positions, kinds, observed = [], [], []
    offers, sources = [], []
    for i, row in enumerate(rows):
        kind = owner.concept_id_at_row(row)
        if kind is None or alloc.order_of(kind) < 1:
            continue
        positions.append(i)
        kinds.append(kind)
        observed.append(by_name.get(kind, len(names)) if numerical else present.get(kind, 0.))
        for symbol in things:
            members = table.members(symbol)
            if not members or owner._row_order(owner._concept_row_of(members[0])) >= owner._row_order(row):
                continue
            if (row, store.column_of_thing(('sym', symbol))) not in store._index:
                offers.append((kind, symbol))
                sources.append(len(kinds) - 1)
    occupied = held._known_count()
    if not offers and occupied == 0:
        return []
    levels = levels.index_select(0, held.refs.new_tensor(positions))
    if numerical:
        observed = torch.nn.functional.pad(values, (0, 1))[ids.new_tensor(observed)].to(held.activity)
        # The first constituent to occur sets the symbol's offer order.
        # The last constituent in that order supplies its observed value.
        width = max(len(row) for _, row in eligible)
        member = ids.new_tensor([row + [-1] * (width - len(row)) for _, row in eligible])
        valid = member >= 0
        seats = member.clamp_min(0)
        seen = ((values[seats] >= PRESENT) | ~valid).all(-1)
        first = ranks[seats].masked_fill(~valid, torch.iinfo(ranks.dtype).max).amin(-1)
        last = ranks[seats].masked_fill(~valid, -1).argmax(-1, keepdim=True)
        order = first.argsort(stable=True)
        symbols = ids.new_tensor(things)
        presence = values[seats.gather(1, last)[:, 0]].to(held.activity)
        symbols = torch.where(seen, symbols, -1)
        legal = torch.zeros(len(kinds), len(things), device=ids.device, dtype=torch.bool)
        if offers:
            index = {symbol: i for i, symbol in enumerate(things)}
            legal[ids.new_tensor(sources), ids.new_tensor([index[symbol] for _, symbol in offers])] = True
        wanted = (levels > observed) & (levels > 0)
        take = (legal & wanted[:, None] & seen).index_select(1, order)
        candidates = ids.new_tensor(kinds)
        ordered = symbols[order].expand(len(kinds), -1)
        pairs = torch.stack((candidates[:, None].expand_as(ordered), ordered), -1)
        pairs = torch.where(take[..., None], pairs, -1).reshape(-1, 2)
    else:
        observed = held.activity.new_tensor(observed)
        wanted = (levels > observed) & (levels > 0)
        pairs = held.refs.new_tensor(offers).reshape(-1, 2)
        pairs = torch.where(wanted[held.refs.new_tensor(sources)][:, None], pairs, -1)
        candidates = held.refs.new_tensor(kinds)
        symbols = held.refs.new_tensor(list(things))
        presence = held.activity.new_tensor(list(things.values()))
    promoted = held.observe(pairs, candidates, torch.where(wanted, levels, 0.), symbols, presence,
        observed=torch.where(wanted, observed, 0.), read_pairs=True)
    if occupied == 0 and held.hold > 1:
        return []
    trial = _current_trial.get()
    if collect is not None:
        collect.append((held, promoted, None, trial))
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
    for (count, *records), (pool, _, rows, trial) in zip(records, pending, strict=True):
        pool._remember_count(count[0])
        pairs = [(left, right) for left, right in records if left >= 0]
        if rows is not None:
            rows.extend(key for key in pairs if key not in rows)
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
