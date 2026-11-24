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

from CandidateMatrix import CandidateMatrix, pair_gradient, substitution_gradient
from util import TheXMLConfig

#: Presence at or above which a concept or a thing is present at a look.
PRESENT = .5


_current_trial = ContextVar('candidate_growth_trial', default=None)


class GrowthTrial:
    """Provisional observations and sparse promotions of one batched reading."""

    def __init__(self, owner):
        self.owner = owner
        self.store = owner._concept_allocator.layer(0)
        self.candidates = None
        self.pairs = None
        self.edges = {}
        self.streams = {}
        self.looks = []
        self.expected = None
        self.costs = {'_expectation_looks': [], '_expectation_percept_looks': []}

    def from_reading(self):
        """An output-only departure inherits the completed reading by value."""
        other = GrowthTrial(self.owner)
        other.streams = {b: stream.from_reading() for b, stream in self.streams.items()}
        other.expected = self.expected
        other.looks = self.looks.copy()
        other.edges = self.edges.copy()
        other.costs = {name: entries.copy() for name, entries in self.costs.items()}
        other.pairs = self.pairs.copy() if self.pairs is not None else None
        if self.candidates is not None:
            other.candidates = {name: matrix.fork() for name, matrix in self.candidates.items()}
        return other

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
            trial.candidates = {name: matrix.fork() for name, matrix in held.items()}
            return trial.candidates
    else:
        trial = None
    if held is None:
        room = int(TheXMLConfig.get('architecture.candidateCapacity', default=64))
        held = dict(pi=CandidateMatrix(room), sigma=CandidateMatrix(room))
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


def presence(owner, field, batch):
    """The positive readings of one stream, for callers outside the look loop."""
    return presences(field)[batch]


def pi_step(owner, present):
    """One step of the Pi's candidates from the concepts present at a look;
    the symbols made."""
    table = owner._concept_allocator.symbol_table()
    order = owner._concept_allocator.order_of
    held = matrices(owner)['pi']
    found = sorted(cid for cid, value in present.items() if value >= PRESENT)
    for left, right in itertools.combinations(found, 2):
        if (order(left) == order(right) >= 0 and (left, right) not in held
                and (left, right) not in pair_rows(owner) and table.fit(left, right) is None):
            held.offer((left, right))
    gradients = {key: pair_gradient(held.value(key),
                                    held.meet(key, present.get(key[0], 0.), present.get(key[1], 0.)))
                 for key in list(held.entries) if max(present.get(key[0], 0.), present.get(key[1], 0.)) > 0}
    promoted, _ = held.step(gradients)
    rows = pair_rows(owner)
    for key, _ in promoted:
        if key not in rows:
            rows.append(key)
    return [key for key, _ in promoted]


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


def sigma_step(owner, present, expected, batch=None):
    """One step of the Sigma's candidates at a look: a kind expected there
    and not read, with a thing present there. The entries written."""
    if batch is not None:
        if expected is None:
            expected = ()
        else:
            column = expected[batch, :, 0]
            rows = (column > 0).nonzero().flatten()
            expected = zip(rows.tolist(), column[rows].tolist(), strict=True)
    alloc = owner._concept_allocator
    table = alloc.symbol_table()
    held = matrices(owner)['sigma']
    things = {}
    for cid, value in present.items():
        if value >= PRESENT:
            for symbol in table.reading(cid):
                if all(present.get(member, 0.) >= PRESENT for member in table.members(symbol)):
                    things[int(symbol)] = value
    wanted = {}
    if things:
        for row, value in expected:
            kind = owner.concept_id_at_row(row)
            if kind is None or alloc.order_of(int(kind)) < 1:
                continue
            observed = present.get(int(kind), 0.)
            if value > observed:
                wanted[int(kind)] = (value, observed)
    store = visible_store(alloc.layer(0))
    for kind in wanted:
        kind_row = owner._concept_row_of(kind)
        for symbol in things:
            members = table.members(symbol)
            if (kind, symbol) in held or not members or owner._row_order(owner._concept_row_of(members[0])) >= owner._row_order(kind_row):
                continue
            column = store.column_of_thing(('sym', symbol))
            if (kind_row, column) in store._index:
                continue
            held.offer((kind, symbol))
    gradients = {}
    for kind, symbol in held.entries:
        if symbol in things:
            e, r = wanted.get(kind, (0., 0.))
            gradients[(kind, symbol)] = substitution_gradient(e, r, things[symbol])
    promoted, _ = held.step(gradients)
    written = []
    for (kind, symbol), value in promoted:
        trial = _current_trial.get()
        promote = trial.promote if trial is not None and trial.owner is owner else owner.kind_entry
        if promote(kind, table.brought[symbol]) is not None:
            written.append((kind, symbol))
    return written


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


def at_look(model, field, extent=None, active=None):
    """The candidates' step at one look, in training. A pair is offered and
    weighed from what the sentence held at the look: its field, and the
    words the omission withheld inside the look's ``extent`` (``[B, W]``)."""
    if not (getattr(model, 'training', False) and getattr(model, '_sentence_training', True)) or field is None:
        return
    owner = model._concept_owner()
    if getattr(owner, 'mtm', None) is None:
        return
    expected = getattr(owner, '_expected_knowing', None)
    held = withheld_words(model, extent) if extent is not None else None
    forecasts = [[] for _ in range(field.evidence.shape[1])]
    if expected is not None:
        column = expected[:len(forecasts), :, 0]
        positive = (column > 0).nonzero()
        values = column[positive[:, 0], positive[:, 1]].tolist()
        for (b, row), value in zip(positive.tolist(), values, strict=True):
            forecasts[b].append((row, value))
    trial = _current_trial.get()
    events = {}
    if trial is not None:
        trial.looks.append(events)
    active = active.tolist() if trial is not None and active is not None else None
    for batch, present in enumerate(presences(field)):
        if active is not None and not active[batch]:
            continue
        sentence = present if not held or not held[batch] else {**present, **held[batch]}
        if trial is not None:
            events[batch] = (present, sentence, forecasts[batch])
            if batch not in trial.streams:
                trial.streams[batch] = GrowthTrial(owner)
        with row_scope(owner, batch):
            pi_step(owner, sentence)
            sigma_step(owner, present, forecasts[batch])
