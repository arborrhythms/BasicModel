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
import itertools

import torch

from CandidateMatrix import CandidateMatrix, pair_gradient, substitution_gradient
from util import TheXMLConfig

#: Presence at or above which a concept or a thing is present at a look.
PRESENT = .5


def matrices(owner):
    """The candidate matrices of ``owner``'s MTM, made on first use, each
    sized by ``architecture.candidateCapacity``."""
    pool = owner.mtm
    held = getattr(pool, 'candidates', None)
    if held is None:
        room = int(TheXMLConfig.get('architecture.candidateCapacity', default=64))
        held = pool.candidates = dict(pi=CandidateMatrix(room), sigma=CandidateMatrix(room))
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
    store = alloc.layer(0)
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
        if owner.kind_entry(kind, table.brought[symbol]) is not None:
            written.append((kind, symbol))
    return written


@torch.no_grad()
def withheld_words(model, extent):
    """The concepts of the words the omission withheld inside a look's
    extent (``extent`` ``[B, W]``, the word places the look holds), each at
    one: ``[{cid: 1.}]`` for each stream. What the sentence held, though
    the stem the look read had them taken out."""
    targets = getattr(model, '_omission_targets', None)
    if targets is None or extent is None:
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


def at_look(model, field, extent=None):
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
        positive = (expected[..., 0] > 0).nonzero()
        values = expected[..., 0][positive[:, 0], positive[:, 1]].tolist()
        for (b, row), value in zip(positive.tolist(), values, strict=True):
            forecasts[b].append((row, value))
    for batch, present in enumerate(presences(field)):
        sentence = present if not held or not held[batch] else {**present, **held[batch]}
        pi_step(owner, sentence)
        sigma_step(owner, present, forecasts[batch])
