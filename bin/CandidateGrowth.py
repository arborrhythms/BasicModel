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


def presence(owner, field, batch):
    """Each concept's presence at a look, ``{cid: value}``: its greatest
    reading on the for lane less what it reads against, at zero or above."""
    ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, batch]
    lanes = field.evidence[:, batch].detach().amax(1)
    value = (lanes[..., 0] - lanes[..., 1]).clamp_min(0)
    return {int(cid): float(v) for cid, v in zip(ids.tolist(), value.tolist()) if int(cid) >= 0 and v > 0}


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
    gradients = {key: pair_gradient(held.value(key), present.get(key[0], 0.), present.get(key[1], 0.))
                 for key in held.entries if max(present.get(key[0], 0.), present.get(key[1], 0.)) > 0}
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


def sigma_step(owner, present, expected, batch):
    """One step of the Sigma's candidates at a look: a kind expected there
    and not read, with a thing present there. The entries written."""
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
    if expected is not None and things:
        rows = (expected[batch, :, 0] > 0).nonzero().flatten().tolist()
        for row in rows:
            kind = owner.concept_id_at_row(row)
            if kind is None or alloc.order_of(int(kind)) < 1:
                continue
            gap = float(expected[batch, row, 0]) - present.get(int(kind), 0.)
            if gap > 0:
                wanted[int(kind)] = (float(expected[batch, row, 0]), present.get(int(kind), 0.))
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
def at_look(model, field):
    """The candidates' step at one look, in training."""
    if not (getattr(model, 'training', False) and getattr(model, '_sentence_training', True)) or field is None:
        return
    owner = model._concept_owner()
    if getattr(owner, 'mtm', None) is None:
        return
    expected = getattr(owner, '_expected_knowing', None)
    for batch in range(field.evidence.shape[1]):
        present = presence(owner, field, batch)
        pi_step(owner, present)
        sigma_step(owner, present, expected, batch)
