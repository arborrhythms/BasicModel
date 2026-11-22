"""The row at 1.00 of doc/ModelAge.md by the output walk (round 6d of item
1.7, thirteenth pass, step 1): the simulation before the build. Each of the
19 words is read alone with no gradient; at the walk's first generated slot
(`OutputOccurrences.generated`, called from `reconstruction`) the
leaf's percept band is filled with a recreated `.what`, and the terminals are
scored against it by `readback_scores`' cosine on the percept band. The
recreated `.what` is the order-zero Sigma's way back (`SparseSigmaLDU.reverse`)
of a concept value, then the first Pi's (`BasicPercepts.kept_percept`), each
first-Pi row's `.what` weighted by its value above zero. Three concept values:
(a) the leaf's own field, as the model's match reads the leaf (`stimulate`);
(b) the open field at the word's reading; (c) the word's concept alone, at
tanh(1). A record: the start is drawn afresh and printed; nothing is asserted
of what a start gives."""
import os
import traceback

import torch
import torch.nn.functional as F

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


def recreated_what(owner, value, *, places=True):
    """``value`` ``[order-zero concepts]`` -> the Sigma's way back -> each
    first-Pi row's kept `.what` (its for lane), weighted by the row's value
    above zero and summed. Returns the `.what` and the four greatest rows
    ``(value, row, is a place row)``. With ``places`` false the place rows'
    `.what` is left out of the sum (fourteenth pass, step 3)."""
    sigma = owner.install_order_zero()
    basic = owner.basic_percepts
    rows = list(range(len(basic.index)))
    values = sigma.reverse(value, rows)
    what = basic.kept_percept(rows)[0][:, 0]
    if not places:
        what = what * torch.tensor([r not in basic.place_rows for r in rows], dtype=what.dtype)[:, None]
    top = sorted(((round(float(v), 3), r, r in basic.place_rows)
                  for r, v in enumerate(values.tolist()) if v), reverse=True)[:4]
    return (values.clamp_min(0)[:, None] * what).sum(0), top


def concept_values(owner, field, reading):
    """A field's signed value (for less against) at ``reading`` on each
    order-zero concept row."""
    sigma = owner.install_order_zero()
    value = torch.zeros(sigma.nOutput)
    for i, cid in enumerate(field.concept_ids.tolist()):
        row = owner._concept_row_of(int(cid)) if cid >= 0 else None
        if row is not None and row < sigma.nOutput:
            value[row] = float(field.evidence[i, 0, reading, 0] - field.evidence[i, 0, reading, 1])
    return value


def test_the_leaf_filled_by_its_concepts_inverse(tmp_path, monkeypatch):
    from FieldProjection import stimulate
    import OutputOccurrences
    import SentenceField
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        leaves, fields = [], []
        generated, reconstruct = OutputOccurrences.generated, SentenceField.reconstruct
        def held_leaf(model, words, counts, truncated, bank, trace, **kw):
            if any(f.name == 'reconstruction' for f in traceback.extract_stack()):
                leaves.append((words.detach().clone(), bank))
            return generated(model, words, counts, truncated, bank, trace, **kw)
        def held_field(model, record, errors, active, **kw):
            fields.append(record.open_field)
            return reconstruct(model, record, errors, active, **kw)
        monkeypatch.setattr(OutputOccurrences, 'generated', held_leaf)
        monkeypatch.setattr(SentenceField, 'reconstruct', held_field)
        from test_item18_expect import read
        owner = model._concept_owner()
        sigma = owner.install_order_zero()
        greatest = dict(built=0, a=0, b=0, c=0, a_no_places=0, b_no_places=0)
        for w in WORDS:
            leaves.clear(); fields.clear()
            with torch.no_grad():
                read(model, w + '.')
                model.End()
                words, bank = leaves[0]
                leaf = words[0, 0]
                width = bank.percept_width
                valid = bank.terminal_valid[0]
                surfaces = [owner.word_surface_for_row(int(r)) for r in bank.rows[0].tolist()]
                own = torch.tensor([bool(valid[i]) and surfaces[i] == w.encode() for i in range(len(valid))])
                def scored(band):
                    s = F.cosine_similarity(band[None, :width].to(bank.codes), bank.codes[0, :, :width], dim=-1).abs()
                    rival = torch.where(valid & ~own, s, -1.)
                    return round(float(s[own].max()), 3), round(float(rival.max()), 3), surfaces[int(rival.argmax())]
                row = owner._concept_row_of(int(owner.interpret.word_for_form(w)))
                alone = torch.zeros(sigma.nOutput)
                alone[row] = torch.tanh(torch.tensor(1.))
                f = fields[0]
                reading = int((f.where[0, :, 1] > f.where[0, :, 0]).nonzero()[0])
                rows = dict(built=(leaf, None),
                            a=recreated_what(owner, concept_values(owner, stimulate(owner, leaf[None, None]), 0)),
                            b=recreated_what(owner, concept_values(owner, f, reading)),
                            c=recreated_what(owner, alone),
                            a_no_places=recreated_what(owner, concept_values(owner, stimulate(owner, leaf[None, None]), 0), places=False),
                            b_no_places=recreated_what(owner, concept_values(owner, f, reading), places=False))
                line = [w, 'leaf .what wires held', int(leaf[:width].ne(0).sum())]
                for key, (band, top) in rows.items():
                    mine, rival, name = scored(band)
                    greatest[key] += mine > rival
                    line += [f'({key}) own {mine} best rival {rival} {name}'] + ([] if top is None else [f'rows {top[:2]}'])
                print(*line)
        print('own terminal greatest, of', len(WORDS), greatest)
        assert all(v <= len(WORDS) for v in greatest.values())
    finally:
        model.End()
