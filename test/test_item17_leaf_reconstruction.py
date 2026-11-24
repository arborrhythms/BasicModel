"""A diagnostic of the perceptual inverse alone (round 6d of item 1.7,
fifteenth pass, step 1, the simulation before the build); not the model
saying a word back, which starts from its own idea (Alec). Each sentence is read eager
with no gradient. For each leaf of the reading (each reading of the
sentence's open field with a bracket, the mark included): its own concept
value (the open field at that reading, for less against) goes through the
order-zero Sigma's way back (`SparseSigmaLDU.reverse`) and the first Pi's
(`BasicPercepts.kept_percept`), place rows' `.what` left out; the recreated
`.what` is matched against the percept codebook (every basic percept's code,
`bank_codes`, place rows left out) by cosine; the leaf's own `.where` wires
(the percept's, `Places.readings`) are matched against the grid
(`WhereRegistry.grid`, the text range) by distance; the matched percept's
spelling is pasted at the matched place. Is the sentence the sentence?
Beside it, the source's own reconstruction of the input ("as built").
A record: the start is drawn afresh and printed; nothing is asserted of
what a start gives."""
import os

import torch
import torch.nn.functional as F

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of
from test_item17_word_from_concept import concept_values, recreated_what
from FieldReverse import spelling

WORDS = words_of(LESSON) + COUNTING[:6]


def codebook(basic):
    """The percept codebook: each spelled, non-place basic percept's code on
    its for lane, and its spelling."""
    rows = [r for r in range(len(basic.index)) if r not in basic.place_rows
            and r not in basic.conjunctions and not basic.is_value(r) and spelling(basic, r)]
    return rows, basic.bank_codes(rows)[:, 0], [spelling(basic, r) for r in rows]


def grid_match(registry, wires):
    """The nearest text address of ``wires`` and its margin, by distance."""
    start, end = registry.slices['text']
    code = registry.grid.detach().float()[start:end]
    d = (code - wires.float()[None]).norm(dim=-1)
    two = d.topk(2, largest=False)
    return int(two.indices[0]), round(float(two.values[1] - two.values[0]), 4)


def leaf_reconstruction(model, record, bank):
    """The sentence pasted leaf by leaf, and a line for each leaf."""
    from Places import readings
    owner = model._concept_owner()
    basic = owner.basic_percepts
    rows, codes, spellings = codebook(basic)
    field = record.open_field
    wires, has, _ = readings(bank)
    out, lines = {}, []
    for k in range(field.where.shape[1]):
        lo, hi = (int(x) for x in field.where[0, k])
        if hi <= lo:
            continue
        what, top = recreated_what(owner, concept_values(owner, field, k), places=False)
        if bool(what.abs().gt(0).any()):
            score = F.cosine_similarity(what[None].to(codes), codes, dim=-1)
            best = score.max()
            tied = (score == best).nonzero().flatten()
            pick = int(tied[torch.randint(len(tied), ())])
            ranked = score.topk(min(2, len(score)))
            margin = round(float(ranked.values[0] - ranked.values[-1]), 3)
            raw = spellings[pick]
        else:
            raw, margin, pick = None, None, None
        at = (bank.where[0, :, 0] == lo) & (bank.where[0, :, 1] == hi)
        if bool(at.any()) and bool(has[0, int(at.int().argmax())]):
            address, where_margin = grid_match(model.where_registry, wires[0, int(at.int().argmax())])
        else:
            address, where_margin = None, None
        lines.append(dict(bracket=(lo, hi), spelled=raw, what_margin=margin, address=address,
                          where_margin=where_margin, place_right=address == lo))
        if raw is not None and address is not None:
            for i, b in enumerate(raw):
                out[address + i] = b
    width = max([hi for d in lines for hi in (d['bracket'][1],)] + [k + 1 for k in out] + [0])
    text = bytes(out.get(i, 32) for i in range(width)).decode('latin1')
    return text, lines


def test_the_sentence_back_leaf_by_leaf(tmp_path, monkeypatch):
    import SentenceField
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        held = []
        reconstruct = SentenceField.reconstruct
        def holding(model, record, errors, active, **kw):
            result = reconstruct(model, record, errors, active, **kw)
            held.append((record, model._sentence_banks[int(record.sentence)],
                         model._input_reconstructions[int(record.sentence)].event))
            return result
        monkeypatch.setattr(SentenceField, 'reconstruct', holding)
        text = lambda t: bytes(int(v) for v in t.reshape(-1).tolist() if 0 < int(v) < 256).decode('latin1')
        tally = dict(words=0, places=0, marks=0, sentences=0)
        for sentence in ('This is Felix.',) + tuple(w + '.' for w in WORDS) + LESSON[1:4]:
            held.clear()
            with torch.no_grad():
                read(model, sentence)
                record, bank, event = held[0]
                got, lines = leaf_reconstruction(model, record, bank)
                model.End()
            words = sentence[:-1].split()
            tally['sentences'] += got == sentence
            print(repr(sentence), 'leaf by leaf ->', repr(got), 'as built ->', repr(text(event[0, 0])))
            for d in lines:
                print('   ', d)
            if len(words) == 1:
                tally['words'] += got[:len(words[0])] == words[0]
                tally['marks'] += got[len(words[0]):] == sentence[-1]
                tally['places'] += all(d['place_right'] for d in lines)
        print('of', len(WORDS), 'words alone:', tally)
        assert tally['sentences'] <= len(WORDS) + 4
    finally:
        model.End()
