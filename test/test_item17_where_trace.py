"""The `.where` of each reading on its way up and back (round 6d of item
1.7, fourteenth pass, step 5, its first measure). One lesson sentence is
read eager with no gradient. At each stage that carries a reading, its
`.where` wires are matched against the grid (`WhereRegistry.code`): the
nearest address, whether it is the reading's own byte start, and the margin
(the second nearest distance less the nearest). Stages that keep the place
as integers are printed as integers. A record: the start is drawn afresh and
printed; nothing is asserted of what a start gives."""
import os
import traceback

import torch

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate

SENTENCE = LESSON[0]


def matched(registry, wires):
    """``wires [..., 4]`` -> (address, margin) by distance to every grid row,
    and the same for each rung (wires 0:2 the long, 2:4 the short)."""
    code = registry.code.detach().float()
    def nearest(v, c):
        d = (c - v[None]).norm(dim=-1)
        two = d.topk(2, largest=False)
        return int(two.indices[0]), round(float(two.values[1] - two.values[0]), 4)
    v = wires.detach().float().reshape(-1)
    return nearest(v, code), nearest(v[:2], code[:, :2]), nearest(v[2:4], code[:, 2:4])


def test_where_of_each_reading(tmp_path, monkeypatch):
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    from Memory import STMChunks
    import OutputOccurrences
    import SentenceField
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        registry = model.where_registry
        text0 = registry.slices['text'][0]
        print('grid', tuple(registry.code.shape), 'slices', dict(registry.slices))
        ps, cs = model.perceptualSpace, model.conceptualSpace
        on, rows = [False], []
        events, phrase = ps._radix_code_events, cs.apply_phrase_rows
        generated, sentence = OutputOccurrences.generated, SentenceField.sentence_generated
        concept_where = None

        def add(stage, wires, truth):
            if on[0]:
                rows.append((stage, wires.detach().float().clone(), truth))

        def evented(event, part_offsets=None):
            out = events(event, part_offsets)
            if part_offsets is not None and on[0]:
                idx = list(ps._radix_where_indices)
                offsets = part_offsets.reshape(-1)
                flat = out.reshape(-1, out.shape[-1])
                for k in range(min(len(offsets), flat.shape[0])):
                    if int(offsets[k]) >= 0:
                        add(f'percept event, part at byte {int(offsets[k])} (PartSpace._radix_code_events)',
                            flat[k, idx], text0 + int(offsets[k]))
            return out

        def phrased(state, phrase_row, applied):
            out = phrase(state, phrase_row, applied)
            if on[0]:
                index = STMChunks.stack_indices(out)[0]
                depth = int(out[STMChunks.DEPTH][0])
                for s in range(depth):
                    p = int(index[s])
                    if p >= 0:
                        add(f'STM slot {s} of {depth}: field 11 (chunk_where)', out[STMChunks.WHERE][0, p], None)
                        add(f'STM slot {s} of {depth}: concept wires [{concept_where}:{concept_where + 4}]',
                            out[0][0, p, concept_where:concept_where + 4], None)
            return out

        def record_of(model, record, **kw):
            if on[0]:
                f = record.open_field
                live = (f.where[0, :, 1] > f.where[0, :, 0]).nonzero().flatten().tolist()
                print('conceptual field .where (integers, [start, end) per reading):',
                      [tuple(int(x) for x in f.where[0, r]) for r in live])
                for k in range(int(record.end_depth[0])):
                    add(f'record.end_slots[{k}] concept wires', record.end_slots[0, k, concept_where:concept_where + 4], None)
            return sentence(model, record, **kw)

        def walk_leaf(model, words, counts, truncated, bank, trace, **kw):
            if any(f.name == 'reconstruct_generation' for f in traceback.extract_stack()):
                for k in range(words.shape[1]):
                    add(f'walk leaf {k} concept wires', words[0, k, concept_where:concept_where + 4], None)
            return generated(model, words, counts, truncated, bank, trace, **kw)

        monkeypatch.setattr(ps, '_radix_code_events', evented)
        monkeypatch.setattr(cs, 'apply_phrase_rows', phrased)
        monkeypatch.setattr(SentenceField, 'sentence_generated', record_of)
        monkeypatch.setattr(OutputOccurrences, 'generated', walk_leaf)
        when_start = model.symbolSpace.ltm_store.when_start
        concept_where = when_start - 4
        on[0] = True
        with torch.no_grad():
            read(model, SENTENCE)
            offsets = model.inputSpace._ar_word_part_offsets[0, :, 0].tolist()
            model.End()
        on[0] = False
        starts = [text0 + o for o in offsets if o >= 0]
        print('sentence', repr(SENTENCE), 'word byte starts (grid addresses):', starts)
        print('| stage | own address | nearest (margin) | long rung (margin) | short rung (margin) | wires |')
        print('| --- | ---: | ---: | ---: | ---: | --- |')
        seen = set()
        for stage, wires, truth in rows:
            if not bool(wires.abs().gt(0).any()):
                key = (stage, 'zero')
                if key not in seen:
                    seen.add(key)
                    print(f'| {stage} | {truth} | no wires lit | | | |')
                continue
            both, long, short = matched(registry, wires)
            key = (stage, both[0])
            if key in seen:
                continue
            seen.add(key)
            right = '' if truth is None else (' right' if both[0] == truth else ' WRONG')
            print(f'| {stage} | {truth if truth is not None else "-"} | {both[0]} ({both[1]}){right} | '
                  f'{long[0]} ({long[1]}) | {short[0]} ({short[1]}) | {[round(float(x), 3) for x in wires]} |')
        assert rows
    finally:
        model.End()
