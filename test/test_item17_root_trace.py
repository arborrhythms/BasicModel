"""The trace of one word's concept from its look to the output walk's leaf
(round 6d of item 1.7, fourteenth pass, step 1). "This." is read alone,
eager, with no gradient, untrained and after rounds of the 19 words. At each
stage the vector's `.what` (the percept band) is compared with the word's own
code and with the best of the other 18 words' codes (cosine), its lit `.what`
wires are counted and its `.where`/`.when` wires are printed. A record: the
start is drawn afresh and printed; nothing is asserted of what a start
gives."""
import os
import traceback

import torch
import torch.nn.functional as F

from test_item17_own_reading import COUNTING, taught
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]
WORD = 'This'


class Trace:
    """Each stage's vector, in the order the reading meets it."""

    def __init__(self, model, monkeypatch):
        import OutputOccurrences
        import SentenceField
        from Memory import STMChunks
        self.on, self.rows = False, []
        cs, language = model.conceptualSpace, model.languageSpace
        interpret = model._concept_owner().interpret
        promote, activate = cs.promote_symbol_reference, interpret.activate
        choose, phrase = language.choose_operation, cs.apply_phrase_rows
        generated, sentence = OutputOccurrences.generated, SentenceField.sentence_generated
        self.choice = None

        def top(state):
            view = STMChunks.stack_view(state)
            depth = int(view[1][0])
            return view[0][0, max(depth - 1, 0)], depth

        def promoted(*args, **kw):
            event, orders = promote(*args, **kw)
            self.add('look: symbolic event (promote_symbol_reference)', event[0, 0])
            return event, orders

        def activated(atoms, presence, *, evidence):
            value = activate(atoms, presence, evidence=evidence)
            caller = traceback.extract_stack()[-2]
            where = f'{caller.name}:{caller.lineno}'
            if caller.name != 'stage_cs_lang':
                return value
            self.add(f'leaf ({where}): word_atoms (presence {float(presence.reshape(-1)[0]):.3f}, '
                     f'evidence {[round(float(x), 3) for x in evidence.reshape(-1)[:2]]})', atoms[0])
            self.add(f'leaf ({where}): interpret.activate', value[0])
            return value

        def rule(kind, op):
            rules = {1: language._compose_binary_rules, 2: language._compose_unary_rules}.get(kind, ())
            return getattr(rules[op], 'method_name', type(rules[op]).__name__) if 0 <= op < len(rules) else '-'

        def chosen(state, row_gate, **kw):
            view = STMChunks.stack_view(state)
            depth = int(view[1][0])
            if bool(row_gate.reshape(-1)[0]):
                for i in range(min(depth, 2)):
                    self.add(f'operand {i} of {depth} (stack slot {depth - 1 - i}) before the choice',
                             view[0][0, depth - 1 - i])
            result = choose(state, row_gate, **kw)
            c = result[0] if isinstance(result, tuple) else result
            kind, op = int(c.kind[0]), int(c.local_op[0])
            self.choice = (('closing ' if kw.get('closing') else 'round ')
                           + f'kind {kind} op {op} ({rule(kind, op)}) applied {bool(c.applied[0])}')
            return result

        def phrased(state, phrase_row, applied):
            out = phrase(state, phrase_row, applied)
            value, depth = top(out)
            self.add(f'STM top after {self.choice} (depth {depth})', value)
            return out

        def walk_leaf(model, words, counts, truncated, bank, trace, **kw):
            if any(f.name == 'reconstruction' for f in traceback.extract_stack()):
                self.add('walk leaf (OutputOccurrences.generated)', words[0, 0])
            return generated(model, words, counts, truncated, bank, trace, **kw)

        def record_of(model, record, **kw):
            depth = int(record.end_depth[0])
            self.add(f'record.root (end_depth {depth})', record.root[0])
            for k in range(min(depth, 3)):
                self.add(f'record.end_slots[{k}]', record.end_slots[0, k])
            if record.meaning_root is not None:
                self.add('record.meaning_root', record.meaning_root[0])
            return sentence(model, record, **kw)

        monkeypatch.setattr(cs, 'promote_symbol_reference', promoted)
        monkeypatch.setattr(interpret, 'activate', activated)
        monkeypatch.setattr(language, 'choose_operation', chosen)
        monkeypatch.setattr(cs, 'apply_phrase_rows', phrased)
        monkeypatch.setattr(OutputOccurrences, 'generated', walk_leaf)
        monkeypatch.setattr(SentenceField, 'sentence_generated', record_of)

    def add(self, name, value):
        if self.on and value.ndim:
            if value.ndim > 1:
                name += f' {tuple(value.shape)}[0]'
            value = value.reshape(-1, value.shape[-1])[0]
            self.rows.append((name, value.detach().float().clone()))


def codes_of(owner):
    rows = [owner._concept_row_of(int(owner.interpret.word_for_form(w))) for w in WORDS]
    return owner.similarity_codebook.lookup_rows(rows).detach().float()


def table(trace, codes, width):
    own = WORDS.index(WORD)
    print(f'| stage | cos own | best rival | cos rival | lit .what of {width} | .where/.when |')
    print('| --- | ---: | --- | ---: | ---: | --- |')
    for name, value in trace.rows:
        what = value[:width]
        s = F.cosine_similarity(what[None], codes[:, :width], dim=-1)
        rival = s.clone(); rival[own] = -2
        r = int(rival.argmax())
        tail = [round(float(x), 3) for x in value[width:width + 8]]
        print(f'| {name} | {float(s[own]):.3f} | {WORDS[r]} | {float(s[r]):.3f} | '
              f'{int(what.abs().gt(1e-6).sum())} | {tail} |')


def test_the_word_from_its_look_to_the_walks_leaf(tmp_path, monkeypatch):
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
        owner = model._concept_owner()
        width = owner.basic_percepts.width
        trace = Trace(model, monkeypatch)
        optimizer = model.getOptimizer(lr=.01)
        rounds = int(os.environ.get('ROUNDS', '5'))
        for k in (0, rounds):
            if k:
                for _ in range(rounds):
                    for w in WORDS:
                        taught(model, optimizer, w + '.')
            trace.rows.clear(); trace.on = True
            with torch.no_grad():
                read(model, WORD + '.')
                model.End()
            trace.on = False
            print(f'\nstart {start}, after {k} rounds; concept width {trace.rows[0][1].numel()}, '
                  f'percept band {width}')
            table(trace, codes_of(owner), width)
        assert trace.rows
    finally:
        model.End()
