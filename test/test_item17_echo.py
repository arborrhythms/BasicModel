"""The echo test (round 6d of item 1.7, eighteenth pass, item 10; sigma-pi
§19.136): each of the lesson's 19 words alone is the input and the supplied
answer, with no mark or with its mark: what is put in is expected out. Four
cells, each a test of the suite asserting that what the model returns
spells its input, input to output on the model as it runs:

- with the grammar (the lesson's model), the word and the word with its
  mark, each read with the looks the sentence's path takes of itself (one
  for the word, two for the word and its mark); the answer returned
  (`BatchResult.outputPred`) is read;
- with no grammar (a model that reads, `_grammar_free_text_model`), the
  same two, read as one whole (one look); that model has no answer in
  words, so its cells are read by the text its reconstruction puts back.

The pair is supplied as a supervised question is: the data's answers table
(`text_answers['train']`, the text at the question's index) with
`has_supervised_outputs` true and the question `What.supervised(i)`. The
supplied answer enters only the cost (A); a measure takes no step
(`train=False`). Beside each reading, read only: the reconstruction's text,
R and A (the exploit path's `parts`), the looks (the input's word each took
and the row said) and the extents of the last look's slots, whether the
sentence's state has a meaning (`entry.meaning`), the answer's meanings and
its source (`derivation`), and what the output walk emitted. Taught by the
model's own training step on the same pairs, round by round, until every
text is echoed in two rounds running or for ``ROUNDS`` rounds (3 by
default; the measure's patience, no bar of a gate). A start is drawn afresh
and printed; nothing is asserted of what a start gives but the echo."""
import os

import torch

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


def text(t):
    return bytes(int(v) for v in t.detach().reshape(-1).tolist() if 0 < int(v) < 256).decode('latin1')


def answer_text(value):
    if value is None or value.is_floating_point():
        return None
    return bytes(value.detach().reshape(-1).tolist()).rstrip(b'\0').decode('latin1')


class Observed:
    """Reads, never writes: the reconstruction's text, the answer's
    resolution, the walk's emitted words and the sentence's two costs."""
    def __init__(self, model, monkeypatch):
        import SentenceField
        from OutputOccurrences import Generated
        self.row = {}
        one, resolve, texts = SentenceField.reconstruction, model._resolve_answer, Generated.texts
        import WholePaths
        finish = WholePaths.finish

        def reconstructed(model_, field, **kw):
            result = one(model_, field, **kw)
            self.row['reconstruction'] = text(result[1].materialize()[0])
            return result

        def resolved(understanding, question, **kw):
            fields = understanding.sentence_states or ()
            self.row['state'] = ['none' if entry is None else
                                 'no meaning' if entry.meaning is None else 'meaning' for entry in fields]
            result = resolve(understanding, question, **kw)
            derivation = result[0] if isinstance(result, tuple) else result
            meanings = getattr(derivation, 'answer_meanings', None)
            self.row['answer meanings'] = None if meanings is None else [len(m) for m in meanings]
            self.row['answer source'] = getattr(derivation, 'row_sources', None)
            return result

        def emitted(generation, surface):
            words = texts(generation, surface)
            self.row.setdefault('walk emitted', []).append((generation.counts.tolist(), words))
            return words

        def finished(*args, **kw):
            path = finish(*args, **kw)
            # The path's two primary costs (`SentenceCredit.components`):
            # R, the expectation, A. The first path is the exploit.
            self.row.setdefault('R, A', tuple(round(float(path.parts[0, i]), 4) for i in (0, 2)))
            return path
        monkeypatch.setattr(SentenceField, 'reconstruction', reconstructed)
        monkeypatch.setattr(model, '_resolve_answer', resolved)
        monkeypatch.setattr(Generated, 'texts', emitted)
        monkeypatch.setattr(WholePaths, 'finish', finished)
        from SentenceLooks import Looks
        run = Looks.run

        def looked(path, *args, **kw):
            result = run(path, *args, **kw)
            # The first path's looks: the input's word each look took
            # (its index, -1 none) and the row said (-1 none); and the
            # extents (start, end) of the slots of the last look's field.
            if 'looks' not in self.row:
                self.row['looks'] = [(int(d['index'].reshape(-1)[0]), int(d['said'].reshape(-1)[0]))
                                     for d in path.decisions]
                where = getattr(path.fields[-1]['field'], 'where', None) if path.fields else None
                self.row['extents'] = None if where is None else [
                    tuple(int(v) for v in pair) for pair in where[0].reshape(-1, 2).tolist() if pair[1] > pair[0]]
            return result
        monkeypatch.setattr(Looks, 'run', looked)


def echo(model, index, word, *, train, optimizer):
    from What import What
    question = What.supervised(index, split='train')
    with torch.set_grad_enabled(train):
        batch, _ = model.runBatch(train=train, optimizer=optimizer, batchSize=1, split='train',
                                  questions=(question,),
                                  batch_override=(model.inputSpace.prepInput([word]), torch.empty(1, 0)))
    return batch


def read_once(model, index, word, *, answered, train, optimizer):
    """One reading; what the model returns: the answer where the model
    answers, else ``None`` (its reconstruction is read)."""
    from test_item18_expect import read
    try:
        if answered:
            batch = echo(model, index, word, train=train, optimizer=optimizer)
            return answer_text(None if batch is None else batch.outputPred)
        with torch.set_grad_enabled(train):
            read(model, word, training=train, optimizer=optimizer)
        return None
    finally:
        model.End()


def cell(tmp_path, monkeypatch, *, mark, grammar):
    from reading_fixtures import use_eager_reading
    from test_item17_one_reconstruction import _grammar_free_text_model
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    texts = tuple(w + mark for w in WORDS)
    print('start', start, '| cell: the word', 'with its mark' if mark else 'with no mark',
          '| grammar' if grammar else '| no grammar (read by the reconstruction)')
    model = (model_for if grammar else _grammar_free_text_model)(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        # The same words in every cell: each nominated alone and with its mark.
        nominate(model, WORDS + tuple(w + '.' for w in WORDS))
        if grammar:
            data = model.inputSpace.data
            monkeypatch.setattr(data, 'has_supervised_outputs', True)
            monkeypatch.setattr(data, 'text_answers', {'train': list(texts), 'validation': []})
        observed = Observed(model, monkeypatch)
        optimizer = model.getOptimizer(lr=.01)
        rounds = int(os.environ.get('ROUNDS', '3'))
        running, curve, echoed = 0, [], 0
        for r in range(rounds + 1):
            if r:
                for index, word in enumerate(texts):
                    read_once(model, index, word, answered=grammar, train=True, optimizer=optimizer)
            echoed, R, A = 0, [], []
            for index, word in enumerate(texts):
                observed.row.clear()
                with torch.no_grad():
                    answer = read_once(model, index, word, answered=grammar, train=False, optimizer=optimizer)
                said = answer if grammar else observed.row.get('reconstruction')
                echoed += said == word
                if 'R, A' in observed.row:
                    R.append(observed.row['R, A'][0]); A.append(observed.row['R, A'][1])
                print(f'   {word!r}: returned {said!r} | answer {answer!r} | '
                      + ' | '.join(f'{k} {v}' for k, v in observed.row.items()))
            mean = (round(sum(R) / len(R), 4) if R else None, round(sum(A) / len(A), 4) if A else None)
            curve.append((r, echoed, *mean))
            print(f' round {r}, start {start}: echoed {echoed} of {len(texts)} | mean R, A {mean}', flush=True)
            running = running + 1 if echoed == len(texts) else 0
            if running == 2:
                break
        print('curve (round, echoed, mean R, mean A):', start, curve)
        assert echoed == len(texts), f'{echoed} of {len(texts)} echoed; curve {curve}'
    finally:
        model.End()


def test_echo_a_word_with_its_own_look(tmp_path, monkeypatch):
    cell(tmp_path, monkeypatch, mark='', grammar=True)


def test_echo_a_word_and_its_mark_with_its_own_looks(tmp_path, monkeypatch):
    cell(tmp_path, monkeypatch, mark='.', grammar=True)


def test_echo_a_word_read_whole_by_a_model_that_reads(tmp_path, monkeypatch):
    cell(tmp_path, monkeypatch, mark='', grammar=False)


def test_echo_a_word_and_its_mark_read_whole_by_a_model_that_reads(tmp_path, monkeypatch):
    cell(tmp_path, monkeypatch, mark='.', grammar=False)
