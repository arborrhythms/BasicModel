"""The row at 1.00 of doc/ModelAge.md: a word said back, alone (round 6d of
item 1.7, twelfth pass, steps 1 and 2). Records, not repairs: a start drawn
afresh and printed; no assertion on the outcome of a draw or of learning."""
import os

import torch

from test_item17_own_reading import Paths, fresh_start, reading_of, taught
from test_item17_places import LESSON, minimal, model_for, nominate, words_of
from test_item17_own_reading import COUNTING

WORDS = words_of(LESSON) + COUNTING[:6]


class Probe:
    """At each look of a presentation: the request's range, each unread
    word's share, the cost of the word said, and the gradient reaching the
    word said (at the say read's units) and the request."""
    def __init__(self, monkeypatch):
        import SentenceLooks
        from RequestRead import RequestRead
        self.looks, self.inputs = [], []
        attend, said = SentenceLooks.attend, RequestRead.said
        probe = self
        def requested(model, units):
            request = attend(model, units)
            if request.requires_grad:
                request.retain_grad()
            probe.looks.append(dict(request=request))
            return request
        def spoken(read, x, seen):
            probe.inputs.append((x.detach().clone(), seen.detach().clone()))
            hidden = torch.relu(read.second(torch.cat((x, seen.to(x)), -1)))
            if hidden.requires_grad:
                hidden.retain_grad()
            if probe.looks:
                V = read.rows
                probe.looks[-1].update(units=hidden, seen=seen.detach(),
                                       seen_row=int(seen[0, :V].argmax()), seen_mass=round(float(seen[0, :V].sum()), 3),
                                       opened_row=int(x[0, 0:2 * V:2].argmax()), opened_max=round(float(x[0, 0:2 * V:2].max()), 3))
            logits = read.say(hidden)
            if probe.looks:
                probe.looks[-1].update(raw=int(logits[0].argmax()), raw_top=round(float(logits[0].max()), 3))
            return logits
        import SentenceField
        reconstruct = SentenceField.reconstruct
        self.rebuilt = []
        def rebuilt(model, record, errors, active, **kw):
            result = reconstruct(model, record, errors, active, **kw)
            from Omission import target as omission_target
            targets = omission_target(model)
            raw = (model._staged_concepts_in if targets is None else targets['raw'])[0, 0]
            text = lambda t: bytes(int(v) for v in t.detach().reshape(-1).tolist() if 0 < int(v) < 256).decode('latin1')
            probe.rebuilt.append(dict(raw=text(raw), input=text(model._input_reconstructions[int(record.sentence)].event[0, 0]),
                                      walk=text(model._sentence_order_zero_reverse.materialize()[0, 0])))
            return result
        monkeypatch.setattr(SentenceField, 'reconstruct', rebuilt)
        monkeypatch.setattr(SentenceLooks, 'attend', requested)
        monkeypatch.setattr(RequestRead, 'said', spoken)

    def summary(self):
        grad = lambda t: None if t is None or t.grad is None else round(float(t.grad.abs().sum()), 6)
        rows = []
        for look in self.looks:
            r = look['request'].detach()[..., 0]
            rows.append(dict(asked=(round(float(r.min()), 4), round(float(r.max()), 4)),
                             request_grad=grad(look['request']), units_grad=grad(look.get('units')),
                             **{k: look[k] for k in ('seen_row', 'seen_mass', 'opened_row', 'opened_max', 'raw', 'raw_top') if k in look}))
        return rows


def presentation(model, paths, optimizer, word, probe):
    paths.seen.clear()
    probe.looks.clear()
    taught(model, optimizer, word + '.')
    path = paths.seen[0]
    out = path.output
    cost = None if out is None else round(float(out.cost()[0]), 4)
    shares = [d['scores'][0].tolist() for d in path.decisions]
    return dict(cost=cost, looks=probe.summary(), scores=shares)


def test_one_word_and_then_the_nineteen(tmp_path, monkeypatch):
    """Step 1: one word taught alone, then the 19; printed by presentation."""
    start = fresh_start()
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        paths = Paths(monkeypatch)
        probe = Probe(monkeypatch)
        if os.environ.get('NO_SAY_COST'):
            # Compared in the test: the looks with no output cost of the say read.
            from SentenceLooks import Looks
            run = Looks.run
            def quiet(path, arrive, **kw):
                result = run(path, arrive, **kw)
                path.teaching = False
                return result
            monkeypatch.setattr(Looks, 'run', quiet)
        optimizer = model.getOptimizer(lr=.01)
        print('start', start)
        for k in range(int(os.environ.get('ONE_WORD_PRESENTATIONS', '12'))):
            print('one', k, presentation(model, paths, optimizer, 'Felix', probe))
            probe.looks.clear(); probe.rebuilt.clear()
            r = reading_of(model, paths, 'Felix.')
            print('  rebuilt', probe.rebuilt[-1:])
            print('  teaching', paths.seen[0].teaching, 'said', r['said'], 'targets', r['targets'], 'right', r['said_right'], probe.summary())
        rounds = int(os.environ.get('NINETEEN_ROUNDS', '0'))
        dump = os.environ.get('SAID_DUMP')
        if dump:
            # The say read's inputs at the first look of each word, read at
            # rate nothing, and its targets: to fit the read alone.
            rows = []
            for w in WORDS:
                probe.inputs.clear()
                r = reading_of(model, paths, w + '.')
                rows.append((probe.inputs[0][0], probe.inputs[0][1], r['targets'][0]))
            torch.save(rows, dump)
        def measured(k):
            said = walk = whole = 0
            missed = []
            for w in WORDS:
                probe.rebuilt.clear()
                said += reading_of(model, paths, w + '.')['said_right']
                r = probe.rebuilt[-1]
                walk += r['walk'] == w
                whole += r['walk'] == r['raw']
                if r['walk'] != w:
                    missed.append((w, r['walk']))
            print('round', k, 'say read right', said, 'walk gives the word', walk, 'walk gives word and mark', whole,
                  'missed', missed[:6])
        for mark in ('.', '?'):
            try:
                probe.rebuilt.clear()
                r = reading_of(model, paths, mark)
                print('mark alone', repr(mark), probe.rebuilt[-1:], 'looks', r['looks'], 'words', r['words'])
            except Exception as error:
                model.End()
                print('mark alone', repr(mark), 'raised', type(error).__name__, str(error)[:200])
        measured(-1)
        for k in range(rounds):
            costs = [presentation(model, paths, optimizer, w, probe)['cost'] for w in WORDS]
            print('costs', k, costs)
            measured(k)
    finally:
        model.End()
