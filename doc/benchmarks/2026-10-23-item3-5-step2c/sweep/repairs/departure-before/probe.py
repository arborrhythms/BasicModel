"""Observe the original draw, the fixture's forced draw, and both trial reads."""
import json
import os
from pathlib import Path

import pytest
from arrival_capture import plain


@pytest.fixture(autouse=True)
def observe_departure(request, monkeypatch):
    if 'test_zero_meaning_departures_and_one_reader_step' not in request.node.name:
        return
    import SentenceCredit
    from SentenceFork import SentenceFork
    from Models import BasicModel
    from identity_keep_probe import clause_graph
    target = Path(os.environ['CONTRACT_CAPTURE']).parent/'departure-probe.json'
    records = []
    def record(value):
        records.append(plain(value))
        target.write_text(json.dumps(records, indent=2)+'\n')
    departure, start = SentenceCredit.departure, SentenceFork.start
    cost, commit = BasicModel._sentence_path_cost, BasicModel._commit_sentence
    epoch = BasicModel.runEpoch
    forks = []
    def draw(*args, **kwargs):
        result = departure(*args, **kwargs)
        record(dict(call='original_draw', draw=result))
        return result
    def begin(fork, draw):
        result = start(fork, draw)
        forks.append(fork)
        record(dict(call='forced_draw_at_fork', draw=draw,
                    slot=fork.slot, action=fork.action, selected=fork.selected))
        return result
    def score(model, *args, **kwargs):
        result = cost(model, *args, **kwargs)
        record(dict(call='trial', trial=model._sentence_trial,
            clauses=[clause_graph(c) for c in result[2]['clauses']],
            answer=model._sentence_answer_cost,
            registry=model._sentence_cost_registry.breakdown()))
        return result
    def keep(model, state, sid, active, observations, predictions, wins):
        record(dict(call='keep', wins=wins, credit=model._last_sentence_credit))
        return commit(model, state, sid, active, observations, predictions, wins)
    def run(model, *args, **kwargs):
        try:
            return epoch(model, *args, **kwargs)
        finally:
            record(dict(call='exit', forks=[dict(slot=f.slot, action=f.action,
                selected=f.selected, joined=f.joined, taken=f.taken) for f in forks]))
    monkeypatch.setattr(SentenceCredit, 'departure', draw)
    monkeypatch.setattr(SentenceFork, 'start', begin)
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', score)
    monkeypatch.setattr(BasicModel, '_commit_sentence', keep)
    monkeypatch.setattr(BasicModel, 'runEpoch', run)
