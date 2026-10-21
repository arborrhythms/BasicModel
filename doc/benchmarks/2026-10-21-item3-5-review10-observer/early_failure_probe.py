"""Retain already-computed choices when a batch exits before its after hook."""
import json
from pathlib import Path
import pytest
from arrival_capture import plain, snapshot
from arrival_keep_probe import reading, clause_graph


@pytest.fixture(autouse=True)
def capture_early_failure(request, monkeypatch, capture_arrival, observe_arrival_keep):
    if not hasattr(request.node, '_arrival_directory'):
        return
    from Models import BasicModel
    observed = []
    cost, batch = BasicModel._sentence_path_cost, BasicModel.runBatch
    def priced(model, *args, **kwargs):
        result = cost(model, *args, **kwargs)
        value = result[2]
        observed.append(plain(dict(trial=model._sentence_trial,
            meanings=[reading(item) for item in value['meanings']],
            clauses=[clause_graph(item) for item in value['clauses']],
            plans=[dict(cost=item['cost'], decisions=item['decisions'], arrivals=item['arrivals'])
                   for item in value.get('reference_plans', ())],
            answer=model._sentence_answer_cost,
            registry=model._sentence_cost_registry.breakdown())))
        return result
    def run(model, *args, **kwargs):
        try:
            return batch(model, *args, **kwargs)
        except BaseException as error:
            directory = Path(request.node._arrival_directory)
            (directory/'failure-store.json').write_text(json.dumps(plain(snapshot(model)),indent=2)+'\n')
            (directory/'failure-early-keep.json').write_text(json.dumps(plain(dict(
                error=repr(error), readings=observed,
                credit=getattr(model, '_last_sentence_credit', None),
                field=getattr(model, '_last_sentence_field', None),
                thought=getattr(model, '_last_thought_comparison', None),
                reference_close_audit=getattr(model, '_reference_close_audit', None))),indent=2)+'\n')
            raise
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', priced)
    monkeypatch.setattr(BasicModel, 'runBatch', run)
