"""Observe the kept close and capacity eviction without changing choices."""
import gzip
import json
import os
from pathlib import Path

import pytest
from arrival_capture import plain, meaning
from identity_keep_probe import clause_graph


@pytest.fixture(autouse=True)
def observe_capacity(request, monkeypatch):
    if 'ordinary_driver_two_three_eight_stream_groups' not in request.node.name:
        return
    from Models import BasicModel
    from MediumTermMemory import MediumTermMemory
    from ReferenceClose import ReferenceClose
    root = Path(os.environ['CONTRACT_CAPTURE']).parent/'capacity-probe'
    root.mkdir()
    count, journal = 0, []
    original_commit, original_open = BasicModel._commit_sentence, MediumTermMemory.open
    original_events = ReferenceClose._commit_events

    def opening(pool, *args, **kwargs):
        before = set(pool.entries)
        result = original_open(pool, *args, **kwargs)
        journal.append(dict(operation='open', occurrence=args[0], token=result,
            evicted=tuple(before-set(pool.entries)), count=len(pool.entries), capacity=pool.capacity))
        return result

    def events(resolver, plan, written, **kwargs):
        pool = resolver.components.owner.mtm
        introductions = {}
        for item in plan['items']:
            node = item['written_node']
            if node.identity is None or not node.identity['handle']:
                continue
            occurrence = written.get(('occurrence', id(node)), written.get('root_occurrence')
                if node is plan['prepared_root'] else None)
            introductions[node.identity['handle']] = dict(occurrence=occurrence,
                retained=pool.token_of(occurrence), origin=node.identity['origins'][node.identity['handle']-1])
        journal.append(dict(operation='events', introductions=introductions,
            mentions=len(plan['items']), pool_count=len(pool.entries), pool_capacity=pool.capacity))
        return original_events(resolver, plan, written, **kwargs)

    def commit(model, state, sid, active, observations, predictions, wins):
        nonlocal count
        count += 1
        record = dict(source=os.environ['ARRIVAL_KEEP_SOURCE'], sentence=sid,
            inputs=model.inputSpace.data.train_input, sources=getattr(model,'_sentence_sources',None),
            keep_decision=wins, credit=getattr(model,'_last_sentence_credit',None),
            trials=[dict(meanings=[meaning(m) for m in view['meanings']],
                clauses=[clause_graph(c) for c in view['clauses']],
                plans=[{k:p.get(k) for k in ('items','decisions','arrivals','cost')}
                    for p in view.get('reference_plans',())]) for view in observations])
        try:
            return original_commit(model,state,sid,active,observations,predictions,wins)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            record['operations'] = list(journal)
            (root/f'close-{count:03}.json.gz').write_bytes(gzip.compress(
                json.dumps(plain(record),indent=2).encode(),mtime=0))
            journal.clear()
    monkeypatch.setattr(MediumTermMemory,'open',opening)
    monkeypatch.setattr(ReferenceClose,'_commit_events',events)
    monkeypatch.setattr(BasicModel,'_commit_sentence',commit)
