"""Observe every MM_grammar closing plan; capture the test's actual RNG.

No preflight, seating, random draw or model result is changed. Distinct plan
shapes are retained once, with all calls counted in the summary.
"""
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def capture_capacity(request, monkeypatch):
    if request.node.name != 'test_mm_grammar_learns_xor_signal':
        return
    from PartClosing import PartClosing
    folder = Path(os.environ['CAPACITY_RECEIPT'])
    folder.mkdir(parents=True, exist_ok=False)
    saved = os.environ.get('CAPACITY_REPLAY')
    if saved:
        rng = torch.load(saved, weights_only=False)
        torch.set_rng_state(rng['torch'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=np.random.get_state()), folder/'initial-rng.pt')
    import ClauseJournal
    finish_clause = ClauseJournal.finish_clause
    journals = set()

    def finish(language, program, **kwargs):
        clause = finish_clause(language, program, **kwargs)
        item = {name: None if getattr(program, name, None) is None else
                getattr(program, name).detach().cpu().tolist()
                for name in ('concept_ids', 'reference_ids', 'reference_orders',
                    'leaf_orders', 'actions', 'operation_refs', 'word_ids')}
        item.update(binding_choices=program.binding_choices,
                    close_references=program.close_references)
        key = json.dumps(item, sort_keys=True)
        if key not in journals:
            journals.add(key)
            with (folder/'journals.jsonl').open('a') as output:
                output.write(key+'\n')
        return clause

    monkeypatch.setattr(ClauseJournal, 'finish_clause', finish)
    original = PartClosing.preflight
    shapes = set()
    summary = dict(calls=0, seat_true=0, lower_nodes=0, lower_edges=0,
                   errors=0, inventory=set())

    def preflight(closing):
        summary['calls'] += 1
        summary['inventory'].add(int(closing.owner.nVectors))
        plan = [dict(symbol=r['symbol'], seat=r['seat'], edge=r['edge'],
            pair=r['pair'], types=r['types'], order=r['value'].order,
            reference=r['value'].reference, word=r['value'].word_id,
            operation=r['value'].operation, formal_name=r['value'].formal_name,
            children=len(r['value'].children)) for r in closing._plan]
        if os.environ.get('CAPACITY_IDENTITIES') == '1':
            native = [(int(identity), closing.owner._csw_row_of(int(identity)))
                      for identity in closing.alloc.placement if closing._native(identity)]
            native = [(identity, row) for identity, row in native if row is not None]
            codes = torch.stack([closing._point(identity) for identity, _ in native])
            identities = []
            for record in closing._plan:
                value = record['value']
                distance = torch.linalg.vector_norm(codes.to(value.point)-value.point, dim=-1)
                identities.append(dict(symbol=record['symbol'], reference=value.reference,
                    seat=record['seat'], operation=value.operation,
                    minimum_distance=float(distance.min()),
                    exact=[identity for (identity, _), gap in zip(native, distance) if float(gap) <= 1e-5]))
            with (folder/'identities.jsonl').open('a') as output:
                output.write(json.dumps(dict(call=summary['calls'], native=native,
                                              identities=identities))+'\n')
        summary['seat_true'] += sum(bool(row['seat']) for row in plan)
        summary['lower_nodes'] += sum(row['formal_name'] == 'lower' for row in plan)
        summary['lower_edges'] += sum(row['formal_name'] == 'lower' and row['edge'] is not None
                                      for row in plan)
        key = json.dumps(plan, sort_keys=True)
        if key not in shapes:
            shapes.add(key)
            with (folder/'plans.jsonl').open('a') as output:
                output.write(json.dumps(dict(call=summary['calls'],
                    next_id=closing.alloc.next_id, planned_end=closing._next,
                    capacity=closing.owner.nVectors, pools=closing.owner._order_caps(),
                    plan=plan))+'\n')
        try:
            return original(closing)
        except BaseException:
            summary['errors'] += 1
            raise

    monkeypatch.setattr(PartClosing, 'preflight', preflight)
    yield
    summary['inventory'] = sorted(summary['inventory'])
    summary['distinct_plans'] = len(shapes)
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
