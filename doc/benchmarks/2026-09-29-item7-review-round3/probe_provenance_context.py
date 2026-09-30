"""Observe the existing withdrawal/retention contract without changing it."""
import json
from pathlib import Path
import warnings

from test_ltm_consolidation import TestRuntimeUserIngestion


def test_observe_provisioning_and_resubmission():
    model, store, view = TestRuntimeUserIngestion()._fresh()
    observations = []
    original = store.clear_origin

    def snapshot():
        return [dict(index=i, occurrence=int(store.occurrence_id[i]),
                     identity=int(store.row_ids[i]), refs=store.refs[i].tolist(),
                     origin=int(store.origin[i]), kind=store.row(i)['kind'],
                     trust=float(store.trust[i]), text=store.text_of(i),
                     timestamp=float(store.timestamp[i]), relation=int(store.rel_type[i]))
                for i in range(len(store))]

    def observe(origin, **kwargs):
        before = snapshot()
        result = original(origin, **kwargs)
        observations.append(dict(roots=kwargs.get('retained_occurrences'),
                                 before=before, after=snapshot(), removed=result))
        return result

    store.clear_origin = observe
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.store_truths([{'content': 'hello world', 'trust': .9},
                            {'content': 'loving there', 'trust': -.5}])
        model.store_truths([{'content': 'hello there', 'trust': .7}])
    report = dict(observations=observations, final=snapshot(),
                  view_sources=view._sources, view_count=int(view.count))
    (Path(__file__).parent / 'provenance-context.json').write_text(
        json.dumps(report, indent=2) + '\n')
    active = [i for i in store.rows_of_origin(store.ORIGIN_USER).tolist()
              if store.row(i)['kind'] == 'fact']
    assert len(active) == 1
    assert store.text_of(active[0]) == 'hello there'
    for i in store.rows_of_origin(store.ORIGIN_USER).tolist():
        if i not in active:
            assert store.row(i)['kind'] == 'unverified'
            assert float(store.trust[i]) == 0.
    assert 'hello world' not in view._sources
