"""Failure-only native state observer; no seed, budget or execution changes."""
import json
import os
from pathlib import Path
import pytest


def plain(value, depth=0):
    import torch
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.tolist() if value.numel() < 2000 else dict(shape=list(value.shape),
            min=float(value.min()), max=float(value.max()), nonzero=int(value.count_nonzero()))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth > 4:
        return str(type(value))
    if isinstance(value, dict):
        return {str(k):plain(v,depth+1) for k,v in value.items()}
    if isinstance(value, (tuple,list,set)):
        return [plain(v,depth+1) for v in value]
    return repr(value)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    data = dict(test=item.nodeid, frames=[])
    for frame in call.excinfo.traceback:
        local = frame.frame.f_locals
        if frame.name in ('compose', 'reduce', 'assert_cube'):
            data['frames'].append(dict(name=frame.name,values=plain({k:local[k] for k in
                ('value','left','right','window') if k in local})))
        if frame.path == item.path:
            data['locals'] = plain({k:local[k] for k in
                ('identities','references','document','distinct','contents','occurrences','snapshots') if k in local})
            if 'resolver' in local:
                resolver = local['resolver']; store=resolver.store
                data['participants'] = plain(resolver._participants())
                data['rows'] = [plain(dict(id=int(store.row_ids[r]),order=int(store.order[r]),
                    refs=store.raw_meaning_of(r).role_refs,point=store.point_of_row(int(store.row_ids[r])),
                    metadata=store.raw_meaning_of(r).metadata())) for r in range(len(store))]
            model = local.get('model')
            if model is not None:
                owner=model._concept_owner();alloc=owner._concept_allocator
                data['word_state']=plain(dict(rows=alloc.layer()._tensor_rows,
                    features=alloc.layer().features._index, weights=alloc.layer().features.values,
                    definitions=owner.definitions._rows, pending=owner.interpret._pending,
                    native_poles=getattr(model,'_attention_native_poles',None),
                    field_ids=getattr(owner,'_cs_field_concept_ids',None),
                    field_rows=getattr(owner,'_cs_field_rows',None),
                    public=getattr(getattr(model,'symbol_cache',None),'_concept_activations',None),
                    literal_ids=getattr(getattr(model,'_sentence_primed_bank',None),'literal_ids',None)))
    with Path(os.environ['SWEEP_DIAGNOSTIC']).open('a') as out:
        out.write(json.dumps(data)+'\n')
