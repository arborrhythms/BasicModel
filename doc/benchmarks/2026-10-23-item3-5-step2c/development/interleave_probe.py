"""Observe the captured shared-owner failure without changing a choice."""
import gzip
import json
import os
from pathlib import Path
import pytest
from arrival_capture import plain

@pytest.fixture(autouse=True)
def observe_introductions(request, monkeypatch):
    if 'test_one_native_read_retains_shared_owner' not in request.node.name:
        return
    from ReferenceClose import ReferenceClose
    from Models import BasicModel
    from identity_keep_probe import clause_graph
    root = Path(os.environ['CONTRACT_CAPTURE']).parent/'interleave-probe'
    root.mkdir()
    sites, commit, preview = ReferenceClose.sites, BasicModel._commit_sentence, ReferenceClose.preview
    records = []
    def write():
        (root/'observations.json.gz').write_bytes(gzip.compress(json.dumps(plain(records),indent=2).encode(),mtime=0))
    def observed_sites(resolver, clause, **kwargs):
        result = sites(resolver,clause,**kwargs)
        records.append(dict(call='sites',sites=[dict(identity=s['node'].identity,
            request=s['node'].reference_request,order=s['node'].order,
            operation=getattr(s['node'].composition,'operation',None)) for s in result]))
        return result
    def observed_preview(resolver,clause,*args,**kwargs):
        try:
            return preview(resolver,clause,*args,**kwargs)
        except Exception as error:
            records.append(dict(call='preview-error',error=repr(error),clause=clause_graph(clause)))
            raise
        finally:
            write()
    def observed_commit(model,state,sid,active,observations,predictions,wins):
        records.append(dict(call='keep',sentence=sid,decision=wins,
            credit=getattr(model,'_last_sentence_credit',None),
            trials=[dict(clauses=[clause_graph(c) for c in o['clauses']],plans=o.get('reference_plans')) for o in observations]))
        write()
        return commit(model,state,sid,active,observations,predictions,wins)
    monkeypatch.setattr(ReferenceClose,'sites',observed_sites)
    monkeypatch.setattr(ReferenceClose,'preview',observed_preview)
    monkeypatch.setattr(BasicModel,'_commit_sentence',observed_commit)
