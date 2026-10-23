"""Observe the two order stamps at the actual identity admission boundary."""
import json
import os
from pathlib import Path
import pytest
from arrival_capture import plain

@pytest.fixture(autouse=True)
def observe_admission(request,monkeypatch):
    from Models import FunctionalPeerSTM
    from MTMOverlay import MTMOverlay
    resolve,push=FunctionalPeerSTM.resolve_top_reference,MTMOverlay.push
    records=[]
    def output():
        (Path(os.environ['CONTRACT_CAPTURE']).parent/'admission-probe.json').write_text(json.dumps(plain(records),indent=2)+'\n')
    def resolved(state,*args,**kwargs):
        value=resolve(state,*args,**kwargs)
        records.append(dict(call='resolve',before=state[2][:,0],after=value[2][:,0],
                            object_order=args[2],gate=args[4]))
        output()
        return value
    def pushed(state,active,value,**kwargs):
        result=push(state,active,value,**kwargs)
        records.append(dict(call='push',order=kwargs['order'],index=kwargs['index'],active=active,
                            handle=result[2][:,0,0]))
        output()
        return result
    monkeypatch.setattr(FunctionalPeerSTM,'resolve_top_reference',staticmethod(resolved))
    monkeypatch.setattr(MTMOverlay,'push',staticmethod(pushed))
