"""Replay and observe the frozen rung-zero failure without changing choices."""
import json,os,random
from pathlib import Path
import numpy as np
import pytest,torch

@pytest.fixture(autouse=True)
def replay_rung0(monkeypatch):
    root=Path(os.environ['RUNG0_CAPTURE']);root.mkdir(parents=True,exist_ok=False)
    state=torch.load(os.environ['RUNG0_REPLAY'],weights_only=False)
    torch.set_rng_state(state['torch']);random.setstate(state['python']);np.random.set_state(state['numpy'])
    from Models import BasicModel
    original=BasicModel._prime_sentence_symbols
    calls=[]
    def probe(model,sid):
        try:
            result=original(model,sid)
        except Exception as error:
            trace=error.__traceback__
            while trace and trace.tb_frame.f_code.co_name!='_prime_sentence_symbols':trace=trace.tb_next
            values={} if trace is None else trace.tb_frame.f_locals
            owner=model._concept_owner();identity=values.get('identity')
            tensors={k:v.detach().cpu().clone() for k,v in values.items() if torch.is_tensor(v)}
            torch.save(dict(locals=tensors,owner=owner.state_dict()),root/'failed-read.pt')
            rows=[]
            for b,batch in enumerate(values.get('surfaces',())):
                for j,raw in enumerate(batch):
                    selected=int(values['selected'][b,j]); form=int(values['form_rows'][b,j])
                    decoded=None if identity is None else identity.read(values['sparse_forms'][b,j,:identity.width])
                    rows.append(dict(stream=b,column=j,selected=selected,form_row=form,
                        symbol=owner.concept_id_at_row(selected) if selected>=0 else None,
                        form_symbol=owner.concept_id_at_row(form) if form>=0 else None,
                        expected=None if raw is None else raw.hex(),
                        decoded=None if decoded is None else decoded.hex()))
            from Spaces import _concept_alloc_of
            alloc=_concept_alloc_of(owner)
            diagnostics=dict(pending=owner.interpret._pending, descriptions=alloc.word_descriptions, words={str(row['symbol']):owner.definitions.words(row['symbol']) for row in rows if row['symbol'] is not None}, definitions=dict(owner.definitions._descriptions))
            (root/'definitions.txt').write_text(repr(diagnostics))
            data=dict(sid=int(sid),calls=calls,error=str(error),audit=getattr(model,'_last_rung0_identity_audit',None),
                rows=rows,word_ids=sorted(owner._word_definition_ids()),
                csw_rows={str(k):v for k,v in owner._csw_rows.items()})
            (root/'failure.json').write_text(json.dumps(data,indent=2)+'\n')
            raise
        calls.append(dict(sid=int(sid),audit=getattr(model,'_last_rung0_identity_audit',None)))
        (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        return result
    monkeypatch.setattr(BasicModel,'_prime_sentence_symbols',probe)
