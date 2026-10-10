"""Positive adapter control using a forced journal and the real clause writer.

This checks measurement plumbing, never learned parsing or identity accuracy.
Run from the BasicModel root with PYTHONPATH=bin:test and MODEL_COMPILE=none.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from ClauseJournal import finish_clause
from identity_measurement import native_observer
from reading_fixtures import record_reading, resolve_reading_references
from test_clause_acceptance import SentenceFixture

torch.set_num_threads(1)
with pytest.MonkeyPatch.context() as patch:
    fixture=SentenceFixture(patch)
    entry=fixture.program(('lift',('lower','a','cat'),'runs'))
    entry=record_reading(fixture.language,resolve_reading_references(
        fixture.language,entry,forced={1:-1}))
    clause=finish_clause(fixture.language,entry,registry=fixture.registry)
    model=SimpleNamespace(languageSpace=fixture.language,
        symbolSpace=SimpleNamespace(ltm_store=fixture.store),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1,3,dtype=torch.bool),
            _packed_sentence_ids=torch.zeros(1,3,dtype=torch.long)),
        _sentence_observation=lambda *a,**k:dict(entries=(entry,)),
        _derivation_program=lambda *a:(torch.arange(3)[None],))
    with native_observer(model) as captured:
        model._sentence_observation(None,0,None,admit=True)
        before=dict(captured[0][0]['tokens'][1])
        written={}
        fixture.store.write_clause(clause,stream=0,written_rows=written)
        after=dict(captured[0][0]['tokens'][1])
        index=fixture.store.index_of_row(after['reference'])
        def nodes(node):
            row=written.get(id(node))
            return [dict(order=node.order,word=node.subject_word_id,row=row,
                identity=None if row is None else int(fixture.store.row_ids[row]))]+[
                entry for child in (*node.children,*node.companions) for entry in nodes(child)]
        facts=dict(before=before,after=after,nodes=nodes(clause))
        Path(__file__).with_name('observer-check-facts.json').write_text(json.dumps(facts,indent=2)+'\n')
        print(json.dumps(facts),flush=True)
        assert before['requested'] and before['reference']==-1
        assert after['source']=='written individual' and index is not None
        assert after['reference']==int(fixture.store.row_ids[index])
    result=dict(protocol='forced journal / real writer; adapter control only',
        before=before,after=after,written_row=index,passed=True)
    Path(__file__).with_name('observer-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
