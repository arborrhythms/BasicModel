"""Measure actual predictor context reads in two unchanged small configurations.

MM_ladder explicitly disables sentence expectation, so its reconstruction
receipt reports no predictor reads. These supplementary native readings use
the default enabled predictor; they change no configuration or random seed.
"""
import json
from pathlib import Path
import pytest
from test_mm_xor import _fresh_model

HERE=Path(__file__).resolve().parent


@pytest.mark.parametrize('configuration',['MM_xor.xml','MM_grammar.xml'])
def test_definition_share_of_native_predictor_reads(monkeypatch,configuration):
    from Layers import InterSentenceLayer
    reports=[]
    original=InterSentenceLayer.expect_next_meaning
    def observe(self,b=0,**kwargs):
        before=self._inter_last_meaning[b]
        value=original(self,b,**kwargs)
        pending=self._inter_last_meaning[b]
        if pending is not None and pending is not before:
            store=self._ltm_store
            sources=tuple(getattr(pending,'source_occurrences',()))
            kinds=[]
            for occurrence in sources:
                row=None if store is None else store._index_occurrences.get(occurrence)
                kinds.append(None if row is None else int(store.rel_type[row]))
            reports.append(dict(source_count=len(sources),resolved_count=sum(x is not None for x in kinds),
                                definition_count=sum(x==getattr(store,'REL_DEF',-1) for x in kinds),
                                relation_kinds=kinds))
        return value
    monkeypatch.setattr(InterSentenceLayer,'expect_next_meaning',observe)
    m,_,_=_fresh_model(str(Path('data')/configuration))
    try:
        m.runEpoch(optimizer=None,batchSize=2,split='validation',max_batches=4)
    finally:
        store=m.symbolSpace.ltm_store
        payload=dict(configuration=configuration,seed=None,new_estimates=reports,
            store_rows=0 if store is None else len(store),
            definitions=0 if store is None else int((store.rel_type[:len(store)]==store.REL_DEF).sum()))
        (HERE/(configuration+'.definition-context.json')).write_text(json.dumps(payload,indent=2)+'\n')
        m.End()
        m.symbolSpace.soft_reset()
