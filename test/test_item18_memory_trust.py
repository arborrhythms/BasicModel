"""Content recall and kept-tree authority certificates."""
from types import SimpleNamespace
import pytest
import torch


def test_episode_query_uses_signed_references_and_survives_compaction():
    from Memory import LTM0
    from EpisodicMemory import write,query
    from Understanding import ConceptualField
    store = LTM0(12,capacity=8,episodic_concepts=3)
    for i,(ids,values) in enumerate([([10,20,-1],[1.,.5,0.]),([10,20,-1],[.5,-1.,0.]),([30,-1,-1],[1.,0.,0.])]):
        row=store.append_idea(torch.ones(12),kind='episode',document_key=('entry',i))
        write(store,row,(torch.tensor([ids]),torch.tensor([values])),new=True)
    def cue(pairs):
        return ConceptualField(torch.tensor([10,20]),torch.tensor(pairs)[:,None,None,:],None,None,None)
    found=query(store,cue([[1.,0.],[1.,0.]]),max_records=8)
    assert [frame['index'] for frame in found['value']] == [0,1]
    assert [frame['match'] for frame in found['value']] == [.75,.25]
    found=query(store,cue([[0.,0.],[0.,1.]]),max_records=8)
    assert [frame['index'] for frame in found['value']] == [1]
    store.compact_rows([1,2])
    assert 10 in store.episodic_concepts[0, 0]
    assert [frame['index'] for frame in query(store,cue([[0.,0.],[0.,1.]]),max_records=8)['value']] == [0]


def test_ltm1_drops_named_legacy_column_on_checkpoint():
    from Memory import LTM1
    original=LTM1(capacity=8)
    state=original.state_dict()
    state['percept']=torch.ones(8,8)
    restored=LTM1(capacity=8)
    with pytest.warns(UserWarning,match='dropped retired percept'):
        restored.load_state_dict(state)
    assert not hasattr(restored,'percept')
    assert restored.percept_migration == 'dropped percept'


def test_operator_gates_follow_kept_ancestors_and_checkpoint(monkeypatch):
    import Language
    from TrustGates import TrustGates
    from util import TheXMLConfig
    rules=[SimpleNamespace(method_name=name,surface_name=None,lhs=lhs,clause_form=form)
           for name,lhs,form in [('lift','S','S'),('compound','NP',None),('adverb','VP',None),('interpret','W',None)]]
    monkeypatch.setattr(Language.TheGrammar,'_ensure_configured',lambda:None)
    monkeypatch.setattr(Language.TheGrammar,'rules_upward',rules)
    monkeypatch.setattr(Language.TheGrammar,'ws_start_patterns',(('S',),))
    original=TheXMLConfig.space
    monkeypatch.setattr(TheXMLConfig,'space',lambda space,key,*args,**kw:dict(closedTrust=.1,threshold=.75)
                        if key=='gates' else original(space,key,*args,**kw))
    gates=TrustGates()
    assert gates.open['lift'] and gates.open['interpret'] and not gates.open['compound']
    gates.observe('lift',{'compound'})
    gates.observe('compound',{'adverb'})
    readiness={'lift':.8,'compound':.3}
    occupations=SimpleNamespace(operations=tuple(readiness),operation=lambda name:dict(measurable=True,definiteness=(readiness[name],)))
    values=gates.measurements(None,occupations)
    assert gates.advance('compound',values['compound'],when=4)
    assert not gates.advance('adverb',values['adverb'],when=4)
    assert not gates.advance('compound',0.,when=5)
    restored=TrustGates();restored.restore(gates.state())
    assert restored.open == gates.open and restored.parents == gates.parents and restored.opened_at['compound']==4
    assert restored.cap(named=False,operation='adverb') == .1
