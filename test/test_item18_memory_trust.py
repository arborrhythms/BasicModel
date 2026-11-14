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
        row=store.append_idea(torch.ones(12))
        write(store,row,(torch.tensor(ids),torch.tensor(values)))
    def cue(pairs):
        return ConceptualField(torch.tensor([10,20]),torch.tensor(pairs)[:,None,None,:],None,None,None)
    found=query(store,cue([[1.,0.],[1.,0.]]),max_records=8)
    assert [frame['index'] for frame in found['value']] == [0,1]
    assert [frame['match'] for frame in found['value']] == [.75,.25]
    found=query(store,cue([[0.,0.],[0.,1.]]),max_records=8)
    assert [frame['index'] for frame in found['value']] == [1]
    store.compact_rows([1,2])
    assert 10 in store.episodic_concepts[0]
    assert [frame['index'] for frame in query(store,cue([[0.,0.],[0.,1.]]),max_records=8)['value']] == [0]


def test_ltm1_drops_named_legacy_column_on_checkpoint():
    from Memory import LTM1
    original=LTM1(capacity=8)
    state=original.state_dict()
    state['percept']=torch.ones(8,8)
    restored=LTM1(capacity=8)
    with pytest.warns(UserWarning,match='dropped retired field or cache percept'):
        restored.load_state_dict(state)
    assert not hasattr(restored,'percept')
