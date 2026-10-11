"""The active-column metric and matched factorial control are audit contracts."""
from collections import Counter
import torch
from identity_experiment import identification, recovery_data, recovery_value, restore_warmup
from test_independent_components import dictionary, witness


def test_identification_requires_the_matching_column_to_be_active_in_this_row():
    atoms=torch.eye(4)[:2]
    columns=dictionary();witness(columns,atoms[0]);witness(columns,atoms[1],start=4)
    # Both atoms have a perfect column. Only the first participates in this row.
    result=identification(columns,atoms[0],atoms,source_limit=2)
    assert result['hits']==[True,False] and result['correct']==1 and result['total']==2
    # A memorized whole is not recovery of its atoms.
    whole=dictionary();witness(whole,atoms.sum(0))
    assert identification(whole,atoms.sum(0),atoms,source_limit=2)['correct']==0


def test_recovery_scores_the_completed_idea_including_its_fused_predicate():
    from types import SimpleNamespace
    atoms = torch.eye(4)[:3]
    chart = SimpleNamespace(project=lambda x:x, expand=lambda x:x)
    meaning = SimpleNamespace(roles=atoms)
    clause = SimpleNamespace(point=atoms.sum(0))
    columns=dictionary()
    for i,atom in enumerate(atoms):witness(columns,atom,start=4*i)
    fused = recovery_value(chart,meaning,clause)
    assert identification(columns,fused,atoms,source_limit=3)['hits']==[True,True,True]
    clause.point = None
    relative = recovery_value(chart,meaning,clause)
    assert identification(columns,relative,atoms,source_limit=3)['hits']==[True,False,True]


def test_recovery_control_has_equal_budgets_and_held_out_factor_combinations():
    streams,labels=recovery_data()
    assert len(streams['factorial'])==len(streams['confounded'])==144
    assert len(streams['heldout'])==48
    for name in streams:
        assert set(Counter(labels[d['id']]['support'] for d in streams[name]).values())=={len(streams[name])//3}
    seen=set()
    for doc in streams['factorial']:
        row=labels[doc['id']];seen.add((row['support'],tuple(row['atoms'])))
    for doc in streams['heldout']:
        row=labels[doc['id']]
        assert (row['support'],tuple(row['atoms'])) not in seen
    pairs={}
    for doc in streams['confounded']:
        row=labels[doc['id']]
        for kind,prop in zip(row['kinds'],row['properties']):pairs.setdefault(kind,set()).add(prop)
    assert len(pairs)==4 and all(len(v)==1 for v in pairs.values())


def test_warmup_restore_keeps_the_lazy_comparison_bank_and_all_saved_tensors(tmp_path):
    import pytest
    import Models
    from identity_experiment import make_model, model_hashes
    native_restore=Models.BaseModel.load_weights
    checkpoint=tmp_path/'warm.ckpt'
    with pytest.MonkeyPatch.context() as patch:
        folder=tmp_path/'source';folder.mkdir()
        model=make_model(folder,patch)
        try:
            assert model._concept_owner().components._ready().meaning_pairs>0
            model.comparison_reader.sync(model)
            assert model.comparison_reader.weights
            with torch.no_grad():
                for parameter in model.comparison_reader.parameters():parameter.add_(.25)
            model.save_weights(str(checkpoint))
            expected=model_hashes(model)
        finally:model.End()
    with pytest.MonkeyPatch.context() as patch:
        folder=tmp_path/'restored';folder.mkdir()
        model=make_model(folder,patch)
        try:
            assert not model.comparison_reader.weights
            restore_warmup(model,checkpoint,native_restore)
            assert model.comparison_reader.weights
            assert model_hashes(model)==expected
        finally:model.End()
