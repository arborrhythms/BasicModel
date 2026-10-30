"""Fixed sparse reservations preserve projection and fail before mutation."""
from types import SimpleNamespace
import pytest
import torch


def test_explicit_bridge_capacity_preserves_prefix_and_fails_before_targets(monkeypatch):
    from test_cs_sparse_weights import _cs
    from Spaces import ConceptualSpace, _concept_alloc_of
    from util import TheXMLConfig
    _cs(nS=64, order=0)
    TheXMLConfig.data['ConceptualSpace']['primingBridgeCapacity'] = 1
    cs = ConceptualSpace([4, 8], [64, 8], [64, 8])
    alloc = _concept_alloc_of(cs)
    assert cs._priming_bridge_src.numel() == 1
    pointer = cs._priming_bridge_src.data_ptr()
    alloc.word_descriptions[31] = dict(parts=((4, 31),), wholes=())
    monkeypatch.setattr(cs, '_word_definition_ids', lambda: (31,))
    monkeypatch.setattr(cs, '_csw_row_of', lambda identity: 63)
    monkeypatch.setattr(cs.definitions, 'objects', lambda word: (101,))
    monkeypatch.setattr(cs.interpret, 'no_object', lambda word: False)
    cs._priming_boosts.fill_(1.)
    cs._priming_boosts[0, 63] = 3.
    monkeypatch.setattr(cs, 'priming_weights', lambda **kwargs: cs._priming_boosts[:1])
    target = torch.ones(1, 64)
    tower = SimpleNamespace(_priming_dim=lambda: 64, _priming_decay=1.,
        _priming_surface=lambda *args, **kwargs: target)
    cs.project_priming_to_towers(tower, None, batch=1)
    expected = torch.ones_like(target)
    expected[:, [4, 31]] = 3.
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    saved = tuple(value.clone() for value in (cs._priming_bridge_src,
        cs._priming_bridge_dst, cs._priming_bridge_bits))
    revision = cs._priming_bridge_revision
    alloc.word_descriptions[31] = dict(parts=((4, 32),), wholes=())
    cs.interpret._description_revision += 1
    with pytest.raises(RuntimeError, match='constructed vocabulary capacity'):
        cs.project_priming_to_towers(tower, None, batch=1)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    for value, before in zip((cs._priming_bridge_src, cs._priming_bridge_dst,
                             cs._priming_bridge_bits), saved):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    assert cs._priming_bridge_src.data_ptr() == pointer
    assert cs._priming_bridge_revision == revision


def test_derived_cache_reuses_storage_and_shared_construction_does_not_read_all_rows(monkeypatch):
    from torch import nn
    from test_cs_sparse_weights import _cs
    from Spaces import Codebook, ConceptualSpace
    cs = _cs(nS=32, order=0)
    cb = Codebook()
    cb.W = nn.Parameter(torch.full((32, 8), .25))
    pointer = cb.W.data_ptr()
    cb.enable_derived_codes(cs, percept_width=4, content_width=8)
    assert cb.W.data_ptr() == pointer
    assert not cb.W.requires_grad and not isinstance(cb.W, nn.Parameter)
    assert cb.W.count_nonzero() == 0
    monkeypatch.setattr(cb, 'getW', lambda: pytest.fail('construction materialized all derived rows'))
    shared = ConceptualSpace([4, 8], [32, 8], [32, 8], shared_similarity_codebook=cb)
    assert shared.similarity_codebook is cb
    from Models import BaseModel
    owner = SimpleNamespace(_aligned_inventory_capacity=lambda: 32,
                            conceptualSpaces=[cs, shared])
    cb.vq = SimpleNamespace(set_active_rows=lambda _: None)
    cs.similarity_codebook = cb
    assert BaseModel._aligned_capacity_codebooks(owner) == [cb]


def test_relation_and_equality_capacities_are_independent_and_handles_fail_before_write():
    from Layers import ConceptAllocator
    alloc=ConceptAllocator(lambda order:(16,16,torch.device('cpu')),lambda:2,
        relation_capacity=11,member_capacity=3,member_width=7)
    assert alloc.parts.capacity==11 and alloc.singleton_values.shape==(3,7)
    alloc.next_id=11
    before=alloc.parts.refs.clone()
    with pytest.raises(OverflowError,match='capacity'):
        alloc._allocate_part_symbol()
    assert alloc.next_id==11 and torch.equal(alloc.parts.refs,before)


def test_adam_reservation_does_not_train_and_matches_lazy_first_step():
    from corpus_capacity import materialize_optimizer
    a=torch.nn.Parameter(torch.tensor([.2,.6]));b=torch.nn.Parameter(a.detach().clone())
    early=torch.optim.Adam([a],lr=.01);lazy=torch.optim.Adam([b],lr=.01)
    before=a.detach().clone()
    assert materialize_optimizer(early)>0
    assert torch.equal(a,before) and early.state[a]['step']==0
    for parameter,optimizer in ((a,early),(b,lazy)):
        parameter.square().sum().backward();optimizer.step()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key,value in early.state[a].items():
        torch.testing.assert_close(value,lazy.state[b][key],rtol=0,atol=0)


def test_capacity_workload_uses_native_populated_windows_without_repacking():
    from Corpus import CurriculumStreamDataset
    from corpus_capacity import populated_cursor
    dataset=CurriculumStreamDataset(['a','b','c','d','e'],3,
        stages=[0,0,1,1,1],document_ids=list(range(5)))
    before=dataset.cursor_state()
    cursor,workload=populated_cursor(dataset)
    assert dataset.cursor_state()==before
    assert workload['skipped_training_sentences']==2
    assert workload['windows']==[(2,3),(3,4),(4,5)]
    dataset.restore_cursor(cursor)
    rows,_,_=dataset.next_packed_tick(256,lambda text:1)
    assert rows==[['c'],['d'],['e']]


def test_capacity_materializes_the_native_mixed_optimizer_without_an_update():
    from Optimizer import Adam,SGD,RowLocalAdam,MultiOptimizer
    from corpus_capacity import materialize_optimizer
    before=[torch.nn.Parameter(torch.full((4,2),.3)) for _ in range(3)]
    after=[torch.nn.Parameter(value.detach().clone()) for value in before]
    def optimizer(values):
        return MultiOptimizer([Adam([values[0]],lr=.01),SGD([values[1]],lr=.01),
                               RowLocalAdam([values[2]],lr=.01)])
    eager,lazy=optimizer(before),optimizer(after)
    assert materialize_optimizer(eager)>0
    assert not eager.optimizers[-1].state
    for values,opt in ((before,eager),(after,lazy)):
        values[0].square().sum().backward();values[1].square().sum().backward()
        values[2].grad=torch.sparse_coo_tensor(torch.tensor([[1]]),torch.ones(1,2),size=(4,2))
        opt.step()
    assert materialize_optimizer(eager)==materialize_optimizer(lazy)
    for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_strict_allocator_restore_checks_constructed_capacities_before_write(eager_reading):
    import copy
    from test_meronomy_ladder import _build_ladder
    from Spaces import _concept_alloc_of
    model=_build_ladder()
    try:
        extras=model._collect_structural_extras()
        # Every allocator-bearing stage is restored through the same native seam.
        assert extras
        alloc=_concept_alloc_of(model._concept_owner())
        before=alloc.next_id
        # This fixture calls the allocator reader directly with the real
        # collected construction description, changing no live reservation.
        def find(value):
            if isinstance(value,dict):
                if 'construction' in value and 'next_id' in value:return value
                for child in value.values():
                    found=find(child)
                    if found is not None:return found
            if isinstance(value,(tuple,list)):
                for child in value:
                    found=find(child)
                    if found is not None:return found
        saved=copy.deepcopy(find(extras));assert saved is not None
        saved['construction']['relation_capacity']+=1
        with pytest.raises(ValueError,match='construction|capacity|capacities'):
            model._restore_allocator_extras(model._concept_owner(),saved)
        assert alloc.next_id==before
    finally:
        model.End();model.symbolSpace.soft_reset()
