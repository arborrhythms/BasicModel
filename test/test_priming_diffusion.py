"""Predictor spread, retained membership/placement, and diffusion retirement."""
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from LTM1 import LTM1
from Priming import part_edges


def owner(points):
    native = nn.Module()
    native.similarity_codebook = SimpleNamespace(mereology=SimpleNamespace(
        placement=F.normalize(points.detach(), dim=-1), meaning_pairs=points.shape[1],
        placement_witnesses=torch.ones(len(points), dtype=torch.long, device=points.device),
        placement_derived=torch.zeros(len(points), dtype=torch.bool, device=points.device),
        placement_valid=torch.zeros(len(points), dtype=torch.bool, device=points.device)))
    native._concept_allocator = SimpleNamespace(parts=LTM1(256), relate_idx={('part-base', 2): (2, 1), ('part-base', 3): (2, 2)})
    native._priming_edge_src = torch.full((512,), -1, dtype=torch.long)
    native._priming_edge_dst = torch.full((512,), -1, dtype=torch.long)
    native._priming_edge_weight = torch.ones(512)
    native._priming_edge_count = native._priming_edge_rebuilds = 0
    native._priming_edge_revision = None
    native._csw_row_of = lambda symbol: symbol - 1 if 0 < symbol <= len(points) else None
    return native

def directions(angles, *, dtype=torch.float64):
    radians = torch.tensor(angles, dtype=dtype) * math.pi / 180
    return torch.stack((radians.cos(), radians.sin()), -1)

def test_membership_cache_follows_native_addresses_evidence_forgetting_and_load():
    native = owner(torch.ones(4, 2))
    parts = native._concept_allocator.parts
    parts.add(1, 2, symbol=100, truth_plus=.8, truth_minus=.3)
    parts.add(2, 3, symbol=101, truth_plus=.6)
    # A definition and an occurrence relation are not membership.
    parts.add(1, 4, symbol=102, truth_plus=1.)
    parts.add((parts.OCCURRENCE, -7), (parts.SYMBOL, 2), symbol=103, truth_plus=1.)
    pointers = tuple(t.data_ptr() for t in (native._priming_edge_src, native._priming_edge_dst, native._priming_edge_weight))
    source, target, weight = part_edges(native)
    assert list(zip(source.tolist(), target.tolist())) == [(0, 1), (1, 2)]
    torch.testing.assert_close(weight, torch.tensor([.5, .6]))
    rebuilds = native._priming_edge_rebuilds
    part_edges(native)
    parts.add((parts.OCCURRENCE, -8), (parts.SYMBOL, 2))
    part_edges(native)
    assert native._priming_edge_rebuilds == rebuilds+1
    parts.add(1, 2, truth_minus=.7)
    assert part_edges(native)[2][0] == pytest.approx(.1)
    assert native._priming_edge_rebuilds == rebuilds+2
    restored = LTM1(parts.capacity)
    restored.load_state_dict(parts.state_dict())
    native._concept_allocator.parts = restored
    restored.remove(1, 2)
    source, target, weight = part_edges(native)
    assert list(zip(source.tolist(), target.tolist())) == [(1, 2)]
    torch.testing.assert_close(weight, torch.tensor([.6]))
    assert pointers == tuple(t.data_ptr() for t in (native._priming_edge_src, native._priming_edge_dst, native._priming_edge_weight))

def class_fixture():
    from test_review13_subspaces import fixture, word
    from ConceptIndex import index_part_row
    cs, cb, _, alloc = fixture(meaning_pairs=2)
    a, b, base = [word(cs, [(4, 1.)]) for _ in range(3)]
    symbol = index_part_row(cs, cs.concept_id_at_row(a), cs.concept_id_at_row(base))
    assert index_part_row(cs, cs.concept_id_at_row(b), symbol) == symbol
    row = cs._csw_row_of(symbol)
    derived = cb.mereology
    derived.placement[[a,b]] = directions([0, 10], dtype=torch.float32)
    derived.placement_witnesses[[a,b]] = 1
    relation = alloc.parts.find(cs.concept_id_at_row(b), symbol)
    alloc.parts.add(cs.concept_id_at_row(b), symbol, truth_minus=.5)
    return cs, derived, (a,b,row), relation

def test_class_centroid_weight_witnesses_cross_order_reach_and_context_skip():
    cs, derived, (a,b,c), _ = class_fixture()
    assert derived.placement_derived[c] and not derived.placement_valid[c]
    derived.place_rows([c], torch.ones(cs.nVectors), b'not-a-context-placement')
    assert derived.placement_witnesses[c] == 0
    derived.derive_class_placements()
    expected = F.normalize(derived.placement[a] + .5*derived.placement[b], dim=0)
    torch.testing.assert_close(derived.placement[c], expected)
    assert derived.placement_witnesses[c] == 2 and derived.placement_valid[c]
    surface = cs.prime_seen(torch.tensor([[a]]))
    assert surface[0,c] == 1  # seen words do not synthesize predictions
    assert cs._concept_allocator.order_of(cs.concept_id_at_row(c)) == 1
    assert cs._priming_seen[0,c] == 1
    pointers = [value.data_ptr() for value in (derived.placement, derived.placement_derived, derived.placement_valid)]
    derived.derive_class_placements()
    assert pointers == [value.data_ptr() for value in (derived.placement, derived.placement_derived, derived.placement_valid)]

def test_class_flags_roundtrip_with_fixed_shapes():
    import copy
    cs,derived,_,_=class_fixture()
    derived.derive_class_placements()
    saved=derived.placement_extras(); state=copy.deepcopy(derived.state_dict())
    pointers=derived.placement_derived.data_ptr(),derived.placement_valid.data_ptr()
    derived.placement_derived.zero_();derived.placement_valid.zero_()
    derived.load_placement_extras(saved)
    assert torch.equal(derived.placement_derived,saved['derived'])
    assert torch.equal(derived.placement_valid,saved['valid'])
    derived.load_state_dict(state)
    assert pointers==(derived.placement_derived.data_ptr(),derived.placement_valid.data_ptr())
    with pytest.raises(ValueError,match='constructed column'):
        derived.load_placement_extras(dict(saved,valid=torch.zeros(1,dtype=torch.bool)))


def test_noncontiguous_class_placement_remains_independent_of_spread():
    cs, derived, (a,b,c), _ = class_fixture()
    derived.placement[b] = torch.tensor([0.,1.])
    derived.derive_class_placements()
    assert not derived.placement_valid[c]
    cs._expected_knowing[0,c] = torch.tensor([.3,.8])
    before = cs._expected_knowing.clone()
    cs.prime_seen(torch.tensor([[a]]))
    assert torch.equal(before,cs._expected_knowing)
    assert cs.placement_context(0)[a] == 1 and cs.placement_context(0)[c] == 0
    assert cs.relevance_weights(batch=1)[0,c] == pytest.approx(1.8)


def test_retired_functions_surfaces_switch_and_configuration_are_absent():
    import Priming, ConceptSpread
    from Spaces import Space, ConceptualSpace
    from test_review13_subspaces import fixture
    cs, *_ = fixture()
    for name in ('configure','propagate','diffuse','geometric'):
        assert not hasattr(Priming,name)
    assert not hasattr(ConceptSpread,'location_measurement')
    for name in ('_priming_boosts','_geometric_radius','_geometric_spread','_priming_spread'):
        assert not hasattr(cs,name)
    for cls in (Space,ConceptualSpace):
        for name in ('_priming_edges','priming_row_orders','_bounded_priming','_priming_surface'):
            assert not hasattr(cls,name)
    root=Path(__file__).resolve().parents[1]
    for name in ('geometricSpread','geometricNeighbours','geometricRadius','geometricRadiusRate',
                 'geometricRadiusMin','geometricRadiusMax','primingMaxBoost','primingSpread'):
        assert '<'+name+'>' not in (root/'data/model.xml').read_text()
        assert 'name="'+name+'"' not in (root/'data/model.xsd').read_text()
        assert '| `'+name+'` |' not in (root/'doc/Params.md').read_text()
    assert 'BASICMODEL_EXPECTATION_MEASUREMENT' not in (root/'bin/ConceptSpread.py').read_text()
    assert callable(part_edges)


@pytest.mark.parametrize('decay', [-.1, 1.1, float('nan'), float('inf')])
def test_seen_decay_validation_fails_loud(decay):
    from test_priming_energy import _chain_cs
    cs, *_ = _chain_cs()
    before = cs._priming_seen.clone()
    with pytest.raises(ValueError,match='decay'):
        cs.prime_seen(torch.tensor([[3]]),decay=decay)
    assert torch.equal(before,cs._priming_seen)


def test_seen_ledger_guard_decay_bump_desire_and_fired_rows(monkeypatch):
    from test_priming_energy import _chain_cs
    from Models import BasicModel
    cs, near, _ = _chain_cs(batch_size=3)
    active=torch.tensor([True,True,False])
    pointers=tuple(v.data_ptr() for v in (cs._priming_seen,cs._expected_knowing,cs._desire))
    cs._priming_seen[0,6]=1.5
    expected=cs._priming_seen.clone();expected[:2]=1+(expected[:2]-1)*.8
    expected[0,3]+=1;expected[1,6]+=1
    cs._expected_knowing[0,near]=torch.tensor([.25,.5])
    frozen=cs._expected_knowing.clone()
    result=cs.prime_seen(torch.tensor([[3],[6],[5]]),decay=.8,active_rows=active)
    assert torch.equal(result,cs.relevance_weights(batch=3))
    assert torch.equal(cs._priming_seen,expected) and torch.equal(frozen,cs._expected_knowing)
    for b in range(3):assert torch.equal(cs.placement_context(b),(expected[b]-1).clamp_min(0))
    cs.prime_desire(torch.tensor([[near],[near],[near]]),valence=.5,bump=2.)
    assert torch.equal(cs._priming_seen,expected) and torch.equal(frozen,cs._expected_knowing)
    assert cs._desire[:,near].eq(1).all()
    object.__setattr__(cs,'_cs_level_rows',[torch.tensor([3])])
    monkeypatch.setattr(cs,'_field_inventory_rows',lambda device:torch.arange(cs.nVectors,device=device))
    monkeypatch.setattr(cs,'project_priming_to_towers',lambda *args,**kwargs:None)
    model=SimpleNamespace(wholeSpaces=[],_concept_owner=lambda:cs,_priming_batch_size=lambda:1)
    BasicModel._prime_seen_step(model)
    assert cs._priming_seen[0,3]>expected[0,3]
    assert torch.equal(frozen,cs._expected_knowing)
    assert pointers==tuple(v.data_ptr() for v in (cs._priming_seen,cs._expected_knowing,cs._desire))
    with pytest.raises(RuntimeError,match='resizing is forbidden'):
        cs.prime_seen(torch.zeros(4,1,dtype=torch.long))


def test_one_precomposition_prediction_and_one_centroid_close_per_sentence(monkeypatch,eager_reading):
    import ConceptSpread
    from test_mm_xor import _fresh_model
    model,*_=_fresh_model('data/XOR_grammar.xml')
    derived=model._concept_owner().similarity_codebook.mereology
    events=[];prime=model._prime_sentence_symbols;predict=ConceptSpread.open_prediction;derive=derived.derive_class_placements
    def sentence(sid):
        events.append(('sentence',sid));return prime(sid)
    def opened(model,active):
        events.append(('prediction',active.tolist()));return predict(model,active)
    def centroid():
        events.append(('centroid',));return derive()
    monkeypatch.setattr(model,'_prime_sentence_symbols',sentence)
    monkeypatch.setattr(ConceptSpread,'open_prediction',opened)
    monkeypatch.setattr(derived,'derive_class_placements',centroid)
    try:
        with torch.no_grad():model.forward(model.inputSpace.prepInput(['hello world','loving there']))
        assert [event[0] for event in events]==['sentence','prediction','centroid']
        assert events[1][1]==[True,True]
    finally:model.End();model.symbolSpace.soft_reset()


def test_predictor_allocation_has_no_inventory_square(tmp_path):
    import json,os
    from Expectation import ActivationPredictor
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    V,D,B=257,7,3
    predictor=ActivationPredictor(D)
    keys=torch.ones(B,V,D);context=torch.zeros(B,2*D);knowing=torch.zeros(B,V,2)
    outputs=[]
    class AllocationRead(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if torch.is_tensor(value):
                    shape=list(value.shape)
                    assert shape.count(V)<2,(str(func),shape)
                    outputs.append(dict(operation=str(func),shape=shape,bytes=value.numel()*value.element_size()))
            return result
    with AllocationRead():result=predictor(keys,context,knowing).sigmoid()
    assert result.shape==(B,V,2)
    folder=Path(os.environ.get('ITEM25_RECEIPT',tmp_path));folder.mkdir(parents=True,exist_ok=True)
    (folder/'predictor-allocation.json').write_text(json.dumps(dict(torch_version=str(torch.__version__), dimensions=dict(streams=B,inventory=V,width=D),
        pairwise_inventory_tensor=False,allocation_read=outputs),indent=2)+'\n')
