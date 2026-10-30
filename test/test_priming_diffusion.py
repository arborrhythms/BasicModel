"""Item 2.9: location-step, membership, guard and retirement certificates."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from LTM1 import LTM1
from Priming import configure, part_edges, propagate


def owner(points, *, spread=.3, neighbours=8, radius=15.):
    native = nn.Module()
    configure(native)
    native._geometric_spread = spread
    native._geometric_neighbours = neighbours
    native._geometric_radius.fill_(math.radians(radius))
    native._geometric_radius_rate = 0.
    native.similarity_codebook = SimpleNamespace(mereology=SimpleNamespace(
        placement=F.normalize(points.detach(), dim=-1), meaning_pairs=points.shape[1],
        placement_witnesses=torch.ones(len(points), dtype=torch.long, device=points.device),
        placement_derived=torch.zeros(len(points), dtype=torch.bool, device=points.device),
        placement_valid=torch.zeros(len(points), dtype=torch.bool, device=points.device)))
    native._geometric_radius = native._geometric_radius.to(points)
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


def test_location_outflow_preserves_energy_codes_rng_and_inactive_streams():
    points = directions([0, 10, 80]).requires_grad_()
    native = owner(points)
    before_codes, before_rng = points.detach().clone(), torch.get_rng_state().clone()
    surface = torch.tensor([[2., 1., 1.], [1., 1., 4.], [1., 3., 1.]], dtype=torch.float64)
    propagate(native, surface, torch.tensor([True, False, True]))
    torch.testing.assert_close(surface, surface.new_tensor([[1.7, 1.3, 1.], [1., 1., 4.], [1.6, 2.4, 1.]]))
    audit = native._last_priming_diffusion
    assert set(audit) == {'before', 'after', 'budget_fraction', 'receivers', 'radius'}
    assert audit['receivers'].tolist() == [1, 0, 1]
    torch.testing.assert_close(audit['before'], audit['after'], rtol=0, atol=1e-9 * surface.shape[1])
    torch.testing.assert_close(points, before_codes, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert points.grad is None and not surface.requires_grad


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


def test_hated_rows_cannot_overdraw_positive_excess_or_create_energy_at_floor():
    native = owner(torch.ones(41, 2), spread=1.)
    surface = torch.zeros(1, 41)
    surface[0, -1] = 2.
    propagate(native, surface, torch.tensor([True]))
    assert bool((surface >= 0).all())
    torch.testing.assert_close(surface.double().sum(), torch.tensor(2., dtype=torch.double), rtol=0, atol=1e-9*41)
    assert native._last_priming_diffusion['budget_fraction'].item() == 1
    # No positive excess means that negative desires cannot drive the field.
    surface.clamp_(max=1)
    before = surface.clone()
    propagate(native, surface, torch.tensor([True]))
    assert torch.equal(before, surface)


def test_six_row_cosine_mask_and_fixed_cap_match_direct_computation(monkeypatch):
    native = owner(directions([0, 10, 20, 30, 5, 2]), spread=.2, neighbours=2, radius=35)
    codes = native.similarity_codebook.mereology
    codes.placement_witnesses[4] = 0
    codes.placement_derived[5] = True
    surface = torch.tensor([[3., 1., 1., 1., 1., 1.]], dtype=torch.double)
    z = sum((surface[0, j]-1).clamp_min(0) * codes.placement[j] for j in [0])
    cosine = torch.tensor([torch.dot(row, z)/z.norm() for row in codes.placement])
    cosine[[0, 4, 5]] = 0
    expected = cosine * ((cosine > 0) & (cosine >= math.cos(math.radians(35))))
    original = torch.Tensor.topk
    calls = []
    def topk(value, k, *args, **kwargs):
        calls.append((value.clone(), k))
        return original(value, k, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'topk', topk)
    propagate(native, surface, torch.tensor([True]))
    assert len(calls) == 1 and calls[0][1] == 2
    torch.testing.assert_close(calls[0][0][0], expected)
    assert native._last_priming_diffusion['receivers'].tolist() == [3]
    assert surface[0, [3, 4, 5]].eq(1).all()
    torch.testing.assert_close(surface[0, 1:3]-1, .4 * expected[1:3]/expected[1:3].sum())
    assert surface[0, 0] == pytest.approx(2.6)


@pytest.mark.parametrize('count,expected', [(5, .98), (1, 1.02), (2, 1.)])
def test_controller_uses_active_mean_before_cap_for_ten_sentences(count, expected):
    native = owner(directions([0]+[.1]*count+[80]*(6-count)), neighbours=2, radius=15)
    native._geometric_radius_rate = .02
    pointer = native._geometric_radius.data_ptr()
    for step in range(10):
        surface = torch.ones(3, 7, dtype=torch.double)
        surface[:2, 0] = 2
        # Inactive streams must not dilute the mean or alter their surface.
        propagate(native, surface, torch.tensor([True, True, False]))
        assert surface[2].eq(1).all()
        assert native._last_priming_diffusion['receivers'][:2].tolist() == [count, count]
        assert native._geometric_radius.item() == pytest.approx(math.radians(15)*expected**(step+1))
    assert pointer == native._geometric_radius.data_ptr()
    for radius in (1., 90.):
        native._geometric_radius.fill_(math.radians(radius))
        propagate(native, torch.ones(3, 7, dtype=torch.double), torch.tensor([False]*3))
        assert native._geometric_radius.item() == pytest.approx(math.radians(radius))
    native._geometric_radius.fill_(native._geometric_radius_min if count > 2 else native._geometric_radius_max)
    surface = torch.ones(1, 7, dtype=torch.double); surface[0,0] = 2
    propagate(native, surface, torch.tensor([True]))
    assert native._geometric_radius_min <= native._geometric_radius <= native._geometric_radius_max


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
    assert surface[0,c] > 1
    assert cs._concept_allocator.order_of(cs.concept_id_at_row(c)) == 1
    assert cs._priming_seen[0,c] == 1
    pointers = [value.data_ptr() for value in (derived.placement, derived.placement_derived, derived.placement_valid)]
    derived.derive_class_placements()
    assert pointers == [value.data_ptr() for value in (derived.placement, derived.placement_derived, derived.placement_valid)]


def test_noncontiguous_class_cannot_supply_field_or_receive_but_members_can():
    cs, derived, (a,b,c), _ = class_fixture()
    derived.placement[b] = torch.tensor([0.,1.])
    derived.derive_class_placements()
    assert not derived.placement_valid[c]
    cs.prime_seen(torch.tensor([[c]]))
    assert cs._priming_boosts[0,c] == 2
    assert cs._priming_boosts[0,[a,b]].eq(1).all()
    cs._priming_boosts.fill_(1)
    # A fourth base row near a receives; the invalid centroid does not.
    from test_review13_subspaces import word
    peer = word(cs, [(4,1.)])
    derived.placement[peer] = derived.placement[a]
    derived.placement_witnesses[peer] = 1
    cs.prime_seen(torch.tensor([[a]]))
    assert cs._priming_boosts[0,peer] > 1 and cs._priming_boosts[0,c] == 1
    cs._priming_boosts.fill_(1)
    cs.prime_seen(torch.tensor([[peer]]))
    assert cs._priming_boosts[0,a] > 1


def test_retired_functions_and_surface_methods_are_absent():
    import Priming
    from Spaces import Space, ConceptualSpace
    for name in ('diffuse', 'geometric'):
        assert not hasattr(Priming, name), name
    for cls in (Space, ConceptualSpace):
        for name in ('_priming_edges', 'priming_row_orders'):
            assert not hasattr(cls, name), (cls, name)


def test_retired_spread_attribute_and_configuration_are_absent():
    from test_review13_subspaces import fixture
    cs, *_ = fixture()
    assert not hasattr(cs, '_priming_spread')
    root = Path(__file__).resolve().parents[1]
    assert 'primingSpread' not in (root/'data/model.xml').read_text()
    assert 'primingSpread' not in (root/'bin/Models.py').read_text()
    # Item 2.9 R2 retains the old name as history in geometricSpread's
    # description; the retired parameter must have no configuration row.
    assert '| `primingSpread` |' not in (root/'doc/Params.md').read_text()
    assert hasattr(cs, '_priming_edge_src') and callable(part_edges)


@pytest.mark.parametrize('knob,value', [('geometricSpread', 2), ('geometricNeighbours', -1),
    ('geometricRadius', float('nan')), ('geometricRadiusRate', 1), ('geometricRadiusMin', 0), ('geometricRadiusMax', 91)])
def test_bad_geometric_construction_fails_loud(monkeypatch, knob, value):
    from util import TheXMLConfig
    original = TheXMLConfig.space
    monkeypatch.setattr(TheXMLConfig, 'space', lambda section, name, default=None: value if name == knob else original(section,name,default))
    with pytest.raises(ValueError):
        configure(nn.Module())


def test_seen_ledger_guard_decay_bump_desire_and_fired_rows(monkeypatch):
    import Priming
    from test_priming_energy import _chain_cs
    from Models import BasicModel
    cs, near, _ = _chain_cs(batch_size=3)
    active = torch.tensor([True,True,False])
    pointers = cs._priming_seen.data_ptr(), cs._priming_boosts.data_ptr()
    cs._priming_seen[0,6] = cs._priming_boosts[0,6] = 1.5
    before = cs._priming_seen.clone()
    expected = before.clone()
    expected[:2] = 1 + (before[:2]-1)*.8
    expected[0,3] += 1
    expected[1,6] += 1
    calls = []
    def observed(owner, surface, flags):
        ledger = owner._priming_seen.clone()
        torch.testing.assert_close(ledger,expected,rtol=0,atol=0)
        propagate(owner,surface,flags)
        assert torch.equal(ledger,owner._priming_seen)
        calls.append(flags.clone())
    monkeypatch.setattr(Priming,'propagate',observed)
    result = cs.prime_seen(torch.tensor([[3],[6],[5]]),decay=.8,active_rows=active)
    assert torch.equal(result,cs.relevance_weights(batch=3)) and len(calls)==1
    assert torch.equal(cs._priming_seen,expected)
    audit, radius = cs._last_priming_diffusion, cs._geometric_radius.clone()
    for b in range(3):
        assert torch.equal(cs.placement_context(b),(expected[b]-1).clamp_min(0))
    stored = cs._priming_boosts.clone()
    cs.prime_desire(torch.tensor([[near],[near],[near]]), valence=.5, bump=2.)
    assert torch.equal(cs._priming_seen,expected)
    torch.testing.assert_close(cs._priming_boosts.double().sum(-1),stored.double().sum(-1)+1,rtol=0,atol=1e-9*cs.nVectors)
    assert cs._last_priming_diffusion is audit and torch.equal(cs._geometric_radius,radius)
    # Exercise the actual fired-row method, including its unregistered-CS arm.
    object.__setattr__(cs,'_cs_level_rows',[torch.tensor([3])])
    monkeypatch.setattr(cs,'_field_inventory_rows',lambda device: torch.arange(cs.nVectors,device=device))
    monkeypatch.setattr(cs,'project_priming_to_towers',lambda *args,**kwargs: None)
    model = SimpleNamespace(wholeSpaces=[],_concept_owner=lambda:cs,
        _priming_batch_size=lambda:1,priming_spread=.05)
    BasicModel._prime_seen_step(model)
    assert len(calls)==1 and cs._last_priming_diffusion is audit
    assert torch.equal(cs._geometric_radius,radius)
    assert pointers==(cs._priming_seen.data_ptr(),cs._priming_boosts.data_ptr())
    assert cs._priming_seen[0,3] > expected[0,3]
    first=cs.relevance_weights(); snap=cs._priming_boosts.clone(); seen=cs._priming_seen.clone()
    assert torch.equal(first,cs.relevance_weights())
    assert torch.equal(snap,cs._priming_boosts) and torch.equal(seen,cs._priming_seen)
    with pytest.raises(RuntimeError,match='resizing is forbidden'):
        cs.prime_seen(torch.zeros(4,1,dtype=torch.long))


def test_one_precomposition_step_and_one_centroid_close_per_sentence(monkeypatch, eager_reading):
    import Priming
    from test_mm_xor import _fresh_model
    model, *_ = _fresh_model('data/XOR_grammar.xml')
    cs=model._concept_owner()
    derived=cs.similarity_codebook.mereology
    events=[]
    prime=model._prime_sentence_symbols
    derive=derived.derive_class_placements
    def sentence(sid):
        events.append(('sentence',sid))
        return prime(sid)
    def step(owner,surface,active):
        events.append(('step',tuple(surface.shape),active.tolist()))
        # The registered words are bumped before the location calculation.
        rows=owner._sentence_seen_rows[:len(surface)]
        for b in active.nonzero().flatten().tolist():
            assert owner._priming_seen[b,rows[b][rows[b]>=0]].gt(1).all()
        return propagate(owner,surface,active)
    def centroid():
        events.append(('centroid',))
        return derive()
    monkeypatch.setattr(model,'_prime_sentence_symbols',sentence)
    monkeypatch.setattr(Priming,'propagate',step)
    monkeypatch.setattr(derived,'derive_class_placements',centroid)
    with torch.no_grad():
        model.forward(model.inputSpace.prepInput(['hello world','loving there']))
    assert [event[0] for event in events] == ['sentence','step','centroid']
    assert events[1][1][0] == 2 and events[1][2] == [True,True]


def test_cost_certificate_has_no_pairwise_inventory_tensor(tmp_path):
    import json, os
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    V,P,B=257,7,3
    native=owner(torch.ones(V,P),neighbours=8)
    surface=torch.ones(B,V); surface[:2,0]=2
    outputs=[]
    class AllocationRead(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if torch.is_tensor(value):
                    shape=list(value.shape)
                    assert shape != [V,V], (str(func),shape)
                    outputs.append(dict(operation=str(func),shape=shape,bytes=value.numel()*value.element_size()))
            return result
    with AllocationRead():
        propagate(native,surface,torch.tensor([True,True,False]))
    assert any(item['shape']==[B,P] for item in outputs)
    assert any(item['shape']==[B,V] for item in outputs)
    assert all(item['bytes']<=max(B*V,V*P)*8 for item in outputs)
    folder=Path(os.environ.get('ITEM29_RECEIPT',tmp_path))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'cost-certificate.json').write_text(json.dumps(dict(
        dimensions=dict(streams=B,inventory=V,meaning_pairs=P),
        pairwise_inventory_tensor=False,allocation_read=outputs),indent=2)+'\n')


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
