"""Item 2.8: native form layout, membership-derived codes and certificates."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from PerceptProperties import PrimitiveProperties
from WordIdentity import WordIdentity, resolve_identity_layout


def property_book(rows=4, identity=3, band=0):
    from Spaces import Codebook
    book = Codebook()
    book.W = nn.Parameter(torch.zeros(rows, identity + 256 + band))
    book.primitive_properties = PrimitiveProperties(rows, property_width=256)
    book.enable_property_codes(identity_width=identity, property_width=256)
    return book


def extended_fixture():
    from test_review13_subspaces import fixture, word
    cs, cb, ps, alloc = fixture(property_width=256)
    ws = property_book(4)
    with torch.no_grad():
        ws.primitive_properties.members[0, [97, 98]] = 1
        ws.primitive_properties.members[1, [97, 98, 99]] = 1
        ws.primitive_properties.observed[:2] = True
        ps.W[4, :3] = torch.tensor([.2, .2, .2])
        ps.W[7, :3] = torch.tensor([.8, .3, .2])
    ws.begin_forward()
    cs._model.wholeSpace = NS(subspace=NS(what=ws))
    a, b = word(cs, [(4, 1.)]), word(cs, [(4, 1.), (7, 1.)])
    cs.add_concept_feature(a, 'ws', 0, 1.)
    cs.add_concept_feature(a, 'ws', 1, 1.)
    cs.add_concept_feature(b, 'ws', 1, 1.)
    return cs, cb, ps, ws, alloc, a, b


def test_construction_resolves_the_split_and_preserves_numeric_layout():
    config = {'architecture': {'data': {'dataType': 'embedding'}},
              'PartSpace': {'nDim': 9}, 'ConceptualSpace': {'nDim': 137}}
    resolve_identity_layout(config)
    assert config['PartSpace']['propertyDim'] == 256
    assert config['PartSpace']['propertyOnes'] == 1
    for name in ('InputSpace', 'PartSpace', 'WholeSpace'):
        assert config[name]['nDim'] == 96 + 256 + 8
    assert config['ConceptualSpace']['nDim'] == 96 + 256 + 8 + 128
    first = copy.deepcopy(config)
    resolve_identity_layout(config)
    assert config == first
    numeric = {'architecture': {'data': {'dataType': 'numeric'}},
               'PartSpace': {'nDim': 9}, 'ConceptualSpace': {'nDim': 137}}
    before = copy.deepcopy(numeric)
    resolve_identity_layout(numeric)
    assert numeric == before


@pytest.mark.parametrize('width,ones', [(2, 3), (0, 1), (256, 0)])
def test_invalid_property_construction_fails_loud(width, ones):
    with pytest.raises(ValueError, match='propertyDim'):
        resolve_identity_layout({'architecture': {'data': {'dataType': 'embedding'}},
                                 'PartSpace': {'propertyDim': width, 'propertyOnes': ones}})


def test_exact_byte_union_is_monotone_and_differentiable_for_weighted_members():
    properties = PrimitiveProperties(4, property_width=256)
    with torch.no_grad():
        properties.members[0, 97] = .2
        properties.members[1, 97:99] = torch.tensor([.4, .3])
        properties.members[2, 97:100] = torch.tensor([.6, .5, .4])
        properties.members[3, 97:101] = torch.tensor([.8, .7, .6, .5])
    torch.testing.assert_close(properties.byte_bits, torch.eye(256), rtol=0, atol=0)
    codes = properties.codes()
    torch.testing.assert_close(codes, properties.members)
    assert (codes[:-1] <= codes[1:]).all()
    codes[:, 97:101].sum().backward()
    assert properties.members.grad[:, 97:101].eq(1).all()
    torch.testing.assert_close(properties.codes(torch.tensor([1, 3])), codes[[1, 3]])


def test_hashed_union_matches_direct_max_without_a_broadcast_workspace():
    properties = PrimitiveProperties(4, property_width=7, ones=2)
    with torch.no_grad():
        properties.members.copy_(torch.arange(1024).reshape(4, 256) / 1024)
    direct = (properties.coefficients()[:, :, None] * properties.byte_bits).amax(1)
    torch.testing.assert_close(properties.codes(), direct)


def test_tower1_refresh_preserves_storage_and_routes_gradient_to_members():
    book = property_book(band=8)
    properties = book.primitive_properties
    pointer = book.W.data_ptr()
    revision = properties.revision
    properties.teach(0, [97, 98], [.5, 1.], steps=1, rate=1.)
    assert properties.revision > revision
    book.begin_forward()
    assert book.W.data_ptr() == pointer and not isinstance(book.W, nn.Parameter)
    torch.testing.assert_close(book.W, torch.cat((torch.ones(4, 3), properties.codes(), torch.zeros(4, 8)), -1))
    book.lookup_rows(0)[3 + 97].backward()
    assert properties.members.grad[0, 97] == 1
    properties.members.grad = None
    book.begin_forward()
    book.getW()[:, 3 + 98].sum().backward()
    assert properties.members.grad[:, 98].eq(1).all()
    assert book.W.data_ptr() == pointer
    revision = properties.revision
    with torch.no_grad():
        properties.members[0, 97] = .75
    assert properties.revision > revision
    book.begin_forward()
    assert book.W[0, 3 + 97] == .75


def test_fixed_property_fills_refresh_on_membership_revision():
    from Spaces import Codebook
    basis = Codebook()
    basis.W = nn.Parameter(torch.zeros(4, 352))
    identity = WordIdentity(NS(dim=352, _capacity=4, _basis=basis), property_width=256)
    atom = ('pair', 'a')
    identity.atoms[atom] = 0
    identity.words.update({b'a': (atom,), b'z': (('pair', 'z'),)})
    identity.word_rows.update({b'a': 1, b'z': 2})
    identity.atoms[('pair', 'z')] = 3
    properties = PrimitiveProperties(2, property_width=256)
    properties.teach(0, [97, 98], [1., 1.], steps=1, rate=1.)
    pointers = basis.W.data_ptr(), basis.fixed_row_codes.data_ptr()
    identity.refresh_property_fills(properties)
    torch.testing.assert_close(basis.W[1, 96:], properties.fill(b'a'))
    assert basis.W[2, 96:].eq(1).all()
    before = basis.W[:, :96].detach().clone()
    revision = properties.revision
    properties.teach(1, [97, 99], [.5, .8], steps=1, rate=1.)
    assert properties.revision > revision
    identity.refresh_property_fills()
    p = properties.on_counts(torch.nn.functional.one_hot(torch.tensor(97), 256).float())
    expected = (1-p[:, None]*(1-properties.codes())).amin(0)
    torch.testing.assert_close(basis.W[1, 96:], expected)
    torch.testing.assert_close(basis.W[:, :96], before, rtol=0, atol=0)
    assert pointers == (basis.W.data_ptr(), basis.fixed_row_codes.data_ptr())


def test_forms_join_identity_and_meet_properties_and_preserve_retained_rows():
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    width = cb.mereology.percept_width
    assert width == cb.mereology.identity_width + cb.mereology.property_width
    expected = torch.cat((ps.W[4, :3], ws.W[:2, 3:].amin(0)))
    torch.testing.assert_close(cb.lookup_rows(a)[:width], expected)
    from test_review13_subspaces import word
    no_wholes = word(cs, [(4, 1.)])
    assert cb.lookup_rows(no_wholes)[3:width].eq(1).all()
    retained = cs.concept_id_at_row(b)
    alloc.symbol_types.setdefault(retained, set()).add('concept')
    with torch.no_grad():
        cb.W[b, :width].fill_(.37)
    torch.testing.assert_close(cb.lookup_rows(b)[:width], torch.full((width,), .37))


@pytest.mark.parametrize('occupied', (2, 3))
def test_thought_description_operands_keep_cube_forms_and_existing_tail(
        tmp_path, monkeypatch, occupied):
    from Layers import WhatInteractionMemory
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_query_vp_boundaries import _context
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1, word_buckets='16')
    try:
        registry = model.grammatical_thoughts
        cs = registry.space
        book = cs.similarity_codebook
        width = book.mereology.percept_width
        payload = torch.ones(3, book.W.shape[-1], requires_grad=True)
        with torch.no_grad():
            payload[2, :width] = 0
            payload[:, width:] = torch.tensor([-2., .5, 3.])[:, None]
        description = with_slots(
            ConceptualMeaning(payload, torch.arange(3) < occupied), (('referent', 0),))
        total = description.roles.sum(0)
        old = total / description.role_mask.sum().sqrt()
        assert old[:width].max() > 1  # the captured 2/sqrt(3) failure
        expected = torch.cat((total[:width] / occupied, old[width:]))
        memory = WhatInteractionMemory(batch=1, capacity=8, detach_mode='episode')
        entry = memory.begin_thought_episode(description, work_budget=8)
        reference = memory.thought_reference(entry)
        context = _context(cs, memory=memory)
        requests = [registry.form('ask', description),
                    registry.form('bind', description),
                    registry.form('ask', reference, context=context)]
        requests.extend(item.request for item in registry.controller_candidates(
            description, description, description,
            descriptions=((description, reference),)) if item.semantic_id == 'ask')
        assert len(requests) == 4
        for request in requests:
            torch.testing.assert_close(request.roles[0], expected)
            assert request.roles[0, :width].ge(0).all()
            assert request.roles[0, :width].le(1).all()
        requests[0].roles[0].sum().backward()
        torch.testing.assert_close(payload.grad[:occupied, :width],
                                   torch.full((occupied, width), 1 / occupied))
        torch.testing.assert_close(payload.grad[:occupied, width:],
                                   torch.full_like(payload.grad[:occupied, width:], occupied ** -.5))
        assert payload.grad[occupied:].eq(0).all()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_containment_certificate_and_constructed_before_violation():
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    report = cb.mereology.containment_audit()
    for name in ('tower0', 'tower1', 'cross', 'order0'):
        assert report[name]['violating_pairs'] == 0, (name, report[name])
    assert report['order0']['pairs'] == 1 and report['order0']['non_strict_pairs'] == 0
    assert report['tower1']['pairs'] > 0
    assert report['cross']['pairs'] == 0 and not report['cross']['applicable']
    values = dict(zip((a, b), cb.lookup_rows(torch.tensor([a, b]))))
    displaced = {r: v.clone() for r, v in values.items()}
    displaced[a][0] = 1
    audit = cb.mereology.containment_audit(before=displaced, after=values)
    assert audit['before']['order0']['violating_pairs'] == 1
    assert audit['after']['order0']['violating_pairs'] == 0


def test_one_call_conjoins_both_subspaces_and_wholeof_is_dual():
    from Layers import Ops
    small, big = torch.tensor([.2, .1]), torch.tensor([.8, .9])
    crossed = torch.tensor([.1, .95])
    assert Ops.partOf(small, big) and Ops.wholeOf(big, small)
    assert not Ops.partOf(big, small)
    assert not Ops.partOf(small, crossed) and not Ops.partOf(crossed, small)
    same = small.clone()
    assert Ops.partOf(small, same) and Ops.partOf(same, small)
    assert torch.equal(small, same)  # Antisymmetry is on vectors, not row ids.
    middle = (small + big) / 2
    assert Ops.partOf(small, middle) and Ops.partOf(middle, big)
    assert Ops.partOf(small, big) and Ops.partOf(small, small)


def test_certificate_orders_weights_allows_ties_and_imposes_no_part_type_order():
    from Layers import Ops
    from test_review13_subspaces import word
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    properties = ws.primitive_properties
    with torch.no_grad():
        properties.members.zero_()
        properties.members[0, 97] = .5
        properties.members[1, 97] = 1
        properties.members[2, 97] = .5
        properties.observed[:3] = True
        ps.W[8].copy_(ps.W[4])
    ws.begin_forward()
    # A soft meet need not be contained in every selected type.
    assert properties.fill(b'a')[97] == .75
    assert properties.codes()[0, 97] == .5
    report = cb.mereology.containment_audit()
    assert report['tower1']['pairs'] == 4
    assert report['tower1']['support_pairs_without_weight_order'] == 2
    assert report['tower1']['violating_pairs'] == 0
    assert not report['cross']['applicable'] and report['cross']['pairs'] == 0
    left = word(cs, [(4, 1.)])
    right = word(cs, [(4, 1.), (8, 1.)])
    codes = cb.lookup_rows(torch.tensor([left, right]))[:, :cb.mereology.percept_width]
    assert Ops.partOf(codes[0], codes[1]) and Ops.partOf(codes[1], codes[0])
    report = cb.mereology.containment_audit()
    assert report['order0']['violating_pairs'] == 0
    assert report['order0']['non_strict_pairs'] > 0  # Informational, not a failure.


def test_certificate_requires_weighted_whole_nesting_for_a_soft_meet():
    from test_review13_subspaces import word
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    left, right = word(cs, [(4, 1.)]), word(cs, [(4, 1.)])
    cs.add_concept_feature(left, 'ws', 0, .25)
    cs.add_concept_feature(right, 'ws', 0, .75)
    report = cb.mereology.containment_audit()
    assert report['support_pairs_without_weight_order'] > 0
    assert report['order0']['violating_pairs'] == 0


def test_overlapping_feature_groups_keep_soft_meet_presence_in_the_cube():
    from test_review13_subspaces import word
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    row = word(cs, [(4, 1.)])
    cs.add_concept_feature(row, 'ws', (0, 1), 1.)
    cs.add_concept_feature(row, 'ws', 1, 1.)
    assert (1, 1, 2.) in cb.mereology._definitions([row])[row]
    value = cb.lookup_rows(row)[:cb.mereology.percept_width]
    expected = torch.cat((ps.W[4, :3], ws.W[:2, 3:].amin(0)))
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert (value >= 0).all() and (value <= 1).all()


def test_retirement_certificates():
    from MereologicalCodes import MereologicalCodes
    import SymbolCentroid
    for name in ('_interval', 'centroid_forms', 'centroid_audit'):
        assert not hasattr(MereologicalCodes, name)
    assert not hasattr(SymbolCentroid, 'place')
    from test_review13_subspaces import fixture
    assert not hasattr(fixture()[1].mereology, '_centroid_cache')
    receipt = os.environ.get('ITEM28_RECEIPT')
    if receipt:
        Path(receipt, 'retirement-certificates.json').write_text(json.dumps(dict(
            absent=['MereologicalCodes._interval', 'MereologicalCodes.centroid_forms',
                    'MereologicalCodes.centroid_audit', 'MereologicalCodes._centroid_cache',
                    'SymbolCentroid.place']), indent=2)+'\n')


def test_checkpoint_rejects_a_different_subspace_layout():
    from MereologicalCodes import MereologicalCodes
    _, cb, _, _, _, _, _ = extended_fixture()
    saved = cb.mereology.state_dict()
    target = MereologicalCodes(cb.mereology.owner, cb.W, percept_width=259,
                               percept_event_width=259, identity_width=4, property_width=255)
    with pytest.raises(ValueError, match='subspaces'):
        target.load_state_dict(saved)
    old = {key: value for key, value in saved.items() if key != '_form_layout'}
    with pytest.raises(ValueError, match='subspaces'):
        cb.mereology.load_state_dict(old)


def test_cost_certificate_has_no_pairwise_or_three_axis_union_workspace():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    class Allocations(TorchDispatchMode):
        def __init__(self):
            self.shapes = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            self.shapes.extend(list(t.shape) for t in tree_flatten(result)[0] if torch.is_tensor(t))
            return result
    book = property_book(rows=17)
    hashed = PrimitiveProperties(17, property_width=7, ones=2)
    allocation = Allocations()
    with allocation:
        book.begin_forward()
        book.primitive_properties.fill(b'ab')
        hashed.codes()
    assert [17, 17] not in allocation.shapes
    assert all(len(s) < 3 for s in allocation.shapes), allocation.shapes
    receipt = os.environ.get('ITEM28_RECEIPT')
    if receipt:
        Path(receipt, 'cost-certificate.json').write_text(json.dumps(dict(
            rows=17, property_width=256, allocations=allocation.shapes,
            pairwise_tensor=False, rows_bytes_width_tensor=False), indent=2)+'\n')


def test_native_lesson_certificate(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1, word_buckets='16')
    try:
        identity = model.perceptualSpace.percept_store.identity
        with identity.nested_lesson('s'):
            model.forward(model.inputSpace.prepInput(['cat cats']))
        cb = model._concept_owner().similarity_codebook
        report = cb.mereology.containment_audit()
        for group in ('tower0', 'tower1', 'cross', 'order0'):
            assert report[group]['violating_pairs'] == 0, (group, report[group])
        assert report['order0']['pairs'] > 0
        native_rows = []
        for raw in (b'cat', b'cats'):
            row = next(r for r in cb.mereology._definitions(
                model._concept_owner()._concept_allocator.layer()._tensor_row_keys)
                if model._concept_owner().word_surface_for_row(r) == raw)
            torch.testing.assert_close(cb.lookup_rows(row)[:identity.width], identity.form(raw).to(cb.W))
            native_rows.append(row)
        from Layers import Ops
        small, large = cb.lookup_rows(torch.tensor(native_rows))[:, :cb.mereology.percept_width]
        assert Ops.partOf(small, large) and bool((small < large).any())
        receipt = os.environ.get('ITEM28_RECEIPT')
        if receipt:
            Path(receipt, 'native-containment.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
