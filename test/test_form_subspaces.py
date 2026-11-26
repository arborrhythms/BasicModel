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

#: The wholes half (sigma-pi rule 50): 64 coordinates, a byte at its group's.
P = 64


def at(properties, byte, *others):
    """The coordinate of ``byte``'s group in the wholes half, which no byte
    of ``others`` shares."""
    coordinate = properties.byte_coordinate[byte]
    assert all(properties.byte_coordinate[other] != coordinate for other in others)
    return coordinate


def property_book(rows=4, identity=3, band=0):
    from Spaces import Codebook
    book = Codebook()
    book.W = nn.Parameter(torch.zeros(rows, identity + P + band))
    book.primitive_properties = PrimitiveProperties(rows, property_width=P)
    book.enable_property_codes(identity_width=identity, property_width=P)
    return book


def extended_fixture():
    from test_review13_subspaces import fixture, word
    cs, cb, ps, alloc = fixture(property_width=P)
    ws = property_book(4)
    with torch.no_grad():
        ws.primitive_properties.members[0, [97, 98]] = 1
        ws.primitive_properties.members[1, [97, 98, 99]] = 1
        ws.primitive_properties.observed[:2] = True
        ps.W[4, :3] = torch.tensor([.2, .2, .2])
        ps.W[7, :3] = torch.tensor([.8, .3, .2])
    ws.primitive_properties.regroup()
    ws.begin_forward()
    cs._model.wholeSpace = NS(subspace=NS(what=ws))
    from BasicPercepts import BasicPercepts
    basic = BasicPercepts(NS(_basis=ps), 3, P, 64)
    basic.bind(ws)
    object.__setattr__(cs, '_basic_percepts', basic)
    a, b = word(cs, [(4, 1.)], (0, 1)), word(cs, [(4, 1.), (7, 1.)], (1,))
    return cs, cb, ps, ws, alloc, a, b


def test_construction_resolves_the_split_and_preserves_numeric_layout():
    from architecture import WHERE_WIRES, WHEN_WIRES
    band = WHERE_WIRES + WHEN_WIRES  # 8 -> 16
    config = {'architecture': {'data': {'dataType': 'embedding'}},
              'PartSpace': {'nDim': 1 + band}, 'ConceptualSpace': {'nDim': 129 + band}}
    resolve_identity_layout(config)
    assert config['PartSpace']['propertyDim'] == 64
    for name in ('InputSpace', 'PartSpace', 'WholeSpace'):
        assert config[name]['nDim'] == 64 + 64 + band
    assert config['ConceptualSpace']['nDim'] == 64 + 64 + band + 128
    first = copy.deepcopy(config)
    resolve_identity_layout(config)
    assert config == first
    numeric = {'architecture': {'data': {'dataType': 'numeric'}},
               'PartSpace': {'nDim': 9}, 'ConceptualSpace': {'nDim': 137}}
    before = copy.deepcopy(numeric)
    resolve_identity_layout(numeric)
    assert numeric == before


def test_invalid_property_construction_fails_loud():
    with pytest.raises(ValueError, match='propertyDim'):
        resolve_identity_layout({'architecture': {'data': {'dataType': 'embedding'}},
                                 'PartSpace': {'propertyDim': 0}})


def test_catalog_byte_union_is_monotone_and_differentiable_for_weighted_members():
    properties = PrimitiveProperties(4, property_width=P)
    with torch.no_grad():
        properties.members[0, 97] = .2
        properties.members[1, 97:99] = torch.tensor([.4, .3])
        properties.members[2, 97:100] = torch.tensor([.6, .5, .4])
        properties.members[3, 97:101] = torch.tensor([.8, .7, .6, .5])
    properties.regroup()
    # Every byte is one at its group's coordinate of the wholes half; a
    # property's code is the OR (greatest) of its members' codes.
    assert properties.byte_bits.sum(1).eq(1).all()
    codes = properties.codes()
    direct = (properties.members[:, :, None] * properties.byte_bits).amax(1)
    torch.testing.assert_close(codes, direct)
    assert (codes[:-1] <= codes[1:]).all()
    columns = sorted({properties.byte_coordinate[byte] for byte in range(97, 101)})
    codes[:, columns].sum().backward()
    assert properties.members.grad[:, 97:101].gt(0).any(0).all()
    torch.testing.assert_close(properties.codes(torch.tensor([1, 3])), codes[[1, 3]])


def test_grouped_union_matches_direct_max_without_a_broadcast_workspace():
    properties = PrimitiveProperties(4, property_width=7)
    with torch.no_grad():
        properties.members.copy_(torch.arange(1024).reshape(4, 256) / 1024)
    properties.regroup()
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
    book.lookup_rows(0)[3 + at(properties, 97, 98)].backward()
    assert properties.members.grad[0, 97] == 1
    properties.members.grad = None
    book.begin_forward()
    book.getW()[:, 3 + at(properties, 98, 97)].sum().backward()
    # Row 0 holds 98 alone at its coordinate; an untaught row's zero members
    # tie there and share the derivative.
    assert properties.members.grad[0, 98] == 1 and properties.members.grad[:, 98].gt(0).all()
    assert book.W.data_ptr() == pointer
    revision = properties.revision
    with torch.no_grad():
        properties.members[0, 97] = .75
    assert properties.revision > revision
    book.begin_forward()
    assert book.W[0, 3 + at(properties, 97, 98)] == .75


def test_retired_property_fills_leave_live_basic_percept_reads():
    from Spaces import Codebook
    basis = Codebook()
    basis.W = nn.Parameter(torch.zeros(4, 128))
    identity = WordIdentity(NS(dim=128, _capacity=4, _basis=basis), property_width=P)
    atom = ('part', 'a')
    identity.atoms[atom] = 0
    identity.words.update({b'a': (atom,), b'z': (('part', 'z'),)})
    identity.word_rows.update({b'a': 1, b'z': 2})
    identity.atoms[('part', 'z')] = 3
    properties = PrimitiveProperties(2, property_width=P)
    properties.teach(0, [97, 98], [1., 1.], steps=1, rate=1.)
    pointers = basis.W.data_ptr(), basis.fixed_row_codes.data_ptr()
    identity.refresh_property_fills(properties)
    assert not basis.W[:, 64:].any()
    identity.basic.bind(NS(lookup_rows=lambda rows: torch.cat((torch.ones(2, 64), properties.codes()), -1)[rows]))
    row = identity.basic.reserve((0,), (0,), word=10)
    torch.testing.assert_close(identity.basic.code(row)[64:], properties.codes()[0])
    before = basis.W[:, :64].detach().clone()
    revision = properties.revision
    properties.teach(1, [97, 99], [.5, .8], steps=1, rate=1.)
    assert properties.revision > revision
    identity.refresh_property_fills()
    p = properties.on_counts(torch.nn.functional.one_hot(torch.tensor(97), 256).float())
    expected = (1-p[:, None]*(1-properties.codes())).amin(0)
    assert not basis.W[:, 64:].any()
    torch.testing.assert_close(identity.basic.code(row)[64:], properties.codes()[0])
    torch.testing.assert_close(basis.W[:, :64], before, rtol=0, atol=0)
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
        assert len(requests) == 4  # System 1 offers one read, not one point per order.
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
    basic = cs.basic_percepts
    values = {row: basic.code(row).detach() for row in basic.index.values()}
    displaced = {r: v.clone() for r, v in values.items()}
    displaced[basic.words[cs.concept_id_at_row(a)]][0] = 1
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


def test_basic_certificate_allows_ties_and_imposes_no_part_type_order():
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
    properties.regroup()
    ws.begin_forward()
    # A soft meet need not be contained in every selected type.
    assert properties.fill(b'a')[at(properties, 97)] == .75
    assert properties.codes()[0, at(properties, 97)] == .5
    report = cb.mereology.containment_audit()
    assert report['tower1']['pairs'] == 3
    assert report['premises'] == 'basic_percepts'
    assert report['tower1']['violating_pairs'] == 0
    assert not report['cross']['applicable'] and report['cross']['pairs'] == 0
    left = word(cs, [(4, 1.)])
    right = word(cs, [(4, 1.), (8, 1.)])
    codes = cb.lookup_rows(torch.tensor([left, right]))[:, :cb.mereology.percept_width]
    assert Ops.partOf(codes[0], codes[1]) and Ops.partOf(codes[1], codes[0])
    report = cb.mereology.containment_audit()
    assert report['order0']['violating_pairs'] == 0
    assert report['order0']['non_strict_pairs'] > 0  # Informational, not a failure.


def test_concept_weights_do_not_change_basic_containment_premises():
    from test_review13_subspaces import word
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    left, right = word(cs, [(4, 1.)]), word(cs, [(4, 1.)])
    cs.add_concept_feature(left, 'ws', 0, .25)
    cs.add_concept_feature(right, 'ws', 0, .75)
    report = cb.mereology.containment_audit()
    assert report['support_pairs_without_weight_order'] == 0
    assert report['premises'] == 'basic_percepts'
    assert report['order0']['violating_pairs'] == 0


def test_overlapping_feature_groups_collect_basic_percepts_in_the_cube():
    from test_review13_subspaces import word
    cs, cb, ps, ws, alloc, a, b = extended_fixture()
    row = word(cs, [(4, 1.)], (0, 1))
    cs.add_concept_feature(row, 'ws', (0, 1), 1.)
    cs.add_concept_feature(row, 'ws', 1, 1.)
    bp = cs.basic_percepts.words[cs.concept_id_at_row(row)]
    assert any(member == bp for member, weight in cb.mereology._definitions([row])[row])
    value = cb.lookup_rows(row)[:cb.mereology.percept_width]
    # The property selector collects both basic percepts carrying property 1.
    # Each contributes its own tied identity/property centroid once.
    expected = (torch.cat((ps.W[4, :3], ws.W[:2, 3:].amin(0))) +
                torch.cat((ps.W[[4, 7], :3].amax(0), ws.W[1, 3:]))) / 2
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
    target = MereologicalCodes(cb.mereology.owner, cb.W, percept_width=3 + P,
                               percept_event_width=3 + P, identity_width=4, property_width=P - 1)
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
    # 37 rows: no rows-by-bytes slice of the union is square by coincidence.
    book = property_book(rows=37)
    hashed = PrimitiveProperties(37, property_width=7)
    allocation = Allocations()
    with allocation:
        book.begin_forward()
        book.primitive_properties.fill(b'ab')
        hashed.codes()
    assert [37, 37] not in allocation.shapes
    assert all(len(s) < 3 for s in allocation.shapes), allocation.shapes
    receipt = os.environ.get('ITEM28_RECEIPT')
    if receipt:
        Path(receipt, 'cost-certificate.json').write_text(json.dumps(dict(
            rows=37, property_width=P, allocations=allocation.shapes,
            pairwise_tensor=False, rows_bytes_width_tensor=False), indent=2)+'\n')


def test_native_lesson_certificate(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1, word_buckets='16')
    try:
        identity = model.perceptualSpace.percept_store.identity
        # The forms are words from the reading in which MTM nominates them.
        for _ in range(model._concept_owner().concept_mint_count):
            model.forward(model.inputSpace.prepInput(['cat cats']))
            model.End()
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


def test_contrast_table_reads_exactly_one_and_zero():
    """Each row of the table of wholes read against every byte, every row and
    some forms by ``min_k max(1 - p_k, r_k)``: exactly 1 where the row holds
    the percept (on every byte of it) and 0 where it does not."""
    from Spaces import _CANONICAL_PROPERTY_ROWS, _CHAR_CLASS_RANGES
    names = [name for name, _ in _CANONICAL_PROPERTY_ROWS]
    properties = PrimitiveProperties(len(names) + 1, property_width=P)
    for row, (name, kind) in enumerate(_CANONICAL_PROPERTY_ROWS):
        if name == 'word':
            properties.define_word(row)
            continue
        targets = torch.zeros(256)
        for lo, hi in _CHAR_CLASS_RANGES[int(kind)]:
            targets[lo:hi + 1] = 1.
        properties.teach(row, torch.arange(256), targets)
    properties.define_by_rows(len(names), (names.index('letter'), names.index('capital')))
    holds = properties.coefficients().detach() > .5
    codes = properties.codes().detach()

    def read(p, r):
        return torch.maximum(1 - p, r).amin(-1)

    assert len(set(properties.byte_coordinate)) == 8
    assert torch.equal(read(properties.byte_bits[None], codes[:, None]), holds.float())
    inside = (~holds[:, None] | holds[None]).all(-1)
    assert torch.equal(read(codes[:, None], codes[None]), inside.float())
    for raw in (b'a1', b'ab', b'A', b'12', b'Ab', b'a b'):
        form = properties.byte_bits[list(raw)].amax(0)
        assert torch.equal(read(form, codes), holds[:, list(raw)].all(1).float()), raw
    mixed = properties.byte_bits[list(b'a1')].amax(0)
    assert read(mixed, codes[names.index('letter')]) == 0 and read(mixed, codes[names.index('digit')]) == 0
    assert read(mixed, properties.top()) == 1
