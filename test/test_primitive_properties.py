"""Primitive property learning and concept evidence grounded in extents."""
import torch
import Spaces
import pytest

from test_cs_sparse_weights import _cs, _mint_row, _mint_field


def test_different_positions_in_one_extent_supply_the_two_symbols():
    from test_concept_memberships import binary_features
    cs = _cs()
    cs.conceptual_pi = True
    native, _ = binary_features(cs, torch.tensor([[49, 48, 49, 49]]))
    extents = torch.tensor([[[0, 2], [2, 4]]])
    read = cs.cs_read_memberships(native, extents)
    torch.testing.assert_close(read[0, 0], torch.tensor([[1., 1.], [1., 0.]]))
    assert cs._cs_position_evidence.shape == (cs._order_caps()[0], 1, 2, 8, 2)


def test_live_property_membership_is_a_learned_byte_definition(tmp_path):
    from test_wholespace_property_migration import _small_property_model
    model = _small_property_model(tmp_path)
    basis = model.wholeSpace.subspace.what
    assert hasattr(basis, 'primitive_properties'), 'fixed property tags are not a learned basis'
    assert basis.primitive_properties.members.shape == (8, 256)
    assert isinstance(basis.primitive_properties.members, torch.nn.Parameter)


def test_arbitrary_byte_subset_learns_without_a_predefined_name():
    from PerceptProperties import PrimitiveProperties
    basis = PrimitiveProperties(1)
    atoms = torch.arange(256)
    target = ((atoms % 7 == 3) | (atoms == 201)).float()
    optimizer = torch.optim.SGD(basis.parameters(), lr=.5)
    for _ in range(32):
        optimizer.zero_grad()
        loss = .5 * (basis(atoms)[:, 0] - target).square().sum()
        loss.backward()
        optimizer.step()
        basis.project()
    torch.testing.assert_close(basis(atoms)[:, 0], target, atol=1e-6, rtol=0)
    mixture = torch.zeros(2, 256)
    mixture[0, 3] = .4
    mixture[0, 10] = .5
    mixture[1, 4] = 1.
    torch.testing.assert_close(basis.from_primitives(mixture)[:, 0], torch.tensor([.5, 0.]))


def test_complement_requires_an_observation():
    from PerceptProperties import PrimitiveProperties
    basis = PrimitiveProperties(1)
    basis.teach(0, [48, 49], [0., 1.])
    observed = torch.tensor([[True, True, False]])
    values = torch.tensor([[48, 49, 48]])
    torch.testing.assert_close(basis(values, observed)[..., 0], torch.tensor([[0., 1., 0.]]))
    torch.testing.assert_close(basis.complement(values, observed)[..., 0], torch.tensor([[1., 0., 0.]]))


def test_property_union_and_part_conjunction_from_primitive_counts():
    from PerceptProperties import PrimitiveProperties, counts_in_spans
    basis = PrimitiveProperties(1)
    basis.teach(0, [48, 49], [0., 1.])
    values = torch.tensor([[48, 48], [48, 49], [49, 48], [49, 49], [0, 0]])
    spans = torch.tensor([[[0, 2]]]).expand(5, -1, -1)
    counts = counts_in_spans(values, spans, observed=values != 0)
    torch.testing.assert_close(basis.on_counts(counts, conjunctive=False).flatten(),
                               torch.tensor([0., 1., 1., 1., 0.]))
    torch.testing.assert_close(basis.on_counts(counts).flatten(),
                               torch.tensor([0., 0., 0., 1., 0.]))


def test_property_reverse_distributes_only_to_its_primitive_members():
    from PerceptProperties import PrimitiveProperties
    basis = PrimitiveProperties(1)
    basis.teach(0, [65, 66], [1., 1.])
    reverse = basis.reverse(torch.tensor([[.8]]))
    torch.testing.assert_close(reverse[0, 65:67], torch.tensor([.4, .4]))
    assert float(reverse.sum()) == pytest.approx(.8)
    assert int(torch.count_nonzero(reverse)) == 2


def test_word_read_trains_definitions_and_preserves_ordered_witness(tmp_path):
    from test_wholespace_property_migration import _small_property_model
    model = _small_property_model(tmp_path)
    ws = model.wholeSpace
    counts = ws.stage_word_primitives([['Ab', '7']], torch.ones(1, 2, dtype=torch.bool))
    event = ws.compute_word_property_event(counts[:, 0])
    event.square().sum().backward()
    primitive = ws.subspace.what.primitive_properties
    assert primitive.members.grad is not None
    assert primitive.members.grad[:, [65, 98]].abs().sum() > 0
    assert counts[0, 0].sum() == 2
    assert counts[0, 1, :, 55].sum() == 1
    assert ws._staged_word_property_spans[0, 0, 0].tolist() == [0, 1]


def test_property_definition_has_one_optimizer_and_checkpoint_owner(tmp_path):
    from test_wholespace_property_migration import _small_property_model
    model = _small_property_model(tmp_path)
    primitive = model.wholeSpace.subspace.what.primitive_properties
    optimizer = model.getOptimizer(lr=.001)
    assert sum(p is primitive.members for g in optimizer.param_groups for p in g['params']) == 1
    keys = [k for k, v in model.state_dict().items() if v.data_ptr() == primitive.members.data_ptr()]
    assert len(keys) == 1
    with torch.no_grad():
        primitive.members[0, 65] = .7
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    extras = model._collect_structural_extras()
    with torch.no_grad():
        primitive.members[0, 65] = .2
    model.load_state_dict(saved)
    model._restore_structural_extras(extras)
    assert float(primitive.members[0, 65]) == pytest.approx(.7)


def test_old_acquired_predicate_migrates_into_primitive_parameters(tmp_path):
    from test_wholespace_property_migration import _small_property_model
    model = _small_property_model(tmp_path)
    extras = model._collect_structural_extras()
    entry = extras['whole_properties'][0]
    entry.setdefault('attributes', {})['_row_bytes'] = {'7': [97, 101]}
    model._restore_structural_extras(extras)
    ws = model.wholeSpaces[0]
    target = torch.zeros(256)
    target[[97, 101]] = 1.
    torch.testing.assert_close(ws.subspace.what.primitive_properties.members[7], target)
    assert '_row_bytes' not in ws.__dict__
    assert '_row_bytes' not in model._collect_structural_extras()['whole_properties'][0].get('attributes', {})


def test_property_priming_projects_direct_surface_rows(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    ps, ws = model.perceptualSpace, model.wholeSpaces[-1]
    row = _mint_row(cs, 0, 101)
    cs._priming_bridge_put(101, None, [1, 3], [0, 2])
    cs.prime_seen(torch.tensor([row]), decay=1., bump=1.)
    cs.project_priming_to_towers(ps, ws)
    assert set((ps.priming_weights() > 1).nonzero().flatten().tolist()) == {1, 3}
    assert set((ws.priming_weights() > 1).nonzero().flatten().tolist()) == {0, 2}
    assert not hasattr(ws, '_pos_kind')


def test_located_copresence_writes_each_witnessed_pole():
    import Spaces
    from test_attention_promotion import _fixture, _pool_rows
    cs, rows = _fixture(pi=True)
    a, b, negative = [r for _, r in rows[:3]]
    evidence = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    evidence[[a, b], 0, 0, 0] = 1.
    evidence[negative, 0, 0, 1] = 1.
    cs._promo_last_acts = evidence
    cs._cs_level_rows = [torch.arange(cs._order_caps()[0])[:, None]]
    cs._cs_position_evidence = evidence[:cs._order_caps()[0]].unsqueeze(-2)
    cs._cs_position_spans = torch.tensor([[[0, 1]]])
    cs._cs_extents = torch.tensor([[[0, 1]]])
    cs.promotion_observe()
    assigned = _pool_rows(cs)
    assert len(assigned) == 1
    store = Spaces._concept_alloc_of(cs).layer()
    negative_column = negative + store.nOutput + 1
    assert dict(cs.concept_weights(assigned[0], conjunctive=True)) == {a: 1., b: 1.}
    assert dict(cs.concept_weights(assigned[0], conjunctive=True, negated=True)) == {negative: 1.}
    assert store.conjunctive.locations[assigned[0], negative_column] == ((0, 1),)


def test_extent_read_keeps_missing_distinct_from_observed_zero():
    from ConceptEvidence import in_extents
    pairs = torch.tensor([[[[1., 0.], [0., 1.]]]])
    positions = torch.tensor([[[0, 1], [1, 2]]])
    extent = torch.tensor([[[0, 2]]])
    field, retained = in_extents(pairs, positions, extent)
    torch.testing.assert_close(field.flatten(), torch.tensor([1., 1.]))
    assert retained.shape == (1, 1, 1, 2, 2)
    for index, expected in ((0, [1., 0.]), (1, [0., 1.])):
        field, _ = in_extents(pairs[..., index:index+1, :], positions[:, index:index+1], extent)
        torch.testing.assert_close(field.flatten(), torch.tensor(expected))


def test_grounded_read_has_one_owner_and_checkpoint_keeps_positions(tmp_path):
    from test_grounded_xor import grounded_model
    model, x = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    cs.add_concept_feature(0, 'ws', 0, .7)
    model.forward(x)
    carrier = model._combine_last_cs_sub
    optimizer = model.getOptimizer(lr=.001)
    parameter = Spaces._concept_alloc_of(cs).layer().features.values
    assert sum(p is parameter for g in optimizer.param_groups for p in g['params']) == 1
    assert not any(v.data_ptr() == parameter.data_ptr() for v in model.state_dict().values())
    evidence = carrier._concept_activations.clone()
    positions = carrier._concept_position_evidence.clone()
    spans = carrier._concept_position_spans.clone()
    extents = carrier._concept_extents.clone()
    saved = model._collect_structural_extras()
    for name in ('activations', 'position_evidence', 'position_spans', 'extents'):
        object.__setattr__(carrier, '_concept_' + name, None)
    model._restore_structural_extras(saved)
    restored = model.conceptualSpaces[-1].subspace
    torch.testing.assert_close(restored._concept_activations, evidence)
    torch.testing.assert_close(restored._concept_position_evidence, positions)
    torch.testing.assert_close(restored._concept_position_spans, spans)
    torch.testing.assert_close(restored._concept_extents, extents)
    leg = model.symbolSpace.forward_concept_to_symbol(restored)
    torch.testing.assert_close(leg._concept_position_spans, spans)
    assert spans[0, :2].tolist() == [[0, 1], [1, 2]]
    model.End()




def test_candidate_growth_preserves_frozen_field_definitions():
    import Spaces
    cs = _cs()
    frozen = _mint_field(cs, 101)
    live = _mint_field(cs, 102)
    cs.add_concept_edge(frozen, 0, .5, conjunctive=True)
    cs.freeze_concept(101)
    cs.add_concept_edge(live, 0, .5, conjunctive=True)
    cs._prepare_part_learning()
    matrix = Spaces._concept_alloc_of(cs).layer().conjunctive
    candidate = matrix._index[live, matrix.nOutput + 1]
    cs._hebbian_strengthen(102)
    assert float(matrix.values[candidate]) == 0.
    matrix.values.sum().backward()
    assert float(matrix.values.grad[matrix._index[frozen, 0]]) == 0.
    assert float(matrix.values.grad[matrix._index[live, 0]]) == 1.
