"""Review regressions: located cases, negative witnessing and open attention."""
from types import SimpleNamespace

import torch

import Spaces
from Models import BasicModel
from test_cs_sparse_weights import _cs, _mint_row
from test_concept_memberships import binary_features


def test_located_conjunction_matches_requested_poles_before_union():
    cs = _cs()
    cs.conceptual_pi = True
    assert _mint_row(cs, 0, 100) == 0
    row = _mint_row(cs, 0, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True, locations=((0, 1),))
    cs.add_concept_edge(row, 0, 1., conjunctive=True, negated=True,
                        locations=((1, 2),))
    native, extents = binary_features(cs, torch.tensor([[48, 48], [48, 49], [49, 48], [49, 49]]))
    read = cs.cs_read_memberships(native, extents)
    torch.testing.assert_close(read[row, :, 0, 0], torch.tensor([0., 0., 1., 0.]))
    # Neither a symbol union nor two separate subjects can supply this case.
    split = torch.tensor([[[0, 1], [1, 2]]]).expand(4, -1, -1)
    read = cs.cs_read_memberships(native, split)
    assert read[row, ..., 0].count_nonzero() == 0
    saved = Spaces._concept_alloc_of(cs).layer().parts_extras()
    other = _cs()
    restored = Spaces._concept_alloc_of(other).layer()
    restored.load_parts_extras(saved)
    assert restored.conjunctive.locations == Spaces._concept_alloc_of(cs).layer().conjunctive.locations


def test_passback_scales_unknown_parts_without_erasing_location():
    cs = SimpleNamespace(_order_caps=lambda: (2,), cs_percept_attribution=lambda *a, **k:
        (torch.tensor([0]), torch.tensor([[[[1., 0.]]]]),
         torch.tensor([[[0, 1], [1, 2]]])))
    model = SimpleNamespace(subsymbolic_loop={1}, conceptualSpaces=[cs],
        conceptualSpace=cs, _subsymbolic_field=torch.zeros(2, 1, 1, 2),
        perceptualSpace=SimpleNamespace(nWhat=2, _forward_input={
            'part_spans': torch.tensor([[[0, 1], [1, 2]]])}),
        _staged_concepts_in=torch.tensor([[[65, 66]]]))
    sub = Spaces.SubSpace(inputShape=(2, 4), outputShape=(2, 4), nInputDim=4, nOutputDim=4)
    event = torch.tensor([[[.4, .8, -.2, .6], [.8, .4, .6, -.2]]])
    sub.set_event(event)
    result = BasicModel._passback_scope_ps(model, 1, sub, None).materialize()
    assert bool((result[..., :2] > 0).all())
    torch.testing.assert_close(result[0, 0, :2], event[0, 0, :2])
    torch.testing.assert_close(result[0, 1, :2], .5 * event[0, 1, :2])
    torch.testing.assert_close(result[..., 2:], event[..., 2:])
    torch.testing.assert_close(sub.materialize(), event)


def test_bound_field_can_include_unwritten_inventory_addresses():
    cs = _cs(nS=128, order=2)
    store = Spaces._concept_alloc_of(cs).layer()
    # A definition-free feature store may precede the expanded inventory.
    store.grow_inventory(128)
    store.features.nOutput = 112
    view = cs._concept_field_store(torch.tensor([0, 112, 127, -1]))
    assert not view.feature_defined.any()


def test_sentence_boundary_writes_present_percepts_at_the_negative_pole():
    from PerceptProperties import PrimitiveProperties
    cs = _cs()
    cid = cs.new_concept()
    row = cs._csw_concept_row(0, cid)
    for part in (7, 8):
        cs.add_concept_feature(row, 'ps', part, 0.)
    cs.add_concept_feature(row, 'ws', 0, -1.)
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48], [1.])
    spans = torch.tensor([[[0, 1]]])
    cs.cs_read_memberships((torch.tensor([[7]]), spans, primitive,
                            torch.tensor([[48]]), spans), spans)
    torch.testing.assert_close(cs._cs_position_evidence[0, 0, 0, 0], torch.tensor([0., 1.]))
    cs.Reset(hard=True)
    features = Spaces._concept_alloc_of(cs).layer().features
    written = {col for (_, col), i in features._index.items() if float(features.values[i]) > 0}
    assert 4 * 7 + 1 in written
    assert 4 * 7 not in written
    assert 4 * 8 not in written and 4 * 8 + 1 not in written


def test_boundary_candidates_keep_located_brackets_and_do_not_veto():
    cs = _cs()
    cs.conceptual_pi = True
    _mint_row(cs, 0, 100)
    row = _mint_row(cs, 0, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True, locations=((0, 1),))
    # An unwritten source outside the bound field must not veto this pattern.
    cs.add_concept_edge(row, 60, 0., conjunctive=True, locations=((1, 2),))
    native, extents = binary_features(cs, torch.tensor([[49, 48]]))
    before = cs.cs_read_memberships(native, extents)
    assert float(before[row, 0, 0, 0]) == 1.
    cs._prepare_part_learning()
    matrix = Spaces._concept_alloc_of(cs).layer().conjunctive
    opposite = matrix.nOutput + 1
    assert matrix.locations[row, opposite] == ((0, 1),)
    after = cs.cs_read_memberships(native, extents)
    torch.testing.assert_close(after, before)


def test_boundary_candidates_stay_in_the_preceding_symbolic_order():
    cs = _cs()
    _mint_row(cs, 0, 100)
    one = _mint_row(cs, 1, 101)
    two = _mint_row(cs, 2, 102)
    cs.add_concept_edge(one, 0, 1.)
    cs.add_concept_edge(two, one, 1.)
    field = torch.zeros(cs._order_caps()[0], 1, 1, 2)
    field[0, 0, 0, 0] = 1.
    object.__setattr__(cs, '_cs_last_a0', field)
    cs._prepare_part_learning()
    matrix = Spaces._concept_alloc_of(cs).layer()
    assert all(col % (matrix.nOutput + 1) != 0
               for row, col in matrix._index if row == two)


def test_primed_reading_writes_the_shared_field_scope():
    canonical = SimpleNamespace(_stage0_indices=torch.tensor([[2, 5]]),
                                priming_weights=lambda: torch.tensor([1., 1., 1., 1., 1., 6.]))
    whole = SimpleNamespace(_staged_analysis_spans=torch.tensor([[[0, 5], [6, 11]]]),
                            _priming_target=lambda: canonical)
    owner = SimpleNamespace()
    model = SimpleNamespace(wholeSpaces=[whole], conceptualSpace=owner,
                            _staged_concepts_in=torch.zeros(1, 1, 16))
    BasicModel._primed_reading_step(model)
    torch.testing.assert_close(owner._passback_scope_where,
                               torch.tensor([[6., 11.]]) / 16)
