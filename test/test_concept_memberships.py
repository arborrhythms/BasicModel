"""Order-zero definitions read native features, independently of codes."""
from basic_fixtures import feature as basic_feature, order_zero_expected, published_presence
import pytest
import torch

import Spaces
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs, _mint_row, _mint_field


def binary_features(cs, raw):
    primitive = PrimitiveProperties(2)
    primitive.teach(0, [48, 49], [0., 1.])
    primitive.teach(1, [48, 49], [1., 0.])
    basic_feature(cs, 0, 'ws', 0, 1., parts=(48, 49))
    basic_feature(cs, 0, 'ws', 1, -1., parts=(48, 49))
    b, length = raw.shape
    ix = torch.arange(length)
    spans = torch.stack((ix, ix + 1), -1)[None].expand(b, -1, -1)
    extents = torch.tensor([[[0, length]]]).expand(b, -1, -1)
    return (raw, spans, primitive, raw, spans), extents


def test_one_property_has_the_four_extent_corners():
    cs = _cs()
    native, extents = binary_features(cs, torch.tensor([[49, 49], [48, 48], [48, 49], [49, 48]]))
    actual = cs.cs_read_memberships(native, extents)
    # Each row of a basic percept that carries the property reads its part's
    # presence times the property's lanes where the part lies (rule 6):
    # property 0 holds at 49 and not at 48; property 1 the reverse.
    raw = torch.tensor([[49, 49], [48, 48], [48, 49], [49, 48]])
    basic, field = cs.basic_percepts, cs._percept_field
    at = {(0, 49): [1., 0.], (0, 48): [0., 1.], (1, 49): [0., 1.], (1, 48): [1., 0.]}
    for prop in (0, 1):
        for part in (48, 49):
            index = field.keys.index(('bp', basic.index[((part,), (prop,))]))
            presence = (raw == part).any(-1).float()[:, None]
            torch.testing.assert_close(field.values[index, :, 0].detach(),
                                       presence * torch.tensor(at[prop, part]))
    # The four corners, true, false, both and both, are read across the
    # concept's members' rows: for where only for is given, against where
    # only against, both where each is.
    corners = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 1.]])
    torch.testing.assert_close(cs.order_zero_both(field, [0])[0, :, 0], corners.amin(-1))
    # The concept is one signed number, for less against.
    for b in range(4):
        _, lanes = order_zero_expected(cs, 0, published_presence(cs, b))
        torch.testing.assert_close(actual[0, b, 0], lanes.to(actual), atol=1e-5, rtol=1e-5)


def test_unrelated_memberships_stay_exactly_zero_at_every_scope():
    cs = _cs()
    for length in (1, 8, 256, 1024):
        native, extents = binary_features(cs, torch.full((1, length), 65))
        actual = cs.cs_read_memberships(native, extents)
        assert actual.count_nonzero() == 0
        assert cs._percept_field.events.count_nonzero() == 0




def test_a_located_part_need_not_pervade_but_the_whole_property_must():
    cs = _cs()
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48, 49], [0., 1.])
    # A PS percept and a WS property have independent native addresses.
    basic_feature(cs, 0, 'ps', 7, .5)
    basic_feature(cs, 0, 'ws', 0, 1., parts=(7, 8))
    raw = torch.tensor([[49, 49], [49, 48], [48, 48]])
    part_ids = torch.tensor([[7, 8], [7, 8], [8, 8]])
    parts = torch.tensor([[[0, 1], [1, 2]]]).expand(3, -1, -1)
    whole = torch.tensor([[[0, 2]]]).expand(3, -1, -1)
    cs.cs_read_memberships((part_ids, parts, primitive, raw, whole), whole)
    # The part at [0,1] suffices at [0,2]; the property must hold at both
    # positions. A missing property adds no counterevidence to the present part.
    # Inspect this occurrence before the extent readout union.
    torch.testing.assert_close(cs._percept_field.event_evidence([('bp', cs.basic_percepts.index[((part,), (0,))]) for part in (7, 8)]).amax(0)[:, 0, -1],
                               torch.tensor([[1., 0.], [0., 0.], [0., 1.]]))
    # A missing byte supplies neither pole, including a missing complement.
    raw[2, 1] = 0
    cs.cs_read_memberships((part_ids, parts, primitive, raw, whole), whole)
    assert cs._percept_field.event_evidence([('bp', cs.basic_percepts.index[((part,), (0,))]) for part in (7, 8)]).amax(0)[2, 0, -1].count_nonzero() == 0


@pytest.mark.parametrize('literal', [(7,), (7, 8)])
def test_part_containment_belongs_to_the_subject_extent(literal):
    cs = _cs()
    basic_feature(cs, 0, 'ps', literal, 1.)
    raw = torch.full((1, 8), 65)
    ids = torch.tensor([[7, 8, 9, 9, 9, 9, 9, 9]])
    starts = torch.arange(8)
    positions = torch.stack((starts, starts + 1), -1)[None]
    extents = torch.tensor([[[0, 4], [4, 8]]])
    actual = cs.cs_read_memberships((ids, positions, None, raw, positions), extents)
    # Other positions inside a subject do not deny its present part: the
    # concept reads its own row for, at tanh(1), and nothing in the extent
    # that lacks it.
    y, lanes = order_zero_expected(cs, 0, published_presence(cs, 0, 0))
    assert y > .76
    torch.testing.assert_close(actual[0, 0, 0], lanes.to(actual), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[0, 0, 1], torch.zeros(2).to(actual))
    assert cs._percept_field.events[0, 0, 0, :, 1].count_nonzero() == 0
    assert cs._percept_field.events[0, 0, 1, :, 0].count_nonzero() == 0
    # Absent parts supply neither pole, even in a completely observed subject.
    missing = cs.cs_read_memberships((ids[:, :6], positions[:, :6], None,
                                      raw, positions), extents)
    assert missing[0, 0, 1].count_nonzero() == 0
    if len(literal) > 1:
        separated = torch.tensor([[[0, 1], [1, 4]]])
        apart = cs.cs_read_memberships((ids, positions, None, raw, positions), separated)
        assert apart[0, 0, :, 0].count_nonzero() == 0
    empty = cs.cs_read_memberships((ids, positions, None, raw, positions), extents[:, :0])
    assert empty.shape[2] == cs._percept_field.events.shape[2] == 0


def test_raw_analysis_layout_clips_regions_to_observed_input():
    from types import SimpleNamespace
    raw = torch.zeros(4, 1, 4096, dtype=torch.long)
    for row, text in enumerate((b'hello world', b'bye', b'', b'a' * 600)):
        raw[row, 0, :len(text)] = torch.tensor(list(text), dtype=torch.long)
    positions, extents = Spaces.WholeSpace.concept_evidence_layout(
        SimpleNamespace(), raw, 8)
    expected = torch.zeros(4, 4096, 2, dtype=torch.long)
    for row, length in enumerate((11, 3, 0, 600)):
        expected[row, :length, 0] = torch.arange(length)
        expected[row, :length, 1] = torch.arange(1, length + 1)
    torch.testing.assert_close(positions, expected)
    torch.testing.assert_close(extents[:, 0], torch.tensor([[0, 11], [0, 3], [0, 0], [0, 600]]))


def test_fractional_feature_weights_use_per_pole_min_and_max_extent_union():
    cs = _cs()
    primitive = PrimitiveProperties(2)
    with torch.no_grad():
        primitive.members[:, 65] = torch.tensor([.25, .36])
    basic_feature(cs, 0, 'ws', 0, .5, parts=(65,))
    basic_feature(cs, 0, 'ws', 1, -1., parts=(65,))
    raw = torch.tensor([[65, 65]])
    spans = torch.tensor([[[0, 1], [1, 2]]])
    extent = torch.tensor([[[0, 2]]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), extent)
    # A fractional property keeps both lanes at the rows of the basic
    # percepts that carry it: the part present, times the property's lanes.
    field = cs._percept_field
    rows = [field.keys.index(('bp', cs.basic_percepts.index[((65,), (prop,))])) for prop in (0, 1)]
    torch.testing.assert_close(field.values[rows, 0, 0].detach(), torch.tensor([[.25, .75], [.36, .64]]))
    # The concept is one signed number, for less against, read by its column.
    _, lanes = order_zero_expected(cs, 0, published_presence(cs))
    torch.testing.assert_close(read[0, 0, 0], lanes.to(read), atol=1e-5, rtol=1e-5)


def test_the_member_record_has_no_penalty_and_an_old_feature_table_is_read_once():
    cs = _cs()
    for feature, weight in enumerate((.8, -.6, .4)):
        basic_feature(cs, 0, 'ws', feature, weight, parts=(65,))
    alloc = Spaces._concept_alloc_of(cs)
    store = alloc.layer()
    record = alloc.order_zero_members
    # The record is no parameter: it has no penalty and no gradient.
    assert cs.definition_sparsity_loss(1.) is None
    assert store.definition_matrices() == (store,)
    assert not any('features' in key for key in cs.state_dict())
    percepts = [cs.basic_percepts.index[((65,), (feature,))] for feature in range(3)]
    assert dict(record) == {(0, percepts[0], False): .8, (0, percepts[1], True): .6,
                            (0, percepts[2], False): .4}
    # A checkpoint's old feature table is read once into the record; an
    # edge at zero (a learning candidate) is recorded at zero.
    old = dict(version=6, buffers={}, matrices={'features': dict(
        rows=[0, 0, 0, 0], cols=[2 * percepts[0], 2 * percepts[1] + 1, 2 * percepts[2], 2 * percepts[2] + 1],
        nInput=2 * cs.basic_percepts.capacity, values=torch.tensor([.8, .6, .4, 0.]))})
    read = store.saved_feature_record(old)
    assert {key: value for key, value in read.items() if value > 0} == pytest.approx(dict(record))
    assert read[0, percepts[2], True] == 0.
    assert 'matrices' not in store.parts_extras()


def test_pruning_a_feature_preserves_the_other_tower_at_the_same_address():
    cs = _cs()
    row = _mint_row(cs, 0, 100)
    basic_feature(cs, row, 'ps', 7, .5)
    basic_feature(cs, row, 'ws', 7, -.5, parts=(65,))
    cs._prepare_part_learning()
    cs._drop_concept_edge(100, 7, side='part')
    record = Spaces._concept_alloc_of(cs).order_zero_members
    bp = cs.basic_percepts.index[((65,), (7,))]
    assert set(record) == {(row, bp, True)}
    cs._drop_concept_edge(100, 7, side='whole')
    assert not record


def test_member_record_keeps_frozen_rows_out_of_the_learned_columns(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    frozen = _mint_row(cs, 0, 100)
    live = _mint_row(cs, 0, 101)
    basic_feature(cs, frozen, 'ws', 0, .5, parts=(65,))
    basic_feature(cs, live, 'ws', 1, .5, parts=(65,))
    cs.freeze_concept(100)
    model.getOptimizer(lr=.001)
    record = Spaces._concept_alloc_of(cs).order_zero_members
    assert {row for row, _, _ in record} == {frozen, live}
    # The record is no parameter; the columns it started are the order-zero
    # Sigma's, and a frozen concept's column receives no gradient.
    assert not any(isinstance(value, torch.Tensor) for value in record.values())
    cs._prepare_part_learning()
    assert {row for row, _, _ in record} == {frozen, live}
    cs.order_zero.weight.sum().backward()
    assert float(cs.order_zero.weight.grad[:, frozen].abs().sum()) == 0
    assert float(cs.order_zero.weight.grad[:, live].abs().sum()) > 0


def test_definition_boundary_preserves_gradient_owned_codes(tmp_path):
    from test_grounded_xor import grounded_model
    model, x = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    basic_feature(cs, 0, 'ws', 0, 1., parts=(65,))
    codes = cs.similarity_codebook.getW().detach().clone()
    cs.Reset(hard=True)
    torch.testing.assert_close(cs.similarity_codebook.getW(), codes, rtol=0, atol=0)
    model.forward(x)
    before = cs._cs_last_a0.clone()
    with torch.no_grad():
        cs.similarity_codebook.getW().normal_()
    model.End()
    model.forward(x)
    torch.testing.assert_close(cs._cs_last_a0, before, atol=0, rtol=0)
    model.End()
