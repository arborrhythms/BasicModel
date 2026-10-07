from types import SimpleNamespace

import torch

import Spaces

from Models import BasicModel

from test_cs_sparse_weights import _cs, _mint_row

from test_concept_memberships import binary_features


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
    read = cs.cs_read_memberships((torch.tensor([[7]]), spans, primitive,
                            torch.tensor([[48]]), spans), spans)
    torch.testing.assert_close(read[0, 0, 0], torch.tensor([0., 1.]))
    cs.Reset(hard=True)
    features = Spaces._concept_alloc_of(cs).layer().features
    written = {col for (_, col), i in features._index.items() if float(features.values[i]) > 0}
    assert 4 * 7 + 1 in written
    assert 4 * 7 not in written
    assert 4 * 8 not in written and 4 * 8 + 1 not in written


def test_boundary_candidates_keep_bracket_semantics_and_do_not_veto():
    cs = _cs()
    cs.conceptual_pi = True
    _mint_row(cs, 0, 100)
    row = _mint_row(cs, 0, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    # An unwritten source outside the bound field must not veto this pattern.
    cs.add_concept_edge(row, 60, 0., conjunctive=True)
    native, extents = binary_features(cs, torch.tensor([[49, 48]]))
    before = cs.cs_read_memberships(native, extents)
    assert float(before[row, 0, 0, 0]) == 1.
    cs._prepare_part_learning()
    matrix = Spaces._concept_alloc_of(cs).layer().conjunctive
    opposite = matrix.nOutput + 1
    assert (row, opposite) in matrix._index
    assert not hasattr(matrix, "locations")
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


def test_primed_reading_writes_the_shared_field_scope(tmp_path, eager_reading):
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=8)
    raw=model.inputSpace.prepInput(['first last'])
    with torch.no_grad():model._lex_embed_stem(raw)
    table=model._attention_words.table
    assert table.done.any()
    last=(table.done.long()*torch.arange(1,table.done.shape[1]+1)).argmax(-1)
    selected=table.intervals[torch.arange(len(last)),last]
    torch.testing.assert_close(model.conceptualSpace._passback_scope_where,
                               selected.to(raw.dtype)/raw.shape[-1])
    assert model.conceptualSpace._passback_scope_space.tolist()==[0]
