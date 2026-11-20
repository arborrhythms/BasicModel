from basic_fixtures import feature as basic_feature, order_zero_expected, published_presence
from order_fixtures import force_order
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
    force_order(cs, cid, 0)
    row = cs._csw_concept_row(0, cid)
    for part in (7, 8):
        basic_feature(cs, row, 'ps', part, 0.)
    basic_feature(cs, row, 'ws', 0, -1., parts=(7,))
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48], [1.])
    spans = torch.tensor([[[0, 1]]])
    read = cs.cs_read_memberships((torch.tensor([[7]]), spans, primitive,
                            torch.tensor([[48]]), spans), spans)
    # The Not member is present: the concept reads against, by its column.
    y, lanes = order_zero_expected(cs, row, published_presence(cs))
    assert y < -.76
    torch.testing.assert_close(read[0, 0, 0], lanes.to(read), atol=1e-5, rtol=1e-5)
    cs.Reset(hard=True)
    record = Spaces._concept_alloc_of(cs).order_zero_members
    written = {2 * percept + int(negated) for (_, percept, negated), weight in record.items() if weight > 0}
    present = cs.basic_percepts.index[((7,), ())]
    absent = cs.basic_percepts.index[((8,), ())]
    assert 2 * present + 1 in written
    assert 2 * present not in written
    assert 2 * absent not in written and 2 * absent + 1 not in written




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


def test_primed_reading_writes_the_shared_field_scope(tmp_path, eager_reading, monkeypatch):
    from test_packed_reconstruction_parity import build_model
    from reading_fixtures import read_in_source_order
    model=build_model(tmp_path,word_capacity=8)
    # Since the loop of looks attention has no accept act of its own: a
    # word is done when grammar interprets it at a look, inside the
    # sentence, and each look's perception writes the scope. The stem's
    # attention-only read therefore marks nothing done. Read the sentence,
    # with its word decisions supplied, and check the scope it leaves.
    read_in_source_order(model, monkeypatch)
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['first last'])
    raw=model.inputSpace.prepInput(['first last'])
    with torch.no_grad():model.understand(raw)
    table=model._attention_words.table
    assert table.done.any()
    last=(table.done.long()*torch.arange(1,table.done.shape[1]+1)).argmax(-1)
    selected=table.intervals[torch.arange(len(last)),last]
    torch.testing.assert_close(model.conceptualSpace._passback_scope_where,
                               selected.to(raw.dtype)/raw.shape[-1])
    assert model.conceptualSpace._passback_scope_space.tolist()==[0]
