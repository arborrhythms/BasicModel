"""Order-zero sigma over pi uses native percepts as its literals."""
import pytest
import torch
from types import SimpleNamespace

import Spaces
from Layers import RadixLayer
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs


@pytest.mark.parametrize('conceptual_pi', [False, True])
def test_order0_unions_alternative_percept_conjunctions(conceptual_pi):
    cs = _cs()
    cs.conceptual_pi = conceptual_pi
    primitive = PrimitiveProperties(4)
    with torch.no_grad():
        primitive.members[:, 65] = torch.tensor([.25, .36, .81, .49])
    cs.add_concept_feature(0, 'ws', 0, 1.)
    cs.add_concept_feature(0, 'ws', 1, .5)
    cs.add_concept_feature(1, 'ws', 2, .5)
    cs.add_concept_feature(1, 'ws', 3, 1.)
    cs.add_concept_edge(0, 1, 1.)
    raw = torch.tensor([[65]])
    spans = torch.tensor([[[0, 1]]])
    result = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    pi = torch.tensor([.25 * .36 ** .5, .81 ** .5 * .49])
    complement = torch.tensor([.75 * .64 ** .5, .19 ** .5 * .51])
    expected = torch.stack((1 - (1 - pi).prod(), complement.prod()))
    torch.testing.assert_close(result[0, 0, 0], expected)
    # The alternative is an ordinary concept on the same row inventory.
    torch.testing.assert_close(result[1, 0, 0], torch.stack((pi[1], complement[1])))


def test_parts_keep_order_multiplicity_and_location_across_radix_tilings():
    store = RadixLayer(4, initial_cap=16)
    a, b, x = [store.insert(s) for s in (b'a', b'b', b'x')]
    ab, aa, ba = [store.insert(s) for s in (b'ab', b'aa', b'ba')]
    # The same ordered part occurs in two tilings and inside a larger span.
    ids = torch.tensor([[a, b, x], [ab, x, -1], [b, a, x], [a, a, x]])
    parts = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                          [[0, 2], [2, 3], [0, 0]],
                          [[0, 1], [1, 2], [2, 3]],
                          [[0, 1], [1, 2], [2, 3]]])
    spans = torch.tensor([[[0, 3], [1, 3], [0, 4]]]).expand(4, -1, -1)
    pair = Spaces.PartSpace.part_memberships(store, torch.tensor([ab, aa, ba]),
                                            ids, parts, spans, 4)
    torch.testing.assert_close(pair[:, :, 0, 0], torch.tensor(
        [[True, True, False, False], [False, False, False, True],
         [False, False, True, False]]))
    assert not pair[:, :, 1, 0].any()  # no start outside the candidate span
    assert not pair[:, :, 2, 1].any()  # missing observation cannot deny a part
    assert pair[:, :, 0, 1].equal(~pair[:, :, 0, 0])


def test_witnesses_write_alternatives_without_conjoining_their_literals():
    cs = _cs()
    A, _, _ = cs.create_word_object_meta([7], [1], key='word')
    store = Spaces._concept_alloc_of(cs).layer()
    row = store.row_of(('snap', A))
    cs.create_word_object_meta([8], [2], key='word')
    [(alternative, weight)] = cs.concept_weights(row)
    assert weight == 1.
    assert {col for r, col in store.features._index if r == row} == {28, 6}
    assert {col for r, col in store.features._index if r == alternative} == {32, 10}
    before = (store.nnz, store.features.nnz, Spaces._concept_alloc_of(cs).next_id)
    cs.create_word_object_meta([8], [2], key='word')
    assert before == (store.nnz, store.features.nnz, Spaces._concept_alloc_of(cs).next_id)
    from test_structural_checkpoint import _model_with
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs()
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    other = Spaces._concept_alloc_of(restored).layer()
    assert other.features._index == store.features._index
    assert restored.concept_weights(row) == cs.concept_weights(row)
    torch.testing.assert_close(other.features.values, store.features.values)


def test_native_word_writer_keeps_letters_inside_one_fused_part(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, pool=8)
    ps, cs, ws = model.perceptualSpace, model.conceptualSpaces[0], model.wholeSpaces[0]
    letters = ps.percept_store.spell_out(b'love')
    whole = ws.property_rows_for_bytes(b'love')
    assert whole
    A, _, _ = cs.create_word_object_meta(letters, whole, key='love')
    [fused] = cs.concept_parts(A)
    assert ps.percept_store.bytes_for(fused) == b'love'
    store = Spaces._concept_alloc_of(cs).layer()
    row = store.row_of(('snap', A))
    assert [col // 4 for r, col in store.features._index
            if r == row and col % 4 == 0] == [fused]
    raw = torch.tensor([[108, 111, 118, 101]])
    parts = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]])
    extent = torch.tensor([[[0, 4]]])
    result = cs.cs_read_memberships((torch.tensor([letters]), parts,
        ws.subspace.what.primitive_properties, raw, extent), extent)
    assert result[row, 0, 0, 0] == 1
    assert ps.fuse_parts(letters) == [fused]


def test_alternatives_do_not_trigger_conjunctive_overcollection():
    cs = _cs()
    for n in range(6):
        A, _, _ = cs.create_word_object_meta([20 + n], [10 + n], key='word')
    store = Spaces._concept_alloc_of(cs).layer()
    before = dict(store.features._index)
    cs.refine_over_collected()
    assert A not in Spaces._concept_alloc_of(cs).retired
    assert store.features._index == before
