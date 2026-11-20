"""Order-zero sigma over pi uses native percepts as its literals."""
from basic_fixtures import feature as basic_feature, order_zero_expected, published_presence
import pytest
import torch
from types import SimpleNamespace

import Spaces
from Layers import RadixLayer
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs
from nomination_fixtures import recurrent_interpretation


@pytest.mark.parametrize('conceptual_pi', [False, True])
def test_order0_unions_alternative_percept_conjunctions(conceptual_pi):
    cs = _cs()
    primitive = PrimitiveProperties(4)
    with torch.no_grad():
        primitive.members[:, 65] = torch.tensor([.25, .36, .81, .49])
    basic_feature(cs, 0, 'ws', 0, 1., parts=(65,))
    basic_feature(cs, 0, 'ws', 1, .5, parts=(65,))
    basic_feature(cs, 1, 'ws', 2, .5, parts=(65,))
    basic_feature(cs, 1, 'ws', 3, 1., parts=(65,))
    basic_feature(cs, 0, 'ws', 2, 1., negated=True, parts=(65,))
    basic_feature(cs, 0, 'ws', 3, .5, negated=True, parts=(65,))
    basic_feature(cs, 1, 'ws', 0, .5, negated=True, parts=(65,))
    basic_feature(cs, 1, 'ws', 1, 1., negated=True, parts=(65,))
    cs.add_concept_edge(0, 1, 1.)
    raw = torch.tensor([[65]])
    spans = torch.tensor([[[0, 1]]])
    result = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    basic, field = cs.basic_percepts, cs._percept_field
    rows = [basic.index[((65,), (prop,))] for prop in range(4)]
    members = primitive.members[:, 65].detach()
    # Each property keeps both lanes at the basic percept that carries it:
    # the row is its part's presence (1) times its property on each lane.
    slots = [field.keys.index(('bp', r)) for r in rows]
    torch.testing.assert_close(field.values[slots, 0, 0], torch.stack((members, 1 - members), -1))
    # Each concept has an entry on each member's row at one; a negated
    # literal is a Not entry (rule 62).
    def entries(signs):
        return sorted((r, 1., sign < 0) for r, sign in zip(rows, signs))
    sigma = cs.order_zero
    assert sorted(sigma.entries(0)) == entries((1, 1, -1, -1))
    assert sorted(sigma.entries(1)) == entries((-1, -1, 1, 1))
    # Each concept, the alternative an ordinary one on the same inventory,
    # is one signed number read by its own column, for less against.
    presence = published_presence(cs)
    for concept in (0, 1):
        _, lanes = order_zero_expected(cs, concept, presence)
        torch.testing.assert_close(result[concept, 0, 0], lanes.to(result), atol=1e-5, rtol=1e-5)


def test_part_literals_keep_order_multiplicity_and_location_in_canonical_ids():
    store = RadixLayer(4, initial_cap=16)
    a, b, x = [store.insert(s) for s in (b'a', b'b', b'x')]
    ab, aa, ba = [store.insert(s) for s in (b'ab', b'aa', b'ba')]
    # A literal matches the canonical retained ids, not another tiling of its bytes.
    ids = torch.tensor([[a, b, x], [ab, x, -1], [b, a, x], [a, a, x]])
    parts = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                          [[0, 2], [2, 3], [0, 0]],
                          [[0, 1], [1, 2], [2, 3]],
                          [[0, 1], [1, 2], [2, 3]]])
    spans = torch.tensor([[[0, 3], [1, 3], [0, 4]]]).expand(4, -1, -1)
    pair = Spaces.PartSpace.part_memberships(store, [(a, b), (a, a), (b, a), ab],
                                            ids, parts, spans, 4)
    torch.testing.assert_close(pair[:, :, 0, 0], torch.tensor(
        [[True, False, False, False], [False, False, False, True],
         [False, False, True, False], [False, True, False, False]]))
    assert not pair[:, :, 1, 0].any()  # no start outside the candidate span
    assert not pair[:, :, 2, 1].any()  # missing observation cannot deny a part
    assert pair[:, :, 0, 1].equal(~pair[:, :, 0, 0])


def test_witnesses_write_alternatives_without_conjoining_their_literals():
    cs = _cs()
    word, A = recurrent_interpretation(cs, [7], [1], key='word')
    store = Spaces._concept_alloc_of(cs).layer()
    row = cs._concept_row_of(A)
    recurrent_interpretation(cs, [8], [2], key='word')
    basic = cs.basic_percepts
    first, second = basic.index[((7,), (1,))], basic.index[((8,), (2,))]
    # The second witness joins the concept as an alternative, an entry on
    # its row: no
    # concept is made for it, no edge, and no conjunction of the two.
    assert cs.concept_weights(row) == []
    record = Spaces._concept_alloc_of(cs).order_zero_members
    assert {(p, n) for r, p, n in record if r == row} == {(first, False), (second, False)}
    assert not basic.conjunctions
    sigma = cs.order_zero
    assert sorted(sigma.entries(row)) == [(first, 1., False), (second, 1., False)]
    before = (store.nnz, len(record), Spaces._concept_alloc_of(cs).next_id)
    weights = sigma.value.detach().clone()
    recurrent_interpretation(cs, [8], [2], key='word')
    assert before == (store.nnz, len(record), Spaces._concept_alloc_of(cs).next_id)
    assert torch.equal(sigma.value.detach(), weights)
    from test_structural_checkpoint import _model_with
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs()
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    assert dict(Spaces._concept_alloc_of(restored).order_zero_members) == dict(record)
    assert restored.concept_weights(row) == cs.concept_weights(row)


def test_word_group_becomes_a_recurrent_row_without_changing_its_read(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, pool=8)
    ps, cs, ws = model.perceptualSpace, model.conceptualSpaces[0], model.wholeSpace
    letters = ps.percept_store.spell_out(b'love')
    whole = ws.property_rows_for_bytes(b'love')
    assert whole
    word, A = recurrent_interpretation(cs, letters, whole, key='love')
    identity = ps.percept_store.identity
    atom_ids = tuple(identity.admit(b'love'))
    assert set(cs.concept_parts(word)) == set(atom_ids)
    fused = identity.word_rows[b'love']
    store = Spaces._concept_alloc_of(cs).layer()
    row = cs._concept_row_of(A)
    bp = cs.basic_percepts.words[word]
    assert cs.basic_percepts.occurrences[bp][0] == atom_ids
    raw = torch.tensor([[108, 111, 118, 101]])
    parts = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]])
    extent = torch.tensor([[[0, 4]]])
    result = cs.cs_read_memberships((torch.tensor([letters]), parts,
        ws.subspace.what.primitive_properties, raw, extent), extent)
    # The word's concept reads its own row, at tanh(1) and as its column gives it.
    y, lanes = order_zero_expected(cs, row, published_presence(cs))
    assert y > .76
    torch.testing.assert_close(result[row, 0, 0], lanes.to(result), atol=1e-5, rtol=1e-5)
    after = cs.cs_read_memberships((torch.tensor([[fused]]), extent,
        ws.subspace.what.primitive_properties, raw, extent), extent)
    torch.testing.assert_close(after, result)
    record = Spaces._concept_alloc_of(cs).order_zero_members
    count = (store.nnz, len(record), Spaces._concept_alloc_of(cs).next_id)
    recurrent_interpretation(cs, [fused], whole, key='love')
    assert count == (store.nnz, len(record), Spaces._concept_alloc_of(cs).next_id)
    cs.refine_over_collected()
    assert A not in Spaces._concept_alloc_of(cs).retired


def test_alternatives_do_not_trigger_conjunctive_overcollection():
    cs = _cs()
    for n in range(6):
        word, A = recurrent_interpretation(cs, [20 + n], [10 + n], key='word')
    store = Spaces._concept_alloc_of(cs).layer()
    before = dict(Spaces._concept_alloc_of(cs).order_zero_members)
    cs.refine_over_collected()
    assert A not in Spaces._concept_alloc_of(cs).retired
    assert dict(Spaces._concept_alloc_of(cs).order_zero_members) == before
