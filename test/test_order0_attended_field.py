"""11b corrections: recurrent parts and a transient attended concept field."""
import torch

import Spaces
from Layers import RadixLayer
from test_cs_sparse_weights import _cs


def test_first_witness_keeps_existing_parts(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    ps = model.perceptualSpace
    parts = ps.percept_store.spell_out(b'abba')
    size = ps.percept_store._size
    assert ps.fuse_parts(parts) == parts
    assert ps.percept_store.get_id(b'abba') is None
    assert ps.percept_store._size == size


def test_containment_reads_canonical_ids_without_expanding_bytes():
    store = RadixLayer(4, initial_cap=16)
    a, b, ab = [store.insert(s) for s in (b'a', b'b', b'ab')]
    def forbidden(_):
        raise AssertionError('the conceptual read must not expand percept bytes')
    store.bytes_for = forbidden
    ids = torch.tensor([[a, b], [ab, -1]])
    parts = torch.tensor([[[0, 1], [1, 2]], [[0, 2], [0, 0]]])
    extent = torch.tensor([[[0, 2]]]).expand(2, -1, -1)
    pair = Spaces.PartSpace.part_memberships(
        store, torch.tensor([a, ab]), ids, parts, extent, 2)
    assert pair[:, :, 0, 0].tolist() == [[True, False], [False, True]]


def test_inventory_outlives_the_eight_attended_rows():
    cs = _cs(nS=64, order=3)
    cs.outputShape[0] = 8
    assert cs._order_caps()[0] == 8
    definitions = []
    for n in range(12):
        cid = cs.new_concept()
        row = cs._csw_concept_row(0, cid)
        assert row is not None, 'a field cap must not cap persistent definitions'
        cs.add_concept_feature(row, 'ps', 100 + n, 1.)
        definitions.append((cid, row))
    extent = torch.tensor([[[0, 1]]])
    for n in (11, 0, 9):
        native = (torch.tensor([[100 + n]]), extent, None,
                  torch.tensor([[65]]), extent)
        read = cs.cs_read_memberships(native, extent)
        assert read.shape[0] == 8
        ids = cs._cs_field_concept_ids
        slot = (ids == definitions[n][0]).nonzero().flatten()
        assert len(slot) == 1
        assert read[slot[0], 0, 0, 0] == 1
        assert int(read[..., 0].count_nonzero()) == 1
    assert all(cs._csw_row_of(cid) == row for cid, row in definitions)


def test_formation_stops_aggregation_until_the_next_turn():
    store = RadixLayer(4, initial_cap=16, promotion_threshold=2)
    store.spell_out(b'abc')
    store.begin_turn()
    assert store.observe_chunk(b'ab') is None
    ab = store.observe_chunk(b'ab')
    assert ab is not None
    assert store.observe_chunk(b'abc') is None
    assert store.observe_chunk(b'abc') is None
    assert store.get_id(b'abc') is None
    store.begin_turn()
    abc = store.observe_chunk(b'abc')
    assert abc is not None
    assert store.part_groups[abc] == (ab, store.get_id(b'c'))


def test_bound_snapshot_and_checkpoint_keep_concept_ids():
    from types import SimpleNamespace
    from test_structural_checkpoint import _model_with
    cs = _cs(nS=64, order=3)
    cs.outputShape[0] = 8
    definitions = []
    for n in range(12):
        cid = cs.new_concept()
        row = cs._csw_concept_row(0, cid)
        cs.add_concept_feature(row, 'ps', 100 + n, 1.)
        definitions.append((cid, row))
    extent = torch.tensor([[[0, 1]]])
    def read(n):
        return cs.cs_read_memberships((torch.tensor([[100 + n]]), extent,
            None, torch.tensor([[65]]), extent), extent)
    _, evidence = cs.cs_forward_content(read(11), cs.similarity_codebook.getW())
    ids, addresses = cs._cs_field_concept_ids.clone(), cs._cs_field_rows.clone()
    codes = cs._field_codes(cs.similarity_codebook.getW()).detach()
    for name, value in (('activations', evidence), ('ids', ids),
                        ('inventory_rows', addresses), ('codes', codes)):
        object.__setattr__(cs.subspace, '_concept_' + name, value)
    object.__setattr__(cs.subspace, '_concept_code_owner', 0)
    object.__setattr__(cs.subspace, '_thought_occurrence', None)
    read(0)
    assert ids[0] == definitions[11][0] != cs._cs_field_concept_ids[0]
    from AccessibleMind import apply_thought_effect
    from Queries import ThoughtResult
    from QueryWork import QueryWorkBudget
    from types import MappingProxyType
    from test_accessible_mind import _meaning
    result = ThoughtResult('what', 'conceptual-subgoal', 'set', 'retrieval', _meaning(),
        MappingProxyType({'frames': ({'reference': ('sym', definitions[11][0])},)}))
    apply_thought_effect(SimpleNamespace(conceptualSpace=cs, conceptualSpaces=[cs]),
                         result, row=0, work=QueryWorkBudget(32))
    assert cs.subspace._concept_activations[0, 0, -1, 0] == 1
    torch.testing.assert_close(cs.subspace._concept_ids, ids)
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs(nS=64, order=3)
    restored.outputShape[0] = 8
    restored.similarity_codebook.W.data.copy_(cs.similarity_codebook.W)
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    torch.testing.assert_close(restored.subspace._concept_ids, ids)
    torch.testing.assert_close(restored.subspace._concept_codes, codes)
    assert restored._csw_row_of(definitions[11][0]) == definitions[11][1]


def test_more_than_eight_words_reuse_the_attended_field(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, inventory=128)
    cs = model.conceptualSpaces[0]
    words = ('ant', 'bat', 'cat', 'dog', 'elk', 'fox', 'gnu', 'hare', 'ibex', 'jay', 'koala', 'lynx')
    for word in words:
        raw = torch.zeros(1, 1, 8, dtype=torch.long)
        raw[0, 0, :len(word)] = torch.tensor(list(word.encode()))
        model.forward(raw)
        cs.Reset(hard=True)
        model.End()
    for word in reversed(words):
        raw = torch.zeros(1, 1, 8, dtype=torch.long)
        raw[0, 0, :len(word)] = torch.tensor(list(word.encode()))
        model.forward(raw)
        cid = cs._word_obj_meta[word][0]
        carrier = model._combine_last_cs_sub
        slot = (carrier._concept_ids == cid).nonzero().flatten()
        assert len(slot) == 1, word
        assert carrier._concept_activations[slot[0], 0, :, 0].max() > 0, word
        assert cs._cs_last_a0.shape[0] == 8
        leg = model.symbolSpace.forward_concept_to_symbol(carrier)
        assert leg._symbol_indices[slot[0]].tolist() == [2 * cid, 2 * cid + 1]
        model.End()
    batch = torch.zeros(len(words), 1, 8, dtype=torch.long)
    for b, word in enumerate(words):
        batch[b, 0, :len(word)] = torch.tensor(list(word.encode()))
    model.forward(batch)
    carrier = model._combine_last_cs_sub
    leg = model.symbolSpace.forward_concept_to_symbol(carrier)
    for b, word in enumerate(words):
        cid = cs._word_obj_meta[word][0]
        slot = (carrier._concept_ids[:, b] == cid).nonzero().flatten()
        assert len(slot) == 1, word
        assert carrier._concept_activations[slot[0], b, :, 0].max() > 0, word
        assert leg._symbol_indices[slot[0], b].tolist() == [2 * cid, 2 * cid + 1]
    model.End()


def test_batch_members_bind_independently_within_eight_rows():
    cs = _cs(nS=64, order=3)
    cs.outputShape[0] = 8
    cids = []
    for n in range(12):
        cid = cs.new_concept()
        row = cs._csw_concept_row(0, cid)
        cs.add_concept_feature(row, 'ps', 100 + n, 1.)
        cids.append(cid)
    spans = torch.tensor([[[0, 1]]]).expand(12, -1, -1)
    a0 = cs.cs_read_memberships((torch.arange(100, 112)[:, None], spans,
        None, torch.full((12, 1), 65), spans), spans)
    assert cs._cs_field_concept_ids[0].tolist() == cids
    assert a0[0, :, 0, 0].tolist() == [1.] * 12
    content, field = cs.cs_forward_content(a0, cs.similarity_codebook.getW())
    assert content.shape[:2] == (12, 2 * sum(cs._order_caps()))
    assert field.shape[:2] == (sum(cs._order_caps()), 12)


def test_unattended_concept_is_unknown_and_does_not_veto_observed_evidence():
    cs = _cs(nS=16, order=1)
    a, b, conjunction = [cs.new_concept() for _ in range(3)]
    ar, br = [cs._csw_concept_row(0, cid) for cid in (a, b)]
    row = cs._csw_concept_row(0, conjunction)
    cs.add_concept_feature(ar, 'ps', 10, 1.)
    cs.add_concept_feature(br, 'ps', 11, 1.)
    cs.add_concept_edge(row, ar, 1., conjunctive=True)
    cs.add_concept_edge(row, br, 1., conjunctive=True)
    cs.conceptual_pi = True
    extent = torch.tensor([[[0, 1]]])
    a0 = cs.cs_read_memberships((torch.tensor([[10]]), extent, None,
        torch.tensor([[65]]), extent), extent)
    _, field = cs.cs_forward_content(a0, cs.similarity_codebook.getW())
    assert a0[0, 0, 0, 0] == 1
    assert b not in cs._cs_field_concept_ids
    slot = (cs._cs_field_concept_ids == conjunction).nonzero().flatten().item()
    torch.testing.assert_close(field[slot, 0, 0], torch.tensor([1., 0.]))


def test_location_writer_keeps_an_ordered_group_before_recurrence(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    ps, cs = model.perceptualSpace, model.conceptualSpaces[0]
    parts = ps.percept_store.spell_out(b'abba')
    cs._populate_cs_symbols(torch.tensor([parts]), torch.arange(4)[None],
                            torch.tensor([[[0, 4]]]))
    store = Spaces._concept_alloc_of(cs).layer(0)
    assert list(store.feature_groups.values()) == [tuple(parts)]
    assert ps.percept_store.get_id(b'abba') is None


def test_definition_group_tracks_new_overlapping_canonical_parts():
    store = RadixLayer(4, initial_cap=32, promotion_threshold=2)
    a, b, c = store.spell_out(b"abc")
    store.begin_turn()
    store.observe_chunk(b"bc")
    bc = store.observe_chunk(b"bc")
    old_literal = store.canonical_parts([a, b, c])
    assert old_literal == [a, bc]
    store.begin_turn()
    store.observe_chunk(b"ab")
    ab = store.observe_chunk(b"ab")
    assert store.spell_out(b"abc") == [ab, c]
    assert store.canonical_parts(old_literal) == [ab, c]
    restored = RadixLayer(4, initial_cap=32, promotion_threshold=2)
    restored.load_vocab_extras(store.vocab_extras())
    def forbidden(_):
        raise AssertionError('definition migration must use id ancestry')
    restored.bytes_for = forbidden
    assert restored.canonical_parts(old_literal) == [ab, c]
