"""§14 addendum: two spaces paired by symbols, with no rows for letters."""
import pytest
import torch
from test_review13_subspaces import fixture, word


def test_native_part_evidence_does_not_allocate_concepts():
    cs, cb, ps, alloc = fixture()
    before = alloc.next_id
    row = word(cs, [((4, 7), .5)])
    assert alloc.next_id == before + 1
    assert set(alloc.layer()._tensor_row_keys) == {row}
    torch.testing.assert_close(cb.lookup_rows(row)[:3], ps.W[[4, 7]].amax(0))


def test_identical_empty_meanings_do_not_require_identical_forms():
    cs, cb, ps, _ = fixture()
    a, b = word(cs, [(4, 1.)]), word(cs, [(7, 1.)])
    paired = cb.lookup_rows(torch.tensor([a, b]))
    assert not torch.equal(paired[0, :3], paired[1, :3])
    torch.testing.assert_close(paired[0, 3:], paired[1, 3:])
    assert not paired[:, 3:].any()


@pytest.mark.parametrize('config,capacity', [('XOR_grammar', 6), ('MM_grammar', 8)])
def test_small_inventory_pairs_words_without_allocating_letter_rows(config, capacity):
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model(f'data/{config}.xml')
    admitted = []
    stage = model._stage_reading_word_concepts
    def observe_admission():
        stage()
        owner = model._concept_owner()
        rows = model.inputSpace._ar_grammar_object_rows
        word_rows = set(rows[rows >= 0].tolist())
        assert len(word_rows) == 4
        assert set(owner._concept_allocator.layer()._tensor_row_keys) == word_rows
        admitted.append(word_rows)
    model._stage_reading_word_concepts = observe_admission
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(
                ['hello world', 'hello there', 'loving world', 'loving there']))
        owner = model._concept_owner()
        cb = owner.similarity_codebook
        rows = model.inputSpace._ar_grammar_object_rows
        word_rows = set(rows[rows >= 0].tolist())
        assert owner.nVectors == capacity and cb.W.shape[0] == capacity
        assert len(word_rows) == 4
        assert admitted and admitted[-1] == word_rows
        # A later grammatical part relation may symbolize an existing word
        # object at order one. Such a row is not a concept for a letter.
        for row in set(owner._concept_allocator.layer()._tensor_row_keys) - word_rows:
            cid = owner.concept_id_at_row(row)
            assert owner._concept_source_order(cid) > 0
            assert all(isinstance(part, tuple) and part[0] == 'sym'
                       for part in owner.concept_parts(cid))
        assert cb.mereology.context_width == 128
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
