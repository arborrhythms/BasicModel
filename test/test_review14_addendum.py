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


@pytest.mark.parametrize('config,capacity', [('XOR_grammar', 18), ('MM_grammar', 20)])
def test_small_inventory_pairs_words_without_allocating_letter_rows(config, capacity, monkeypatch):
    from reading_fixtures import keep_existing_references
    keep_existing_references(monkeypatch)
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model(f'data/{config}.xml')
    constructed = set(model._concept_owner()._concept_allocator.layer()._tensor_row_keys)
    texts = ['hello world', 'hello there', 'loving world', 'loving there']
    # MTM nominates the words on their concept_mint_count-th reading
    # (sigma-pi §21, rule 20): the readings before it admit no word.
    from nomination_fixtures import readings_before_nomination
    readings_before_nomination(model, lambda: model.forward(model.inputSpace.prepInput(texts)))
    admitted = []
    stage = model._stage_reading_word_concepts
    def observe_admission():
        stage()
        owner = model._concept_owner()
        rows = model.inputSpace._ar_grammar_object_rows
        word_rows = set(rows[rows >= 0].tolist())
        assert len(word_rows) == 4
        assert set(owner._concept_allocator.layer()._tensor_row_keys) == word_rows | constructed
        admitted.append(word_rows)
    model._stage_reading_word_concepts = observe_admission
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(texts))
        owner = model._concept_owner()
        cb = owner.similarity_codebook
        rows = model.inputSpace._ar_grammar_object_rows
        word_rows = set(rows[rows >= 0].tolist())
        # The configured capacity is the room of order zero; each order
        # above has half the one below, one row at least; the declared
        # seats come on top: the grammar's arguments and operators and the
        # classes of words and objects above order zero (sigma-pi §21,
        # rules 20 and 21; §23.4, step 10), the frozen thought predicates
        # at order zero.
        alloc = owner._concept_allocator
        seats = tuple(owner._order_reservations)
        assert sum(seats[1:]) == (len(alloc.formal_arguments) + len(alloc.declared_classes)
                                  + len(alloc.operator_concepts))
        assert owner._order_caps()[0] == capacity + seats[0]
        taper = sum(max(1, capacity >> order) for order in range(1, max(owner._order_limit, len(seats) - 1) + 1))
        assert owner.nVectors == capacity + taper + sum(seats)
        assert cb.W.shape[0] == owner.nVectors
        assert len(word_rows) == 4
        assert admitted and admitted[-1] == word_rows
        # A kept relation may seat an argument at its parsed order, including
        # zero. Covers and these arguments are concepts, never letter words.
        for row in set(owner._concept_allocator.layer()._tensor_row_keys) - word_rows:
            cid = owner.concept_id_at_row(row)
            if owner.is_frozen(cid):
                # §27.2 seats the grammar's fixed thought identities during
                # construction; none is a letter admitted by this reading.
                assert row in constructed
                assert 'word' not in owner._concept_allocator.symbol_types.get(cid, ())
                continue
            assert 'concept' in owner._concept_allocator.symbol_types.get(cid, ())
            for tag, identity in owner.concept_parts(cid):
                assert tag in ('sym', 'occ')
                if tag == 'occ':
                    store = model.symbolSpace.ltm_store
                    index = store.index_of_row(identity)
                    assert index is not None
                    # A named row can now read a later definition. A retained
                    # collection keeps the order of its selected members
                    # (§18.6), independently of that containing row's order.
                    token = owner.mtm.token_of(store.occurrence_of(index))
                    order = (store.order_of_row(index) if token is None else
                             owner.mtm.entries[token].refinement['member_order'])
                    assert order + 1 == owner._concept_allocator.order_of(cid)
        assert cb.mereology.context_width == 128
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
