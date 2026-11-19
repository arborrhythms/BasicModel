"""Identity by construction, dense binding only, and trust as an evidence pair."""
from Memory import STM0
import copy
import pytest
import torch
from nomination_fixtures import recurrent_word


def identity_bank(capacity=512):
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    store = RadixLayer(64, initial_cap=capacity)
    store.identity = WordIdentity(store)
    return store.identity


def test_forms_anagrams_length_and_nontrivial_containment():
    bank = identity_bank()
    words = [b'circus', b'cursic', b'cirrus', b'bana', b'banana', b'cat',
             b'concat', b'aba', b'ababa', b'an', b'and', b'ant']
    for word in words:
        ids = bank.admit(word)
        rows = bank.store._basis.lookup_rows(torch.tensor(ids))
        torch.testing.assert_close(rows.amax(0), bank.form(word), rtol=0, atol=0)
        assert bank.read(bank.form(word)) == word
    assert len({bank.key(bank.form(w)) for w in words}) == len(words)
    for a, b in [(b'bana', b'banana'), (b'cat', b'concat'), (b'aba', b'ababa')]:
        assert set(bank.words[a]) < set(bank.words[b])
        assert (bank.form(a) <= bank.form(b)).all()
    # With no end marks, a form inside a form is inside its code.
    for b in [b'and', b'ant']:
        assert set(bank.words[b'an']) < set(bank.words[b])
        assert (bank.form(b'an') <= bank.form(b)).all()
    # The anagrams are told apart by an added part.
    assert set(bank.words[b'circus']) != set(bank.words[b'cursic'])
    for n in range(1, 33):
        bank.admit(b'a'*n)
        # No block of lengths: the code is the parts half alone (rule 50).
        assert bank.form(b'a'*n).shape[-1] == bank.pair_dim == 64
    assert len({bank.key(bank.form(b'a'*n)) for n in range(1, 33)}) == 32
    assert bank.audit()['containment_violations'] == 0


def test_mint_rekeys_existing_word_and_checkpoints_without_global_rng():
    bank = identity_bank()
    state = torch.get_rng_state().clone()
    bank.admit(b'calaba')
    old_row = bank.word_rows[b'calaba']
    bank.admit(b'cabala')
    assert torch.equal(state, torch.get_rng_state())
    # The two have the same parts and the same runs of two; the first run
    # of three that one holds and the other does not becomes a part.
    assert bank.added == {'bal': ('calaba', 'cabala')}
    assert ('part', 'bal') in bank.words[b'cabala'] and ('part', 'bal') not in bank.words[b'calaba']
    assert bank.word_rows[b'calaba'] == old_row
    assert not torch.equal(bank.form(b'calaba'), bank.form(b'cabala'))
    for word in [b'calaba', b'cabala']:
        assert bank.read(bank.form(word)) == word
    restored = identity_bank()
    restored.store.load_state_dict(bank.store.state_dict())
    restored.store.load_vocab_extras(bank.store.vocab_extras())
    assert restored.audit() == bank.audit()
    assert not hasattr(restored, 'projection') and not hasattr(bank, 'binding')


def test_fixed_atoms_survive_optimizer_and_composition_uses_the_native_cube():
    bank = identity_bank()
    ids = bank.admit(b'hello')
    before = bank.form(b'hello')
    weights = bank.store._basis.W
    optimizer = torch.optim.AdamW([weights], lr=.1, weight_decay=.1)
    bank.store._basis.lookup_rows(torch.tensor(ids)).sum().backward()
    optimizer.step()
    torch.testing.assert_close(bank.store._basis.lookup_rows(torch.tensor(ids)).amax(0), before, rtol=0, atol=0)
    other = identity_bank()
    assert not hasattr(bank, 'projection') and not hasattr(other, 'projection')
    assert not list(bank.parameters())
    assert bool(((before >= 0) & (before <= 1)).all())
    assert before[:64].norm() > 0 and before[64:].count_nonzero() == 0
    assert bank.read(before) == b'hello'


def test_spelling_reads_the_stored_table():
    from WordIdentity import base_atoms
    bank = identity_bank()
    bank.admit(b'calaba'); bank.admit(b'cabala')
    assert bank.spelling(bank.admit(b'cabala')) == b'cabala'
    assert bank.spelling(bank.admit(b'calaba')) == b'calaba'
    # A code that is no stored form has no spelling.
    unseen = identity_bank()
    rows = [unseen._atom(atom) for atom in base_atoms(b'circus')]
    assert not unseen.words and unseen.spelling(rows) is None


@pytest.mark.parametrize('separable', [True, False])
def test_masked_part_extends_bits_and_an_inseparable_row_is_shared(monkeypatch, separable):
    import WordIdentity
    bank = identity_bank()
    bank.admit(b'calaba')
    form = bank.form(b'calaba')[:64]
    used = form.nonzero().flatten().tolist()
    missing = (~form.bool()).nonzero().flatten().tolist()
    stream = WordIdentity.bit_stream
    def controlled(atom, dimension):
        if atom != b'bal':
            return stream(atom, dimension)
        # Its first three bits are calaba's: the rows stay equal at three.
        prefix = used[:3] + ([missing[0]] if separable else [])
        return tuple(prefix+[i for i in range(dimension) if i not in prefix and (separable or i in used)])[:dimension]
    monkeypatch.setattr(WordIdentity, 'bit_stream', controlled)
    bank.admit(b'cabala')
    assert list(bank.added) == ['bal']
    if separable:
        assert bank.bits[('part', 'bal')] == 4 and not bank.shared
        assert not torch.equal(bank.form(b'calaba'), bank.form(b'cabala'))
    else:
        # The parts tell the two apart; the row of fixed width is shared.
        assert bank.shared == {b'cabala': b'calaba'}
        assert torch.equal(bank.form(b'calaba'), bank.form(b'cabala'))
        assert bank.word_rows[b'calaba'] != bank.word_rows[b'cabala']
        assert set(bank.words[b'calaba']) != set(bank.words[b'cabala'])


@pytest.mark.parametrize('complete_first', [False, True])
def test_live_mint_rekeys_pending_and_completed_definitions(tmp_path, complete_first):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, 8)
    cs = model._concept_owner()
    bank = model.perceptualSpace.percept_store.identity
    try:
        first = recurrent_word(cs.interpret, [], [], form='calaba')
        from reading_fixtures import supply_object
        if complete_first:
            supply_object(cs, first)
        second = recurrent_word(cs.interpret, [], [], form='cabala')
        if not complete_first:
            supply_object(cs, first)
        supply_object(cs, second)
        assert first != second
        for raw, word in [(b'calaba',first),(b'cabala',second)]:
            assert cs.definitions.word(identity=bank.key(bank.form(raw)).hex()) == word
            assert set(cs.definitions.description(word)['parts'][0]) == set(bank.admit(raw))
            assert bank.read(bank.form(raw)) == raw
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_numeric_mm_keeps_its_original_geometry():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/MM_text.xml')
    try:
        assert model.perceptualSpace.subspace.muxedSize == 14
        assert getattr(model.perceptualSpace.percept_store, 'identity', None) is None
    finally:
        model.End(); model.symbolSpace.soft_reset()


@pytest.mark.parametrize('trust,polarity,expected', [(.7,True,(.7,0.)),(-.4,True,(0.,.4)),
    (.7,False,(.7,0.)),(-.4,False,(0.,.4)),(0.,True,(0.,0.))])
def test_trust_to_pair_and_negation(trust, polarity, expected):
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    store = STM0(4, capacity=4, relation_capacity=8192)
    meaning = ConceptualMeaning(torch.ones(3,4), torch.tensor([True,False,False]),
                                 polarity=polarity)
    row = store.append_meaning(meaning, trust=trust)
    torch.testing.assert_close(torch.tensor(store.evidence_of(row)), torch.tensor(expected))
    assert 'trust' in store.state_dict()
    assert float(store.trust[row]) == pytest.approx(expected[0]-expected[1])


def test_live_grammar_uses_sparse_index_dense_nonzero_roots_and_exact_rung0(monkeypatch):
    from test_mm_xor import _fresh_model
    from Language import ConjunctionLayer, DisjunctionLayer
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    try:
        # The forms are words, nominated by MTM before the epoch.
        from nomination_fixtures import nominated_forms
        nominated_forms(model, ['hello world', 'loving there'])
        optimizer = model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer, batchSize=4, split='train')
        identity = model.perceptualSpace.percept_store.identity
        owner = model._concept_owner()
        words = [b'hello', b'world', b'loving', b'there']
        dense = []
        for word in words:
            key = identity.key(identity.form(word)).hex()
            native = owner.interpret.word_for_form(word)
            assert native is not None
            obj = owner.interpret.forward(native)
            row = owner._csw_row_of(obj)
            sparse = owner.similarity_codebook.lookup_rows(row)
            torch.testing.assert_close(sparse[:identity.width],identity.form(word),rtol=0,atol=0)
            dense.append(sparse)
        for layer in [ConjunctionLayer(), DisjunctionLayer()]:
            roots = torch.stack([layer.compose(dense[a],dense[b]) for a,b in [(0,1),(0,3),(2,1),(2,3)]])
            assert (roots.norm(dim=-1)>0).all()
        record = model._last_sentence_understanding
        assert record is not None
        assert record.primed.forms is not None
        assert model._last_rung0_identity_audit['errors'] == 0
        assert identity.audit()['reconstruction_errors'] == 0
        assert model._last_sentence_credit['components'][:,:,0].isfinite().all()
        from Forgetting import model_boundary
        model_boundary(model, epoch_boundary=True)
        assert model._last_sentence_understanding is record  # Point 20 removes epoch presentation rollback.
    finally:
        model.End(); model.symbolSpace.soft_reset()
