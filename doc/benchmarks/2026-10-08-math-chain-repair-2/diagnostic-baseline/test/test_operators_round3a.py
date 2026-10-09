"""Identity by construction, dense binding only, and trust as an evidence pair."""
import copy
import pytest
import torch


def identity_bank(capacity=512):
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    store = RadixLayer(96, initial_cap=capacity)
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
    for b in [b'and', b'ant']:
        assert not set(bank.words[b'an']) <= set(bank.words[b])
    for n in range(1, 33):
        bank.admit(b'a'*n)
        assert bank.form(b'a'*n)[64:].tolist() == [1.]*n + [0.]*(32-n)
    assert len({bank.key(bank.form(b'a'*n)) for n in range(1, 33)}) == 32
    assert bank.audit()['containment_violations'] == 0


def test_mint_rekeys_existing_word_and_checkpoints_without_global_rng():
    bank = identity_bank()
    state = torch.get_rng_state().clone()
    bank.admit(b'calaba')
    old_row = bank.word_rows[b'calaba']
    bank.admit(b'cabala')
    assert torch.equal(state, torch.get_rng_state())
    assert bank.mints and bank.mints[0]['atoms'] == ['cal@1', 'cab@1']
    assert bank.mints[0]['bit_counts'][0] >= 3
    assert bank.word_rows[b'calaba'] == old_row
    assert not torch.equal(bank.form(b'calaba'), bank.form(b'cabala'))
    for word in [b'calaba', b'cabala']:
        assert bank.read(bank.form(word)) == word
    restored = identity_bank()
    restored.store.load_state_dict(bank.store.state_dict())
    restored.store.load_vocab_extras(bank.store.vocab_extras())
    assert restored.audit() == bank.audit()
    assert torch.equal(restored.projection, bank.projection)


def test_fixed_atoms_survive_optimizer_and_projection_is_only_binding_view():
    bank = identity_bank()
    ids = bank.admit(b'hello')
    before = bank.form(b'hello')
    weights = bank.store._basis.W
    optimizer = torch.optim.AdamW([weights], lr=.1, weight_decay=.1)
    bank.store._basis.lookup_rows(torch.tensor(ids)).sum().backward()
    optimizer.step()
    torch.testing.assert_close(bank.store._basis.lookup_rows(torch.tensor(ids)).amax(0), before, rtol=0, atol=0)
    other = identity_bank()
    assert torch.equal(bank.projection, other.projection)
    assert not list(bank.parameters())
    dense = bank.binding(before)
    assert dense[:64].norm() > 0 and dense[64:].count_nonzero() == 0
    assert (dense[:64] < 0).any()
    assert bank.read(before) == b'hello'


def test_unidentified_pair_assembly_and_index_resolves_ambiguity():
    from WordIdentity import base_atoms, assemble_pairs
    for word in [b'circus', b'banana']:
        pairs = [v for k,v in base_atoms(word) if k == 'pair']
        assert assemble_pairs(pairs, len(word)) == word
    bank = identity_bank()
    bank.admit(b'calaba'); bank.admit(b'cabala')
    pairs = [v for k,v in base_atoms(b'calaba') if k == 'pair']
    with pytest.raises(ValueError, match='minted index'):
        assemble_pairs(pairs, 6)
    assert assemble_pairs(pairs, 6, index=bank, form=bank.form(b'cabala')) == b'cabala'
    assert bank.read_parts(bank.admit(b'cabala')) == b'cabala'
    unseen = identity_bank()
    rows = [unseen._atom(atom) for atom in base_atoms(b'circus')]
    assert not unseen.words and unseen.read_parts(rows) == b'circus'


@pytest.mark.parametrize('quadruple', [False, True])
def test_masked_mint_extends_bits_and_has_declared_quadruple_fallback(monkeypatch, quadruple):
    import WordIdentity
    bank = identity_bank()
    bank.admit(b'calaba')
    form = bank.form(b'calaba')[:64]
    used = form.nonzero().flatten().tolist()
    missing = (~form.bool()).nonzero().flatten().tolist()
    stream = WordIdentity.bit_stream
    def controlled(atom, dimension):
        if b'@' not in atom:
            return stream(atom, dimension)
        value = atom.decode().split('@')[0]
        if quadruple and len(value) == 3:
            return tuple(range(dimension))  # both additions remain identical
        selected = missing[0] if value.startswith('cal') else missing[1]
        prefix = used[:3] + [selected]
        return tuple(prefix+[i for i in range(dimension) if i not in prefix])
    monkeypatch.setattr(WordIdentity, 'bit_stream', controlled)
    bank.admit(b'cabala')
    mint, = bank.mints
    assert mint['size'] == (4 if quadruple else 3)
    assert mint['bit_counts'] == [4,4]
    assert not torch.equal(bank.form(b'calaba'), bank.form(b'cabala'))


@pytest.mark.parametrize('complete_first', [False, True])
def test_live_mint_rekeys_pending_and_completed_definitions(tmp_path, complete_first):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, 8)
    cs = model._concept_owner()
    bank = model.perceptualSpace.percept_store.identity
    try:
        first = cs.interpret.lookup_word([], [], form='calaba')
        if complete_first:
            cs.interpret.forward(first)
        second = cs.interpret.lookup_word([], [], form='cabala')
        cs.interpret.forward(first)
        cs.interpret.forward(second)
        assert first != second
        for raw, word in [(b'calaba',first),(b'cabala',second)]:
            assert cs.definitions.word(identity=bank.key(bank.form(raw)).hex()) == word
            assert set(cs.definitions.description(word)['parts'][0]) == set(bank.admit(raw))
            assert bank.read(bank.form(raw)) == raw
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_numeric_mm_keeps_its_original_geometry():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/MM_xor.xml')
    try:
        assert model.perceptualSpace.subspace.muxedSize == 14
        assert getattr(model.perceptualSpace.percept_store, 'identity', None) is None
    finally:
        model.End(); model.symbolSpace.soft_reset()


@pytest.mark.parametrize('trust,polarity,expected', [(.7,True,(.7,0.)),(-.4,True,(0.,.4)),
    (.7,False,(0.,.7)),(-.4,False,(.4,0.)),(0.,True,(0.,0.))])
def test_trust_to_pair_and_negation(trust, polarity, expected):
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    store = TernaryTruthStore(4, capacity=4)
    meaning = ConceptualMeaning(torch.ones(3,4), torch.tensor([True,False,False]),
                                mode='assertive', polarity=polarity)
    row = store.append_meaning(meaning, trust=trust)
    torch.testing.assert_close(torch.stack((store.c_plus[row],store.c_minus[row])), torch.tensor(expected))
    assert 'trust' not in store.state_dict()
    assert float(store.trust[row]) == pytest.approx(expected[0]-expected[1])


def test_live_grammar_uses_sparse_index_dense_nonzero_roots_and_exact_rung0():
    from test_mm_xor import _fresh_model
    from Language import ConjunctionLayer, DisjunctionLayer
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    try:
        optimizer = model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer, batchSize=4, split='train')
        identity = model.perceptualSpace.percept_store.identity
        owner = model._concept_owner()
        words = [b'hello', b'world', b'loving', b'there']
        dense = []
        for word in words:
            key = identity.key(identity.form(word)).hex()
            native = owner.definitions.word(identity=key)
            assert native is not None
            obj = owner.definitions.deref(native)
            row = owner._csw_row_of(obj)
            sparse = owner.similarity_codebook.lookup_rows(row)
            torch.testing.assert_close(sparse[:96],identity.form(word),rtol=0,atol=0)
            dense.append(owner.interpret.binding_atoms(sparse))
        for layer in [ConjunctionLayer(), DisjunctionLayer()]:
            roots = torch.stack([layer.compose(dense[a],dense[b]) for a,b in [(0,1),(0,3),(2,1),(2,3)]])
            assert (roots.norm(dim=-1)>0).all()
        record = model._last_sentence_understanding
        assert record is not None
        assert record.primed.forms is not None
        assert model._last_rung0_identity_audit['errors'] == 0
        assert identity.audit()['reconstruction_errors'] == 0
        assert model._last_sentence_credit['components'][:,:,0].isfinite().all()
    finally:
        model.End(); model.symbolSpace.soft_reset()
