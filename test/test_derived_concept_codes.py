"""§13.4: occurrence conduction, native ownership and checkpoint contracts.

The outgoing lookup/blend tests are retained in review13-settled-before.
The settled native-subspace arithmetic is in test_review13_subspaces.py.
"""
from types import SimpleNamespace
import torch
from torch import nn
from test_review13_subspaces import fixture as native_fixture, word


def fixture():
    cs, cb, _, alloc = native_fixture(meaning_pairs=2, batch_size=2)
    return cs, cb, alloc


def test_rows_conduct_to_other_words_without_crossing_batch_streams():
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    cs, cb, alloc = fixture()
    a, b, c = [word(cs, [(p, 1.)]) for p in (4, 7, 8)]
    store = LTM0(3, capacity=8)
    cs.symbolSpace = SimpleNamespace(ltm_store=store)
    from ClauseRow import attach_clause_index
    attach_clause_index(SimpleNamespace(), store, cs)
    root = alloc.parts.add(cs.concept_id_at_row(a), cs.concept_id_at_row(b))
    row = store.append_meaning(ConceptualMeaning(torch.eye(3), torch.tensor([True,False,False]),
         sentence_kind='idea', role_refs=(('sym', root), None, None)), evidence=(1.,0.))
    cb.mereology.placement[[a,b,c]] = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    cb.mereology.placement_witnesses[[a,b,c]] = 1
    # Recorded relations do not conduct; nearby placements do.
    cs.prime_seen(torch.tensor([[a],[c]]))
    surface = cs.prime_seen(torch.tensor([[a],[c]]))
    assert surface[0,b] > 1 and surface[1,b] == 1
    assert not hasattr(cs, '_last_occurrence_priming')
    assert len(store) == 1


def test_frozen_lookup_does_not_admit_symbol_concept_pairs():
    cs, cb, alloc = fixture()
    row = word(cs, [(4, 1.)])
    before = alloc.next_id, tuple(alloc.layer()._tensor_row_keys)
    cs._online_learning_frozen = True
    cb.lookup_rows(torch.tensor([row, 20]))
    assert (alloc.next_id, tuple(alloc.layer()._tensor_row_keys)) == before


def test_actual_serial_leaf_pairs_forms_and_meanings_and_perception_owns_forms():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    model.train()
    model.forward(model.inputSpace.prepInput(['hello world', 'hello there']))
    owner = model._concept_owner()
    cb = owner.similarity_codebook
    assert not isinstance(cb.W, nn.Parameter)
    rows = model.inputSpace._ar_grammar_object_rows
    atoms = model.inputSpace._ar_grammar_object_atoms
    store = owner._closed_clause_store()
    covers = {owner._csw_row_of(store.raw_meaning_of(i).role_refs[0][1])
              for i in range(len(store)) if store.row(i)['kind'] == 'cover'}
    assert len(covers) == 2
    assert set(owner._concept_allocator.layer()._tensor_row_keys) == set(rows[rows>=0].tolist()) | covers
    torch.testing.assert_close(atoms[rows>=0][..., :96], cb.lookup_rows(rows[rows>=0])[..., :96])
    assert cb.mereology.placement_witnesses[rows[rows>=0]].gt(0).all()
    assert not atoms[rows>=0][..., 104:].any()
    opt = model.getOptimizer(lr=.01)
    groups = model.objective_parameter_groups(opt)
    assert any(p is model.perceptualSpace.subspace.what.W for p in groups['reconstruction'])
    assert all(p is not model.perceptualSpace.subspace.what.W for p in groups['output'])


def test_readback_audit_distinguishes_code_priming_and_ties():
    from SentenceUnderstanding import PrimedSymbols, readback_decisions
    rows = torch.tensor([[0,1],[0,1],[0,1]])
    codes = torch.tensor([[[1.,0.],[0.,1.]], [[1.,0.],[1.,0.]], [[1.,0.],[1.,0.]]])
    bank = PrimedSymbols(rows, codes, torch.tensor([[1.,1.],[1.,2.],[1.,1.]]),
        torch.ones_like(rows, dtype=torch.bool), rows[:,:,None], torch.ones(3,2,1,dtype=torch.bool))
    decisions = readback_decisions(torch.tensor([[[1.,0.]]]*3), torch.ones(3,dtype=torch.long), bank,
                                  torch.ones(3,dtype=torch.bool))
    assert [row['decided_by'] for row in decisions] == ['code', 'priming', 'tie']
    assert [row['changed_winner'] for row in decisions] == [False, True, False]


def test_percept_lookup_checkpoint_has_no_independent_word_state():
    cs, cb, _ = fixture()
    row = word(cs, [(4, .75), (7, -.25)])
    state = cb.state_dict()
    other, other_cb, _ = fixture()
    other_row = word(other, [(4, .75), (7, -.25)])
    other_cb.load_state_dict(state)
    torch.testing.assert_close(other_cb.lookup_rows(other_row), cb.lookup_rows(row))
    assert all('W' != name for name, _ in other_cb.named_parameters())
