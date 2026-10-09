"""October 8 repair contracts; no arithmetic executor or seeded learner."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def test_interpret_calls_bind_and_bare_forward_choices_share_the_chooser():
    from Interpret import InterpretLayer
    from Language import MLPTransformChooser
    owner = SimpleNamespace(bind=lambda value, **kw: InterpretLayer.bind(None, value, **kw))
    value = torch.randn(8, 1, 4).tanh()
    arguments = dict(identity=torch.full((8, 1), 19, dtype=torch.long),
        scope=torch.zeros(8, 1, dtype=torch.long),
        candidate_ids=torch.full((8, 1, 1), 31, dtype=torch.long),
        candidate_values=torch.randn(8, 1, 1, 4),
        candidate_relations=torch.zeros(8, 1, 1, dtype=torch.bool),
        available=torch.ones(8, 1, 1, dtype=torch.bool), associations=torch.empty(0, 2, dtype=torch.long),
        mode=None, active=torch.ones(8, 1, dtype=torch.bool))
    options = InterpretLayer.forward(owner, value, binding=arguments)
    assert [int(item[1][0, 0]) for item in options] == [31, -1, 0, 19]
    assert [bool(item[3][0, 0]) for item in options] == [True, True, True, False]
    chooser = MLPTransformChooser(d_model=4, n_copy=1, n_op=1, binding_choices=True)
    context = torch.cat((torch.stack((value, -value), 2), value.new_zeros(8, 1, 2, 4)), -1)
    _, scores = chooser.score_unary(value, value[:, :, None].expand(-1, -1, 2, -1),
        None, None, op_indices=torch.zeros(2, dtype=torch.long), reference_ctx=context)
    probability = scores.softmax(-1)[..., 1]
    assert bool(((probability > .49) & (probability < .51)).all())
    loss = -probability.log().mean()
    loss.backward()
    assert chooser.reference_choice.weight.grad.norm() > 0
    # Same backward lookup, with no declared surface class of any kind.
    found = InterpretLayer.forward(owner, value, binding=dict(arguments,
        associations=torch.tensor([[19, 31]])))
    assert len(found) == len(options)
    assert all(torch.equal(a[3], b[3]) for a, b in zip(found, options))
    assert all(bool(item[4].all()) for item in found)
    assert not any(bool(item[4].any()) for item in options)
    definite = InterpretLayer.forward(owner, value, binding=dict(arguments, mode='bind', available=torch.zeros_like(arguments['available'])))
    assert [int(item[1][0,0]) for item in definite if bool(item[3][0,0])] == [0]
    minted = InterpretLayer.forward(owner, value, binding=dict(arguments, mode='mint'))
    assert [int(item[1][0,0]) for item in minted if bool(item[3][0,0])] == [-1]


def test_driver_batch_groups_resize_priming_before_reads_without_broadcast():
    from MathChainTraining import document_batches
    from math_chain_corpus import MathChainCorpus, flatten
    from test_priming_energy import _chain_cs
    cs, _, _ = _chain_cs()
    _, _, addresses = flatten(MathChainCorpus().presentation()['train'], supplied=False)
    sizes = [len(rows) for rows, end in document_batches(addresses, 8) if end]
    assert {2, 3, 8}.issubset(sizes)
    # A provisioning write must not leak row zero into the first batch.
    cs.prime_seen(torch.tensor([[3]]))
    previous = 1
    for batch in sizes:
        weights = cs.priming_weights(batch=batch)
        assert weights.shape == (batch, cs._priming_dim())
        if batch != previous:
            assert torch.equal(weights, torch.ones_like(weights))
        cs.prime_seen(torch.arange(batch)[:, None] + 8)
        weights = cs.priming_weights(batch=batch)
        for row in range(batch):
            assert weights[row, row + 8] > 1
            if batch != previous:
                assert all(weights[other, row + 8] == 1 for other in range(batch) if other != row)
        previous = batch
    cs._priming_boosts = torch.ones(5)
    with pytest.raises(ValueError, match='surface'):
        cs.priming_weights(batch=8)


def test_unfold_reads_the_readers_priming_row():
    from MemoryIndex import unfold_idea
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(),
        reverse_inverses=lambda ops: (), generate_policy_logits=lambda point: point.new_zeros(1, 1))
    basis = torch.eye(3)
    weights = torch.tensor([[2., 1., 1.], [1., 2., 1.]])
    for row in range(2):
        result = unfold_idea(language, basis, basis[row], 4, activation=lambda: weights, stream=row)
        assert result['codes'] == (row,) and result['complete']
        assert not unfold_idea(language, basis, basis[1-row], 4,
                               activation=lambda: weights, stream=row)['complete']
    with pytest.raises(ValueError, match='stream'):
        unfold_idea(language, basis, basis[0], 4, activation=lambda: weights, stream=2)


@pytest.mark.parametrize('forward_first', [True, False])
def test_pending_relation_fills_in_place_in_both_arrival_orders(forward_first):
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from test_clause_storage import clause_store, idea_clause
    store, _ = clause_store()
    meaning = with_slots(ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool)),
                         (('referent', 0),), pair=(1., 0.))
    data = dict(meaning.bindings)
    data['_forward_references'] = ((0, 17),)
    meaning = replace(meaning, bindings=data)
    occurrence = None
    if not forward_first:
        occurrence = store.write_clause(idea_clause(), document_key='arrival', sentence_index=1)
    refs = (-1 if occurrence is None else int(store.row_ids[occurrence]), 3, 2)
    if occurrence is not None:
        meaning = replace(meaning, role_refs=(store.occurrence_of(occurrence), ('sym', 3), ('sym', 2)))
    row = store.write_clause(Clause(meaning, relation='part', refs=refs),
                             document_key='pending', sentence_index=2)
    address, document, timestamp = store.occurrence_of(row), store.document_keys[row].clone(), store.timestamp[row].clone()
    if forward_first:
        assert store.row(row)['pending'] and int(store.refs[row, 0]) == -1
        assert store._forward_addresses((17,)) == {int(store.address_keys[row])}
        # Checkpoint restore replaces the durable metadata owner. The
        # derived postings must be rebuilt, with no row or stream registry.
        extras = store.semantic_extras()
        store.load_state_dict(store.state_dict())
        store.load_semantic_extras(extras)
        occurrence = store.write_clause(idea_clause(), document_key='arrival', sentence_index=1)
        store.fill_forward_references((17,), int(store.row_ids[occurrence]), store.slots[occurrence, 0])
    assert not store.row(row)['pending']
    assert store.occurrence_of(row) == address
    torch.testing.assert_close(store.document_keys[row], document, rtol=0, atol=0)
    torch.testing.assert_close(store.timestamp[row], timestamp, rtol=0, atol=0)
    assert int(store.refs[row, 0]) == int(store.row_ids[occurrence])
    if forward_first:
        assert not store._forward_addresses((17,))


def test_configured_store_still_raises_when_full():
    from Layers import TernaryTruthStore
    store = TernaryTruthStore(4, capacity=1)
    store.append_idea(torch.ones(4), document_key='first')
    with pytest.raises(OverflowError, match='capacity 1 exhausted'):
        store.append_idea(torch.zeros(4), document_key='second')


def test_retrieval_prior_rejects_a_stale_single_stream_surface():
    from Attention import BracketKeys
    with pytest.raises(ValueError, match='priming'):
        BracketKeys._codebook_retrieval_prior(torch.ones(8, 2, 4),
            torch.eye(4), None, torch.ones(1, 4))


def test_proposal_menu_never_offers_taxonomy_with_occurrence_operands():
    from test_normal_thought_controller import _catalog_world
    model, registry, _, part, whole = _catalog_world()
    request = registry.form('isPart', part, whole)
    bad = replace(request, role_refs=(('ltm', 'example', 123),
                                      request.role_refs[1], ('ltm', 'example', 456)))
    menu = registry.controller_candidates(bad, bad, bad)
    assert not any(item.semantic_id == 'isPart' for item in menu)
    assert any(item.semantic_id == 'isPart' for item in
               registry.controller_candidates(request, request, request))


def test_answer_rendering_does_not_reopen_a_completed_closing():
    from test_normal_thought_controller import _catalog_world
    from Understanding import SentenceEndState
    from ThoughtReferences import question
    model, registry, _, part, whole = _catalog_world()
    source = question(registry.form('isPart', part, whole), (('evidence', -1),))
    field = SentenceEndState(source, query=source, thought_completed=True)
    assert model._run_selected_sentence_thoughts((field,)) == ()
    assert field.detached().thought_completed


def test_later_binding_cannot_fill_from_an_absent_role():
    from Meaning import ConceptualMeaning
    from ThoughtReferences import question,fill,open_slots
    source=question(ConceptualMeaning(torch.ones(3,4),torch.ones(3,dtype=torch.bool)),
                    (('referent',2),))
    supplied=ConceptualMeaning.from_description(torch.ones(4))
    result=fill(source,dict(meaning=supplied,support_true=1.,support_false=0.))
    assert ('referent',2) in open_slots(result)
    assert result.role_refs[2] is None


def test_ltm_capacity_takes_precedence_over_truth_view_capacity(tmp_path):
    from test_math_chain import ROOT, build_model
    config = tmp_path/'capacity.xml'
    config.write_text((ROOT/'data/MM_math_chain.xml').read_text()
        .replace('<ltmCapacity>131072</ltmCapacity>', '<ltmCapacity>128</ltmCapacity>')
        .replace('<truthMaxEntries>65536</truthMaxEntries>', '<truthMaxEntries>7</truthMaxEntries>'))
    model = build_model(config)
    try:
        assert model.symbolSpace.ltm_store.capacity == 128
        assert model.symbolSpace.truth_layer.truths.shape[0] == 7
    finally:
        model.End()


def test_ordinary_driver_two_three_eight_stream_groups_after_single_write(tmp_path):
    from math_chain_corpus import MathChainCorpus, flatten
    from MathChainTraining import present
    from test_math_chain import ROOT, build_model
    config = tmp_path/'groups.xml'
    config.write_text((ROOT/'data/MM_math_chain.xml').read_text()
        .replace('<ltmCapacity>131072</ltmCapacity>', '<ltmCapacity>8192</ltmCapacity>')
        .replace('<attentionBudget>32</attentionBudget>', '<attentionBudget>0</attentionBudget>'))
    model = build_model(config)
    data = model.inputSpace.data
    corpus = MathChainCorpus()
    # These are the real tail groups (10 mod 8 and 11 mod 8) and a full
    # counting group, retaining their real sentence lengths and driver.
    docs = corpus.presentation()['train']
    groups = ((corpus.counting()[0],),
              tuple(doc for doc in docs if doc.pair is not None and doc.pair[1] == 3)[8:],
              tuple(doc for doc in docs if doc.pair is not None and doc.pair[1] == 0)[8:],
              corpus.counting()[::2][:8])
    assert tuple(map(len, groups)) == (1, 2, 3, 8)
    observed = []
    def after(model, split, rows, result):
        batch = len(rows)
        weights = model._concept_owner().priming_weights(batch=batch)
        assert weights is None or weights.shape[0] == batch
        observed.append(batch)
    try:
        for group in groups:
            texts, labels, addresses = flatten(group, supplied=False)
            data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
            data.text_answers['train'] = labels
            data.source_addresses['train'] = [dict(value, split='train') for value in addresses]
            assert present(model, split='train', after_batch=after)['sentences'] == len(texts)
        assert tuple(dict.fromkeys(observed)) == (1, 2, 3, 8)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
