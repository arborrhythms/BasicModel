"""Accessible-mind §6: expectation changes conception, never observation."""
import copy
import os
os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

from Layers import MeaningExpectation, ExpectationComparison, TernaryTruthStore
from Meaning import ConceptualMeaning
from meaning_fixtures import record_observation
from test_sentence_expectation import layer, observe


def test_empty_role_has_residual_and_trains_the_predictor():
    model = layer()
    observe(model, torch.ones(3, 4))
    prediction = model.expect_next_meaning()
    prediction.roles.retain_grad()
    mask = torch.tensor([True, True, False])
    target = torch.nn.Parameter(torch.ones(3, 4))
    model.observe_stm_end_state([3], [target], layout="infix", role_masks=[mask])
    comparison = model.last_expectation_comparison()
    torch.testing.assert_close(comparison.residual[2], -prediction.roles[2])
    model.consume_inter_loss().backward()
    assert prediction.roles.grad[2].norm() > 0
    assert target.grad is None


@pytest.mark.parametrize("gain", [0., .5, 1.])
def test_negative_image_identity_presence_and_object_mask(gain):
    from Meaning import negative_image
    observed = torch.eye(3)
    estimate = observed.clone()
    presence = torch.tensor([1., 1., 0.])
    mask = torch.tensor([0., 1., 0.])
    conceived, image = negative_image(observed, estimate, presence,
                                      gain=gain, object_mask=mask)
    torch.testing.assert_close(conceived - image, observed)
    torch.testing.assert_close(conceived[0], (1 - gain) * observed[0])
    torch.testing.assert_close(conceived[1:], observed[1:])


def test_conceived_absence_and_no_credit_to_gain_mask_or_reading():
    from Meaning import negative_image
    observed = torch.eye(3, requires_grad=True)
    estimate = torch.ones(3, 3, requires_grad=True)
    gain = torch.tensor(1., requires_grad=True)
    mask = torch.zeros(3, requires_grad=True)
    conceived, image = negative_image(observed * 0, estimate, torch.ones(3),
                                      gain=gain, object_mask=mask)
    torch.testing.assert_close(conceived, -estimate)
    assert not conceived.requires_grad and not image.requires_grad
    cold, image = negative_image(observed, None, None, gain=gain)
    torch.testing.assert_close(cold, observed)
    assert not image.any()


def test_retained_pair_keeps_full_estimate_and_normalized_surprise():
    from Meaning import expectation_surprise, negative_image
    store = TernaryTruthStore(4, capacity=8)
    source = ConceptualMeaning(torch.ones(3, 4), torch.ones(3, dtype=torch.bool))
    first = store.append_meaning(source, kind="observation", sentence_index=len(store))
    prediction = MeaningExpectation(torch.full((3, 4), .5), torch.tensor([9., -9., 9.]))
    mask = torch.tensor([True, True, False])
    actual = ConceptualMeaning(torch.ones(3, 4), mask)
    comparison = ExpectationComparison(
        prediction, actual.roles, mask, actual.roles - prediction.roles,
        mask.float() - prediction.presence_logits.sigmoid(), "doc",
        (store.occurrence_of(first),), ("external", "doc"))
    row = record_observation(store, actual.roles, 3,
                                   meaning=actual, expectation=comparison)
    assert len(store) == 3
    pair = store.expectation_pair(row, gain=.5, object_mask=torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(pair["estimate"].roles, prediction.roles)
    torch.testing.assert_close(pair["observation"].roles, actual.roles)
    torch.testing.assert_close(pair["residual"], comparison.residual)
    torch.testing.assert_close(pair["conceived"] - pair["negative_image"], actual.roles)
    expected = expectation_surprise(actual.roles, prediction.roles,
        actual.role_mask, prediction.presence_logits.sigmoid())
    assert store.row(row)["surprise"] == pytest.approx(float(expected))
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    assert restored.row(row)["surprise"] == store.row(row)["surprise"]


def test_old_checkpoint_unknown_surprise_is_not_zero():
    store = TernaryTruthStore(4, capacity=8)
    store.append_idea(torch.ones(4), sentence_index=len(store))
    state = store.state_dict()
    state.pop("surprise", None)
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(state, strict=True)
    assert restored.row(0)["surprise"] == -1


def test_predictor_loss_and_gradients_are_identical_at_every_gain():
    from Meaning import negative_image
    results = []
    for gain in (0., 1.):
        model = layer()
        observe(model, torch.eye(3, 4))
        observe(model, torch.ones(3, 4), mask=torch.tensor([True, False, True]))
        comparison = model.last_expectation_comparison()
        negative_image(comparison.observed, comparison.estimate.roles,
                       comparison.estimate.presence_logits.sigmoid(), gain=gain)
        loss = model.consume_inter_loss()
        loss.backward()
        results.append((loss.detach(), [p.grad for p in model._inter_predictor.parameters()]))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_declared_not_is_a_thought_act_not_an_observation(tmp_path, monkeypatch):
    from test_normal_thought_controller import _catalog_world
    from ThoughtReferences import with_slots, evidence_pair
    from ThoughtClosing import absence
    model, registry, memory, part, whole = _catalog_world()
    assert 'not' in registry.executable_operation_ids
    source = with_slots(registry.form('part', part, whole, mode='assertive'), (), pair=(.3,.7))
    request = registry.form('not', source)
    store = TernaryTruthStore(8, capacity=8)
    model.symbolSpace.ltm_store = store
    observation = store.append_meaning(source,kind='observation')
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(request,work_budget=32)
    assert result.result.semantic_id == 'not'
    assert evidence_pair(result.meaning) == (.7,.3)
    assert len(store) == 2 and store.row(1)['kind'] == 'inference'
    assert store.row(observation)['kind'] == 'observation'
    torch.testing.assert_close(store.meaning_of(observation).roles,source.roles)


def _anticipating_model():
    from Layers import BracketExpectation
    from test_normal_thought_controller import _catalog_world
    model, registry, memory, part, whole = _catalog_world()
    meaning = registry.form("part", part, whole)
    width = meaning.roles.shape[-1]
    store = TernaryTruthStore(width, capacity=64)
    discourse = BracketExpectation(4, 8, width, concept_dim=width,
                                   expectation_scope="structured")
    discourse._ltm_store = store
    model.symbolSpace.ltm_store = store
    model.symbolSpace.expectation = discourse
    model.expectation_policy_weight = .2
    model.expectation_query_budget = 64
    observe(discourse, meaning.roles)
    index = record_observation(store, meaning.roles, 3, meaning=meaning)
    discourse.bind_observation_occurrence(0, store.occurrence_of(index))
    return model, discourse, meaning


def test_expectation_updates_predictors_outside_the_departure_comparison(monkeypatch):
    """Expectation trains its predictor; R + A judges the departure."""
    from test_item6_2_thinking import test_expectation_trains_predictor_without_moving_departure_chooser
    test_expectation_trains_predictor_without_moving_departure_chooser(monkeypatch)


def test_metadata_comes_from_the_preceding_occurrence_not_the_target():
    from dataclasses import replace
    model, discourse, meaning = _anticipating_model()
    store = discourse._ltm_store
    prior = replace(meaning, bindings={"variable": "prior"}, scope={"where": "before"})
    observe(discourse, prior.roles)
    row = record_observation(store, prior.roles, 3, meaning=prior)
    discourse.bind_observation_occurrence(0, store.occurrence_of(row))
    estimate = discourse.expect_next_meaning()
    target = replace(meaning, bindings={"variable": "target"}, scope={"where": "after"})
    observe(discourse, target.roles)
    comparison = discourse.last_expectation_comparison()
    row = record_observation(store, target.roles, 3, meaning=target, expectation=comparison)
    pair = store.expectation_pair(row)
    assert pair["estimate"].bindings == prior.bindings != target.bindings
    assert pair["estimate"].scope == prior.scope != target.scope
    assert estimate.bindings == comparison.estimate.bindings


def test_native_unlabelled_batch_trains_the_same_chooser(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_reverse_traversal import _select_completed_binary_path
    torch.manual_seed(946)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8", batch_size=1, input_width=16,
        training_overrides={"expectationPolicyWeight": .2, "expectationQueryBudget": 64,
                            "expectationGain": 0., "reconstructInLoop": True,
                            "reconstructionPlacement": "eager", "intraLossWeight": 0.},
        architecture_overrides={"ltmConsolidation": True})
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    # Residual thought credit requires completed preceding observations.
    # Control compose only; the thought chooser under test remains trainable.
    _select_completed_binary_path(model)
    # Ordinary input reconstruction and unlabelled residual credit share the
    # real runBatch optimizer; there is no desired answer in this presentation.
    monkeypatch.setattr(model.inputSpace.data, "has_supervised_outputs", False)
    optimizer = model.getOptimizer(lr=.001)
    for step, text in enumerate(("a b", "a c", "b c")):
        inputs = model.inputSpace.prepInput([text])
        result, _ = model.runBatch(train=True, batchNum=step, batchSize=1,
            split="train", optimizer=optimizer,
            batch_override=(inputs, torch.empty(1, 0)))
        assert result is not None
    # Bound declaratives use compose and global expectation, with no separate policy.
    assert not hasattr(model, 'selected_thought_choosers')
    chooser = model.symbolSpace.languageLayer.operation_layer
    assert chooser.chooser.mlp[-1].weight.norm() > 0
    assert not model.__dict__.get("_selected_thought_policy_records")
    assert model._expectation_policy_loss() is None


def test_surprise_normalizes_observed_or_expected_roles_without_stance():
    from Meaning import expectation_surprise
    observed = torch.zeros(3, 4, requires_grad=True)
    estimate = torch.tensor([[1.] * 4, [2.] * 4, [9.] * 4], requires_grad=True)
    mask = torch.tensor([True, False, False])
    presence = torch.tensor([0., .5, 0.], requires_grad=True)
    # Observed roles have unit weight; otherwise use expected presence.
    # The half-expected empty role counts; unused padding does not dilute it.
    score = expectation_surprise(observed, estimate, mask, presence)
    error = (1. + .5 * 4.) / 1.5
    assert float(score) == pytest.approx(error / (1 + error))
    assert not score.requires_grad
    assert expectation_surprise(observed, observed, mask, presence) == 0
    assert expectation_surprise(observed, estimate, torch.zeros(3, dtype=torch.bool),
                                torch.zeros(3)) == 0


@pytest.mark.parametrize("capacity,sources", [(8, True), (2, True), (8, False)])
def test_idea_and_relation_rows_have_equal_surprise_per_role(capacity, sources):
    scores = []
    for depth in (1, 3):
        store = TernaryTruthStore(4, capacity=capacity)
        mask = torch.arange(3) < depth
        actual = ConceptualMeaning(torch.ones(3, 4), mask)
        source = store.append_meaning(actual, kind="observation", sentence_index=len(store))
        prediction = MeaningExpectation(torch.zeros(3, 4),
                                        torch.where(mask, 100., -100.))
        comparison = ExpectationComparison(prediction, actual.roles, mask,
            actual.roles - prediction.roles, mask.float() - prediction.presence_logits.sigmoid(),
            "doc", (store.occurrence_of(source),) if sources else (), ("external", "doc"))
        row = record_observation(store, actual.roles, depth,
                                       meaning=actual, expectation=comparison)
        scores.append(store.row(row)["surprise"])
    assert scores == pytest.approx([.5, .5])


@pytest.mark.parametrize("state", ["live", "restored", "compacted"])
def test_pair_lookup_uses_occurrence_index_in_both_directions(state, monkeypatch):
    store = TernaryTruthStore(4, capacity=64)
    meaning = ConceptualMeaning(torch.ones(3, 4), torch.ones(3, dtype=torch.bool))
    for _ in range(24):
        row = store.append_idea(torch.ones(4), sentence_index=len(store))
        store.set_origin(row, store.ORIGIN_USER)
    source = store.append_meaning(meaning, kind="observation", sentence_index=len(store))
    e, o = store.append_expectation_pair(meaning, meaning, presence_logits=torch.ones(3),
        source_occurrences=(store.occurrence_of(source),))
    eref, oref = store.occurrence_of(e), store.occurrence_of(o)
    if state == "restored":
        restored = TernaryTruthStore(4, capacity=64)
        restored.load_state_dict(copy.deepcopy(store.state_dict()))
        restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
        store = restored
    elif state == "compacted":
        assert store.clear_origin(store.ORIGIN_USER) == 24
    e, o = store._index_occurrences[eref], store._index_occurrences[oref]
    occurrence_of = store.occurrence_of
    def indexed_only(row):
        assert row in (e, o), "pair lookup scanned an unrelated row"
        return occurrence_of(row)
    monkeypatch.setattr(store, "occurrence_of", indexed_only)
    for row in (e, o):
        pair = store.expectation_pair(row)
        assert pair["estimate_occurrence"] == eref
        assert pair["observation_occurrence"] == oref
        assert not pair["residual"].any()


def test_detached_pending_prediction_replays_without_an_optimizer_step():
    model = layer()
    observe(model, torch.ones(3, 4))
    model.expect_next_meaning()
    model.detach_prediction_context()
    observe(model, torch.zeros(3, 4))
    loss = model.consume_inter_loss()
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model._inter_predictor.parameters())


def test_delayed_contrastive_credit_replays_the_current_predictor():
    model = layer()
    model.set_inter_loss_weight(0.)
    model.set_inter_contrastive(1.)
    observe(model, torch.eye(3, 4))
    original = model.expect_next_meaning().roles.detach().clone()
    model.detach_prediction_context()
    with torch.no_grad():
        next(model._inter_predictor.parameters()).add_(.001)
    observe(model, -torch.eye(3, 4))
    torch.testing.assert_close(model.last_expectation_comparison().estimate.roles, original)
    loss = model.consume_inter_contrastive_loss()
    assert loss is not None and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model._inter_predictor.parameters())


def test_disabling_expectation_erases_pending_credit_and_starts_cold():
    model, discourse, meaning = _anticipating_model()
    model.eval()
    discourse.expect_next_meaning(record=True)
    assert discourse._inter_last_meaning[0] is not None
    discourse.set_expectation_enabled(False)
    assert discourse._inter_last_meaning == [None]
    assert discourse.consume_expectation_policy_outcomes() == []
    discourse.set_expectation_enabled(True)
    assert discourse.expect_next_meaning() is None


def test_open_question_and_nested_what_spare_the_same_role():
    from Queries import QueryWorkBudget
    model, discourse, meaning = _anticipating_model()
    registry = model.grammatical_thoughts
    source, whole = meaning.role_refs[0], meaning.role_refs[2]
    question = registry.form("isPart", whole, open_roles=("I1",))
    observe(discourse, meaning.roles)
    model.expectation_gain = 1.
    store = model.symbolSpace.ltm_store
    index = store.append_meaning(question, kind="question", sentence_index=len(store))
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(question, row=0,
            work=QueryWorkBudget(64), continuation=None)
        outer = registry.form("ask", store.occurrence_of(index), context=context)
        direct = model._selected_thought_expectation(question, row=0, work=QueryWorkBudget(64))
        nested = model._selected_thought_expectation(outer, row=0, work=QueryWorkBudget(64))
    torch.testing.assert_close(direct, nested)
    width = question.roles.shape[-1]
    assert direct[-4:-1].tolist() == [1., 0., 0.]
    torch.testing.assert_close(direct[:width], discourse.last_expectation_comparison().observed[0])


def test_anticipation_ignores_incoming_staging_and_other_streams():
    from types import SimpleNamespace
    with torch.no_grad():
        model, discourse, meaning = _anticipating_model()
    model.eval()
    perturbed = copy.deepcopy(model)
    before = discourse.expect_next_meaning(record=True)
    # Content staged for an arriving or later packed sentence is unavailable
    # to prior-only thought, including the most recent mutable program slot.
    perturbed._last_understanding = SimpleNamespace(sentence_states=(SimpleNamespace(leaves=torch.full((9, meaning.roles.shape[-1]), 999.)),))
    perturbed.inputSpace = SimpleNamespace(_ar_embedded_N=torch.full((2, 8, 16), -999.))
    perturbed.symbolSpace.ltm_store.append_meaning(meaning, kind="observation", stream=1, sentence_index=len(perturbed.symbolSpace.ltm_store))
    after = perturbed.symbolSpace.expectation.expect_next_meaning(record=True)
    torch.testing.assert_close(before.roles, after.roles, rtol=0, atol=0)
    assert not model._what_memory().thought_history()
    assert not perturbed._what_memory().thought_history()


@pytest.mark.parametrize("budget", [1, 64])
def test_nested_object_mask_reads_a_later_thought_on_the_shared_budget(budget):
    from QueryWork import QueryWorkBudget
    model, discourse, meaning = _anticipating_model()
    observe(discourse, meaning.roles)
    registry = model.grammatical_thoughts
    question = registry.form("isPart", meaning.role_refs[2], open_roles=("I1",))
    memory = model._what_memory()
    memory.begin_thought_episode(meaning, work_budget=8)
    memory.commit_thought(question, operation="part")
    memory.finish_thought(question)
    memory.end_what_episode()
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(question, row=0,
            work=QueryWorkBudget(64), continuation=None)
        outer = registry.form("ask", memory.thought_reference(memory.thought_history()[1]), context=context)
        meter = QueryWorkBudget(budget)
        nested = model._selected_thought_expectation(outer, row=0, work=meter)
        if budget == 64:
            direct = model._selected_thought_expectation(question, row=0,
                work=QueryWorkBudget(64))
            torch.testing.assert_close(nested, direct)
            assert meter.spent == 2
        else:
            assert meter.remaining == 0
            assert nested[-4:-1].tolist() == [0., 0., 0.]


def test_generate_sentence_uses_positive_prediction_only_as_generate_seed(monkeypatch):
    model, discourse, meaning = _anticipating_model()
    prediction = discourse.expect_next_meaning()
    expected = (prediction.roles * prediction.presence_logits.sigmoid()[:, None])[None]
    seen = []
    monkeypatch.setattr(model, "_walk_operand", lambda value, **kw: value)
    monkeypatch.setattr(model, "_walk_budget", lambda: 4)
    def walk(idea, *args):
        seen.append(idea)
        return idea, torch.tensor([1]), None, None
    monkeypatch.setattr(model, "_compiled_output_walk", lambda: walk)
    monkeypatch.setattr(model, "_generated_word_text", lambda *args: ("a wheel",))
    assert model.generate_sentence() == ["a", "wheel"]
    torch.testing.assert_close(seen[0], expected)
    assert not seen[0].requires_grad


def test_previous_chooser_checkpoint_preserves_logits_with_zero_new_columns():
    from Models import BasicModel
    from Language import SelectedThoughtChooser
    old = SelectedThoughtChooser(context_dim=32, hidden=16)
    prefix = 'selected_thought_choosers.8.'
    state = {prefix+name: value.clone() for name,value in old.state_dict().items()}
    expected = set(state)
    restored = BasicModel()
    assert restored._materialize_answer_path_from_checkpoint(state) == 0
    assert not state and restored._pending_thought_policy_reset == expected
    assert not hasattr(restored,'selected_thought_choosers')


def test_native_nonzero_composition_is_bit_identical_at_every_stance(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
    snapshots = []
    for enabled, staged, gain in ((True, False, 1.), (True, True, 0.), (True, True, 1.), (False, False, 1.)):
        torch.manual_seed(948)
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8", batch_size=1,
            training_overrides={"reconstructInLoop": False, "sentenceExpectation": enabled, "expectationGain": gain},
            architecture_overrides={"attentionBudget": 32})
        model.eval()
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        with torch.no_grad():
            # A nonzero readout makes the snapshot sensitive to content;
            # the zero-init cursor bootstrap alone could hide leaked inputs.
            model.answer_attention.scorer[-1].weight.fill_(.25)
            _stage_fullgraph_tensor_peer(model, ["a bicycle has a wheel"])
            if staged:
                discourse = model.symbolSpace.expectation
                observe(discourse, torch.ones(3, discourse.concept_dim))
                assert discourse.expect_next_meaning() is not None
            cs = model.conceptualSpace
            # Seed ordinary intent through its owner so the priority surface
            # is live and nonuniform; use the real staged percepts for scope.
            cs.prime_desire(torch.tensor([0, 1]), valence=1.)
            before_priority = model._assemble_relevance_priority(cs, 0, None, None).clone()
            __import__('ModelAttention').stage_input(model)
            before_scope = cs._passback_scope_where.clone()
            out = model._forward_with_compiled_sentence_state(None)
            model._publish_compiled_sentence_state(out)
            # Inspect the numerical kernel directly; no completed row stores
            # the operation journal. This is a caller-owned comparison only.
            trace = model._reconstruction_stack()
            captured = (out[19].detach().clone(), trace._choice_rule_ids.detach().clone(),
                        trace._choice_arities.detach().clone(), trace._choice_mask.detach().clone(),
                        model._tensor_pushed_ideas.detach().clone())
            priority = model._assemble_relevance_priority(cs, 0, None, None)
            __import__('ModelAttention').stage_input(model)
            scope = cs._passback_scope_where
            assert priority.abs().sum() > 0 and priority.max() > priority.min()
            assert scope.numel() == 2 and scope.reshape(-1)[1] > scope.reshape(-1)[0]
        snapshots.append(captured +
            (model.conceptualSpace.similarity_codebook.W.clone(),
             model.inputSpace._ar_embedded_N.detach().clone(),
             before_priority, before_scope, priority.clone(), scope.clone()))
        assert captured[-1].abs().sum() > 0
    for snapshot in snapshots[1:]:
        torch.testing.assert_close(snapshot, snapshots[0], rtol=0, atol=0)
