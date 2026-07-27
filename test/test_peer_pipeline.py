"""Peer-pipelined runtime ownership and ordering pins."""
import os
import sys
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import pytest
import torch
from types import SimpleNamespace

from Spaces import (
    ConceptualSpace,
    LanguageBinaryChoice,
    LanguageUnaryChoice,
    SubSpace,
    SubSpaceView,
    guard_peer_views,
)
from Models import StaticPeerPipeline, TensorPeerWhilePipeline
from Language import LanguageSpace, RoutingState
from Layers import ShortTermMemory


class _DeterministicBinaryChooser(torch.nn.Module):
    def forward(self, window):
        parent = window.sum(dim=1, keepdim=True)
        B = int(window.shape[0])
        routing = {
            "chosen_reduced": parent,
            "copy_score": window.new_zeros(B, 2, 1),
            "reduce_score": window.new_full((B, 1, 1), 10.0),
            "action_kind": torch.ones(
                B, 1, dtype=torch.long, device=window.device),
            "src_left": torch.full(
                (B, 1), -1, dtype=torch.long, device=window.device),
            "reduce_marginal_op": window.new_tensor(
                [[[0.25, 0.75]]]).expand(B, 1, 2),
            "local_structural_loss": window.new_full((B, 1), 0.125),
        }
        return parent, parent, routing


class _DeterministicUnaryChooser(torch.nn.Module):
    def forward(self, window):
        candidate = -window
        B = int(window.shape[0])
        routing = {
            "apply_mask": torch.ones(
                B, 1, 1, dtype=torch.bool, device=window.device),
            "action_op": torch.full(
                (B, 1), 3, dtype=torch.long, device=window.device),
            "local_structural_loss": window.new_full((B, 1), 0.25),
        }
        return candidate, candidate, routing


def _language_choice_harness():
    layer = SimpleNamespace(
        _binary_layers={"CS": _DeterministicBinaryChooser()},
        _unary_layers={"CS": _DeterministicUnaryChooser()},
        _binary_rule_ids={"CS": (0, 1)},
        _unary_rule_ids={"CS": (2,)},
    )
    coordinator = SimpleNamespace(languageLayer=layer)
    return LanguageSpace(SimpleNamespace(subspace=coordinator))


def test_language_chooses_and_conceptual_space_alone_applies_stm_update():
    language = _language_choice_harness()
    state = (
        torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        ]),
        torch.tensor([3, 1]),
        torch.tensor([[2, 1, 0], [5, -1, -1]]),
        torch.tensor([[4, 2, 0], [7, -1, -1]]),
        torch.tensor([[12, 11, 10], [15, -1, -1]]),
        torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.0, 0.0]]),
    )
    before = tuple(value.clone() for value in state)
    versions = tuple(value._version for value in state)
    gate = torch.tensor([True, False])

    binary = language.choose_capacity_binary(
        state, gate, base_tau=0.75)
    seal_binary = language.choose_sentence_seal_binary(
        state, gate, base_tau=0.75)
    assert isinstance(binary, LanguageBinaryChoice)
    for actual, expected in zip(seal_binary, binary):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not hasattr(language, "apply_binary_language_choice")
    assert tuple(value._version for value in state) == versions
    for actual, expected in zip(state, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    (binary_state, binary_applied, binary_op,
     binary_valid, binary_loss) = (
        ConceptualSpace.apply_binary_language_choice(state, binary))
    assert tuple(value._version for value in state) == versions
    for actual, expected in zip(state, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        binary_state[0][0],
        torch.tensor([[4.0, 6.0], [5.0, 6.0], [0.0, 0.0]]))
    torch.testing.assert_close(binary_state[1], torch.tensor([2, 1]))
    torch.testing.assert_close(binary_state[2][0], torch.tensor([2, 0, -1]))
    torch.testing.assert_close(binary_state[3][0], torch.tensor([5, 0, -1]))
    torch.testing.assert_close(
        binary_state[4][0], torch.tensor([-1, 10, -1]))
    torch.testing.assert_close(
        binary_state[5][0], torch.tensor([0.0, 0.7, 0.0]))
    torch.testing.assert_close(
        binary_applied, torch.tensor([True, False]))
    torch.testing.assert_close(binary_valid, binary_applied)
    torch.testing.assert_close(binary_op, torch.tensor([1, 1]))
    torch.testing.assert_close(
        binary_loss, torch.tensor([0.125, 0.125]))

    unary = language.choose_unary(binary_state, gate)
    assert isinstance(unary, LanguageUnaryChoice)
    final_state, unary_applied, unary_op, unary_valid, unary_loss = (
        ConceptualSpace.apply_unary_language_choice(binary_state, unary))
    torch.testing.assert_close(
        final_state[0][0, 0], torch.tensor([-4.0, -6.0]))
    torch.testing.assert_close(final_state[3][0], torch.tensor([6, 0, -1]))
    torch.testing.assert_close(
        unary_applied, torch.tensor([True, False]))
    torch.testing.assert_close(unary_valid, unary_applied)
    torch.testing.assert_close(unary_op, torch.tensor([3, 3]))
    torch.testing.assert_close(
        unary_loss, torch.tensor([0.25, 0.25]))


def test_subspace_view_has_no_setters_and_owner_commits():
    sub = SubSpace(inputShape=(1, 4), outputShape=(1, 4))
    owner = object()
    object.__setattr__(sub, "_owner_space", owner)
    view = sub.view()
    assert not hasattr(view, "set_event")
    assert not hasattr(view, "set_what")
    with pytest.raises(PermissionError):
        sub.commit_event(object(), torch.ones(1, 1, 4))
    sub.commit_event(owner, torch.ones(1, 1, 4))
    assert torch.equal(view.materialize(mode="event"), torch.ones(1, 1, 4))


def test_peer_guard_detects_direct_or_setter_mutation():
    sub = SubSpace(inputShape=(1, 2), outputShape=(1, 2))
    sub.set_event(torch.zeros(1, 1, 2))
    with pytest.raises(RuntimeError, match="peer mutation guard"):
        with guard_peer_views(sub.view()):
            sub.set_event(torch.ones(1, 1, 2))


def test_published_snapshot_is_zero_copy_read_only_and_guarded():
    owner = object()
    event = torch.zeros(1, 1, 2)
    view = SubSpaceView.snapshot(event, owner=owner)
    assert not hasattr(view, "set_event")
    assert view.materialize(mode="event") is event
    with pytest.raises(RuntimeError, match="peer mutation guard"):
        with guard_peer_views(view):
            view.materialize().add_(1)


def test_language_plan_snapshots_routing_before_latch():
    rules = {"SS": [[1, 2]]}
    routing = RoutingState(
        rules_by_space_role=rules,
        selected_rules=[1, 2],
        rule_probs=torch.tensor([[0.25, 0.75]]),
    )
    coordinator = SimpleNamespace(
        current_rules=rules, routing_state=routing, _compose_generation=7)
    language = LanguageSpace(SimpleNamespace(subspace=coordinator))
    plan = language.reduction_plan()
    # A later Language C-stage may replace its mutable coordinator state; the
    # plan held for B[p+2] must continue to describe the source C-stage.
    routing.rule_probs.zero_()
    rules["SS"][0][0] = 99
    assert plan["generation"] == 7
    assert plan["rules"]["SS"] == [[1, 2]]
    assert torch.equal(
        plan["routing"].rule_probs,
        torch.tensor([[0.25, 0.75]]))


def test_cs_alone_commits_symbolic_and_language_peer_results():
    cs = SimpleNamespace()
    cs.CSsym = SubSpace(inputShape=(1, 3), outputShape=(1, 3))
    object.__setattr__(cs.CSsym, "_owner_space", cs)
    cs.accept_language_plan = lambda plan: setattr(
        cs, "_language_feedback_plan", plan)
    event = torch.ones(1, 1, 3)
    symbol = torch.full((1, 1, 3), 2.0)
    plan = {"routing": "latched"}
    result = ConceptualSpace.commit_symbolic_peer_results(
        cs, event, symbol_event=symbol,
        prior_symbolic_event=torch.zeros_like(event), language_plan=plan)
    assert result is cs.CSsym
    assert torch.equal(result.materialize(mode="event"), event)
    assert result._symbol_peer_event is symbol
    assert cs._language_feedback_plan is plan


def test_static_pipeline_warmup_drain_and_two_symbolic_index_latch():
    width = 16
    pipeline = StaticPeerPipeline(width)
    b_seen, c_seen = [], []

    def stage_a(word, index):
        return (word, index)

    def stage_b(a_result, index, feedback):
        b_seen.append((index, feedback))
        return a_result

    def stage_c(b_result, index):
        c_seen.append((index, b_result[0]))
        # Language (C), not B, owns grammar production.  The scheduler must
        # expose this plan to B exactly two symbolic indices later.
        return f"grammar-{index}"

    trace = pipeline.run(list(range(width)), stage_a, stage_b, stage_c)
    assert trace[:3] == [("A", 0), ("A", 1), ("B", 0)]
    assert trace[-1] == ("C", width - 1)
    assert b_seen[0] == (0, None)
    assert b_seen[1] == (1, None)
    assert b_seen[2] == (2, "grammar-0")
    assert c_seen == [(i, i) for i in range(width)]


def _tensor_pipeline_probe(words, active):
    """Return device-side ordering records from the dynamic scheduler."""
    capacity = int(words.shape[1])
    device = words.device
    missing = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device)
    initial_state = (
        missing.clone(),  # A indices
        missing.clone(),  # B indices
        missing.clone(),  # C indices
        missing.clone(),  # feedback observed by B
    )
    empty_a = (
        torch.zeros(
            int(words.shape[0]), int(words.shape[2]),
            dtype=words.dtype, device=device),
        torch.zeros(
            int(words.shape[0]), 1, dtype=torch.bool, device=device),
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    empty_b = (
        torch.zeros_like(empty_a[0]),
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    empty_feedback = (
        torch.full((), -1, dtype=torch.int64, device=device),
    )

    def _put(vector, index, value):
        safe = index.clamp(0, capacity - 1).reshape(1)
        return vector.scatter(0, safe, value.reshape(1))

    def stage_a(word, row_gate, index, live, state):
        del live
        a_seen, b_seen, c_seen, b_feedback = state
        next_state = (
            _put(a_seen, index, index),
            b_seen, c_seen, b_feedback,
        )
        return (word[:, 0, :], row_gate, index), next_state

    def stage_b(a_payload, feedback, index, live, state):
        del live
        a_word, _row_gate, a_index = a_payload
        a_seen, b_seen, c_seen, b_feedback = state
        next_state = (
            a_seen,
            _put(b_seen, index, a_index),
            c_seen,
            _put(b_feedback, index, feedback[0]),
        )
        return (a_word, a_index), next_state

    def stage_c(b_payload, index, live, state):
        del live
        _b_word, b_index = b_payload
        a_seen, b_seen, c_seen, b_feedback = state
        next_state = (
            a_seen, b_seen, _put(c_seen, index, b_index), b_feedback)
        return (b_index,), next_state

    final_state, feedback, trip_count = TensorPeerWhilePipeline().run(
        words, active,
        state=initial_state,
        empty_a=empty_a,
        empty_b=empty_b,
        empty_feedback=empty_feedback,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c)
    return (*final_state, feedback[0], trip_count)


def test_tensor_while_pipeline_dynamic_trip_warmup_drain_and_latch():
    words = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    active = torch.tensor([
        [True, True, True, True, True, False, False, False],
        [True, True, True, False, False, False, False, False],
    ])
    a_seen, b_seen, c_seen, b_feedback, feedback, trip_count = (
        _tensor_pipeline_probe(words, active))
    assert int(trip_count) == 5
    expected = torch.tensor([0, 1, 2, 3, 4, -1, -1, -1])
    torch.testing.assert_close(a_seen, expected)
    torch.testing.assert_close(b_seen, expected)
    torch.testing.assert_close(c_seen, expected)
    torch.testing.assert_close(
        b_feedback, torch.tensor([-1, -1, 0, 1, 2, -1, -1, -1]))
    assert int(feedback) == 4


def test_tensor_while_pipeline_is_one_fullgraph_and_reusable():
    pytest.importorskip("torch._dynamo")
    torch._dynamo.reset()
    compiled = torch.compile(
        _tensor_pipeline_probe, backend="eager", fullgraph=True)
    words = torch.randn(4, 12, 5)
    active_long = (
        torch.arange(12).unsqueeze(0)
        < torch.tensor([11, 7, 4, 9]).unsqueeze(1))
    first = compiled(words, active_long)
    graphs_after_first = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"])
    active_short = (
        torch.arange(12).unsqueeze(0)
        < torch.tensor([3, 6, 2, 5]).unsqueeze(1))
    second = compiled(words, active_short)
    assert int(first[-1]) == 11
    assert int(second[-1]) == 6
    # Both runtime trip counts use the same Dynamo graph. A Python-unrolled
    # loop would specialize on 11 and then compile again for 6.
    assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == (
        graphs_after_first)


def _tensor_banked_pipeline_probe(words, active):
    """Exercise the two age slots and three disjoint owner banks."""
    capacity = int(words.shape[1])
    device = words.device
    missing = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device)
    stm_state = (missing.clone(),)
    a_state = (missing.clone(),)
    b_state = (missing.clone(), missing.clone())
    empty_a = (
        torch.zeros(
            int(words.shape[0]), int(words.shape[2]),
            dtype=words.dtype, device=device),
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    empty_b = (
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    empty_feedback = (
        torch.full((), -1, dtype=torch.int64, device=device),
    )

    def _put(vector, index, value):
        safe = index.clamp(0, capacity - 1).reshape(1)
        return vector.scatter(0, safe, value.reshape(1))

    def stage_a(word, row_gate, index, live, current_a, current_stm):
        del live, current_stm
        next_a = (_put(current_a[0], index, index),)
        return (word[:, 0, :], index), next_a

    def stage_b(a_payload, feedback, index, live,
                current_b, current_stm):
        del live
        _word, source_index = a_payload
        next_b = (
            _put(current_b[0], index, source_index),
            _put(current_b[1], index, feedback[0]),
        )
        next_stm = (
            _put(current_stm[0], index, source_index),
        )
        return (source_index,), next_b, next_stm

    def stage_c(b_payload, index, live):
        del index, live
        return (b_payload[0],)

    final_stm, final_a, final_b, feedback, trip_count = (
        TensorPeerWhilePipeline().run_banked(
            words, active,
            stm_state=stm_state, a_state=a_state, b_state=b_state,
            empty_a=empty_a, empty_b=empty_b,
            empty_feedback=empty_feedback,
            stage_a=stage_a, stage_b=stage_b, stage_c=stage_c))
    return (
        final_stm[0], final_a[0], final_b[0], final_b[1],
        feedback[0], trip_count)


def test_tensor_banked_pipeline_uses_one_slot_per_age_and_latches_by_two():
    words = torch.randn(3, 9, 4)
    active = (
        torch.arange(9).reshape(1, 9)
        < torch.tensor([7, 4, 2]).reshape(3, 1))
    stm, a_state, b_state, b_feedback, feedback, trip_count = (
        _tensor_banked_pipeline_probe(words, active))
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, -1])
    torch.testing.assert_close(stm, expected)
    torch.testing.assert_close(a_state, expected)
    torch.testing.assert_close(b_state, expected)
    torch.testing.assert_close(
        b_feedback, torch.tensor([-1, -1, 0, 1, 2, 3, 4, -1, -1]))
    assert int(feedback) == 6
    assert int(trip_count) == 7


def test_tensor_banked_pipeline_is_one_fullgraph_and_reusable():
    pytest.importorskip("torch._dynamo")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        _tensor_banked_pipeline_probe, backend="eager", fullgraph=True)
    words = torch.randn(4, 12, 5)
    active_long = (
        torch.arange(12).unsqueeze(0)
        < torch.tensor([11, 7, 4, 9]).unsqueeze(1))
    first = compiled(words, active_long)
    graph_count = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"])
    active_short = (
        torch.arange(12).unsqueeze(0)
        < torch.tensor([3, 6, 2, 5]).unsqueeze(1))
    second = compiled(words, active_short)
    assert int(first[-1]) == 11
    assert int(second[-1]) == 6
    assert int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"]) == graph_count


def _tensor_percept_concept_probe(words, active):
    """Record the corrected two-leg word schedule entirely on device."""
    capacity = int(words.shape[1])
    batch = int(words.shape[0])
    device = words.device
    missing = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device)
    stm_state = (missing.clone(),)
    percept_state = (missing.clone(),)
    concept_state = (missing.clone(), missing.clone())
    empty_percept = (
        torch.zeros(
            batch, int(words.shape[2]), dtype=words.dtype, device=device),
        torch.zeros(batch, 1, dtype=torch.bool, device=device),
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    empty_feedback = (
        torch.full((), -1, dtype=torch.int64, device=device),
    )

    def _put(vector, index, value):
        safe = index.clamp(0, capacity - 1).reshape(1)
        return vector.scatter(0, safe, value.reshape(1))

    def stage_percept(word, row_gate, index, live, current):
        del live
        next_state = (_put(current[0], index, index),)
        return (word[:, 0, :], row_gate, index), next_state

    def stage_concept(payload, feedback, index, live,
                      current_concept, current_stm):
        del live
        _word, _row_gate, source_index = payload
        next_concept = (
            _put(current_concept[0], index, source_index),
            _put(current_concept[1], index, feedback[0]),
        )
        next_stm = (_put(current_stm[0], index, source_index),)
        # The feedback source is this completed conceptual/Language index.
        return next_concept, next_stm, (source_index,)

    final_stm, final_percept, final_concept, feedback, trip_count = (
        TensorPeerWhilePipeline().run_percept_concept_banked(
            words, active,
            stm_state=stm_state,
            percept_state=percept_state,
            concept_state=concept_state,
            empty_percept=empty_percept,
            empty_feedback=empty_feedback,
            stage_percept=stage_percept,
            stage_concept=stage_concept))
    return (
        final_stm[0], final_percept[0],
        final_concept[0], final_concept[1],
        feedback[0], trip_count)


def test_percept_concept_pipeline_overlaps_adjacent_words_and_drains_latches():
    words = torch.randn(3, 9, 4)
    active = (
        torch.arange(9).reshape(1, 9)
        < torch.tensor([7, 4, 2]).reshape(3, 1))
    stm, percept, concept, seen_feedback, feedback, trip_count = (
        _tensor_percept_concept_probe(words, active))
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, -1])
    torch.testing.assert_close(percept, expected)
    torch.testing.assert_close(concept, expected)
    torch.testing.assert_close(stm, expected)
    # Concept w sees Language's choice from w-2, never w or w-1.
    torch.testing.assert_close(
        seen_feedback,
        torch.tensor([-1, -1, 0, 1, 2, 3, 4, -1, -1]))
    # Two post-concept ticks drain both grammar latches.
    assert int(feedback) == -1
    assert int(trip_count) == 7


def test_percept_concept_pipeline_is_one_fullgraph_across_runtime_lengths():
    pytest.importorskip("torch._dynamo")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        _tensor_percept_concept_probe, backend="eager", fullgraph=True)
    words = torch.randn(4, 12, 5)
    long_rows = (
        torch.arange(12).reshape(1, 12)
        < torch.tensor([11, 7, 4, 9]).reshape(4, 1))
    first = compiled(words, long_rows)
    graph_count = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"])
    short_rows = (
        torch.arange(12).reshape(1, 12)
        < torch.tensor([3, 6, 2, 5]).reshape(4, 1))
    second = compiled(words, short_rows)
    assert int(first[-1]) == 11
    assert int(second[-1]) == 6
    assert int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"]) == graph_count


def _tensor_cs_lane_probe(words, active):
    capacity = int(words.shape[1])
    device = words.device
    empty = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device)
    stm_state = (empty.clone(),)
    cs_sub_state = (empty.clone(),)
    cs_sym_state = (empty.clone(), empty.clone())
    cs_lang_state = (empty.clone(),)
    empty_cs_sub = (
        torch.full((), -1, dtype=torch.int64, device=device),)
    empty_cs_sym = (
        torch.full((), -1, dtype=torch.int64, device=device),)
    empty_feedback = (
        torch.full((), -1, dtype=torch.int64, device=device),)

    def _put(vector, index, value):
        safe = index.clamp(0, capacity - 1).reshape(1)
        return vector.scatter(0, safe, value.reshape(1))

    def stage_cs_sub(word, row_gate, index, live, current):
        del word, row_gate, live
        return (index,), (_put(current[0], index, index),)

    def stage_cs_sym(payload, feedback, index, live, current):
        del live
        source = payload[0]
        return (
            (source,),
            (
                _put(current[0], index, source),
                _put(current[1], index, feedback[0]),
            ),
        )

    def stage_cs_lang(payload, index, live, current, stm):
        del live
        source = payload[0]
        return (
            (_put(current[0], index, source),),
            (_put(stm[0], index, source),),
            (source,),
        )

    (final_stm, final_cs_sub, final_cs_sym, final_cs_lang,
     feedback, trip_count) = TensorPeerWhilePipeline().run_cs_lanes_banked(
        words, active,
        stm_state=stm_state,
        cs_sub_state=cs_sub_state,
        cs_sym_state=cs_sym_state,
        cs_lang_state=cs_lang_state,
        empty_cs_sub=empty_cs_sub,
        empty_cs_sym=empty_cs_sym,
        empty_feedback=empty_feedback,
        stage_cs_sub=stage_cs_sub,
        stage_cs_sym=stage_cs_sym,
        stage_cs_lang=stage_cs_lang)
    return (
        final_stm[0], final_cs_sub[0],
        final_cs_sym[0], final_cs_sym[1],
        final_cs_lang[0], feedback[0], trip_count)


def test_cs_lane_pipeline_orders_three_stages_and_latches_feedback_by_index():
    words = torch.randn(3, 9, 4)
    active = (
        torch.arange(9).reshape(1, 9)
        < torch.tensor([7, 4, 2]).reshape(3, 1))
    (stm, cs_sub, cs_sym, seen_feedback,
     cs_lang, feedback, trip_count) = _tensor_cs_lane_probe(words, active)
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, -1])
    torch.testing.assert_close(cs_sub, expected)
    torch.testing.assert_close(cs_sym, expected)
    torch.testing.assert_close(cs_lang, expected)
    torch.testing.assert_close(stm, expected)
    # CSLang(w) executes at tick w+2. Its one physical feedback register is
    # read by CSSym(w+2) on the following tick.
    torch.testing.assert_close(
        seen_feedback,
        torch.tensor([-1, -1, 0, 1, 2, 3, 4, -1, -1]))
    assert int(feedback) == -1
    assert int(trip_count) == 7


def test_cs_lane_pipeline_is_one_fullgraph_across_runtime_lengths():
    pytest.importorskip("torch._dynamo")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        _tensor_cs_lane_probe, backend="eager", fullgraph=True)
    words = torch.randn(4, 12, 5)
    long_rows = (
        torch.arange(12).reshape(1, 12)
        < torch.tensor([11, 7, 4, 9]).reshape(4, 1))
    first = compiled(words, long_rows)
    graph_count = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"])
    short_rows = (
        torch.arange(12).reshape(1, 12)
        < torch.tensor([3, 6, 2, 5]).reshape(4, 1))
    second = compiled(words, short_rows)
    assert int(first[-1]) == 11
    assert int(second[-1]) == 6
    assert int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"]) == graph_count


def test_functional_stm_push_matches_owner_commit_and_preserves_masked_rows():
    B, capacity, D = 3, 4, 5
    stm = ShortTermMemory(batch=B, capacity=capacity, concept_dim=D)
    stm.begin_forward(B, dtype=torch.float32)
    # Seed distinguishable state without relying on the operation under test.
    stm._buffer = torch.arange(
        B * capacity * D, dtype=torch.float32).reshape(B, capacity, D)
    stm._depth = torch.tensor([1, 3, 4])
    stm._orders = torch.arange(B * capacity).reshape(B, capacity)
    stm._grammar_orders = stm._orders + 20
    stm._concept_rows = stm._orders + 40
    stm._concept_activations = stm._orders.to(torch.float32) + 0.5
    before = tuple(t.clone() for t in (
        stm._buffer, stm._depth, stm._orders, stm._grammar_orders,
        stm._concept_rows, stm._concept_activations))

    ideas = torch.randn(B, D)
    gate = torch.tensor([[True], [False], [True]])
    inserted_orders = torch.tensor([7, 8, 9])
    inserted_grammar = torch.tensor([1, 2, 3])
    inserted_rows = torch.tensor([70, 80, 90])
    inserted_activations = torch.tensor([0.7, 0.8, 0.9])
    expected = ShortTermMemory.functional_push_step_masked(
        *before, ideas, gate, inserted_orders, inserted_grammar,
        inserted_rows, inserted_activations)

    stm.push_step_masked(
        ideas, gate, orders=inserted_orders,
        grammar_orders=inserted_grammar, concept_row=inserted_rows,
        concept_activation=inserted_activations)
    actual = (
        stm._buffer, stm._depth, stm._orders, stm._grammar_orders,
        stm._concept_rows, stm._concept_activations)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    # The middle row was not active and every state slab is unchanged.
    for actual_tensor, before_tensor in zip(actual, before):
        torch.testing.assert_close(actual_tensor[1], before_tensor[1])


def test_functional_stm_state_is_fullgraph_while_loop_safe():
    B, capacity, D = 4, 5, 3

    def recurrent(ideas, gates):
        buffer = torch.zeros(B, capacity, D)
        depth = torch.zeros(B, dtype=torch.long)
        orders = torch.full((B, capacity), -1, dtype=torch.long)
        grammar = torch.full_like(orders, -1)
        rows = torch.full_like(orders, -1)
        activations = torch.zeros(B, capacity)
        zero = torch.zeros((), dtype=torch.long)

        def cond(index, *state):
            del state
            return index < ideas.shape[1]

        def body(index, *state):
            (current_buffer, current_depth, current_orders,
             current_grammar, current_rows, current_activations) = state
            gather_index = index.reshape(1)
            word = torch.index_select(
                ideas, 1, gather_index)[:, 0, :]
            gate = torch.index_select(gates, 1, gather_index)
            inserted = index.expand(B)
            next_state = ShortTermMemory.functional_push_step_masked(
                current_buffer, current_depth, current_orders,
                current_grammar, current_rows, current_activations,
                word, gate, inserted, inserted, inserted,
                inserted.to(dtype=ideas.dtype))
            return (index + 1, *next_state)

        result = torch.while_loop(
            cond, body,
            (zero, buffer, depth, orders, grammar, rows, activations))
        return result[1:]

    compiled = torch.compile(recurrent, backend="eager", fullgraph=True)
    ideas = torch.randn(B, 7, D)
    gates = (
        torch.arange(7).reshape(1, 7)
        < torch.tensor([7, 5, 3, 1]).reshape(B, 1))
    eager = recurrent(ideas, gates)
    captured = compiled(ideas, gates)
    for eager_tensor, captured_tensor in zip(eager, captured):
        torch.testing.assert_close(eager_tensor, captured_tensor)


@pytest.mark.parametrize("width", (1, 8, 15, 17, 64, 256))
def test_pipeline_requires_canonical_static_buckets(width):
    if width in (8, 64, 256):
        assert StaticPeerPipeline(width).width == width
    else:
        with pytest.raises(ValueError):
            StaticPeerPipeline(width)
