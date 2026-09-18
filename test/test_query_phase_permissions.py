"""Reviewer probes for sentence/query phase ownership and row permissions.

These tests were rebased from the preserved September 17 phase candidate.
They deliberately inject checked query execution into compose, reconstruction,
and output paths: a query is executable only at a completed answer boundary.
"""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Models import BasicModel
from Output import AnswerDerivation
from Queries import BUILTIN_QUERIES, GrammaticalQueryRegistry, QueryContext
from Understanding import AnswerProgram, InputReconstruction, Understanding
from What import What
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _model():
    model = BasicModel()
    model.spaces = []
    object.__setattr__(model, "conceptualSpace", _cs())
    object.__setattr__(model, "perceptualSpace", SimpleNamespace())
    model.reconstruct_in_loop = False
    return model


def _understanding(*completed):
    roles = torch.zeros(3, 8)
    roles[0] = 1
    program = AnswerProgram(
        rows=torch.tensor([0]), word_rows=torch.tensor([0]),
        activations=torch.ones(1), leaves=roles[:1],
        actions=torch.tensor([[0, -1, 0]]), targets=torch.tensor([-1]),
        end_state=roles, concept_ids=torch.tensor([1]),
    )
    return Understanding(
        conceptual_state=roles[None].expand(len(completed), -1, -1),
        answer_program=tuple(program if ready else None for ready in completed),
    )


def _fixture():
    model = _model()
    calls = []
    signature = replace(
        BUILTIN_QUERIES["isEqual"],
        executor=lambda *args: calls.append(args) or {
            "support_true": 1.0, "support_false": 0.0,
        },
    )
    reasoner = TruthGroundedReasoner(model=model)

    def invoke(row=0):
        return signature.invoke(
            QueryContext(reasoner, row=row), torch.ones(8), torch.ones(8)
        )

    return model, invoke, calls


@pytest.mark.parametrize("path", ("compose", "reconstruct", "generate"))
def test_checked_executor_is_masked_throughout_each_sentence_path(monkeypatch, path):
    model, invoke, calls = _fixture()
    understanding = _understanding(True)

    def injected_query(*args, **kwargs):
        invoke()
        raise AssertionError("checked executor escaped the sentence-phase mask")

    if path == "compose":
        monkeypatch.setattr(model, "_forward_per_stage", injected_query)
        call = lambda: model.forward(torch.ones(1, 1, 8))
    elif path == "reconstruct":
        value = torch.ones(1, 1, 8)
        owned = InputReconstruction(value, value, None, None, None, None)
        understanding = replace(understanding, input_reconstruction=owned)
        monkeypatch.setattr(model, "_reverse_event_loss", injected_query)
        call = lambda: model.reverseReconstruct(understanding, target=value)
    else:
        monkeypatch.setattr(model, "_materialize_answer_idea", injected_query)
        call = lambda: model.reverseOutput(
            understanding, AnswerDerivation(None, program=understanding.answer_program)
        )
    with pytest.raises(RuntimeError, match="sentence|query|boundary"):
        call()
    assert calls == []


def test_checked_execution_requires_a_completed_answer_boundary():
    model, invoke, calls = _fixture()
    with pytest.raises(RuntimeError, match="boundary"):
        invoke()
    assert calls == []

    with model._query_boundary_scope((0,)):
        assert invoke()["support_true"] == 1.0
    assert len(calls) == 1


def test_resolution_opens_only_completed_rows_and_closes_after_return(monkeypatch):
    model, invoke, calls = _fixture()
    understanding = _understanding(True, False)
    marker = object()

    def resolve(*args):
        assert invoke(0)["support_true"] == 1
        with pytest.raises(RuntimeError, match="row|boundary"):
            invoke(1)
        return marker

    monkeypatch.setattr(model, "_resolve_answer", resolve)
    assert model.resolveAnswer(
        understanding, (What.inference(0), What.inference(1))
    ) is marker
    with pytest.raises(RuntimeError, match="boundary"):
        invoke(0)
    assert len(calls) == 1


def test_resolution_exception_cannot_leave_permission_open(monkeypatch):
    model, invoke, calls = _fixture()

    def resolve(*args):
        invoke()
        raise ValueError("injected resolver failure")

    monkeypatch.setattr(model, "_resolve_answer", resolve)
    with pytest.raises(ValueError, match="injected resolver"):
        model.resolveAnswer(_understanding(True), What.inference(0))
    with pytest.raises(RuntimeError, match="boundary"):
        invoke()
    assert len(calls) == 1


def test_nested_resolution_cannot_override_an_active_sentence_phase(monkeypatch):
    model, invoke, calls = _fixture()
    monkeypatch.setattr(model, "_resolve_answer", lambda *args: invoke())
    monkeypatch.setattr(
        model,
        "_forward_per_stage",
        lambda *args: model.resolveAnswer(_understanding(True), What.inference(0)),
    )
    with pytest.raises(RuntimeError, match="sentence|boundary|query"):
        model.forward(torch.ones(1, 1, 8))
    assert calls == []


def test_tied_input_reconstruction_must_finish_before_opening_query_boundary(monkeypatch):
    model, invoke, calls = _fixture()
    model.reconstruct_in_loop = True
    monkeypatch.setattr(model, "_resolve_answer", lambda *args: invoke())
    with pytest.raises(RuntimeError, match="reconstruction"):
        model.resolveAnswer(_understanding(True), What.inference(0))
    assert calls == []


@pytest.mark.parametrize("entry", ("answer_query", "reason_about", "think_about"))
def test_legacy_query_entries_are_masked_inside_forward(monkeypatch, entry):
    model = _model()
    model.reasoning_iterations = model.thinking_budget = 0
    assert getattr(model, entry)(None) is None
    monkeypatch.setattr(
        model, "_forward_per_stage", lambda *args: getattr(model, entry)(None)
    )
    with pytest.raises(RuntimeError, match="sentence|query|queries"):
        model.forward(torch.zeros(1, 1, 8))
    assert getattr(model, entry)(None) is None


@pytest.mark.parametrize(
    "entry", ("explicit_state", "understand_executor", "what_executor")
)
def test_every_input_execution_entry_masks_checked_queries(monkeypatch, entry):
    model = _model()
    # Simulate a completed parent resolution starting a child sentence. The
    # child must mask that parent's permission until it completes.
    model._query_ready_rows = (0,)
    calls = []
    signature = replace(
        BUILTIN_QUERIES["isEqual"], executor=lambda *args: calls.append(args) or {}
    )
    context = QueryContext(TruthGroundedReasoner(model))

    def query(*args, **kwargs):
        signature.invoke(context, torch.ones(8), torch.ones(8))
        raise AssertionError("input execution escaped its phase mask")

    if entry == "explicit_state":
        monkeypatch.setattr(model, "_forward_per_stage", query)
        call = lambda: model._forward_with_compiled_sentence_state(None)
    elif entry == "understand_executor":
        call = lambda: model.understand(torch.ones(1, 1, 8), executor=query)
    else:
        monkeypatch.setattr(model, "_begin_referents", lambda *args: None)
        monkeypatch.setattr(
            model, "_what_grammar_context", lambda *args, **kwargs: (torch.zeros(1, 4), ({},))
        )
        call = lambda: model.what(
            What.inference(0), torch.ones(1, 1, 8), executor=query
        )
    with pytest.raises(RuntimeError, match="sentence|query|queries"):
        call()
    assert calls == []


def test_operand_occurrence_reads_require_permission_before_preparation(monkeypatch):
    import Queries
    from Meaning import ConceptualMeaning

    model = _model()
    registry = GrammaticalQueryRegistry.install(
        model.conceptualSpace,
        SimpleNamespace(
            query_signatures={"exist": Queries.BUILTIN_QUERIES["exist"]},
            rules_upward=(),
        ),
    )
    vp = registry._reference(("ltm-facts", "exist"))
    meaning = ConceptualMeaning(
        torch.ones(3, 8), torch.tensor([True, True, False]),
        mode="interrogative", role_refs=(("ltm", "missing", 0), vp, None),
    )
    reads = []
    monkeypatch.setattr(
        Queries,
        "_occurrence_description",
        lambda *args: reads.append(args) or (_ for _ in ()).throw(
            AssertionError("occurrence read before boundary permission")
        ),
    )
    context = QueryContext(TruthGroundedReasoner(model))
    with pytest.raises(RuntimeError, match="boundary"):
        registry.execute(meaning, context)
    assert reads == []


def test_reconstruction_completion_keeps_query_executors_masked(monkeypatch):
    import Queries

    model = _model()
    model.reconstruct_in_loop = True
    model._recon_completed = True
    model._recon_ideas = torch.zeros(1, 1, 8)
    carrier = SimpleNamespace(set_event=lambda *args: None)
    object.__setattr__(
        model,
        "conceptualSpace",
        SimpleNamespace(
            outputShape=(1, 8),
            subspace=SimpleNamespace(carrier_like=lambda: carrier),
        ),
    )
    monkeypatch.setattr(model, "_synthesis_guard", nullcontext)
    calls = []
    signature = replace(
        Queries.BUILTIN_QUERIES["isEqual"],
        executor=lambda *args: calls.append(args) or {},
    )
    context = QueryContext(TruthGroundedReasoner(model))
    monkeypatch.setattr(
        model,
        "_reverse_input_surface",
        lambda *args: signature.invoke(context, torch.ones(8), torch.ones(8)),
    )
    with pytest.raises(RuntimeError, match="sentence|query|queries"):
        model._complete_input_reconstruction()
    assert calls == []


def test_nested_boundary_can_narrow_but_not_widen_completed_rows(monkeypatch):
    model = _model()
    outer, inner = _understanding(True, False), _understanding(False, True)

    def resolve(understanding, *args):
        if understanding is inner:
            model._assert_query_boundary(1)
        else:
            model._assert_query_boundary(0)
            with pytest.raises(RuntimeError, match="row|boundary"):
                model.resolveAnswer(inner, What.inference(1))
            model._assert_query_boundary(0)
        return "complete"

    monkeypatch.setattr(model, "_resolve_answer", resolve)
    assert model.resolveAnswer(outer, What.inference(0)) == "complete"
    with pytest.raises(RuntimeError, match="boundary"):
        model._assert_query_boundary(0)


def test_nested_sentence_restores_the_outer_completed_boundary(monkeypatch):
    model = _model()
    calls = []

    def forward(*args):
        calls.append("forward")
        model._assert_query_boundary(0)

    def resolve(*args):
        model._assert_query_boundary(0)
        with pytest.raises(RuntimeError, match="sentence|query|queries"):
            model.forward(torch.zeros(1, 1, 8))
        model._assert_query_boundary(0)
        return "complete"

    monkeypatch.setattr(model, "_forward_per_stage", forward)
    monkeypatch.setattr(model, "_resolve_answer", resolve)
    assert model.resolveAnswer(_understanding(True), What.inference(0)) == "complete"
    assert calls == ["forward"]
    with pytest.raises(RuntimeError, match="boundary"):
        model._assert_query_boundary(0)


def test_held_completed_understanding_owns_its_readiness_after_later_staging(monkeypatch):
    model = _model()
    held = _understanding(True, False)
    model._last_understanding = _understanding(False, True)
    monkeypatch.setattr(
        model, "_resolve_answer", lambda *args: model._assert_query_boundary(0)
    )
    assert model.resolveAnswer(held, What.inference(0)) is None


def test_uncommitted_or_empty_program_rows_do_not_open_queries(monkeypatch):
    model = _model()
    empty = SimpleNamespace(leaves=torch.zeros(0, 8))
    for understanding in (
        Understanding(conceptual_state=torch.ones(2, 3, 8)),
        Understanding(answer_program=(empty, None)),
    ):
        monkeypatch.setattr(
            model, "_resolve_answer", lambda *args: model._assert_query_boundary(0)
        )
        with pytest.raises(RuntimeError, match="boundary"):
            model.resolveAnswer(understanding, What.inference(0))


@pytest.mark.parametrize("eager_island", (False, True))
def test_compiled_input_cannot_execute_queries_during_trace_or_eager_island(
    monkeypatch, eager_island
):
    model = _model()
    model._spaces_started_for_forward = False
    model._query_ready_rows = (0,)
    calls = []
    signature = replace(
        BUILTIN_QUERIES["isEqual"], argument_roles=(), argument_kinds=(),
        occupied_roles=(1,), executor=lambda *args: calls.append(args) or {},
    )
    context = QueryContext(TruthGroundedReasoner(model))

    def body(value):
        signature.invoke(context)
        return value.sin()

    if eager_island:
        body = torch.compiler.disable(body)
    monkeypatch.setattr(model, "_forward_per_stage", body)
    compiled = torch.compile(model.forward, backend="eager", fullgraph=not eager_island)
    try:
        with pytest.raises(Exception, match="queries cannot execute inside a sentence path"):
            model.understand(torch.ones(1, 1, 8), executor=compiled)
        assert calls == []
        assert model._query_sentence_depth == 0
    finally:
        torch._dynamo.reset()
