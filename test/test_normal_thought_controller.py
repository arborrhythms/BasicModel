"""Selected grammatical questions execute through ordinary thought history.

These are reviewer probes for the normal boundary controller, not the legacy
``LTMSlot`` parity loop.  They exercise a real checked VP registry, the one
existing interaction-memory owner, and its shared work/cutoff accounting.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch

from Language import Grammar
from Layers import WhatInteractionMemory
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry
from test_cs_symbol_table import _cs
from Understanding import Understanding
from What import WhatQuestion


def _world():
    return _catalog_world()


def _catalog_world():
    """A normal-controller fixture with only the grammar-owned registry."""
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file("complete.grammar")
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    part, whole = ("sym", cs.new_concept()), ("sym", cs.new_concept())
    cs._csw_concept_row(0, part[1])
    cs._csw_concept_row(0, whole[1])
    cs.add_whole(part[1], whole)
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    model = BasicModel()
    model.spaces = []
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        what_memory=memory, grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"
    return model, registry, memory, part, whole


def _with_what_wrapper(language, entry):
    """Mark a completed structural program interrogative through grammar."""
    local = next(
        index for index, rule in enumerate(language._compose_unary_rules)
        if rule.method_name == "what")
    return replace(entry, actions=torch.cat((entry.actions, torch.tensor(
        [[2, local, -1]], dtype=entry.actions.dtype))))


def test_normal_controller_selects_a_catalog_operation_not_used_by_the_parse(
        monkeypatch):
    """A completed ``part`` may select grammar-declared ``equal`` afterwards."""
    model, registry, memory, part, whole = _catalog_world()
    question = registry.form("part", part, whole)
    choices = []

    def choose(_root, _active, actions, **_kwargs):
        choices.append(tuple(
            "conclude" if action is None else action.semantic_id
            for action in actions))
        if len(choices) == 1:
            return next(action for action in actions
                        if action is not None and action.semantic_id == "equal")
        return None

    monkeypatch.setattr(model, "_choose_selected_thought_action", choose)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(question, row=0, work_budget=16)

    assert "equal" in choices[0]
    assert [record.operation for record in result.records if record.kind == "thought"] == [
        "equal", "conclude"]
    assert 0.0 <= result.evidence["support_true"] <= 1.0
    memory.end_what_episode()


def test_selected_boundary_query_records_one_episode_and_its_actual_work():
    model, registry, memory, part, whole = _world()
    question = registry.form("part", part, whole)

    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(question, row=0, work_budget=16)

    assert result.evidence["support_true"] == 1
    assert result.meaning is question
    assert result.work.spent == memory.thought_state().work_spent
    assert result.work.spent <= 16
    records = memory.thought_history()
    ordinary = [record for record in records if record.kind != "cutoff"]
    assert [record.kind for record in ordinary] == [
        "begin", "thought", "thought", "finish"]
    assert [record.operation for record in ordinary[1:3]] == ["part", "conclude"]
    assert ordinary[1].meaning.role_refs == question.role_refs
    assert ordinary[-1].support_true == 1
    assert memory.thought_state().finished
    memory.end_what_episode()


def test_what_subgoal_descends_executes_returns_and_causally_carries_evidence():
    model, registry, memory, part, whole = _world()
    from QueryWork import QueryWorkBudget

    inner = registry.form("part", part, whole)
    prior = memory.begin_thought_episode(inner, work_budget=2)
    memory.finish_thought(inner, support_true=0.0, evidence_kind="subgoal")
    memory.end_what_episode()
    with model._query_boundary_scope((0,)):
        outer = registry.form(
            "what", memory.thought_reference(memory.thought_history()[0]),
            context=model._thought_grammar_context(
                inner, row=0, work=QueryWorkBudget(8), continuation=None),
        )
        result = model.run_selected_thought(outer, row=0, work_budget=64)

    state = memory.thought_state()
    assert state.finished and state.level == 0
    assert result.evidence["support_true"] == 1
    assert result.work.spent == state.work_spent
    records = memory.thought_history()
    latest = records[-7:]
    assert [record.kind for record in latest] == [
        "begin", "thought", "descend", "thought", "return", "thought", "finish",
    ]
    assert latest[1].meaning.role_refs == outer.role_refs
    assert latest[3].meaning.role_refs == inner.role_refs
    assert latest[4].support_true == 1
    assert any(ref[0] == "thought" for ref in latest[4].sources)
    assert [record.operation for record in latest if record.kind == "thought"] == [
        "what", "part", "conclude"]
    memory.end_what_episode()


def test_selected_controller_requires_the_completed_boundary_before_any_read():
    model, registry, _memory, part, whole = _world()
    question = registry.form("part", part, whole)
    try:
        model.run_selected_thought(question, row=0, work_budget=8)
    except RuntimeError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("selected controller escaped the boundary guard")


def test_completed_compose_program_enters_normal_controller_before_legacy_thinking(
        monkeypatch):
    from test_selected_relation_meaning import _program_owner

    cs, grammar, _legacy_registry, language, _leaves, program, _part, _whole = (
        _program_owner(monkeypatch, interrogative=True))
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    model = BasicModel()
    model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=32, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        languageSpace=language, what_memory=memory,
        grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"

    with model._query_boundary_scope((0,)):
        selected = model._run_selected_program_thoughts(
            (program(),), work_budget=16)

    assert len(selected) == 1
    row, result = selected[0]
    assert row == 0 and result.meaning.mode == "interrogative"
    assert [record.operation for record in result.records if record.kind == "thought"] == [
        "part", "conclude"]
    assert memory.thought_state().finished
    memory.end_what_episode()


def test_normal_controller_policy_sees_all_mandatory_roles_and_gets_credit():
    model, registry, memory, part, whole = _world()
    model.selected_thought_policy_weight = 1.0
    question = registry.form("part", part, whole)
    changed_roles = question.roles.clone()
    changed_roles[1].add_(0.75)
    changed_roles[2].sub_(0.5)
    changed = replace(question, roles=changed_roles)
    chooser = model._selected_thought_chooser(question)
    with torch.no_grad():
        for parameter in chooser.parameters():
            parameter.fill_(0.01)
    first = model._selected_thought_context(
        question, question, question, level=0, pressure=0.0)
    second = model._selected_thought_context(
        question, question, changed, level=0, pressure=0.0)
    assert not torch.equal(first, second)
    assert not torch.equal(
        chooser.logits(torch.stack((first, first)), (False, False)),
        chooser.logits(torch.stack((first, second)), (False, False)),
    )

    changed_root_roles = question.roles.clone()
    changed_root_roles[0].add_(0.25)
    changed_root = replace(question, roles=changed_root_roles)
    changed_active_roles = question.roles.clone()
    changed_active_roles[2].sub_(0.125)
    changed_active = replace(question, roles=changed_active_roles)
    assert not torch.equal(
        first,
        model._selected_thought_context(
            changed_root, question, question, level=0, pressure=0.0),
    )
    assert not torch.equal(
        first,
        model._selected_thought_context(
            question, changed_active, question, level=0, pressure=0.0),
    )
    # Unoccupied-role values are physical padding, not controller features.
    unary = registry.form("part", whole, open_roles=("I1",))
    padded = unary.roles.clone()
    padded[0].fill_(99.0)
    unary_with_padding = replace(unary, roles=padded)
    torch.testing.assert_close(
        model._selected_thought_context(
            unary, unary, unary, level=0, pressure=0.0),
        model._selected_thought_context(
            unary, unary, unary_with_padding, level=0, pressure=0.0),
    )

    with model._query_boundary_scope((0,)):
        model.run_selected_thought(question, row=0, work_budget=8)
    loss = model._selected_thought_policy_loss(torch.tensor(0.25))
    assert loss is not None and loss.requires_grad
    loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum())
        for parameter in chooser.parameters()
    )
    memory.end_what_episode()


def test_normal_controller_checkpoint_rebuilds_its_width_owned_policy():
    source, registry, _memory, part, whole = _world()
    question = registry.form("part", part, whole)
    chooser = source._selected_thought_chooser(question)
    with torch.no_grad():
        chooser.mlp[-1].weight.fill_(0.125)
    state = {key: value.detach().clone() for key, value in source.state_dict().items()}

    restored = BasicModel()
    assert getattr(restored, "selected_thought_choosers", None) is None
    assert restored._materialize_answer_path_from_checkpoint(state) == 1
    restored.load_state_dict(state, strict=True)
    loaded = restored.selected_thought_choosers[str(question.roles.shape[-1])]
    for before, after in zip(chooser.parameters(), loaded.parameters()):
        torch.testing.assert_close(before, after)
    ids = [id(parameter) for parameter in restored.synthesis_parameters()]
    assert all(ids.count(id(parameter)) == 1 for parameter in loaded.parameters())


def test_selected_program_resolution_releases_completed_eval_episode(monkeypatch):
    from test_selected_relation_meaning import _program_owner

    cs, grammar, _legacy_registry, language, _leaves, program, _part, _whole = (
        _program_owner(monkeypatch, interrogative=True))
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    model = BasicModel()
    model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        languageSpace=language, what_memory=memory,
        grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"
    model.reconstruct_in_loop = False
    model.eval()
    # The semantic/controller seam is the subject of this probe.  Its answer
    # carrier is deliberately ordinary, rather than a second surface path.
    monkeypatch.setattr(model, "_walk_budget", lambda: 8)
    monkeypatch.setattr(
        model, "_materialize_entries",
        lambda entries, base, budget: (base, torch.empty(
            base.shape[0], 0, device=base.device, dtype=torch.long)),
    )
    monkeypatch.setattr(
        model, "_what_grammar_context",
        lambda questions, **kwargs: (torch.zeros(
            len(questions), 1, device=kwargs["device"], dtype=kwargs["dtype"]), ()),
    )
    monkeypatch.setattr(model, "_select_perceptual_bindings", lambda _: ())
    understanding = Understanding(answer_program=(
        program(),))
    question = WhatQuestion.present(0)

    first = model.resolveAnswer(understanding, question)
    assert len(first.selected_thoughts) == 1
    assert memory.thought_state().finished and not memory.in_episode(0)
    # A later completed sentence may start a new ordinary episode; retaining
    # the finished eval graph would otherwise make this raise before selection.
    second = model.resolveAnswer(understanding, question)
    assert len(second.selected_thoughts) == 1
    episodes = [record.episode for record in memory.thought_history()
                if record.kind == "begin"]
    assert len(episodes) == 2 and episodes[0] != episodes[1]


def test_selected_program_precedes_the_legacy_surface_reasoner(monkeypatch):
    """A grammar-owned completed question never opens the legacy tool path."""
    from test_selected_relation_meaning import _program_owner

    cs, grammar, _legacy_registry, language, _leaves, program, _part, _whole = (
        _program_owner(monkeypatch, interrogative=True))
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    model = BasicModel()
    model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        languageSpace=language, what_memory=memory,
        grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"
    model.reconstruct_in_loop = False
    model.reasoning_iterations = 1
    model.eval()
    monkeypatch.setattr(model, "_walk_budget", lambda: 8)
    monkeypatch.setattr(
        model, "_materialize_entries",
        lambda entries, base, budget: (base, torch.empty(
            base.shape[0], 0, device=base.device, dtype=torch.long)),
    )
    monkeypatch.setattr(
        model, "_what_grammar_context",
        lambda questions, **kwargs: (torch.zeros(
            len(questions), 1, device=kwargs["device"], dtype=kwargs["dtype"]), ()),
    )
    monkeypatch.setattr(model, "_select_perceptual_bindings", lambda _: ())

    legacy_calls = []

    def legacy_reasoner(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return {"posture": "legacy", "confidence": 1.0,
                "support_true": 1.0, "support_false": 0.0}

    monkeypatch.setattr(model, "answer_query", legacy_reasoner)
    understanding = Understanding(answer_program=(program(),))
    question = WhatQuestion.inference(0, prompt="is the part in the whole?")

    result = model.resolveAnswer(understanding, question)

    assert len(result.selected_thoughts) == 1
    assert result.source == "identity"
    assert not legacy_calls
    assert memory.thought_state().finished and not memory.in_episode(0)


def test_normal_what_uses_the_selected_thought_without_a_legacy_slot(monkeypatch):
    """One grammatical boundary must not append a second legacy interaction."""
    from test_selected_relation_meaning import _program_owner

    cs, grammar, _legacy_registry, language, _leaves, program, _part, _whole = (
        _program_owner(monkeypatch, interrogative=True))
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    model = BasicModel()
    model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        languageSpace=language, what_memory=memory,
        grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"
    model.reconstruct_in_loop = False
    model.answer_synthesis = True
    model.eval()
    monkeypatch.setattr(model, "_walk_budget", lambda: 8)
    monkeypatch.setattr(
        model, "_materialize_entries",
        lambda entries, base, budget: (base, torch.empty(
            base.shape[0], 0, device=base.device, dtype=torch.long)),
    )
    monkeypatch.setattr(
        model, "_what_grammar_context",
        lambda questions, **kwargs: (torch.zeros(
            len(questions), 1, device=kwargs["device"], dtype=kwargs["dtype"]),
            ({},) * len(questions)),
    )
    monkeypatch.setattr(model, "_select_perceptual_bindings", lambda _: ())
    understanding = Understanding(answer_program=(program(),))
    produced = torch.zeros(1, 1, 3, dtype=program().leaves.dtype)
    execution = (None, produced, produced)
    monkeypatch.setattr(model, "_capture_understanding", lambda _: understanding)
    monkeypatch.setattr(
        model, "reverseOutput", lambda _understanding, _derivation:
        SimpleNamespace(actual=produced))

    answer = model.what(WhatQuestion.inference(0), execution=execution)

    assert answer.available and answer.ltm_slot is None
    assert len(memory.get_what_slots()) == 0
    assert [record.kind for record in memory.thought_history()] == [
        "begin", "thought", "thought", "finish"]
