"""Selected queries resolve live occurrences from the one interaction owner."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Language import Grammar
from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning
from Queries import GrammaticalQueryRegistry, QueryContext
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _world():
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file("complete.grammar")
    registry = GrammaticalQueryRegistry.install(cs, grammar)
    memory = WhatInteractionMemory(batch=2, capacity=32, detach_mode="episode")
    source = torch.randn(3, 8, requires_grad=True)
    question = ConceptualMeaning(
        source,
        torch.ones(3, dtype=torch.bool),
        mode="interrogative",
        scope={"place": "workshop"},
        bindings={"x": ("sym", 7)},
    )
    entry = memory.begin_thought_episode(question, work_budget=8)
    model = SimpleNamespace(conceptualSpace=cs, symbolSpace=SimpleNamespace(what_memory=memory))
    context = QueryContext(TruthGroundedReasoner(model=model), schedule_subgoal=lambda value: value.roles.square().sum())
    return registry, memory, source, memory.thought_reference(entry), context


def test_what_reads_live_complete_question_without_detaching_its_episode():
    registry, memory, source, reference, context = _world()
    outer = registry.form("what", reference, context=context)
    assert outer.roles.requires_grad and outer.role_refs[0] == reference
    result = registry.execute(outer, context)
    result["value"].backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert memory.thought_state().level == 0
    assert result["resolution_records_scanned"] == 1


def test_reading_a_reference_or_forming_a_candidate_does_not_change_level_or_history():
    registry, memory, _, reference, context = _world()
    before = memory.thought_history()
    for _ in range(3):
        registry.form("what", reference, context=context)
    assert memory.thought_history() == before
    assert memory.thought_state().level == 0 and memory.thought_state().work_spent == 0


def test_query_occurrences_cannot_read_another_batch_row():
    registry, memory, _, reference, context = _world()
    with pytest.raises(ValueError, match="row|unavailable"):
        registry.form("what", reference, context=replace(context, row=1))


def test_query_read_after_restore_retains_all_roles_binding_scope_and_mode():
    registry, memory, _, reference, context = _world()
    snapshot = memory.thought_extras()
    memory.load_thought_extras(snapshot)
    seen = []
    context = replace(context, schedule_subgoal=lambda value: seen.append(value) or value)
    result = registry.execute(registry.form("what", reference, context=context), context)
    assert result["value"] is seen[0]
    assert seen[0].mode == "interrogative"
    assert seen[0].scope == (("place", "workshop"),)
    assert seen[0].bindings == (("x", ("sym", 7)),)
    torch.testing.assert_close(seen[0].roles, memory.thought_history()[0].meaning.roles)
