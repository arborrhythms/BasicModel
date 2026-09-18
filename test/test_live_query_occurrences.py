"""Selected thoughts resolve live occurrences from the one interaction owner."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Language import Grammar
from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning
from Queries import GrammaticalThoughtRegistry
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context


def _world():
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file("complete.grammar")
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
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
    context = _context(
        cs, model=model, memory=memory,
        continuation=lambda value: value.roles.square().sum())
    return registry, memory, source, memory.thought_reference(entry), context


def test_what_reads_live_complete_question_across_a_detached_boundary():
    registry, memory, source, reference, context = _world()
    outer = registry.form("what", reference, context=context)
    assert outer.role_refs[0] == reference
    result = registry.execute(outer, context)
    assert not result.value.requires_grad
    assert source.grad is None
    assert memory.thought_state().level == 0
    assert result.evidence["resolution_records_scanned"] == 1


def test_reading_a_reference_or_forming_a_candidate_does_not_change_level_or_history():
    registry, memory, _, reference, context = _world()
    before = memory.thought_history()
    for _ in range(3):
        registry.form("what", reference, context=context)
    assert memory.thought_history() == before
    assert memory.thought_state().level == 0


def test_thought_occurrences_cannot_read_another_batch_row():
    registry, memory, _, reference, context = _world()
    with pytest.raises(ValueError, match="row|unavailable"):
        registry.form("what", reference, context=replace(context, row=1))


def test_thought_read_after_restore_retains_all_roles_binding_scope_and_mode():
    registry, memory, _, reference, context = _world()
    snapshot = memory.thought_extras()
    memory.load_thought_extras(snapshot)
    seen = []
    context = replace(context, continuation=lambda value: seen.append(value) or value)
    result = registry.execute(registry.form("what", reference, context=context), context)
    assert result.value is not seen[0]
    assert seen[0].mode == "interrogative"
    assert seen[0].scope == (("place", "workshop"),)
    assert seen[0].bindings == (("x", ("sym", 7)),)
    torch.testing.assert_close(seen[0].roles, memory.thought_history()[0].meaning.roles)
    torch.testing.assert_close(result.value.roles, seen[0].roles)
