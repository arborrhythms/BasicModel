"""Persistence, credit and invalid-history boundaries for the existing owner."""

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch

from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning
from Models import BaseModel
from Queries import ThoughtResult


def _meaning(value=1, *, width=8):
    return ConceptualMeaning(
        torch.ones(3, width) * value,
        torch.ones(3, dtype=torch.bool),
        mode="interrogative",
    )


def _memory():
    memory = WhatInteractionMemory(batch=2, capacity=32, detach_mode="episode")
    memory.begin_thought_episode(_meaning(), work_budget=12)
    memory.descend_thought(_meaning(2))
    return memory


def _model(memory):
    model = BaseModel()
    model.name = "LevelledHistoryCheckpoint"
    model.spaces = []
    model.conceptualSpaces = []
    model.wholeSpaces = []
    model.symbolSpace = torch.nn.Module()
    model.symbolSpace.what_memory = memory
    return model


def test_integrated_checkpoint_retains_the_one_memory_owner_and_active_history(tmp_path):
    source = _model(_memory())
    path = tmp_path / "thoughts.ckpt"
    source.save_weights(path)
    memory = WhatInteractionMemory(batch=2, capacity=32, detach_mode="episode")
    target = _model(memory)
    assert target.load_weights(path)
    assert target.symbolSpace.what_memory is memory
    state = memory.thought_state()
    assert state is not None and state.level == 1 and state.work_spent == 1
    assert memory.thought_history()[0].meaning.mode == "interrogative"
    assert memory.thought_state(b=1) is None


def test_integrated_checkpoint_restores_a_typed_checked_result(tmp_path):
    """Thought-result evidence is portable without restoring a reader graph."""
    memory = WhatInteractionMemory(batch=1, capacity=32, detach_mode="episode")
    request = _meaning()
    result = ThoughtResult(
        semantic_id="part", domain="conceptual-taxonomy",
        result_kind="truth", evidence_kind="taxonomy",
        request=request.detached(), evidence=MappingProxyType({
            "support_true": 1.0,
            "support_false": 0.0,
            "incomplete": (),
        }))
    memory.begin_thought_episode(request, work_budget=4)
    memory.commit_thought(
        request, operation="part", result=result,
        support_true=1.0, evidence_kind="taxonomy")
    memory.finish_thought(
        request, result=result, support_true=1.0,
        evidence_kind="taxonomy")
    memory.end_what_episode()

    source = _model(memory)
    path = tmp_path / "typed-thought.ckpt"
    source.save_weights(path)
    restored_memory = WhatInteractionMemory(
        batch=1, capacity=32, detach_mode="episode")
    target = _model(restored_memory)
    assert target.load_weights(path)
    replayed = next(
        record for record in restored_memory.thought_history()
        if record.kind == "thought" and record.operation == "part")
    assert replayed.result is not None
    assert replayed.result.semantic_id == "part"
    assert replayed.result.evidence["support_true"] == 1.0
    assert not replayed.result.request.roles.requires_grad


def test_v1_thought_history_checkpoint_remains_loadable_without_a_result_field():
    """Older history has no typed result and restores as the same history."""
    source = _memory()
    legacy = source.thought_extras()
    legacy["version"] = 1
    for row in legacy["rows"]:
        for record in row:
            if "legacy" not in record:
                record.pop("result")
    target = WhatInteractionMemory(batch=2, capacity=32, detach_mode="episode")
    target.load_thought_extras(legacy)
    assert target.thought_state().level == source.thought_state().level
    assert all(record.result is None for record in target.thought_history())


def test_restored_active_episode_keeps_new_computations_live_until_optimizer_boundary():
    source = _memory()
    memory = WhatInteractionMemory(batch=2, capacity=32, detach_mode="episode")
    memory.load_thought_extras(source.thought_extras())
    assert not memory.thought_state().contexts[-1].meaning.roles.requires_grad
    roles = torch.randn(3, 8, requires_grad=True)
    value = ConceptualMeaning(roles, torch.ones(3, dtype=torch.bool), mode="assertive")
    memory.commit_thought(value, operation="refine")
    assert memory.thought_history()[-1].meaning.roles.requires_grad
    memory.return_thought(value)
    memory.finish_thought(value)
    memory.thought_history()[-1].meaning.roles.square().sum().backward()
    assert roles.grad is not None and roles.grad.abs().sum() > 0


def test_credit_boundary_cannot_detach_an_unfinished_thought_episode():
    memory = _memory()
    with pytest.raises(RuntimeError, match="unfinished|finish|active"):
        memory.end_what_episode()
    assert memory.in_episode() and memory.thought_state().level == 1


def test_nested_meaning_cannot_silently_change_concept_width():
    memory = _memory()
    before = memory.thought_history()
    with pytest.raises(ValueError, match="width"):
        memory.descend_thought(_meaning(width=4))
    assert memory.thought_history() == before


@pytest.mark.parametrize("corruption", ["skip_level", "duplicate_id", "extra_budget", "bad_namespace"])
def test_malformed_history_load_is_atomic(corruption):
    source = _memory()
    saved = source.thought_extras()
    if corruption == "skip_level":
        saved["rows"][0][1]["level"] = 3
    elif corruption == "duplicate_id":
        saved["rows"][0][1]["id"] = saved["rows"][0][0]["id"]
    elif corruption == "extra_budget":
        saved["rows"][0][1]["budget"] = 50
    else:
        saved["namespace"] = "z" * 32
    target = _memory()
    before = target.thought_history()
    with pytest.raises(ValueError):
        target.load_thought_extras(saved)
    assert target.thought_history() == before


def test_occurrence_reads_are_row_local_and_keep_live_payloads():
    memory = _memory()
    occurrence = memory.thought_reference(memory.thought_history()[0])
    with pytest.raises(ValueError, match="row|unavailable"):
        memory.resolve_thought(occurrence, b=1)
    with pytest.raises(ValueError, match="limit|unavailable"):
        memory.resolve_thought(occurrence, max_records=0)
    value, cost = memory.resolve_thought(occurrence, max_records=1)
    assert cost == 1
    torch.testing.assert_close(value.roles, memory.thought_history()[0].meaning.roles)


def test_trim_cannot_remove_an_occurrence_referenced_by_the_active_root():
    memory = WhatInteractionMemory(capacity=7)
    memory.begin_thought_episode(_meaning(), work_budget=3)
    result = memory.finish_thought(replace(_meaning(2), mode="assertive"))
    reference = memory.thought_reference(result)
    memory.end_what_episode()
    root = replace(_meaning(3), role_refs=(reference, None, None))
    memory.begin_thought_episode(root, work_budget=20)
    memory.commit_thought(_meaning(4), operation="refine")
    memory.commit_thought(_meaning(5), operation="refine")
    with pytest.raises(OverflowError, match="referenced|capacity"):
        memory.commit_thought(_meaning(6), operation="refine")
    value, _ = memory.resolve_thought(reference)
    torch.testing.assert_close(value.roles, result.meaning.roles)
