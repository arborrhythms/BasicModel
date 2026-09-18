"""Reviewer probes for ordinary thought history and the shared work budget."""

from dataclasses import replace

import pytest
import torch

from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning


def _meaning(seed=0, *, question=True, scope=None, bindings=None):
    roles = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24 + seed
    return ConceptualMeaning(
        roles,
        torch.ones(3, dtype=torch.bool),
        mode="interrogative" if question else "assertive",
        scope=scope or {},
        bindings=bindings or {},
    )


def test_multiple_ordinary_thoughts_at_root_do_not_finish_or_make_qa_pairs():
    memory = WhatInteractionMemory(batch=1, capacity=32, detach_mode="episode")
    root = _meaning(scope={"place": "workshop"})
    memory.begin_thought_episode(root, work_budget=8)
    memory.commit_thought(_meaning(1, question=False), operation="refine")
    memory.commit_thought(_meaning(2), operation="question")
    state = memory.thought_state()
    assert state.level == 0 and not state.finished
    assert state.work_spent == 2 and state.work_remaining == 6
    assert [record.kind for record in memory.thought_history()] == [
        "begin",
        "thought",
        "thought",
    ]
    assert all(record.level == 0 for record in memory.thought_history())
    memory.finish_thought(_meaning(3, question=False), support_true=0, support_false=0)
    assert memory.thought_state().finished
    assert memory.thought_history()[-1].kind == "finish"


def test_nested_and_sibling_returns_restore_parent_scope_and_pending_roles():
    memory = WhatInteractionMemory(capacity=64)
    root = _meaning(scope={"place": "root"}, bindings={"x": ("sym", 7)})
    child = _meaning(1, scope={"place": "child"}, bindings={"x": ("sym", 8)})
    child_pending = _meaning(
        2, scope={"place": "child-new"}, bindings={"x": ("sym", 9)}
    )
    memory.begin_thought_episode(root, work_budget=20)
    memory.descend_thought(child)
    memory.commit_thought(child_pending, operation="refine")
    memory.descend_thought(_meaning(3))
    memory.return_thought(
        _meaning(4, question=False), support_true=0.6, support_false=0.2
    )
    resumed = memory.thought_state().contexts[-1]
    assert resumed.meaning.scope == child_pending.scope
    assert resumed.meaning.bindings == child_pending.bindings
    torch.testing.assert_close(resumed.meaning.roles, child_pending.roles)
    assert len(resumed.results) == 1
    memory.return_thought(_meaning(5, question=False), support_true=0.4)
    pressure = memory.thought_state().pressure
    memory.descend_thought(_meaning(6, scope={"place": "sibling"}))
    state = memory.thought_state()
    assert state.level == 1 and state.pressure >= pressure
    assert state.contexts[0].meaning.scope == root.scope
    assert state.contexts[0].meaning.bindings == root.bindings
    assert state.contexts[1].meaning.scope == (("place", "sibling"),)
    assert state.contexts[1].results == ()
    assert [r.kind for r in memory.thought_history()][-2:] == ["return", "descend"]


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_budget_cutoff_has_only_depth_returns_plus_one_finish(depth):
    memory = WhatInteractionMemory(capacity=64)
    memory.begin_thought_episode(_meaning(), work_budget=depth + 1)
    for i in range(depth):
        memory.descend_thought(_meaning(i + 1))
    memory.commit_thought(_meaning(7, question=False), operation="refine")
    cutoff = memory.thought_state()
    assert cutoff.work_remaining == 0 and cutoff.cutoff_depth == depth
    with pytest.raises(RuntimeError, match="cutoff|budget"):
        memory.commit_thought(_meaning(8), operation="query")
    with pytest.raises(RuntimeError, match="cutoff|budget"):
        memory.descend_thought(_meaning(9))
    for _ in range(depth):
        memory.return_thought(
            _meaning(10, question=False), support_true=0, support_false=0
        )
    memory.finish_thought(_meaning(11, question=False), support_true=0, support_false=0)
    final = memory.thought_state()
    assert final.finished and final.forced
    assert final.work_spent == depth + 1 and final.drain_count == depth + 1
    assert final.pressure >= cutoff.pressure
    assert memory.thought_history()[-1].support_true == 0
    with pytest.raises(RuntimeError, match="finished|episode"):
        memory.finish_thought(_meaning(12, question=False))


def test_return_underflow_and_new_episode_do_not_corrupt_history():
    memory = WhatInteractionMemory(capacity=32)
    memory.begin_thought_episode(_meaning(), work_budget=8)
    before = memory.thought_history()
    with pytest.raises(ValueError, match="level|root|underflow"):
        memory.return_thought(_meaning(1, question=False))
    with pytest.raises(RuntimeError, match="active|episode"):
        memory.begin_thought_episode(_meaning(2), work_budget=8)
    assert memory.thought_history() == before


def test_replay_and_checkpoint_preserve_active_contexts_and_row_isolation():
    memory = WhatInteractionMemory(batch=2, capacity=64)
    memory.begin_thought_episode(_meaning(scope={"row": "zero"}), b=0, work_budget=20)
    memory.begin_thought_episode(_meaning(scope={"row": "one"}), b=1, work_budget=7)
    memory.descend_thought(_meaning(1, scope={"place": "child"}), b=0)
    memory.commit_thought(
        _meaning(2, bindings={"pending": ("sym", 10)}), b=0, operation="refine"
    )
    restored = WhatInteractionMemory(batch=2, capacity=64)
    restored.load_thought_extras(memory.thought_extras())
    for b in (0, 1):
        left, right = memory.thought_state(b=b), restored.thought_state(b=b)
        assert (left.level, left.work_spent, left.work_remaining, left.episode) == (
            right.level,
            right.work_spent,
            right.work_remaining,
            right.episode,
        )
        for old, new in zip(left.contexts, right.contexts):
            assert old.meaning.metadata() == new.meaning.metadata()
            torch.testing.assert_close(old.meaning.roles, new.meaning.roles)
    restored.return_thought(_meaning(3, question=False), b=0)
    assert restored.thought_state(b=0).level == 0
    assert restored.thought_state(b=1).work_spent == 0


def test_episode_gradients_reach_earlier_roles_then_detach_after_step():
    memory = WhatInteractionMemory(capacity=32, detach_mode="episode")
    source = torch.randn(3, 8, requires_grad=True)
    question = ConceptualMeaning(
        source, torch.ones(3, dtype=torch.bool), mode="interrogative"
    )
    memory.begin_thought_episode(question, work_budget=8)
    child = replace(question, roles=question.roles * 2)
    memory.descend_thought(child)
    carried = memory.thought_state().contexts[-1].meaning
    result = replace(carried, roles=carried.roles * 3, mode="assertive")
    memory.return_thought(result)
    memory.finish_thought(result)
    memory.thought_history()[-1].meaning.roles.square().sum().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    memory.end_what_episode()
    assert all(
        r.meaning is None or not r.meaning.roles.requires_grad
        for r in memory.thought_history()
    )


def test_active_history_cannot_be_trimmed_or_silently_reinterpreted_as_parity():
    memory = WhatInteractionMemory(capacity=7)
    root = _meaning(scope={"keep": "root"})
    memory.begin_thought_episode(root, work_budget=50)
    memory.descend_thought(_meaning(1, scope={"keep": "child"}))
    for i in range(2):
        memory.commit_thought(_meaning(i + 2), operation="refine")
    before = memory.thought_history()
    with pytest.raises(OverflowError, match="capacity|active"):
        memory.commit_thought(_meaning(8), operation="refine")
    assert memory.thought_history() == before
    assert memory.thought_state().contexts[0].meaning.scope == root.scope
    memory.cutoff_thought(reason="memory_capacity")
    memory.return_thought(_meaning(9, question=False))
    memory.finish_thought(_meaning(10, question=False))
    assert memory.thought_state().finished
