"""Legacy adapters cannot discard or corrupt the ordinary history owner."""

from dataclasses import replace

import pytest
import torch

from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning
from What import LTMSlot


def _meaning():
    return ConceptualMeaning(torch.ones(3, 8, requires_grad=True), torch.ones(3, dtype=torch.bool), mode="interrogative")


def test_legacy_credit_begin_cannot_replace_an_active_ordinary_episode():
    memory = WhatInteractionMemory(capacity=32, detach_mode="episode")
    memory.begin_thought_episode(_meaning(), work_budget=4)
    with pytest.raises(RuntimeError, match="ordinary|episode"):
        memory.begin_what_episode()
    assert len(memory._episode_live[0]) == 1


def test_legacy_credit_reservation_can_end_before_forward_establishes_batch():
    memory = WhatInteractionMemory(batch=1, capacity=32, detach_mode="episode")
    memory.begin_what_episode(b=1)
    assert memory.in_episode(b=1)
    assert memory.end_what_episode(b=1) == 0
    assert not memory.in_episode(b=1)


def test_legacy_adapter_detaches_complete_meanings_at_the_credit_boundary():
    memory = WhatInteractionMemory(capacity=32, detach_mode="episode")
    memory.begin_what_episode()
    meaning = _meaning()
    memory.append_what_slot(LTMSlot(input=meaning, output=meaning))
    assert memory.get_what_slots()[0].input.roles.requires_grad
    memory.end_what_episode()
    assert not memory.get_what_slots()[0].input.roles.requires_grad
    assert meaning.roles.requires_grad


def test_checkpoint_snapshot_detaches_legacy_values_without_detaching_the_live_owner():
    memory = WhatInteractionMemory(capacity=32, detach_mode="episode")
    memory.begin_what_episode()
    value = torch.ones(8, requires_grad=True)
    memory.append_what_slot(LTMSlot(input=value, closure_pressure=3.5))
    extra = memory.thought_extras()
    assert not extra["rows"][0][0]["legacy"].input.requires_grad
    assert memory.get_what_slots()[0].input.requires_grad


def test_legacy_checkpoint_replay_preserves_open_context_pressure():
    source = WhatInteractionMemory(capacity=32)
    source.append_what_slot(LTMSlot(input=torch.ones(8), closure_pressure=3.5))
    restored = WhatInteractionMemory(capacity=32)
    restored.load_thought_extras(source.thought_extras())
    assert restored.what_context()["closure_pressure"] == 3.5
    assert restored.what_open_depth() == 1


def test_legacy_checkpoint_return_underflow_is_rejected_atomically():
    source = WhatInteractionMemory(capacity=32)
    source.append_what_slot(LTMSlot(input=torch.ones(8)))
    extra = source.thought_extras()
    extra["rows"][0][0]["legacy"] = LTMSlot(output=torch.ones(8))
    restored = WhatInteractionMemory(capacity=32)
    restored.append_what_slot(LTMSlot(input=torch.zeros(8)))
    before = restored.get_what_slots()
    with pytest.raises(ValueError, match="open|underflow"):
        restored.load_thought_extras(extra)
    assert restored.get_what_slots() == before


def test_nonzero_return_support_requires_its_complete_proposition():
    memory = WhatInteractionMemory(capacity=32)
    memory.begin_thought_episode(_meaning(), work_budget=8)
    memory.descend_thought(_meaning())
    before = memory.thought_history()
    with pytest.raises(ValueError, match="proposition|meaning"):
        memory.return_thought(support_true=0.5)
    assert memory.thought_history() == before
