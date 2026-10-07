"""Legacy adapters cannot discard or corrupt the ordinary history owner."""

from dataclasses import replace

import pytest
import torch

from Layers import WhatInteractionMemory
from Meaning import ConceptualMeaning
from What import LTMSlot


def _meaning():
    return ConceptualMeaning(torch.ones(3, 8, requires_grad=True), torch.ones(3, dtype=torch.bool), mode="interrogative")










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
