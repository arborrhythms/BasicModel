"""Durable history must not keep autograd through legacy question metadata."""

from dataclasses import replace

import torch

from Layers import WhatInteractionMemory
from What import LTMSlot, WhatQuestion


def _live_memory():
    memory = WhatInteractionMemory(capacity=32, detach_mode="episode")
    memory.begin_what_episode()
    prompt = torch.ones(8, requires_grad=True)
    trace = prompt * 2
    question = WhatQuestion.inference(0, prompt={"meaning": prompt})
    memory.append_what_slot(LTMSlot(input=prompt, output=trace, question=question, grammar_trace=({"activation": trace},)))
    return memory, prompt


def test_checkpoint_question_prompt_is_detached_without_cutting_live_credit():
    memory, prompt = _live_memory()
    snapshot = memory.thought_extras()["rows"][0][0]["legacy"]
    assert not snapshot.question.prompt["meaning"].requires_grad
    assert memory.get_what_slots()[0].question.prompt["meaning"].requires_grad
    with torch.no_grad():
        prompt.add_(3)
    assert torch.equal(snapshot.question.prompt["meaning"], torch.ones(8))


def test_restoring_a_legacy_question_detaches_its_prompt():
    memory, prompt = _live_memory()
    extra = memory.thought_extras()
    raw = extra["rows"][0][0]["legacy"]
    extra["rows"][0][0]["legacy"] = replace(raw, question=WhatQuestion.inference(0, prompt=prompt))
    restored = WhatInteractionMemory(capacity=32)
    restored.load_thought_extras(extra)
    assert not restored.get_what_slots()[0].question.prompt.requires_grad
    assert prompt.requires_grad


def test_episode_end_releases_question_and_trace_credit_together():
    memory, prompt = _live_memory()
    memory.end_what_episode()
    stored = memory.get_what_slots()[0]
    assert not stored.question.prompt["meaning"].requires_grad
    assert not stored.grammar_trace[0]["activation"].requires_grad
    assert prompt.requires_grad
