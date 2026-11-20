"""A thought history replayed from an earlier replay of its prefix gives the
state a replay from its first record gives (item 1.75)."""
import pytest
from Layers import WhatInteractionMemory
import Thoughts
from test_levelled_thought_history import _meaning


def _from_the_start(records):
    kept = list(Thoughts._REPLAYED)
    Thoughts._REPLAYED.clear()
    try:
        return Thoughts.replay_thoughts(records)
    finally:
        Thoughts._REPLAYED.clear()
        Thoughts._REPLAYED.extend(kept)


def test_each_step_of_a_history_replays_as_from_the_start():
    memory = WhatInteractionMemory(capacity=64)
    steps = (lambda: memory.begin_thought_episode(_meaning(scope={"place": "root"}), work_budget=20),
             lambda: memory.descend_thought(_meaning(1, scope={"place": "child"})),
             lambda: memory.commit_thought(_meaning(2), operation="refine"),
             lambda: memory.descend_thought(_meaning(3)),
             lambda: memory.return_thought(_meaning(4, question=False), support_true=.6, support_false=.2),
             lambda: memory.return_thought(_meaning(5, question=False), support_true=.4),
             lambda: memory.commit_thought(_meaning(6), operation="question"),
             lambda: memory.finish_thought(_meaning(7, question=False), support_true=0, support_false=0))
    Thoughts._REPLAYED.clear()
    for step in steps:
        step()
        history = memory.thought_history()
        assert memory.thought_state() == _from_the_start(history)
        # Twice: the second is read whole from the earlier replay.
        assert memory.thought_state() == _from_the_start(history)
    assert memory.thought_state().finished


def test_a_bad_record_after_a_replayed_prefix_is_still_refused():
    memory = WhatInteractionMemory(capacity=64)
    memory.begin_thought_episode(_meaning(), work_budget=8)
    memory.commit_thought(_meaning(1), operation="refine")
    history = memory.thought_history()
    Thoughts.replay_thoughts(history)
    with pytest.raises(ValueError, match="chronological"):
        Thoughts.replay_thoughts(list(history) + [history[-1]])
