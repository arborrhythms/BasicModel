"""A word is stable only after its evidence dependencies are settled."""
from types import SimpleNamespace
import torch

from test_attention_thinking_steps import original_attention, exact


def first_legal(*, fields=False):
    def attend(keys, legal, space, **kwargs):
        flat = legal.flatten(1)
        indices = torch.arange(flat.shape[1], device=keys.device)
        priority = -indices + ((indices % 6 >= 3).long()*flat.shape[1] if fields else 0)
        action = priority[None].expand_as(flat).masked_fill(~flat, -10**6).argmax(-1)
        action = torch.where(flat.any(-1), action, flat.shape[1])
        return action, keys.new_ones(len(keys)), flat.sum(-1)>1, dict(
            probability=keys.new_ones(len(keys)), alternative_count=(flat.sum(-1)-1).clamp_min(0))
    return SimpleNamespace(attention_operations=(0, 1, 2, 4) if fields else (0, 1, 2), attend=attend)


def walk(*, field=False, repeated=False):
    import Attention
    chooser = first_legal(fields=field)
    keys = torch.tensor([[[.25, 0.], [0., .5]]])
    spans = torch.tensor([[[0, 3], [4, 7]]])
    known = torch.tensor([[True, True]])
    poles = torch.tensor([[[0., .25] if repeated else [.25, 0.], [0., 0.]]])
    options = dict(budget=32, poles=poles,
                   identities=torch.tensor([[0, 0]]) if repeated else torch.tensor([[0, 1]]))
    wanted = original_attention().narrow_words(chooser, keys, spans, known, **options)
    state = Attention.open_narrowing(chooser, keys, spans, known, **options)
    return state, wanted


def test_isolated_accepted_word_can_be_read_before_the_walk_finishes():
    import Attention
    state, wanted = walk()
    needed = torch.tensor([[True, False]])
    Attention.advance_narrowing(state, needed)
    assert state.steps_taken < 32 and not state.accepted[0, 1]
    assert Attention.narrowing_stable(state, needed).all()
    reading = Attention.close_narrowing(state)
    exact(reading.poles[:, 0], wanted.poles[:, 0])
    exact(reading.accepted[:, 0], wanted.accepted[:, 0])
    Attention.finish_narrowing(state)
    exact(Attention.close_narrowing(state), wanted)


def test_earlier_field_scope_waits_for_the_later_words_descent():
    import Attention
    state, wanted = walk(field=True)
    while not state.accepted[0, 0]:
        Attention.narrow_step(state)
    needed = torch.tensor([[True, False]])
    assert not Attention.narrowing_stable(state, needed).any()
    assert not torch.equal(Attention.close_narrowing(state).poles[:, 0], wanted.poles[:, 0])
    Attention.advance_narrowing(state, needed)
    exact(Attention.close_narrowing(state).poles[:, 0], wanted.poles[:, 0])
    Attention.finish_narrowing(state)
    exact(Attention.close_narrowing(state), wanted)


def test_another_occurrence_can_change_an_already_accepted_identity():
    import Attention
    state, wanted = walk(repeated=True)
    while not state.accepted[0, 0]:
        Attention.narrow_step(state)
    needed = torch.tensor([[True, False]])
    assert not Attention.narrowing_stable(state, needed).any()
    assert not torch.equal(Attention.close_narrowing(state).poles[:, 0], wanted.poles[:, 0])
    Attention.advance_narrowing(state, needed)
    exact(Attention.close_narrowing(state).poles[:, 0], wanted.poles[:, 0])
    Attention.finish_narrowing(state)
    exact(Attention.close_narrowing(state), wanted)
