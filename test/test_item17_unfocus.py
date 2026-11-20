"""A walk does not take an unfocus that returns it to a state it has been in.

The state is the walk's brackets (intervals, valid, space, level, done), the
words accepted and descended, and each bracket's field record, whatever the
walk has spent. (a) At the step right after a descend that splits off a word
not yet known, unfocus is not among the legal actions and descend is. (b)
After a descend and field reductions on what it split off, which change
nothing the undo keeps, the unfocus is not legal either. (c) A walk that
prefers the unfocus wherever it is legal never comes back to a state. The rule
is provisional: attention's bracket table is replaced in round 6. That no
first walk of the lesson repeats a state of its table is measured over runs by
``attention_thinking_certificate.py lesson --runs N`` and printed there.
"""
import torch

A = 6
DESCEND, UNFOCUS = 1, 2


def _walk():
    from Language import OperationSelectionLayer
    import Attention
    chooser = OperationSelectionLayer(d_model=4)
    chooser.attention_operations = (0, 1, 2)
    keys = torch.rand(1, 3, 4)
    spans = torch.tensor([[[0, 3], [4, 6], [7, 12]]])
    known = torch.zeros(1, 3, dtype=torch.bool)
    return Attention, Attention.open_narrowing(chooser, keys, spans, known, budget=32)


def _legal(Attention, walk):
    return Attention.narrowing_view(walk)[8][0]


def test_no_unfocus_right_after_a_descend_onto_an_unknown_word():
    Attention, walk = _walk()
    # The sentence's bracket is slot 0; a descend splits off its first word.
    assert bool(_legal(Attention, walk)[0, DESCEND])
    Attention.narrow_step(walk, lesson_action=torch.tensor([0 * A + DESCEND]))
    legal = _legal(Attention, walk)
    frame = walk.focus_frames[0, 0]
    parent, child = int(frame[0]), int(frame[1])
    assert child >= 0 and parent == 0
    # The child holds one word that is not known: neither, so descend is
    # legal on it; the unfocus that would restore the sentence is not legal
    # anywhere at this step.
    assert bool(legal[child, DESCEND])
    assert not bool(legal[:, UNFOCUS].any())


class _Prefers:
    """Chooses the unfocus wherever it is legal, then a field reduction, then
    a divide or descend: the order in which a walk would loop."""
    attention_operations = tuple(range(A))
    forced = None

    def attend(self, keys, legal, space, **options):
        flat = legal.flatten(1)[0]
        if self.forced is not None:
            assert bool(flat[self.forced])
            return (torch.tensor([self.forced]), torch.ones(1), torch.tensor([False]),
                    {'probability': torch.ones(1), 'alternative_count': torch.zeros(1)})
        order = [UNFOCUS, 3, 4, 5, DESCEND, 0]
        for op in order:
            slots = legal[0, :, op].nonzero()
            if len(slots):
                action = int(slots[0, 0]) * A + op
                break
        else:
            action = int(flat.long().argmax())
        return (torch.tensor([action]), torch.ones(1), torch.tensor([False]),
                {'probability': torch.ones(1), 'alternative_count': torch.zeros(1)})


def _unknown_walk(chooser):
    import Attention
    keys = torch.rand(1, 3, 4)
    spans = torch.tensor([[[0, 3], [4, 6], [7, 12]]])
    known = torch.zeros(1, 3, dtype=torch.bool)
    return Attention, Attention.open_narrowing(chooser, keys, spans, known, budget=32)


def test_no_unfocus_after_reductions_that_the_undo_does_not_keep():
    chooser = _Prefers()
    Attention, walk = _unknown_walk(chooser)
    chooser.forced = 0 * A + DESCEND
    Attention.narrow_step(walk)
    child = int(walk.focus_frames[0, 0, 1])
    # The child holds two words: the field reductions are legal on it.
    assert bool(_legal(Attention, walk)[child, 3])
    for op in (3, 4):
        chooser.forced = child * A + op
        Attention.narrow_step(walk)
    # Two steps after the descend: the unfocus would clear the child's
    # reductions and restore the sentence, the state the walk began in.
    assert walk.steps_taken == 3
    assert not bool(_legal(Attention, walk)[:, UNFOCUS].any())


def test_a_walk_that_prefers_the_unfocus_comes_back_to_no_state():
    Attention, walk = _unknown_walk(_Prefers())
    # As the loop of looks takes it: grammar interprets a pure word at the
    # look, then attention steps, until nothing is legal.
    slot = torch.zeros(1, dtype=torch.long)
    while walk.steps_taken < 32:
        _, allowed = Attention.interpretation_at(walk, slot)
        Attention.record_interpretation(walk, slot, allowed)
        if not bool(_legal(Attention, walk).any()):
            break
        Attention.narrow_step(walk)
        slot = walk.look_slot
    steps = walk.steps_taken
    states = walk.visited[0, :steps]
    assert steps < 32
    assert len({tuple(row.tolist()) for row in states}) == steps
