"""A walk does not take the unfocus that would undo the focus it has just made.

Certificate (a) of the fourth pass of round 4 of item 1.7: at the step right
after a descend that splits off a word not yet known, unfocus is not among the
legal actions and descend is. The rule is provisional: attention's bracket
table is replaced in round 6. Certificate (b), that no first walk of the
lesson repeats a state of its table, is measured over runs by
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
