"""A headless grammar result is its own value, never identity zero's alias."""
from types import SimpleNamespace

import torch

import STMChunks as chunks
from ChunkJournal import IDENTITY, empty_scope
from chunk_fixtures import Walk, choice


def composed(*, head=0):
    walk = Walk((SimpleNamespace(method_name='conjunction', head_role=head),), ())
    prior = torch.tensor([[.9, .8]])
    stm = chunks.empty(1, 8, 2)
    stm, old = chunks.deposit(stm, prior, torch.tensor([True]), identity=0,
        refs=torch.tensor([[[chunks.RELATION, 7], [0, 0]]]), order=0)
    state = walk.empty(prior, 8, state=stm)
    left, right = torch.tensor([[.2, .3]]), torch.tensor([[.4, .1]])
    state = walk.shift(state, left, word=11, index=0, order=0)
    state = walk.shift(state, right, word=12, index=1, order=0)
    result = torch.maximum(left, right)
    state = walk.apply(state, choice(result, head=head, order=0),
                       torch.stack((left, right), 1), 0)
    return walk, state, result, old


def test_headless_composition_has_no_identity():
    _, state, _, _ = composed()
    index = chunks.at(state[0])
    assert chunks.gather(state[0][chunks.IDENTITY], index[:, None]).item() == -1
    assert state[1][0, 0, IDENTITY].item() == -1
    # The scope's other nonzero empty sentinel is its absent clause reference.
    assert empty_scope(state[0][0])[0, 0, 1].item() == -1
    assert state[1][0, 0, 1].item() == -1


def test_refresh_returns_new_headless_result_beside_previous_root():
    _, state, expected, _ = composed()
    _, actual = chunks.refresh(state[0], chunks.at(state[0])[:, None])
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)


def test_three_word_composition_reads_its_headless_operand():
    walk, state, first, _ = composed()
    third = torch.tensor([[.1, .6]])
    state = walk.shift(state, third, word=13, index=2, order=0)
    stm, operands = chunks.refresh(state[0], chunks.stack_indices(state[0])[:, :2])
    state = (stm, *state[1:])
    # The later operation consumes the actual production refresh outputs.
    result = torch.maximum(operands[:, 1], operands[:, 0])
    state = walk.apply(state, choice(result, head=0, order=0),
                       torch.stack((operands[:, 1], operands[:, 0]), 1), 1)
    recorded = state[3][0, 1].reshape(3, 2)
    torch.testing.assert_close(recorded[0], first[0], rtol=0, atol=0)
    torch.testing.assert_close(recorded[2], torch.maximum(first, third)[0], rtol=0, atol=0)


def test_bind_menu_excludes_kept_headless_root():
    _, state, result, _ = composed()
    top = chunks.at(state[0])
    stm = chunks.keep_close(state[0], torch.tensor([True]),
        refs=torch.tensor([[[chunks.RELATION, 8], [0, 0]]]))
    _, _, indices, available = chunks.bind_menu(stm, result[:, None])
    assert not bool(available[indices == top[:, None, None]].any())


def test_headless_result_does_not_inherit_left_identity_with_off_stack_field():
    """§50 item103: inheriting the left's identity is also an invalid repair."""
    walk = Walk((SimpleNamespace(method_name='conjunction', head_role=0),), ())
    left, right = torch.tensor([[.2, .3]]), torch.tensor([[.4, .1]])
    state = walk.empty(left, 8)
    state = walk.shift(state, left, word=11, index=0, order=1)
    identity = state[1][:, 0, IDENTITY].clone()
    assert identity.item() >= 0
    field = torch.tensor([[.9, .8]])
    stm, field_index = chunks.deposit(state[0], field, torch.tensor([True]),
        identity=identity, refs=torch.tensor([[[chunks.RELATION, 7], [0, 0]]]), order=1)
    assert chunks.gather(stm[chunks.STACK], field_index[:, None]).item() == -1
    state = (stm, *state[1:])
    state = walk.shift(state, right, word=12, index=1, order=0)
    result = torch.maximum(left, right)
    state = walk.apply(state, choice(result, head=0, order=0),
                       torch.stack((left, right), 1), 0)
    top = chunks.at(state[0])
    assert state[1][0, 0, IDENTITY].item() == -1
    assert chunks.gather(state[0][chunks.IDENTITY], top[:, None]).item() == -1
    _, refreshed = chunks.refresh(state[0], top[:, None])
    torch.testing.assert_close(refreshed[:, 0], result, rtol=0, atol=0)
    kept = chunks.keep_close(state[0], torch.tensor([True]),
        refs=torch.tensor([[[chunks.RELATION, 8], [0, 0]]]))
    _, _, indices, available = chunks.bind_menu(kept, result[:, None])
    assert not bool(available[indices == top[:, None, None]].any())
