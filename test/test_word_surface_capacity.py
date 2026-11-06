"""Word bytes have a construction bound independent of sentence length."""
from types import SimpleNamespace

import pytest
import torch

from WordSurfaces import capacity, primitive_support, stage


@pytest.mark.parametrize('value', [0, -1, True, 2.5, '2.5', None])
def test_surface_capacity_rejects_nonpositive_or_nonintegral_values(value):
    with pytest.raises(ValueError, match='wordSurfaceCapacity'):
        capacity(value)


def test_fixed_surface_bytes_have_a_separate_eos_reserve_and_compact_support():
    for raw in (b'a', b'a' * 64):
        data, valid, present, support = stage([[raw, None]], 64, device='cpu', eos=True)
        assert data.shape == valid.shape == (1, 2, 65)
        assert data.dtype == torch.uint8 and valid.dtype == torch.bool
        assert int(valid.sum()) == len(raw) and not valid[..., -1].any()
        assert bytes(data[valid].tolist()) == raw
        assert present.tolist() == [[True, False]]
        torch.testing.assert_close(support, primitive_support(data, valid))
        compact = stage([[raw, None]], 64, device='cpu', eos=True, retain_bytes=False)
        assert compact[:2] == (None, None)
        torch.testing.assert_close(compact[2], present)
        torch.testing.assert_close(compact[3], support)
    for retain in (False, True):
        with pytest.raises(ValueError, match='65 bytes.*wordSurfaceCapacity is 64'):
            stage([[b'a' * 65]], 64, device='cpu', eos=True, retain_bytes=retain)


def test_aligned_snapshot_rejects_overlong_candidates_instead_of_clipping():
    from Models import BasicModel
    surfaces = {1: b'a' * 65}
    owner = SimpleNamespace(word_surface_capacity=64,
        inputSpace=SimpleNamespace(_ar_concept_lookup_rows=torch.tensor([[1]])),
        _concept_owner=lambda: SimpleNamespace(word_surface_for_row=surfaces.get))
    with pytest.raises(ValueError, match='65 bytes.*wordSurfaceCapacity is 64'):
        BasicModel._stage_snapshot_bytes(owner)


def test_byte_cost_and_gradient_are_unchanged_with_uint8_storage():
    from Models import BasicModel
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    data, valid, _, _ = stage([[b'ab', b'abc']], 64, device='cpu', eos=True)
    targets, target_valid, _, _ = stage([[b'ab']], 64, device='cpu')
    results = []
    for dtype in (torch.long, torch.uint8):
        idea = torch.tensor([[.2, .8]], requires_grad=True)
        cost = BasicModel._byte_word_cost(owner, idea, torch.tensor(0), torch.eye(2)[None],
            data.to(dtype), valid, targets.to(dtype), target_valid, True)
        gradient, = torch.autograd.grad(cost.sum(), idea)
        results.append((cost.detach(), gradient))
    for before, after in zip(*results):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_named_part_lesson_still_reads_the_uint8_surface():
    from AttentionLesson import PartLesson
    from SentenceUnderstanding import PrimedSymbols
    data, valid, present, support = stage([[b'ab', b'abc']], 64, device='cpu', eos=True)
    codes = torch.eye(2)[None]
    bank = PrimedSymbols(torch.tensor([[0, 1]]), codes, torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool), data, valid,
        surface_present=present, primitive_support=support)
    error, _ = PartLesson(None, ((b'abc',),)).cost(SimpleNamespace(primed=bank),
        (codes[:, 1:], torch.ones(1, dtype=torch.long)))
    assert error.tolist() == [0.]
