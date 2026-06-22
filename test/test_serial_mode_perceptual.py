"""PartSpace serial_mode invariants.

The tests exercise the slide-and-recompute fast path via
full-model forward() calls so the text-mode lex/embed flow
populates the subspaces as the pipeline expects.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

from data import TheData
from Models import BaseModel

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _xor_input():
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ).float().unsqueeze(1)


def _build(serial_mode):
    TheData.load("xor")

    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    m.serial_mode = serial_mode
    m.perceptualSpace.serial_mode = serial_mode
    m.conceptualSpace.serial_mode = serial_mode
    return m


def test_serial_mode_does_not_break_forward():
    """Setting serial_mode=True keeps forward() working on XOR input."""
    m = _build(serial_mode=True)
    out = m.forward(_xor_input())
    assert out is not None
    assert len(out) == 4


def test_serial_mode_matches_non_serial():
    """Serial-mode forward on non-AR produces close (not bit-identical)
    prediction as non-serial.

    Updated 2026-05-20: post-bivector-retirement the serial and non-
    serial paths use different fold orderings in the recurrent cell;
    relax the tolerance from 5e-2 to 1e-1 to absorb the per-batch
    float-ordering drift while still catching gross divergence.

    Updated 2026-05-29: clean-stack STM + LSE soft-max (tau*log(2)
    ≈ 0.069 per stage) widens the drift; bumped atol 1e-1 → 2e-1."""
    m_s = _build(serial_mode=True)
    m_b = _build(serial_mode=False)
    inp = _xor_input()
    out_s = m_s.forward(inp)
    out_b = m_b.forward(inp)
    if isinstance(out_s[2], torch.Tensor) and isinstance(out_b[2], torch.Tensor):
        assert torch.allclose(out_s[2], out_b[2], atol=2e-1), (
            f"serial vs non-serial diverged: max |diff| = "
            f"{(out_s[2] - out_b[2]).abs().max().item()}"
        )


def test_perceptual_serial_mode_flag_propagates():
    m = _build(serial_mode=True)
    assert m.perceptualSpace.serial_mode is True
    m2 = _build(serial_mode=False)
    assert m2.perceptualSpace.serial_mode is False


def test_serial_cache_populated_by_cold_forward():
    """First serial_mode forward (cold) should populate subspace.serial_cache."""
    m = _build(serial_mode=True)
    ps = m.perceptualSpace
    assert ps.subspace.serial_cache.get(id(ps)) is None
    m.forward(_xor_input())
    # Either cold populated cache, or nothing flowed (would mean the
    # upstream subspace was empty and we never reached the cache write).
    assert ps.subspace.serial_cache.get(id(ps)) is not None, (
        "serial_mode cold forward should populate subspace.serial_cache")


def test_serial_cache_cleared_by_reset():
    """Reset() clears subspace.serial_cache so the next forward is cold."""
    m = _build(serial_mode=True)
    ps = m.perceptualSpace
    m.forward(_xor_input())
    assert ps.subspace.serial_cache.get(id(ps)) is not None
    ps.Reset()
    assert ps.subspace.serial_cache.get(id(ps)) is None


def test_warm_path_skips_slot_forward_embed():
    """Warm path calls _slot_forward (codebook on 1 slot) instead of _embed
    over the full buffer.

    We intercept _slot_forward and the codebook forward call to count how
    many positions get quantized on the second call (should be 1, not N).
    """
    m = _build(serial_mode=True)
    ps = m.perceptualSpace
    # First call: cold. Populate cache with a plausible [B, N, D] tensor.
    import torch as _t
    B, N, D = 2, 4, int(ps.outputShape[1])

    # Craft an upstream subspace whose materialize() matches the cache
    # shape so the warm path's shape guard accepts it. Seed the cache on
    # the upstream subspace -- copy_context propagates serial_cache by
    # reference into ps.subspace at forward() entry.
    from Spaces import SubSpace
    from Layers import Error
    class _FakeSubspace:
        def __init__(self, t):
            self._t = t
            self.symbolSpace = None
            self.errors = Error()
            self.serial_cache = {}
            # Microbatch-AR routing attrs propagated by copy_context.
            self.k_axis = False
            self.valid_mask = None
            self.stem_embedded = False
        def is_empty(self):
            return False
        def materialize(self):
            return self._t
        _demuxed = False
        _index = None
    upstream = _FakeSubspace(_t.randn(B, N, D))
    upstream.serial_cache[id(ps)] = _t.zeros(B, N, D)

    # Count _slot_forward calls and their input shape.
    calls = []
    orig_slot = ps._slot_forward
    def _spy(x, quantize=True):
        calls.append(tuple(x.shape))
        return orig_slot(x, quantize=quantize)
    ps._slot_forward = _spy

    result = ps.forward(upstream)
    # Warm path took: one _slot_forward on [B, 1, D].
    assert len(calls) == 1, f"expected 1 slot_forward call, got {len(calls)}"
    assert calls[0] == (B, 1, D), f"slot shape {calls[0]} != ({B}, 1, {D})"
    assert result is ps.subspace
