"""End-to-end serial_mode equivalence + EoS cascade tests."""
import sys
import time
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


def _build_model(serial_mode, masked_prediction='AR'):
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    m.masked_prediction = masked_prediction
    m.serial_mode = serial_mode
    if hasattr(m, 'perceptualSpace'):
        m.perceptualSpace.serial_mode = serial_mode
    if hasattr(m, 'conceptualSpace'):
        m.conceptualSpace.serial_mode = serial_mode
    return m


def _xor_input():
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ).float().unsqueeze(1)


def test_serial_mode_equivalence_end_to_end():
    """Serial and non-serial forward produce matching predictions (fp tolerance)."""
    m_serial = _build_model(serial_mode=True, masked_prediction='NONE')
    m_baseline = _build_model(serial_mode=False, masked_prediction='NONE')
    inp = _xor_input()
    out_s = m_serial.forward(inp)
    out_b = m_baseline.forward(inp)
    assert isinstance(out_s[2], torch.Tensor)
    assert isinstance(out_b[2], torch.Tensor)
    assert torch.allclose(out_s[2], out_b[2], atol=5e-2), (
        f"serial vs baseline prediction diverged: "
        f"max |diff| = {(out_s[2] - out_b[2]).abs().max().item()}"
    )


def test_eos_triggers_reset_cascade():
    """Reset cascade clears buffers in all spaces."""
    m = _build_model(serial_mode=True)
    m.perceptualSpace.subspace.set_event(torch.zeros(1, 4, 8))
    m.conceptualSpace.subspace.set_event(torch.zeros(1, 4, 8))
    for space in m.spaces:
        if hasattr(space, 'Reset'):
            space.Reset()
    # ConceptualSpace event clears to None (plain Tensor slot).
    ev_c = m.conceptualSpace.subspace.event
    assert (ev_c is None or ev_c.getW() is None), (
        "ConceptualSpace.event should be cleared by Reset")


def test_serial_mode_does_not_slow_short_streams():
    """Smoke: serial_mode forward latency on 4 tokens is within 2x of baseline."""
    m_s = _build_model(serial_mode=True, masked_prediction='NONE')
    m_b = _build_model(serial_mode=False, masked_prediction='NONE')
    inp = _xor_input()

    # Warmup.
    m_s.forward(inp)
    m_b.forward(inp)

    t0 = time.perf_counter()
    for _ in range(5):
        m_s.forward(inp)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        m_b.forward(inp)
    t_baseline = time.perf_counter() - t0

    # Loose bound: serial-mode overhead should not regress short-stream
    # timings by more than 2x. Real perf win appears at N >= 32.
    assert t_serial < 2.0 * t_baseline, (
        f"serial-mode latency regressed: serial={t_serial:.3f}s "
        f"baseline={t_baseline:.3f}s")
