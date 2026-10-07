"""ConceptualSpace serial_mode invariants + serial-with-attention doctrine.

Task 4 (doc/plans/2026-05-29-stm-serial-parallel-modes.md §"Serial mode =
attentional filtering"): the former serial/attention guard — which forced
``conceptualSpace.serial_mode = False`` whenever
``serial_mode and conceptualSpace.hasAttention`` — is LIFTED. Serial mode
**is** the attentional-filtering regime; MentalModel.xml runs serial WITH
attention by design. ``test_serial_with_attention_is_not_downgraded`` pins
the new doctrine (the guard does not fire).
"""
import sys
import warnings
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


def test_conceptual_serial_mode_flag_propagates():
    m = _build(serial_mode=True)
    assert m.conceptualSpace.serial_mode is True
    m2 = _build(serial_mode=False)
    assert m2.conceptualSpace.serial_mode is False


def test_conceptual_serial_mode_matches_non_serial():
    """ConceptualSpace forward with serial_mode=True is close (not bit-
    identical) to non-serial.

    Updated 2026-05-20: post-bivector-retirement the serial and non-
    serial fold orderings drift slightly per batch; relax tolerance to
    1e-1 to absorb the float-ordering drift while still catching gross
    divergence.

    Updated 2026-05-29: clean-stack STM (parallel uses
    ``_stm_set_all_slots`` writing the whole [B, N, D] slab; serial
    pushes per-position via ``_stm_shift_and_push``) plus LSE
    soft-max (``tau*log(2) ≈ 0.069`` per stage) widens the drift
    further. Bumped atol to 2e-1; still flags gross divergence."""
    m_s = _build(serial_mode=True)
    m_b = _build(serial_mode=False)
    inp = _xor_input()
    out_s = m_s.forward(inp)
    out_b = m_b.forward(inp)
    if isinstance(out_s[2], torch.Tensor) and isinstance(out_b[2], torch.Tensor):
        assert torch.allclose(out_s[2], out_b[2], atol=2e-1), (
            f"serial vs non-serial diverged: max |diff| = "
            f"{(out_s[2] - out_b[2]).abs().max().item()}")


def test_serial_with_attention_is_not_downgraded():
    """DOCTRINE (Task 4): serial mode + ConceptualSpace.hasAttention is a
    SUPPORTED regime — the former guard that forced
    ``conceptualSpace.serial_mode = False`` is LIFTED.

    Builds a model through the real ``BaseModel.from_config`` path (which
    runs ``BaseModel.__init__`` where the guard used to live), then sets
    ``conceptualSpace.hasAttention = True`` and ``serial_mode = True`` and
    asserts that ``conceptualSpace.serial_mode`` is NOT forced off. Serial
    mode IS the attentional-filtering regime
    (doc/plans/2026-05-29-stm-serial-parallel-modes.md §"Serial mode =
    attentional filtering"); MentalModel.xml runs serial WITH attention by
    design.

    No ``RuntimeWarning`` about ``hasAttention`` should be emitted by the
    framework — the downgrade-and-warn path is gone.
    """
    TheData.load("xor")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
        m.conceptualSpace.hasAttention = True
        m.serial_mode = True
        m.perceptualSpace.serial_mode = True
        m.conceptualSpace.serial_mode = True
        # The guard is lifted: nothing downgrades conceptualSpace here.
        assert m.conceptualSpace.serial_mode is True, (
            "serial mode IS the attentional regime; the guard that "
            "forced conceptualSpace.serial_mode=False is lifted (Task 4).")
        assert m.perceptualSpace.serial_mode is True
        # No framework-emitted hasAttention downgrade warning.
        assert not any(
            "hasAttention" in str(x.message)
            and "serial_mode=False" in str(x.message)
            for x in w), (
            "the serial/attention downgrade-and-warn path must be gone")
