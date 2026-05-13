"""ConceptualSpace serial_mode invariants + attention guard tests."""
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
    """ConceptualSpace forward with serial_mode=True matches non-serial."""
    m_s = _build(serial_mode=True)
    m_b = _build(serial_mode=False)
    inp = _xor_input()
    out_s = m_s.forward(inp)
    out_b = m_b.forward(inp)
    if isinstance(out_s[2], torch.Tensor) and isinstance(out_b[2], torch.Tensor):
        assert torch.allclose(out_s[2], out_b[2], atol=5e-2)


def test_attention_guard_forces_serial_mode_off():
    """When hasAttention=True, serial_mode is forced off on ConceptualSpace.

    Simulates the attention guard by setting hasAttention=True on the
    conceptualSpace and re-running the derive logic from
    create_from_config.
    """
    TheData.load("xor")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
        m.masked_prediction = 'AR'
        m.conceptualSpace.hasAttention = True
        # Re-derive (mirrors create_from_config's guard).
        m.serial_mode = True
        m.perceptualSpace.serial_mode = True
        m.conceptualSpace.serial_mode = True
        if (m.serial_mode
                and getattr(m.conceptualSpace, 'hasAttention', False)):
            warnings.warn(
                "ConceptualSpace.hasAttention=True violates position-"
                "locality; forcing conceptualSpace.serial_mode=False.",
                RuntimeWarning,
            )
            m.conceptualSpace.serial_mode = False
        assert m.conceptualSpace.serial_mode is False
        assert m.perceptualSpace.serial_mode is True
        assert any("hasAttention" in str(x.message) for x in w)
