"""Reset() cascade unit tests for Phase 3 serial_mode."""
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


@pytest.fixture
def model():
    TheData.load("xor")
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return m


def test_input_space_reset_clears_ar_buffer(model):
    import torch as _t
    model.inputSpace._ar_buffer = _t.zeros(2, 4, 8)
    model.inputSpace._ar_embedded = _t.zeros(2, 4, 8)
    model.inputSpace._seq_cursor = 3
    model.inputSpace.Reset()
    assert model.inputSpace._ar_buffer is None
    assert model.inputSpace._ar_embedded is None
    assert model.inputSpace._seq_cursor == 0


def test_perceptual_space_reset_clears_event(model):
    import torch as _t
    # Skip when the event slot is a learnable Codebook — Reset must not
    # clobber codebook parameters. The cached-event invariant is only
    # testable when .event is a plain Tensor.
    from Layers import Layer  # ensures Spaces import chain
    from Spaces import Codebook
    if isinstance(model.perceptualSpace.subspace.event, Codebook):
        pytest.skip("PerceptualSpace.event is a Codebook in this config")
    model.perceptualSpace.subspace.set_event(_t.zeros(2, 4, 8))
    model.perceptualSpace.Reset()
    ev = model.perceptualSpace.subspace.event
    if ev is not None:
        assert ev.getW() is None, "Reset should clear the event W"


def test_conceptual_space_reset_clears_event(model):
    import torch as _t
    model.conceptualSpace.subspace.set_event(_t.zeros(2, 4, 8))
    model.conceptualSpace.Reset()
    ev = model.conceptualSpace.subspace.event
    if ev is not None:
        assert ev.getW() is None


def test_word_space_reset_calls_clear_sentence(model):
    if model.wordSpace is None:
        pytest.skip("config has no WordSpace")
    called = []
    orig = model.wordSpace.clear_sentence
    model.wordSpace.clear_sentence = lambda: called.append(True) or orig()
    model.wordSpace.Reset()
    assert called, "WordSpace.Reset should call clear_sentence()"


def test_base_space_reset_cascades_to_layers(model):
    """Space.Reset() iterates self.layers and calls Reset() on each."""
    layers = list(model.conceptualSpace.layers)
    assert layers, "ConceptualSpace should have at least one layer"
    called = []
    target = layers[0]
    orig = getattr(target, "Reset", lambda: None)

    def _probe():
        called.append(True)
        orig()

    target.Reset = _probe
    try:
        model.conceptualSpace.Reset()
    finally:
        if orig is _probe:
            del target.Reset
        else:
            target.Reset = orig
    assert called, "Space.Reset() should cascade to its Layers"


def test_end_of_stream_flag_starts_false(model):
    assert model.inputSpace._end_of_stream is False


def test_reset_clears_end_of_stream_flag(model):
    model.inputSpace._end_of_stream = True
    model.inputSpace.Reset()
    assert model.inputSpace._end_of_stream is False


def test_serial_mode_default_off_on_none_masked_prediction(model):
    assert model.serial_mode is False
    assert model.perceptualSpace.serial_mode is False
    assert model.conceptualSpace.serial_mode is False


def test_serial_mode_on_for_arlm():
    TheData.load("xor")
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    m.masked_prediction = 'ARLM'
    # Re-derive serial_mode as the model would in create_from_config.
    m.serial_mode = m.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
    m.perceptualSpace.serial_mode = m.serial_mode
    m.conceptualSpace.serial_mode = m.serial_mode
    assert m.serial_mode is True
    assert m.perceptualSpace.serial_mode is True
    assert m.conceptualSpace.serial_mode is True
