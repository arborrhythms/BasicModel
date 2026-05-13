"""Tests for the Stage 2 Task 1 unified outer loop.

The unified outer loop lives in ``BasicModel.forward`` and handles the
flat (non-grammar) configuration. ``useGrammar="all"`` still uses its
specialized path.

These tests verify:
* ``SymbolicSpace.empty_state`` is callable and shape-correct -- the
  unified loop's seed for ``ss``.
* The j-loop runs ``conceptualOrder`` times (via
  ``_unified_j_iterations`` counter).
* ``conceptualOrder==0`` -> zero j-iterations + a single pre-seed C->S
  pass (spec's implicit j=-1); concepts/symbols are still populated.
"""
import os
import warnings

import torch

from Spaces import SymbolicSpace


def test_symbolicspace_empty_state_is_callable():
    """Used to seed ``ss`` in the unified loop."""
    assert callable(getattr(SymbolicSpace, "empty_state", None))


def test_symbolicspace_empty_state_shape():
    """empty_state returns zeros of shape [batch, nOutput, nDim]."""
    space = SymbolicSpace.__new__(SymbolicSpace)
    space.outputShape = (5, 7)
    state = space.empty_state(batch=3)
    assert tuple(state.shape) == (3, 5, 7)
    assert state.abs().sum().item() == 0.0


def _load_mental_model(conceptualOrder: int = 1):
    """Build a BasicModel from MentalModel.xml with the requested
    conceptualOrder, via an XML patch."""
    import xml.etree.ElementTree as ET
    import tempfile
    import Models

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    arch = root.find("architecture")
    co = arch.find("conceptualOrder")
    if co is None:
        co = ET.SubElement(arch, "conceptualOrder")
    co.text = str(conceptualOrder)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tree.write(tmp.name)
    tmp.close()

    model, _ = Models.BasicModel.from_config(tmp.name)
    return model, tmp.name


def _run_single_batch(model):
    """Run one forward pass on two trivial sentences."""
    import Models
    sentences = ['the cat sat on the mat', 'a dog chased the ball']
    outputs = [torch.tensor([0.0]), torch.tensor([1.0])]
    with Models.TheData.runtime_batch(sentences, outputs), \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Range violation")
        warnings.filterwarnings("ignore", message="PiLayer.reverse")
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:2])
        model.eval()
        model.set_sigma(0)
        with torch.no_grad():
            model.forward(x)


def test_unified_loop_runs_conceptualorder_iterations():
    """conceptualOrder=3 -> j-loop fires three times."""
    model, path = _load_mental_model(conceptualOrder=3)
    try:
        _run_single_batch(model)
        assert model._unified_j_iterations == 3
    finally:
        os.unlink(path)


def test_unified_loop_conceptualorder_zero_pre_seed_only():
    """conceptualOrder=0 -> zero j-iterations + a single pre-seed
    C->S pass. concepts/symbols are still populated so OutputSpace has
    state to consume."""
    model, path = _load_mental_model(conceptualOrder=0)
    try:
        _run_single_batch(model)
        assert model._unified_j_iterations == 0
        assert model.concepts is not None
        assert model.symbols is not None
    finally:
        os.unlink(path)
