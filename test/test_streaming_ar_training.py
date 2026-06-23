"""Tests for the streaming AR training loop refactor.

Plan reference: basicmodel/doc/plans/2026-04-20-streaming-ar-training-loop.md
Spec reference: basicmodel/doc/specs/2026-04-20-streaming-ar-training-loop-design.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

from Layers import Layer


def test_layer_has_start_method():
    """Layer base class exposes a Start() method that is callable."""
    layer = Layer(nInput=4, nOutput=4)
    assert callable(getattr(layer, "Start", None))
    layer.Start()   # no-op at the base, must not raise


def test_layer_start_cascades_to_children():
    """Layer.Start() walks self.layers and calls Start() on each child."""
    parent = Layer(nInput=4, nOutput=4)
    child_a = Layer(nInput=4, nOutput=4)
    child_b = Layer(nInput=4, nOutput=4)
    called = []
    child_a.Start = lambda: called.append('a')
    child_b.Start = lambda: called.append('b')
    parent.layers = [child_a, child_b]
    parent.Start()
    assert called == ['a', 'b']


def test_subspace_has_reset_event():
    """SubSpace.reset_event() clears the cached event tensor."""
    from Spaces import SubSpace
    assert callable(getattr(SubSpace, "reset_event", None))


def test_space_has_start_method():
    """Space base class exposes a Start() method."""
    from Spaces import Space
    assert callable(getattr(Space, "Start", None))


def test_arir_requires_reconstruct_not_none():
    """Retired 2026-05-14: ``<maskedPrediction>`` was retired in the
    IR-only refactor, so there is no ARIR mode whose coupling with
    ``<reconstruct>`` needs validating.  ``<reconstruct>`` is now an
    independent forward-only loss selector.
    """
    return  # retired check; see plan §1


import pytest




def test_mentalmodel_forward_populates_inputs_and_symbolic_state():
    """BasicModel.forward(inputData) populates the InputSpace terminal
    subspace and self.symbolic_state as a side effect — the post-refactor
    replacement for the old Start(inputData) entry point.

    Phase 1.5 subsumed the ``self.inputs`` back-ref alias (it was a
    read-only handle to ``inputSpace.subspace``, stamped by the forward).
    This contract test now asserts the same side effect at its true owner
    (``model.inputSpace.subspace`` materializes after forward) -- the
    forward-populates-input-state intent is unchanged; only the alias
    name it checked moved to the owning Space.
    """
    import warnings

    import torch

    import Models
    import Language

    # Build a minimal BasicModel from MM_xor.xml (small + fast).
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MM_xor.xml")
    Language.TheGrammar._configured = False
    model, _ = Models.BasicModel.from_config(src)

    # Fabricate a valid input tensor via getTrainData.
    Models.TheData.load("xor")
    train_input, _ = model.inputSpace.getTrainData()
    x = model.inputSpace.prepInput(train_input[:2])

    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.forward(x)

    # Phase 1.5: assert the InputSpace terminal subspace materializes
    # post-forward (was: the now-subsumed ``model.inputs`` alias).
    assert model.inputSpace.subspace is not None
    assert model.inputSpace.subspace.materialize() is not None
    assert getattr(model, 'symbolic_state', None) is not None
    # symbolic_state shape: [B, nOutput, nDim] from WholeSpace.outputShape
    sshape = tuple(model.symbolic_state.shape)
    assert len(sshape) == 3
    assert sshape[0] == x.shape[0]   # batch dim matches
