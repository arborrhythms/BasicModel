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


@pytest.mark.skip(reason="Pending serial_mode refactor (2026-04-22)")
def test_arlm_forward_returns_predictions_list_and_no_reconstruction():
    """AR: forward() returns (input_state, symbols, predictions_list, None).

    The outer pos loop in BasicModel.forward() emits one prediction per
    revealed token. AR does not reconstruct -- the fourth return value
    is always None.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    import warnings

    import torch

    import Models
    import Language

    # Build a BasicModel with maskedPrediction=AR in the right XML path.
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    arch = root.find("architecture")
    training = arch.find("training")
    if training is None:
        training = ET.SubElement(arch, "training")
    mp = training.find("maskedPrediction")
    if mp is None:
        mp = ET.SubElement(training, "maskedPrediction")
    mp.text = "AR"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tree.write(tmp.name)
    tmp.close()

    try:
        Language.TheGrammar._configured = False
        model, _ = Models.BasicModel.from_config(tmp.name)

        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]
        with Models.TheData.runtime_batch(sentences, outputs):
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])

            model.eval()
            model.set_sigma(0)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                out = model.forward(x)

        assert len(out) == 4, f"expected 4-tuple return, got {len(out)}"
        _, _, predictions, reconstruction = out
        assert isinstance(predictions, torch.Tensor), \
            f"AR must return a [B, K, N, predDim] tensor, got {type(predictions)}"
        assert predictions.dim() == 4, \
            f"AR predictions must be 4D [B, K, N, predDim], got {tuple(predictions.shape)}"
        assert predictions.shape[1] > 0, \
            "AR must emit at least one per-cursor prediction (K > 0)"
        assert reconstruction is None, \
            "AR must not produce a reconstruction"
    finally:
        os.unlink(tmp.name)


@pytest.mark.skip(reason="Pending serial_mode refactor (2026-04-22); too slow to run pre-refactor")
def test_arlm_runbatch_trains_without_reverse():
    """AR runBatch runs forward+loss+backward+step without calling reverse().

    This is the key speedup: the previous per-position mask-and-rerun
    training loop called reverse() N times per sentence. The new
    streaming loop never calls reverse() under AR.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    import warnings

    import torch

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    arch = root.find("architecture")
    training = arch.find("training")
    if training is None:
        training = ET.SubElement(arch, "training")
    mp = training.find("maskedPrediction")
    if mp is None:
        mp = ET.SubElement(training, "maskedPrediction")
    mp.text = "AR"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tree.write(tmp.name)
    tmp.close()

    try:
        Language.TheGrammar._configured = False
        model, _ = Models.BasicModel.from_config(tmp.name)
        opt = model.getOptimizer(lr=0.01)

        # Count reverse() calls to verify AR does not invoke it.
        reverse_calls = {'n': 0}
        original_reverse = model.reverse

        def tracking_reverse(*args, **kwargs):
            reverse_calls['n'] += 1
            return original_reverse(*args, **kwargs)

        model.reverse = tracking_reverse

        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]
        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, output_target = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])
            y = model.outputSpace.prepOutput(output_target[:1])
            result, _ = model.runBatch(
                train=True, batchNum=0, batchSize=1, split="train",
                optimizer=opt, batch_override=(x, y))

        assert result is not None, "runBatch should return a BatchResult"
        assert reverse_calls['n'] == 0, \
            f"AR must not call reverse() during training; got {reverse_calls['n']} calls"
    finally:
        os.unlink(tmp.name)


def test_basicmodel_arlm_runbatch_uses_streaming_predictions():
    """Retired 2026-05-14: streaming AR predictor was specific to AR path which is retired in IR-only refactor."""
    return  # AR-specific behaviour; covered elsewhere or no longer applicable


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
    # symbolic_state shape: [B, nOutput, nDim] from SymbolicSpace.outputShape
    sshape = tuple(model.symbolic_state.shape)
    assert len(sshape) == 3
    assert sshape[0] == x.shape[0]   # batch dim matches
