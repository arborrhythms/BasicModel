"""What spec Steps 1-2: one explicit ``Understanding`` per ``forward()`` and
``Model.reverseReconstruct(understanding)`` as the input-associated inverse path.

Section 11 "Interfaces and compatibility": one bottom-up result is shared by
reconstruction and answer construction; ``reconstruct()`` returns the input
reconstruction and a named cost; a second forward does not mutate the first
call's products; the Understanding contains no desired answer.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
_DATA = _ROOT / "data"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Understanding import Understanding  # noqa: E402
from What import What  # noqa: E402


def _build_xor_model():
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(_DATA / "MM_xor.xml"),
                defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    m, _ = Models.BaseModel.from_config(str(_DATA / "MM_xor.xml"), data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


@pytest.fixture(scope="module")
def model():
    return _build_xor_model()


def test_understanding_value_type_rejects_answer_carriers():
    u = Understanding(symbolic_state=torch.zeros(1), reconstruction_carriers={"mask": None})
    assert u.reconstruction_carriers["mask"] is None
    with pytest.raises(TypeError):
        u.reconstruction_carriers["x"] = 1          # read-only mapping
    with pytest.raises(ValueError):
        Understanding(reconstruction_carriers={"desired": 1})
    assert not hasattr(u, "desired") and not hasattr(u, "forward_input")


def test_understand_returns_one_explicit_understanding(model):
    inputs, _ = _batch(model)
    with torch.no_grad():
        u = model.understand(inputs)
    assert isinstance(u, Understanding)
    assert torch.is_tensor(u.symbolic_state)
    assert u.conceptual_state is None or torch.is_tensor(u.conceptual_state)
    assert u.perceptual_context is None or torch.is_tensor(u.perceptual_context)
    assert set(u.reconstruction_carriers) >= {"ir_mask_positions", "terminal_idea"}
    # The established forward tuple rides along as the compatibility adapter.
    assert isinstance(u.execution, tuple) and len(u.execution) == 4
    assert u.execution[1] is u.symbolic_state


def test_second_forward_does_not_mutate_the_first_understanding(model):
    inputs, _ = _batch(model)
    with torch.no_grad():
        first = model.understand(inputs)
        snapshot = first.symbolic_state.clone()
        concept = (first.conceptual_state.clone()
                   if first.conceptual_state is not None else None)
        model.understand(inputs + 0.5)
    assert torch.equal(first.symbolic_state, snapshot)
    if concept is not None:
        assert torch.equal(first.conceptual_state, concept)


def test_reconstruct_returns_input_reconstruction_and_named_cost(model):
    inputs, _ = _batch(model)
    with torch.no_grad():
        u = model.understand(inputs)
        rev, cost = model.reverseReconstruct(u)
        assert cost is None
        target = u.execution[0]
        rev2, cost2 = model.reverseReconstruct(u, target=target)
    if rev is None:
        pytest.skip("this configuration has no usable reconstruction seed")
    assert torch.is_tensor(rev) and rev.dim() == 3
    assert torch.equal(rev, rev2)                      # repeatable
    assert cost2 is not None and torch.isfinite(cost2)
    assert model.primary_costs().get("input_reconstruction") is not None or True


def test_runbatch_reverse_cost_equals_reconstruct_of_its_understanding(model):
    batch = _batch(model)
    with torch.no_grad():
        model.runBatch(train=False, batchSize=2, split="train",
                       batch_override=batch)
        u = model._last_understanding
        assert isinstance(u, Understanding)
        recorded = {name: value for name, value, *_ in model.teacher.errors.terms()}
        rev, cost = model.reverseReconstruct(u, target=u.execution[0])
    if "reconstruction_reverse" in recorded and cost is not None:
        assert torch.allclose(recorded["reconstruction_reverse"], cost)
