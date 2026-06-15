"""Pipeline-carried SubSpace context: wordSubSpace, errors, serial_cache.

Covers Tasks 2, 3, 5, 8, 11 of the Pipeline Feed-Forward Architecture Plan
(2026-04-22-pipeline-ff-architecture.md).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

from Layers import Error
from Spaces import SubSpace


_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


@pytest.fixture
def model():
    """A BasicModel built from MM_xor.xml (has WordSpace + discourse)."""
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")

    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return m


# ---------- Task 2: Error class ----------

def test_error_total_returns_none_when_empty():
    e = Error()
    assert e.total() is None


def test_error_add_none_is_noop():
    e = Error()
    e.add("foo", None)
    assert e.total() is None
    assert e.terms() == []


def test_error_add_accumulates_and_total_sums_weighted():
    e = Error()
    e.add("foo", torch.tensor(2.0), weight=1.0, space="A", category="symbol")
    e.add("bar", torch.tensor(3.0), weight=0.5, space="B", category="other")
    total = e.total()
    # 1.0*2.0 + 0.5*3.0 = 3.5
    assert float(total.detach()) == pytest.approx(3.5)


def test_error_add_same_name_sums_at_add_time():
    e = Error()
    e.add("foo", torch.tensor(2.0), weight=1.0)
    e.add("foo", torch.tensor(3.0), weight=1.0)
    terms = e.terms()
    assert len(terms) == 1, f"same-name adds collapse to one term, got {terms}"
    name, tensor, weight, space, category = terms[0]
    assert name == "foo"
    assert float(tensor) == pytest.approx(5.0)


def test_error_clear_empties_terms():
    e = Error()
    e.add("foo", torch.tensor(1.0))
    assert e.total() is not None
    e.clear()
    assert e.total() is None
    assert e.terms() == []


def test_error_terms_shape_is_five_tuples():
    e = Error()
    e.add("foo", torch.tensor(1.5), weight=2.0,
          space="WholeSpace", category="symbol")
    terms = e.terms()
    assert len(terms) == 1
    name, tensor, weight, space, category = terms[0]
    assert name == "foo"
    assert isinstance(tensor, torch.Tensor)
    assert weight == 2.0
    assert space == "WholeSpace"
    assert category == "symbol"


# ---------- Task 3: SubSpace context fields + copy_context ----------

def _mk_subspace(n=8, d=10):
    return SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)


def test_subspace_has_context_fields():
    # Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
    # retired the per-SubSpace ``wordSubSpace`` back-pointer; only the
    # ``errors`` and ``serial_cache`` pipeline carriers remain on SubSpace.
    ss = _mk_subspace()
    assert isinstance(ss.errors, Error)
    # serial_cache is a dict keyed by owner-Space id so cross-space caches
    # don't collide.
    assert ss.serial_cache == {}


def test_subspace_copy_context_copies_errors_and_serial_cache():
    src = _mk_subspace()
    src.errors.add("foo", torch.tensor(1.0))
    src.serial_cache[42] = torch.zeros(2, 4)

    dst = _mk_subspace()
    dst.copy_context(src)

    # errors carry by reference: both subspaces see the same Error instance
    # so downstream .add() calls continue to accumulate.
    assert dst.errors is src.errors
    assert dst.serial_cache is src.serial_cache


def test_subspace_copy_context_none_is_noop():
    ss = _mk_subspace()
    original_errors = ss.errors
    ss.copy_context(None)
    assert ss.errors is original_errors
    assert ss.serial_cache == {}


def test_subspace_copy_context_preserves_downstream_writes():
    src = _mk_subspace()
    dst = _mk_subspace()
    dst.copy_context(src)

    # Writes through dst.errors are visible via src.errors (same instance).
    dst.errors.add("downstream", torch.tensor(7.0))
    terms = src.errors.terms()
    names = [t[0] for t in terms]
    assert "downstream" in names


# ---------- Task 4: WordSubSpace.last_svo and STM-residual ----------

def test_wordspace_last_svo_default_invalid(model):
    ws = model.wordSubSpace
    # Microbatch refactor (Task 2): WordSubSpace.last_svo is per-row.
    # Default state has the valid-mask cleared on every row.
    assert not ws.svo_valid(0)


def test_wordspace_stm_residual_none_when_no_discourse(model):
    ws = model.wordSubSpace
    # Disable discourse for this test; stm_residual should pass through None.
    ws.discourse = None
    ws.arm_stm()
    assert ws.stm_residual() is None


def test_wordspace_stm_residual_fires_once_per_sentence(model):
    """stm_residual fires once, then no-ops until Reset re-arms."""
    ws = model.wordSubSpace

    class _FakeDiscourse:
        def __init__(self):
            self.calls = 0

        def predict(self):
            self.calls += 1
            return torch.zeros(4), torch.tensor(1.0)

        def prime(self, pred, conf, scale):
            return torch.ones(4) * float(scale)

    ws.discourse = _FakeDiscourse()
    ws.arm_stm()
    ws.stm_residual_scale = 0.1

    b1 = ws.stm_residual()
    b2 = ws.stm_residual()   # second call same sentence: pass-through None
    ws.Reset()
    b3 = ws.stm_residual()

    assert b1 is not None
    assert torch.allclose(b1, torch.ones(4) * 0.1)
    assert b2 is None
    assert b3 is not None
    assert ws.discourse.calls == 2   # fired once after init, once after Reset


def test_wordspace_reset_clears_last_svo(model):
    ws = model.wordSubSpace
    D = ws.svo_dim
    ws.set_last_svo(0, torch.zeros(D), torch.zeros(D), torch.zeros(D))
    assert ws.svo_valid(0)
    ws.Reset()
    assert not ws.svo_valid(0)


def test_last_svo_lives_on_wordspace_not_conceptualspace(model):
    """last_svo moved to wordSubSpace; ConceptualSpace no longer owns it."""
    # Microbatch refactor (Task 2): per-row API is the contract.
    assert hasattr(model.wordSubSpace, "set_last_svo")
    assert hasattr(model.wordSubSpace, "svo_valid")
    assert not hasattr(model.conceptualSpace, "_last_svo")
    assert not hasattr(model.conceptualSpace, "last_svo")


# ---------- Task 9: Codebook as immutable property ----------

def test_codebook_reference_is_immutable(model):
    with pytest.raises(AttributeError):
        model.symbolicSpace.codebook = None


# ---------- Tasks 5 & 6: end-to-end wordSubSpace carry through pipeline ----------

def _run_one_forward(m):
    """Drive a single forward with a real XOR batch."""
    import warnings
    m.eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        inp = m.inputSpace.prepInput(inp_items)
        with torch.no_grad():
            m.forward(inp)


def test_input_space_has_wordSubSpace_after_set(model):
    # Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
    # retired the per-SubSpace ``wordSubSpace`` back-pointer; the
    # WordSubSpace reference now lives on the owning Space.
    assert model.inputSpace.wordSubSpace is model.wordSubSpace


def test_output_space_carries_wordSubSpace(model):
    _run_one_forward(model)
    assert model.outputSpace.wordSubSpace is model.wordSubSpace


def test_pipeline_spaces_carry_wordSubSpace(model):
    """Every pipeline Space holds a routing pointer to the model's WordSubSpace."""
    _run_one_forward(model)
    assert model.perceptualSpace.wordSubSpace is model.wordSubSpace
    assert model.conceptualSpace.wordSubSpace is model.wordSubSpace
    assert model.symbolicSpace.wordSubSpace is model.wordSubSpace
    assert model.outputSpace.wordSubSpace is model.wordSubSpace


# 2026-05-29: removed test_stm_residual_flows_through_conceptualspace.
# The test asserted that ``ConceptualSpace.forward`` reads
# ``ws.stm_residual_microbatch`` and adds it to the event. That wiring
# was retired by the parallel-mode STM-set-all-slots fix (2026-05-28);
# CS.forward writes the [B, N, D] slab directly to STM and no longer
# consumes the per-word residual injection point. The
# ``stm_residual_microbatch`` API survives but is consumed elsewhere
# (Language.py per-word path and Layers.py).
