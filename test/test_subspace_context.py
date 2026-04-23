"""Pipeline-carried SubSpace context: wordSpace, errors, serial_cache.

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
    """A MentalModel built from MM_xor.xml (has WordSpace + discourse)."""
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    torch.manual_seed(0)
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
          space="SymbolicSpace", category="symbol")
    terms = e.terms()
    assert len(terms) == 1
    name, tensor, weight, space, category = terms[0]
    assert name == "foo"
    assert isinstance(tensor, torch.Tensor)
    assert weight == 2.0
    assert space == "SymbolicSpace"
    assert category == "symbol"


# ---------- Task 3: SubSpace context fields + copy_context ----------

def _mk_subspace(n=8, d=10):
    return SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)


def test_subspace_has_context_fields():
    ss = _mk_subspace()
    assert ss.wordSpace is None
    assert isinstance(ss.errors, Error)
    # serial_cache is a dict keyed by owner-Space id so cross-space caches
    # don't collide.
    assert ss.serial_cache == {}


def test_subspace_copy_context_copies_all_fields():
    src = _mk_subspace()
    sentinel_ws = object()
    src.wordSpace = sentinel_ws
    src.errors.add("foo", torch.tensor(1.0))
    src.serial_cache[42] = torch.zeros(2, 4)

    dst = _mk_subspace()
    dst.copy_context(src)

    assert dst.wordSpace is sentinel_ws
    # errors carry by reference: both subspaces see the same Error instance
    # so downstream .add() calls continue to accumulate.
    assert dst.errors is src.errors
    assert dst.serial_cache is src.serial_cache


def test_subspace_copy_context_none_is_noop():
    ss = _mk_subspace()
    original_errors = ss.errors
    ss.copy_context(None)
    assert ss.errors is original_errors
    assert ss.wordSpace is None
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


# ---------- Task 4: WordSpace.last_svo and STM-residual ----------

def test_wordspace_last_svo_default_invalid(model):
    ws = model.wordSpace
    # Microbatch refactor (Task 2): WordSpace.last_svo is per-row.
    # Default state has the valid-mask cleared on every row.
    assert not ws.svo_valid(0)


def test_wordspace_stm_residual_none_when_no_discourse(model):
    ws = model.wordSpace
    # Disable discourse for this test; stm_residual should pass through None.
    ws.discourse = None
    ws.arm_stm()
    assert ws.stm_residual() is None


def test_wordspace_stm_residual_fires_once_per_sentence(model):
    """stm_residual fires once, then no-ops until Reset re-arms."""
    ws = model.wordSpace

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
    ws = model.wordSpace
    D = ws.svo_dim
    ws.set_last_svo(0, torch.zeros(D), torch.zeros(D), torch.zeros(D))
    assert ws.svo_valid(0)
    ws.Reset()
    assert not ws.svo_valid(0)


def test_last_svo_lives_on_wordspace_not_conceptualspace(model):
    """last_svo moved to wordSpace; ConceptualSpace no longer owns it."""
    # Microbatch refactor (Task 2): per-row API is the contract.
    assert hasattr(model.wordSpace, "set_last_svo")
    assert hasattr(model.wordSpace, "svo_valid")
    assert not hasattr(model.conceptualSpace, "_last_svo")
    assert not hasattr(model.conceptualSpace, "last_svo")


# ---------- Task 9: Codebook as immutable property ----------

def test_codebook_reference_is_immutable(model):
    with pytest.raises(AttributeError):
        model.symbolicSpace.codebook = None


# ---------- Tasks 5 & 6: end-to-end wordSpace carry through pipeline ----------

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


def test_input_subspace_has_wordSpace_after_set(model):
    assert model.inputSpace.subspace.wordSpace is model.wordSpace


def test_output_subspace_carries_wordSpace(model):
    _run_one_forward(model)
    assert model.outputSpace.subspace.wordSpace is model.wordSpace


def test_pipeline_stages_carry_wordSpace(model):
    """Every per-stage subspace picks up wordSpace via copy_context."""
    _run_one_forward(model)
    # PerceptualSpace, ConceptualSpace(s), SymbolicSpace(s), OutputSpace.
    assert model.perceptualSpace.subspace.wordSpace is model.wordSpace
    assert model.conceptualSpace.subspace.wordSpace is model.wordSpace
    assert model.symbolicSpace.subspace.wordSpace is model.wordSpace
    assert model.outputSpace.subspace.wordSpace is model.wordSpace


def test_stm_residual_flows_through_conceptualspace(model):
    """When wordSpace.stm_residual_microbatch returns a non-None bias,
    ConceptualSpace.forward stages a biased tensor into its own subspace."""
    ws = model.wordSpace
    cs = model.conceptualSpace

    # Build an upstream subspace carrying wordSpace + a known event.
    from Spaces import SubSpace
    upstream = SubSpace(cs.inputShape, cs.outputShape,
                        nInputDim=cs.nInputDim, nOutputDim=cs.nOutputDim)
    upstream.copy_context(cs.subspace)  # seed serial_cache/errors dicts
    upstream.wordSpace = ws
    B, N = 2, 4
    D = int(cs.inputShape[1])
    # The body sees a flattened [B*K, N, D] event; here K=1 so BK=B=2.
    # Resize ws._stm_fired to the source batch so the call site derives K=1.
    ws._stm_fired = torch.zeros(B, dtype=torch.bool, device=ws._stm_fired.device)
    event_in = torch.zeros(B, N, D)
    upstream.set_event(event_in)

    # Baseline: no residual. forward() must not mutate upstream.
    ws.discourse = None
    ws.arm_stm()
    ws.stm_residual_microbatch = lambda B_arg, K_arg: None
    cs.forward(upstream)
    baseline_event = cs.subspace.event.getW()

    # Primed: non-None residual. forward() stages event+bias in cs.subspace.
    # The new microbatch contract returns [B*K, D]; the call site
    # broadcasts over N via .unsqueeze(1).
    bias = torch.ones(B, D) * 0.3
    ws.arm_stm()
    ws.stm_residual_microbatch = lambda B_arg, K_arg: bias
    upstream.set_event(event_in)  # reset upstream
    cs.forward(upstream)
    primed_event = cs.subspace.event.getW()

    assert not torch.allclose(baseline_event, primed_event), (
        "STM-residual should change the ConceptualSpace forward event")
