"""Phase 4: CS -> SS is .forward()-mediated.

The symbol bind leg is produced by ``SymbolSpace.forward_concept_to_symbol``
(the concept arrives through ``forward``; the symbol is the row-aligned view of
the concept), NOT by the retired ``ConceptualSpace._build_symbol_leg`` reach
into the WholeSpace meta codebook + the stashed ``_model_symbolSpace`` pointer.
A symbol-tower config in PARALLEL mode runs the 3-stream bind through it.
"""
import inspect
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
import Language

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_build_symbol_leg_is_retired():
    # The CS-resident reach is gone -- the method no longer exists on
    # ConceptualSpace; the SS leg is forward-mediated instead.
    assert not hasattr(Spaces.ConceptualSpace, "_build_symbol_leg")
    assert hasattr(Language.SymbolSpace, "forward_concept_to_symbol")


def test_forward_concept_to_symbol_no_cross_space_reach():
    src = inspect.getsource(
        Language.SymbolSpace.forward_concept_to_symbol)
    head, _, rest = src.partition('"""')           # strip the docstring
    code = head + rest.partition('"""')[2]
    # no reach into WholeSpace / a stashed Space pointer
    assert "_model_symbolSpace" not in code
    assert "_relation_store" not in code
    assert "wholeSpace" not in code and "WholeSpace" not in code


def test_symbol_tower_parallel_forward_smoke():
    """MM_symbol_tower.xml is symbolicOrder=0 (serial=False -> PARALLEL) with
    symbolTower on, so the 3-stream bind runs through the new SS leg. The
    forward must run without error."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_symbol_tower.xml")
    assert m.symbol_tower is True and m.serial is False
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)        # exercises forward_concept_to_symbol in the bind


def test_forward_concept_to_symbol_returns_row_aligned_leg():
    """The leg is the row-aligned view of the concept: same [B, N, D] event as
    the concept it was handed (detached)."""
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace
    # a small concept subspace [B, N, D]
    D = int(ss.subspace.what.getW().shape[-1]) if (
        getattr(ss.subspace, "what", None) is not None
        and hasattr(ss.subspace.what, "getW")
        and ss.subspace.what.getW() is not None) else 8
    B, N = 2, 3
    ev = torch.randn(B, N, D)
    sub = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                          nInputDim=D, nOutputDim=D)
    sub.set_event(ev)
    leg = ss.forward_concept_to_symbol(sub)
    assert leg is not None
    out = leg.materialize()
    assert out.shape == ev.shape
    assert torch.allclose(out, ev)             # row-aligned, detached copy
    assert not out.requires_grad


def test_forward_concept_to_symbol_empty_is_none():
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace
    empty = Spaces.SubSpace(inputShape=(1, 8), outputShape=(1, 8),
                            nInputDim=8, nOutputDim=8)
    assert ss.forward_concept_to_symbol(empty) is None
    assert ss.forward_concept_to_symbol(None) is None


def test_symbol_leg_from_activations_is_0d_times_row_and_carries_grad():
    """Sparse-active contract: with ``_concept_activations`` stamped (the 0-D
    symbols from the sparse forward), the leg is activation x identity-row and
    the GRADIENT flows through the activation (the old detach bug)."""
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace
    W = ss.subspace.what.getW()
    D = int(W.shape[-1])
    B, N = 2, 3
    ev = torch.randn(B, N, D)
    sub = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                          nInputDim=D, nOutputDim=D)
    sub.set_event(ev)
    acts = torch.randn(N, B, requires_grad=True)     # [N, B] signed symbols
    object.__setattr__(sub, "_concept_activations", acts)
    leg = ss.forward_concept_to_symbol(sub)
    assert leg is not None
    out = leg.materialize()
    assert out.shape == (B, N, D) and out.requires_grad
    # leg == activation x (EMA-synced, detached) identity row.
    rows = W[:N].detach()
    want = acts.t().unsqueeze(-1) * rows.unsqueeze(0)
    assert torch.allclose(out[..., :int(rows.shape[-1])], want, atol=1e-5)
    out.sum().backward()
    assert acts.grad is not None and torch.any(acts.grad != 0)
    assert W.grad is None or torch.all(W.grad == 0)  # identity stays EMA-only


def test_symbol_leg_survives_repeated_sync_backward():
    """Regression (anomaly crash, 2026-07-02): each stage's leg build syncs
    the SS codebook IN-PLACE before the next stage's product; backward
    through an earlier stage's leg must survive the later syncs (the
    identity rows are cloned, not a live view of the codebook)."""
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace
    W = ss.subspace.what.getW()
    D = int(W.shape[-1])
    B, N = 1, 3

    def _sub():
        s = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                            nInputDim=D, nOutputDim=D)
        s.set_event(torch.randn(B, N, D))
        return s

    sub0 = _sub()
    acts0 = torch.randn(N, B, requires_grad=True)
    object.__setattr__(sub0, "_concept_activations", acts0)
    leg0 = ss.forward_concept_to_symbol(sub0)         # stage 0 (saved for bwd)
    sub1 = _sub()
    acts1 = torch.randn(N, B, requires_grad=True)
    object.__setattr__(sub1, "_concept_activations", acts1)
    ss.forward_concept_to_symbol(sub1)                # stage 1: re-syncs W
    ss.forward_concept_to_symbol(sub1)                # stage 2: re-syncs W
    leg0.materialize().sum().backward()               # must NOT raise
    assert acts0.grad is not None and torch.any(acts0.grad != 0)


def test_symbol_leg_fallback_without_activations_stays_detached():
    """No stamp (sparse-inactive) -> the legacy detached-copy leg."""
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace
    D = int(ss.subspace.what.getW().shape[-1])
    B, N = 2, 3
    sub = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                          nInputDim=D, nOutputDim=D)
    sub.set_event(torch.randn(B, N, D))
    assert getattr(sub, "_concept_activations", None) is None
    leg = ss.forward_concept_to_symbol(sub)
    assert leg is not None and not leg.materialize().requires_grad
