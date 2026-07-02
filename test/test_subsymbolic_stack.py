"""P4 sigma/pi stacks (doc/plans/2026-07-02-two-phase-loops-sparse-relation.md).

<subsymbolicStack> builds DISTINCT per-pass subsymbolic layers -- PS
sigmas[t] / WS pis[t] for t in 0..subsymbolicOrder-1 (depth IS mereological
order; the stack is the subsymbolic reasoning engine). <subsymbolicNoop>
marks identity slots. Default off: the single reused sigma/pi, with
RNG-NEUTRAL stack construction so gated-off configs keep their init streams.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
from util import TheXMLConfig
from test_basicmodel import _populate_test_config

_D = 8


def _config(stack=False, noop=None, T=3):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=16, nSymbols=16,
        nWords=16, nOutput=16, nWhere=0, nWhen=0,
    )
    TheXMLConfig.set("architecture.subsymbolicOrder", T)
    TheXMLConfig.set("architecture.subsymbolicStack", stack)
    if noop is not None:
        TheXMLConfig.set("architecture.subsymbolicNoop", noop)
    return nP


def _ps(stack=False, noop=None, T=3):
    nP = _config(stack=stack, noop=noop, T=T)
    return Spaces.PartSpace([nP, _D], [16, _D], [16, _D])


def _ws(stack=False, noop=None, T=3):
    nP = _config(stack=stack, noop=noop, T=T)
    return Spaces.WholeSpace([nP, _D], [16, _D], [16, _D])


def test_stack_off_by_default_single_layers():
    ps = _ps()
    ws = _ws()
    assert getattr(ps, "sigmas", None) is None
    assert getattr(ws, "pis", None) is None
    assert ps._sigma_for_pass() is ps.sigma          # the single reused fold
    assert ws._pi_for_pass() is ws.pi


def test_stack_on_builds_distinct_per_pass_layers():
    ps = _ps(stack=True, T=3)
    assert ps.sigmas is not None and len(ps.sigmas) == 3
    assert ps.sigmas[0] is ps.sigma                  # pass 0 IS the base layer
    assert ps.sigmas[1] is not ps.sigma and ps.sigmas[2] is not ps.sigma
    assert ps.sigmas[1] is not ps.sigmas[2]          # DISTINCT layers
    # the fresh layers' params are registered (trainable).
    ids = {id(p) for p in ps.params}
    for ly in ps.sigmas[1:]:
        for p in ly.getParameters():
            assert id(p) in ids
    ws = _ws(stack=True, T=3)
    assert ws.pis is not None and len(ws.pis) == 3
    assert ws.pis[0] is ws.pi


def test_noop_slots_are_identity_and_occupy_their_pass():
    ps = _ps(stack=True, noop="1", T=3)
    assert len(ps.sigmas) == 3                       # slot still occupied
    assert ps.sigmas[1] is None                      # the identity pass
    assert ps._sigma_for_pass(1) is None
    assert ps._sigma_for_pass(0) is ps.sigma
    ws = _ws(stack=True, noop="0,2", T=3)
    assert ws.pis[0] is None and ws.pis[2] is None
    assert ws.pis[1] is not None


def test_stack_construction_is_rng_neutral():
    """The similarity_codebook idiom: building the stack must not shift the
    global RNG stream downstream of construction."""
    torch.manual_seed(1234)
    _ps(stack=False, T=3)
    after_off = torch.rand(4)
    torch.manual_seed(1234)
    _ps(stack=True, T=3)
    after_on = torch.rand(4)
    assert torch.equal(after_off, after_on)


def test_pass_selection_reads_stamped_index():
    ws = _ws(stack=True, noop="2", T=3)
    object.__setattr__(ws, "_pump_pass_idx", 1)
    assert ws._pi_for_pass() is ws.pis[1]
    object.__setattr__(ws, "_pump_pass_idx", 2)
    assert ws._pi_for_pass() is None                 # the no-op pass
    object.__setattr__(ws, "_pump_pass_idx", 7)      # beyond T: clamps
    assert ws._pi_for_pass() is ws.pis[2]


def test_synthesize_feedback_identity_when_off_or_noop():
    ps = _ps(stack=False)
    sub = Spaces.SubSpace(inputShape=(2, _D), outputShape=(2, _D),
                          nInputDim=_D, nOutputDim=_D)
    sub.set_event(torch.randn(1, 2, _D))
    assert ps.synthesize_feedback(sub, 1) is sub     # stack off -> identity
    ps2 = _ps(stack=True, noop="1", T=3)
    assert ps2.synthesize_feedback(sub, 1) is sub    # no-op slot -> identity


def test_synthesize_feedback_applies_pass_layer():
    ps = _ps(stack=True, T=3)
    fold = ps.sigmas[2]
    ev = torch.randn(1, 2, _D)
    sub = Spaces.SubSpace(inputShape=(2, _D), outputShape=(2, _D),
                          nInputDim=_D, nOutputDim=_D)
    sub.set_event(ev)
    out = ps.synthesize_feedback(sub, 2)
    assert out is not sub                            # a fed, fresh SubSpace
    got = out.materialize()
    assert got.shape == ev.shape
    assert not torch.allclose(got, ev)               # the fold applied


def test_stack_layers_are_registered_submodules():
    """The t>=1 stack layers must ride state_dict (checkpoint round-trip)
    and the model-wide .to() -- a plain Python list would silently drop
    their trained weights on save/load and strand them on CPU under the
    build-on-cpu-then-move device path."""
    ps = _ps(stack=True, T=3)
    subs = list(ps._sigma_stack_modules)
    assert subs == [ly for ly in ps.sigmas[1:] if ly is not None]
    registered = set(id(m) for m in ps.modules())
    assert all(id(ly) in registered for ly in subs)
    # state_dict carries the fresh layers' parameters.
    param_ids = {id(p) for p in ps.state_dict().values()}
    for ly in subs:
        for p in ly.getParameters():
            assert any(torch.equal(p.detach(), v) for v in
                       ps.state_dict().values() if v.shape == p.shape)
    ws = _ws(stack=True, noop="1", T=3)
    assert list(ws._pi_stack_modules) == [ly for ly in ws.pis[1:]
                                          if ly is not None]
    assert all(id(ly) in set(id(m) for m in ws.modules())
               for ly in ws._pi_stack_modules)
