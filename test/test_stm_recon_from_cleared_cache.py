"""Full integration: reconstruct words from the held STM idea ALONE,
after deleting SymbolicSubSpace's syntactic cache.

Task 9 (plan §6) of the STM serial/parallel modes plan. The deliverable
goal (verbatim): *"deleting SymbolicSubSpace's syntactic cache and running
reverse should reconstruct the words that best match the held STM idea.
... assert top-k recovered words at each position overlap with input
above the existing atol=2e-1 closeness threshold."*

Harness (mirrors ``test/test_router_fires_per_word.py``):
  * Build the cheap serial-grammar model from ``data/MM_xor_loopback.xml``
    (``<symbolicOrder>1</symbolicOrder>``), load the ``xor`` data.
  * Run ONE real per-word ``model.forward`` -- this fills the C-tier STM,
    populates ``symbolSpace.current_rules`` / ``generate_rules``, and
    reduces the STM to the single sentence idea ``S = model._stm_single_S``
    (``[B, D_c]``).
  * Snapshot ``S`` and the per-position forward word indices
    (``perceptualSpace.subspace._index[:, :, 0]`` -- the frozen-lexicon
    MPHF index the forward placed at each slot).
  * DELETE the SymbolicSubSpace syntactic cache (``current_rules`` -> {} ,
    ``generate_rules`` -> {} , ``recur_pass`` -> 0).
  * Drive the reverse-from-STM legs and decode the top-``k`` nearest
    words per position against the perceptual MPHF codebook, asserting
    top-``k`` overlap with the input above ``atol=2e-1``.

==========  STATUS: Findings A/B FIXED; recon criterion still XFAILS  ==========

Driving the integration on a fresh (untrained) ``MM_xor_loopback`` serial
model originally surfaced TWO numerical issues that blocked per-position
word recovery. BOTH are now FIXED (bin/Layers.py ``PiLayer``), and they
turned out to be the SAME root cause -- the ``nonlinear=False`` PiLayer
fold took ``log`` of a value that can be ``<= 0``:

FINDING A (FIXED) -- forward left the STM (and ``_stm_single_S``) as NaN.
  The per-word forward filled the C-tier STM with NaN because
  ``PiLayer._to_mult`` (``(1+x)/(1-x)``, which then feeds ``log``) clamped
  its input to ``(-1, 1)`` ONLY when ``nonlinear=True``. A percept landing
  outside [-1, 1] (legitimate -- percept normalization runs AFTER
  ``pi.forward``) drove the ratio ``<= 0`` and injected ``log(<=0) = NaN``.
  Fix: the ``_to_mult`` clamp is now unconditional. Regression:
  ``test_forward_stm_idea_is_finite`` (here) + ``test_pilayer_log_domain_finite``.

FINDING B (FIXED) -- the reverse perceptual leg turned a finite seed NaN.
  ``_reverse_body`` preserved finiteness but ``_reverse_perceptual`` ->
  ``PartSpace.reverse`` -> ``_reverse_text`` -> ``PiLayer.reverse``
  did a BARE ``log(y)`` on the signed reverse signal (``nonlinear=False``
  branch). Fix: clamp the reverse log to its positive domain and use the
  overflow-safe ``tanh(lx/2)`` exit (== ``_from_mult(exp(lx))``).
  Regression: ``test_reverse_perceptual_preserves_finiteness_on_finite_seed``
  (here) + ``test_pilayer_log_domain_finite``.

Both guards are DOMAIN clamps, not NaN scrubs: ``clamp`` leaves NaN/Inf
untouched, so a genuine upstream divergence still propagates and fails
loud (user memory: never silently nan_to_num / gate away Inf/NaN).

The remaining blocker is SEPARATE and out of scope: the reverse per-op
inverses are *identity stubs* by design (``SyntacticLayer.reverse``
returns the parent / lossy ``(parent, parent)`` for layers with no
authored inverse -- ``bin/Language.py`` ~4150; ``_reverse_from_S``
docstring: *"we do NOT author per-op reverse math"*). ``_reverse_from_S``
therefore returns a position-COLLAPSED ``[B, 1, D_c]`` surface rather than
fanning ``S`` back across the original word positions -- so even though
the recon is now FINITE, the top-k word overlap is ~0.0. Per the task's
instruction (*"a failing recon test that reveals a real invertibility gap
is more valuable than a vacuous passing one ... do NOT weaken the
assertion just to make it green"*), the end-to-end overlap check stays
pinned as ``xfail`` (NOT relaxed) until the per-op reverses reconstruct
``[B, N, D_c]``.

The NON-xfail assertions below pin everything that DOES hold today: the
forward produces a usable ``S`` shape, finite per-position targets, and a
FINITE held idea; clearing the cache works and the
``_chart_generate_from_stm`` re-derive site re-fires
``symbolSpace.generate`` from the STM snapshot ALONE (rebuilding
``generate_rules``); the ``_reverse_from_S`` leg is drivable from the
cleared-cache state and returns a finite, decodable-width surface; and
both the body AND perceptual reverse legs preserve finiteness. When the
per-op reverses reconstruct ``[B, N, D_c]``, the xfail check will xpass
and its marker should be removed.

TEST-ONLY harness here; the PiLayer fix lives in ``bin/Layers.py``. Uses
the REAL model forward / reverse and the REAL perceptual codebook for word
decode (no reimplementation). NaN is never swallowed (user memory: fail
loud on numerical divergence).
"""

import os
import re
import sys
import tempfile
import warnings

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT, "data")
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import matplotlib
matplotlib.use("Agg")

import Models
import Language
from util import init_config, init_device

_GRAMMAR_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")

ATOL_RECON = 2e-1   # the plan's existing closeness threshold for recon
TOPK = 3            # top-k recovered words per position to compare


# -- harness ---------------------------------------------------------------

def _write_serial_config():
    """Materialize a temp XML overlaying ``<symbolicOrder>1`` (serial) --
    BasicModel.from_config re-reads from disk, so the knob must be on a
    file. Mirrors test_router_fires_per_word._write_config_with_overrides.
    """
    with open(_GRAMMAR_CONFIG, "r") as f:
        text = f.read()
    text = re.sub(
        r"\s*<symbolicOrder>[^<]*</symbolicOrder>\s*\n", "\n", text)
    inject = "<symbolicOrder>1</symbolicOrder>"
    if "<architecture>" in text:
        text = text.replace("<architecture>", f"<architecture>\n    {inject}", 1)
    else:
        text = re.sub(
            r"<model[^>]*>",
            lambda m: m.group(0) + f"\n  <architecture>{inject}</architecture>",
            text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA_DIR)
    tmp.write(text)
    tmp.close()
    return tmp.name


def _make_serial_model():
    """Build the serial-grammar model + load xor data (cheap PS/CS/SS).

    Function-scoped (a FRESH model per test): the reverse path mutates
    ``conceptualSpace.subspace`` (via ``set_event``) and the cache, so a
    shared model would leak state between assertions.
    """
    init_device("cpu")
    cfg = _write_serial_config()
    try:
        init_config(path=cfg, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = Models.BasicModel.from_config(cfg)
        Models.TheData.load("xor")
        model.eval()
        return model
    finally:
        try:
            os.unlink(cfg)
        except OSError:
            pass


def _one_input(model):
    loader = model.inputSpace.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    return model.inputSpace.prepInput(inp_items)


def _run_forward(model):
    """Run one real per-word forward; return (S, forward_word_idx)."""
    x = _one_input(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            model.forward(x)
    S = getattr(model, "_stm_single_S", None)
    ps = model.perceptualSpace
    sub = getattr(ps, "subspace", None)
    active = getattr(sub, "_index", None) if sub is not None else None
    fwd_idx = (active[:, :, 0].long().clone()
               if active is not None and active.dim() == 3 else None)
    return (S.clone() if torch.is_tensor(S) else None), fwd_idx


def _clear_word_cache(ss):
    """Delete SymbolicSubSpace's syntactic cache (the §6 cache fields)."""
    ss.current_rules = {}
    ss.generate_rules = {}
    ss.recur_pass = 0


def _codebook_W(model):
    ps = model.perceptualSpace
    cb = ps._mphf_codebook() if ps is not None else None
    return cb.getW() if cb is not None else None


def _topk_decode(W, recon, k=TOPK):
    """Top-k nearest word indices per position against the codebook W.
    Returns ``[..., k]`` LongTensor, or None when undecodable."""
    if W is None or recon is None or not torch.is_tensor(recon):
        return None
    if recon.shape[-1] != W.shape[-1]:
        # "6+2+2": the reversed surface carries the full event width
        # (.what + .where + .when), while the MPHF word codebook is the bare
        # content (.what) width. Word decode matches on the .what slice, so
        # demux a wider recon down to the codebook width.
        if recon.shape[-1] > W.shape[-1]:
            recon = recon[..., :W.shape[-1]]
        else:
            return None
    flat = recon.reshape(-1, recon.shape[-1])
    d = torch.cdist(flat, W)                          # [N, V]
    kk = min(k, W.shape[0])
    idx = d.topk(kk, dim=-1, largest=False).indices   # [N, kk]
    return idx.reshape(*recon.shape[:-1], kk)


def _topk_overlap(fwd_idx, topk_idx):
    """Fraction of forward positions whose forward index is among the
    top-k recovered indices at that position (common position prefix)."""
    if fwd_idx is None or topk_idx is None:
        return None
    Bk = min(fwd_idx.shape[0], topk_idx.shape[0])
    Nk = min(fwd_idx.shape[1], topk_idx.shape[1])
    f = fwd_idx[:Bk, :Nk].unsqueeze(-1)               # [B, N, 1]
    t = topk_idx[:Bk, :Nk, :]                          # [B, N, k]
    return (t == f).any(dim=-1).float().mean().item()


# -- assertions that DO hold today -----------------------------------------

def test_forward_produces_single_S_and_targets():
    """The per-word forward yields the held idea S [B, D_c] and the
    per-position forward word indices the recon is graded against."""
    model = _make_serial_model()
    S, fwd_idx = _run_forward(model)
    assert S is not None and torch.is_tensor(S), \
        "forward must set model._stm_single_S (the held STM idea)."
    assert S.dim() == 2, f"S must be [B, D_c]; got {tuple(S.shape)}"
    assert fwd_idx is not None and fwd_idx.dim() == 2, \
        "forward must expose per-position word indices (PS._index[:, :, 0])."
    assert int(fwd_idx.numel()) > 0


def test_cache_clears_and_chart_generate_rederives_from_stm():
    """Clearing the syntactic cache works, and the cache RE-DERIVE site
    (``_chart_generate_from_stm``) re-fires ``symbolSpace.generate`` from
    the STM snapshot ALONE -- repopulating ``generate_rules`` -- which is
    exactly the reverse-leg behavior §6 relies on after the cache is
    deleted."""
    model = _make_serial_model()
    _run_forward(model)
    ss = model.symbolSpace
    assert ss is not None
    _clear_word_cache(ss)
    assert ss.current_rules == {} and ss.generate_rules == {} \
        and ss.recur_pass == 0

    fired = {"n": 0}
    orig = ss.generate

    def _spy(*a, **k):
        fired["n"] += 1
        return orig(*a, **k)

    ss.generate = _spy
    try:
        with torch.no_grad():
            model._chart_generate_from_stm()
    finally:
        ss.generate = orig

    assert fired["n"] >= 1, (
        "the reverse-leg cache re-derive (_chart_generate_from_stm) must "
        "re-fire symbolSpace.generate over the STM snapshot.")
    # generate_rules must have been rebuilt from the snapshot alone.
    assert isinstance(ss.generate_rules, dict) and len(ss.generate_rules) > 0, (
        f"generate must repopulate generate_rules from the STM snapshot; "
        f"got {ss.generate_rules!r}")


def test_reverse_from_cleared_cache_is_drivable_and_decodable():
    """After clearing the cache, ``_reverse_from_S(S)`` runs from S ALONE
    and returns a surface whose width matches the perceptual codebook, so
    a nearest-word decode is at least well-defined (shape contract). This
    pins that the reverse-from-STM leg is DRIVABLE end-to-end; the QUALITY
    of the recovered words is the xfail below (Findings A/B)."""
    model = _make_serial_model()
    S, _ = _run_forward(model)
    ss = model.symbolSpace
    _clear_word_cache(ss)
    with torch.no_grad():
        recon = model._reverse_from_S(S)
    assert recon is not None and torch.is_tensor(recon) and recon.dim() == 3, \
        f"reverse-from-S must return a [B, N, D] surface; got {recon!r}"
    W = _codebook_W(model)
    topk = _topk_decode(W, recon)
    assert topk is not None, \
        "reconstruction width must match the MPHF codebook for word decode."
    assert topk.shape[-1] == min(TOPK, W.shape[0])


def test_reverse_body_preserves_finiteness_on_finite_seed():
    """The body reverse leg itself is numerically clean: a FINITE seed
    stays finite through ``_reverse_body``. This localizes Finding B to
    the perceptual leg (next test), not the body."""
    model = _make_serial_model()
    _run_forward(model)
    W = _codebook_W(model)
    assert W is not None
    seed = W[:3].mean(dim=0, keepdim=True)             # [1, D] finite
    assert torch.isfinite(seed).all()
    cs = model.conceptualSpace
    cs.subspace.set_event(seed.unsqueeze(1))           # [1, 1, D]
    with torch.no_grad():
        xb = model._reverse_body(cs.subspace)
    mb = xb.materialize() if hasattr(xb, "materialize") else xb
    assert torch.is_tensor(mb) and torch.isfinite(mb).all(), \
        "_reverse_body must preserve finiteness on a finite seed."


# -- explicit findings (surfaced, NOT swallowed) ---------------------------

def test_forward_stm_idea_is_finite():
    """REGRESSION (was FINDING A): the per-word forward leaves a FINITE
    held STM idea, deterministically (verified across random inits).

    On the untrained MM_xor_loopback serial config the forward USED TO
    fill the C-tier STM with NaN -- an unguarded PiLayer log-domain fold:
    ``PiLayer._to_mult`` clamped its input ONLY when ``nonlinear=True``,
    so a percept landing outside [-1, 1] (legitimate -- percept
    normalization runs AFTER ``pi.forward``) drove ``(1+x)/(1-x) <= 0``
    into ``log`` -> NaN. The non-finiteness was therefore init-dependent
    (it needed a percept past +-1). Fixed by making the ``_to_mult``
    clamp unconditional (bin/Layers.py ``PiLayer._to_mult``); finiteness
    no longer depends on the random init. Both the reduced idea
    ``_stm_single_S`` and the STM snapshot are finite, and the input
    encoding stays finite."""
    model = _make_serial_model()
    x = _one_input(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            out = model.forward(x)
    # Input encoding finite (the corruption was never in the input path).
    input_state = out[0] if isinstance(out, (tuple, list)) and out else None
    if torch.is_tensor(input_state):
        assert torch.isfinite(input_state).all(), \
            "forward input encoding is expected finite."
    S = model._stm_single_S
    stm = model.conceptualSpace.stm
    snap = stm.snapshot() if stm is not None else None
    assert torch.is_tensor(S) and bool(torch.isfinite(S).all()), (
        f"forward must leave a FINITE STM idea _stm_single_S; got {S!r}")
    assert snap is None or (torch.is_tensor(snap)
                            and bool(torch.isfinite(snap).all())), (
        "forward must leave the STM snapshot finite (no NaN in the held "
        "idea the reverse reconstructs from).")


def test_reverse_perceptual_preserves_finiteness_on_finite_seed():
    """REGRESSION (was FINDING B): the perceptual reverse leg keeps a
    FINITE seed finite, deterministically (verified across random inits).

    ``_reverse_body`` keeps the seed finite (prior test); the perceptual
    leg (``_reverse_perceptual`` -> ``PartSpace.reverse`` ->
    ``_reverse_text`` -> ``PiLayer.reverse``) USED TO turn the finite seed
    NaN via an unguarded ``log(y)`` on the signed reverse signal (the
    ``nonlinear=False`` branch). Fixed by clamping the reverse log to its
    positive domain and using the overflow-safe ``tanh(lx/2)`` exit
    (algebraically identical to ``_from_mult(exp(lx))``) -- bin/Layers.py
    ``PiLayer.reverse``. The recovered percepts are finite and in the
    normalized percept range [-1, 1]."""
    model = _make_serial_model()
    _run_forward(model)
    W = _codebook_W(model)
    seed = W[:3].mean(dim=0, keepdim=True)
    cs = model.conceptualSpace
    cs.subspace.set_event(seed.unsqueeze(1))
    with torch.no_grad():
        xb = model._reverse_body(cs.subspace)
        xp = model._reverse_perceptual(xb)
    mp = xp.materialize() if (xp is not None and hasattr(xp, "materialize")) \
        else xp
    assert torch.is_tensor(mp), "perceptual reverse must return a tensor."
    assert bool(torch.isfinite(mp).all()), (
        f"perceptual reverse must keep a finite seed finite; got {mp!r}")
    # Recovered percepts live in the normalized percept range [-1, 1].
    assert bool((mp.abs() <= 1.0 + 1e-4).all()), (
        f"recovered percepts must be in [-1, 1]; got range "
        f"[{mp.min().item():.4f}, {mp.max().item():.4f}]")


@pytest.mark.xfail(
    reason="PLAN RECON CRITERION still blocked -- but now ONLY by the "
           "per-op reverse identity stubs (out of scope: '_reverse_from_S' "
           "does NOT author per-op reverse math). FINDING A (forward STM "
           "NaN) and FINDING B (perceptual reverse NaN) are FIXED (the "
           "PiLayer log-domain guard), so the recon is now FINITE -- but "
           "SyntacticLayer.reverse is an identity stub, so _reverse_from_S "
           "returns a position-COLLAPSED [B,1,D_c] surface (no per-position "
           "fan-out vs the N>1 forward positions). The top-k recovered-word "
           "overlap is therefore 0.0, below atol=2e-1. NOT a cache-clear "
           "fault and NOT a NaN. Remove this xfail once the per-op reverses "
           "reconstruct [B,N,D_c].",
    strict=False)
def test_topk_recovered_words_overlap_input():
    """PLAN SUCCESS CRITERION: top-k recovered words at each position
    overlap the input above the atol=2e-1 closeness threshold, running
    reverse from the STM snapshot ALONE (cache cleared).

    Asserted HONESTLY (not weakened): the recon is now finite (Findings
    A/B fixed) but the identity-stub per-op reverses collapse it to a
    single position, so the overlap is ~0.0 and this xfails -- documenting
    the remaining real gap rather than passing vacuously."""
    model = _make_serial_model()
    S, fwd_idx = _run_forward(model)
    ss = model.symbolSpace
    _clear_word_cache(ss)
    with torch.no_grad():
        recon = model._reverse_from_S(S)
    # Defensive: if the recon were non-finite the decode would be
    # meaningless -- treat overlap as 0.0 so the criterion fails honestly
    # (do NOT mask any NaN by sanitizing it into a passing comparison).
    # Post-fix the recon IS finite; the 0.0 overlap comes from the
    # position collapse, not from NaN.
    if recon is None or not torch.is_tensor(recon) \
            or not bool(torch.isfinite(recon).all()):
        overlap = 0.0
    else:
        W = _codebook_W(model)
        topk = _topk_decode(W, recon)
        overlap = _topk_overlap(fwd_idx, topk)
        overlap = 0.0 if overlap is None else overlap
    # The closeness threshold at the word level: a meaningful
    # reconstruction recovers the forward word within its top-k at a
    # majority of positions, i.e. overlap >= (1 - atol).
    assert overlap >= (1.0 - ATOL_RECON), (
        f"top-k recovered-word overlap with input = {overlap:.3f}; "
        f"expected >= {1.0 - ATOL_RECON:.3f} (atol={ATOL_RECON}). Reverse "
        f"from S alone is not reconstructing the per-position words "
        f"(Findings A/B + identity-stub per-op reverses).")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rxX"]))
