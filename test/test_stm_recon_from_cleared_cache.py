"""Clear the syntactic cache and reconstruct the owned understanding.

The historical STM plan's top-k overlap criterion remains unchanged: every
forward position must be recovered with overlap at least .8. Unseeded
results still vary across that bar, so the original xfail stays. The retired D3
root replay is replaced by the retained Understanding-owned inverse, and
nearest-percept decoding uses the native store's codebook. No target is used
as a reverse carrier. The earlier PiLayer finiteness regressions and cache
re-derivation assertions remain below.
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
    """Materialize a temp XML overlaying ``<serial>true</serial>`` --
    BasicModel.from_config re-reads from disk, so the knob must be on a
    file. Mirrors test_router_fires_per_word._write_config_with_overrides.
    """
    with open(_GRAMMAR_CONFIG, "r") as f:
        text = f.read()
    text = re.sub(
        r"\s*<symbolicOrder>[^<]*</symbolicOrder>\s*\n", "\n", text)
    text = re.sub(
        r"\s*<serial>[^<]*</serial>\s*\n", "\n", text)
    inject = "<serial>true</serial>\n    <symbolicOrder>1</symbolicOrder>"
    if "<architecture>" in text:
        text = text.replace("<architecture>", f"<architecture>\n    {inject}", 1)
    else:
        text = re.sub(
            r"<model[^>]*>",
            lambda m: m.group(0) + f"\n  <architecture>{inject}</architecture>",
            text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False)
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
            result = model.forward(x)
            model._test_understanding = model._capture_understanding(result)
    S = getattr(model, "_stm_single_S", None)
    isp = model.inputSpace
    rows = isp._ar_grammar_object_rows
    active = isp._ar_grammar_leaf_mask & isp._word_active_mask
    # Grammar leaves now denote OBJECT rows. The native percept dictionary
    # no longer manufactures one word ID per input position.
    model._test_word_columns = torch.stack([row.nonzero().reshape(-1) for row in active])
    fwd_idx = rows.gather(1, model._test_word_columns).clone()
    assert bool((fwd_idx >= 0).all()), "each presented word must be admitted"
    return (S.clone() if torch.is_tensor(S) else None), fwd_idx


def _clear_word_cache(ss):
    """Delete SymbolSubSpace's syntactic cache (the §6 cache fields)."""
    ss.current_rules = {}
    ss.generate_rules = {}
    ss.recur_pass = 0


def _codebook_W(model):
    ps = model.perceptualSpace
    cb = ps.percept_store._basis if ps is not None else None
    return cb.getW() if cb is not None else None


def _topk_decode(W, recon, k=TOPK):
    """Top-k nearest word indices per position against the codebook W.
    Returns ``[..., k]`` LongTensor, or None when undecodable."""
    if W is None or recon is None or not torch.is_tensor(recon):
        return None
    if recon.shape[-1] != W.shape[-1]:
        # "6+2+2": the reversed surface carries the full event width
        # (.what + .where + .when), while the native percept codebook is the bare
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
    top-k recovered indices at that position; all rows/positions must exist."""
    if fwd_idx is None or topk_idx is None:
        return None
    if fwd_idx.shape != topk_idx.shape[:2]:
        return 0.0
    f = fwd_idx.unsqueeze(-1)                       # [B, N, 1]
    return (topk_idx == f).any(dim=-1).float().mean().item()


# -- assertions that DO hold today -----------------------------------------

@pytest.mark.parametrize('forward, recovered, expected', [
    ([[7, 11]], [[[7, 1, 2]]], 0.0),
    ([[7]], [[[7, 1, 2], [11, 1, 2]]], 0.0),
    ([[7], [11]], [[[7, 1, 2]]], 0.0),
    ([[7, 11]], [[[7, 1, 2], [11, 1, 2]]], 1.0),
    ([[7, 11]], [[[7, 1, 2], [0, 1, 2]]], 0.5),
])
def test_topk_overlap_requires_every_forward_position(forward, recovered, expected):
    assert _topk_overlap(torch.tensor(forward), torch.tensor(recovered)) == expected


def test_forward_produces_single_S_and_targets():
    """The per-word forward yields the held idea S [B, D_c] and the
    per-position forward word indices the recon is graded against."""
    model = _make_serial_model()
    S, fwd_idx = _run_forward(model)
    assert S is not None and torch.is_tensor(S), \
        "forward must set model._stm_single_S (the held STM idea)."
    assert S.dim() == 2, f"S must be [B, D_c]; got {tuple(S.shape)}"
    assert fwd_idx is not None and fwd_idx.dim() == 2, \
        "forward must expose the native object row of each grammar word."
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
    """After clearing the cache, reverseReconstruct reads its owned understanding
    and returns a surface whose width matches the perceptual codebook, so
    a nearest-word decode is at least well-defined (shape contract). This
    pins that the reverse-from-STM leg is DRIVABLE end-to-end; the QUALITY
    of the recovered words is the .8 overlap assertion below."""
    model = _make_serial_model()
    S, _ = _run_forward(model)
    ss = model.symbolSpace
    _clear_word_cache(ss)
    with torch.no_grad():
        recon, _ = model.reverseReconstruct(model._test_understanding)
    assert recon is not None and torch.is_tensor(recon) and recon.dim() == 3, \
        f"reverse-from-S must return a [B, N, D] surface; got {recon!r}"
    W = _codebook_W(model)
    topk = _topk_decode(W, recon)
    assert topk is not None, \
        "reconstruction width must match the native percept codebook for word decode."
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
    fill the C-space_role STM with NaN -- an unguarded PiLayer log-domain fold:
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


@pytest.mark.xfail(strict=False, reason="Historical .8 per-position top-k reconstruction criterion remains unsettled: recorded runs reached .8 and .5. Same bar on the retained inverse; §16.6 permits XPASS.")
def test_topk_recovered_words_overlap_input():
    """Recover each word from the owned inverse after clearing syntactic caches.

    Keep the historical top-k overlap bar of 1-atol=.8. The grammar now
    consumes object codes, so compare against those same object rows.
    """
    model = _make_serial_model()
    S, fwd_idx = _run_forward(model)
    ss = model.symbolSpace
    _clear_word_cache(ss)
    with torch.no_grad():
        # Consuming the owned inverse after clearing syntactic caches must
        # preserve the per-word grammar result. Only occurrence positions,
        # never target leaf values, select its word columns.
        model.reverseReconstruct(model._test_understanding)
        ideas = model._test_understanding.input_reconstruction.ideas
        columns = model._test_word_columns
        recon = ideas.gather(1, columns[..., None].expand(*columns.shape, ideas.shape[-1]))
    # Defensive: if the recon were non-finite the decode would be
    # meaningless -- treat overlap as 0.0 so the criterion fails honestly
    # (do NOT mask any NaN by sanitizing it into a passing comparison).
    if recon is None or not torch.is_tensor(recon) \
            or not bool(torch.isfinite(recon).all()):
        overlap = 0.0
    else:
        W = model._concept_owner().similarity_codebook.getW()
        topk = _topk_decode(W, recon)
        overlap = _topk_overlap(fwd_idx, topk)
        overlap = 0.0 if overlap is None else overlap
    # The closeness threshold at the word level: a meaningful
    # reconstruction recovers the forward word within its top-k at a
    # majority of positions, i.e. overlap >= (1 - atol).
    assert overlap >= (1.0 - ATOL_RECON), (
        f"top-k recovered-word overlap with input = {overlap:.3f}; "
        f"expected >= {1.0 - ATOL_RECON:.3f} (atol={ATOL_RECON}). Reverse "
        f"from the owned understanding is not reconstructing the per-position words.")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rxX"]))
