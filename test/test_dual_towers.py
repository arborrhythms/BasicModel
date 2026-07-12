"""Symmetric dual towers (2026-07-10 plan, rev 2, Task A).

PS and WS are symmetric duals — atoms vs universe views of the same input —
with the thin ``PerceptualSpace`` intermediate class removed. cpu/eager,
seed-free structural pins (no training in this file).
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import functools
import hashlib
import sys

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config
import Spaces
from Spaces import Space, PartSpace, WholeSpace

# HEAD 2026-07-10 (captured pre-Task-A): structural pins that must survive
# the PerceptualSpace removal byte-for-byte.
_XOR_HEAD_NVEC = [8, 8, 8]
_GRAMMAR_HEAD_NVEC = [4, 2, 2]
_HEAD_SD = {  # (n_keys, sha16 of sorted state_dict key names)
    "data/MM_20M_xor.xml": (1875, "8f4dc25033d09807"),
    "data/MM_20M_grammar.xml": (2664, "3e5ff795238e7c59"),
}


@functools.lru_cache(maxsize=None)
def _build(cfg):
    model, *_ = _build_model(_resolve_config(cfg))
    return model


def test_perceptualspace_class_removed():
    """PS/WS/SS subclass Space directly; the thin base is gone."""
    assert not hasattr(Spaces, "PerceptualSpace")
    assert PartSpace.__bases__ == (Space,)
    assert WholeSpace.__bases__ == (Space,)
    from Language import SymbolSpace
    assert SymbolSpace.__bases__ == (Space,)


def test_null_percept_key_survives_on_partspace():
    """Consumers reference PartSpace.NULL_PERCEPT_KEY; it must keep resolving."""
    assert PartSpace.NULL_PERCEPT_KEY == "__NULL_PERCEPT__"


def test_ws_matches_ps_view_shape():
    """The two towers present identical [8, 1024] views for the callosum."""
    m = _build("data/MM_sparse_concept.xml")
    ps, ws = m.perceptualSpace, m.wholeSpace
    assert int(ps.nOutputDim) == int(ws.nOutputDim) == 1024


def test_off_path_stores_unchanged():
    """Serial + sO=0 CS store sizes keep their HEAD shapes."""
    for cfg, want in (("data/MM_20M_xor.xml", _XOR_HEAD_NVEC),
                      ("data/MM_20M_grammar.xml", _GRAMMAR_HEAD_NVEC)):
        got = [int(cs.nVectors) for cs in _build(cfg).conceptualSpaces]
        assert got == want, (cfg, got)


def test_off_path_state_dict_keys_unchanged():
    """The base held no params: key names must be identical post-removal."""
    for cfg, (n, sha) in _HEAD_SD.items():
        keys = sorted(_build(cfg).state_dict().keys())
        h = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
        assert (len(keys), h) == (n, sha), (cfg, len(keys), h)


def test_dual_forward_signatures():
    """One symmetric signature: forward(in_sub, cs_out=None) on both towers."""
    import inspect
    for cls in (PartSpace, WholeSpace):
        params = list(inspect.signature(cls.forward).parameters)
        assert params == ["self", "in_sub", "cs_out"], (cls.__name__, params)


def _run_one_epoch(cfg):
    m = _build(cfg)
    opt = m.getOptimizer(lr=0.01)
    m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    return m


def test_ws_routes_universe_on_parallel_path():
    """sO>=1 parallel: WS consumes the universe view at EVERY stage."""
    m = _run_one_epoch("data/MM_sparse_concept.xml")
    stamps = [getattr(ws, "_ws_routed_source", None) for ws in m.wholeSpaces]
    assert stamps == ["universe"] * len(stamps), stamps


def test_ws_routing_after_serial_migration():
    """UNCONDITIONAL routing + the VALIDITY law (2026-07-12): a staged
    unity routes universe, period; an ALL-ZERO unity is staged as None at
    the stem and the carrier body routes. Embedding-mode inputs lex zero
    byte buffers today (live-universe byte plumbing is the recorded
    follow-on), so serial per-word routes carrier; when real bytes land,
    universe routing engages with no code change. The parallel pump offers
    its unity raw (glue contract) and stamps universe regardless.
    """
    m = _build("data/MM_20M_grammar.xml")
    for w in m.wholeSpaces:                     # cached models: clear stamps
        object.__setattr__(w, "_ws_routed_source", None)
    m = _run_one_epoch("data/MM_20M_grammar.xml")
    stamps = [getattr(ws, "_ws_routed_source", None) for ws in m.wholeSpaces]
    assert stamps[-1] == "universe", stamps     # LIVE unity, per-word
    import torch as _t
    assert _t.is_tensor(getattr(m, "_ws_universe", None))
    m = _build("data/MM_20M_xor.xml")
    for w in m.wholeSpaces:
        object.__setattr__(w, "_ws_routed_source", None)
    m = _run_one_epoch("data/MM_20M_xor.xml")
    stamps = [getattr(ws, "_ws_routed_source", None) for ws in m.wholeSpaces]
    assert stamps[0] == "universe", stamps      # live unity at the bootstrap


# ---- Tasks B/C: signed snap + feedforward pyramid (rev 2 design §§2-5) ----

def test_signed_snap_no_annihilation():
    """B: the order-0 readout is SIGNED — the epoch-1 all-negative ->
    clamp -> a_0=0 death (probe 2026-07-10) is structurally impossible."""
    import torch
    m = _run_one_epoch("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    a0 = getattr(cs0, "_cs_last_a0", None)
    assert a0 is not None and torch.is_tensor(a0)
    assert float(a0.abs().max()) > 0.0, "order-0 presence must be alive"
    assert bool((a0 < 0).any()) or bool((a0 > 0).any())


def test_pyramid_replaces_wave():
    """C: feedforward per-order folds replace the settling wave."""
    import torch
    m = _run_one_epoch("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    assert getattr(cs0, "_cs_wave_qe", None) is None, "wave statistic retired"
    lv = getattr(cs0, "_cs_level_acts", None)
    assert lv is not None and len(lv) >= 1, "per-rung stats must populate"
    assert float(lv[0]) > 0.0, "rung 0 (order-0 tiles) must be lit"


def test_pyramid_taper_topk_selection():
    """C: per-order top-K taper 8/4/2/1 lands in cs.subspace.index and a
    generic materialize() pulls exactly the selected codes.

    The staging is PER-BATCH state (SubSpace.End() releases it with the
    other per-batch tensors), so the contract is read inside the
    consumption window: drive the symbolic phase directly post-training.
    """
    import torch
    m = _run_one_epoch("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    settled = torch.randn(2, int(cs0.outputShape[0]),
                          int(cs0.subspace.muxedSize))
    content, acts = cs0.cs_symbolic_phase(settled)
    assert acts is not None, "symbolic phase must be active"
    idx = cs0.subspace.get_index()
    assert idx is not None and idx.ndim == 3, "top-K selection must be staged"
    n_sel = int(idx.shape[1])
    K = int(getattr(cs0, "_symbolic_order", 0))
    caps = [8, 4, 2, 1][:K + 1]
    # Caps are CAPS, not quotas: early epochs may not mint every order.
    assert caps[0] <= n_sel <= sum(caps), f"taper range violated: {n_sel}"
    codes = cs0.subspace.materialize()
    assert torch.is_tensor(codes) and codes.shape[1] == n_sel, (
        "generic materialize() must pull exactly the selected codes")
    assert int(codes.shape[0]) == 2, "codes are per-batch [B, n_sel, D]"


def test_pyramid_grads_reach_every_rung():
    """C: gradients flow to the sparse edge values through the FF folds."""
    import torch
    m = _build("data/MM_sparse_concept.xml")
    opt = m.getOptimizer(lr=0.01)
    m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    cs0 = m.conceptualSpaces[0]
    fams = cs0._sparse_families(0)
    vals = [ly.values for ly in fams if getattr(ly, "values", None) is not None]
    assert vals, "the concept store must expose trainable edge values"
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in vals)
