"""Canonical abstraction order (todo "make abstraction order canonical").

The ramsification record is part of the normal codebook contract, not an
optional sidecar:

  * ALLOCATION -- ``Codebook.create`` allocates the ``[V, max_order]``
    fold-provenance table for every built codebook, ``max_order = max(1,
    architecture.subsymbolicOrder)``; no ``<mereologyRaise>`` flag needed.
  * LIVE STAMPING -- the sigma/pi processing sites stamp rows as a normal
    consequence of processing: RadixLayer.insert (multi-byte percept =
    sigma product), the stage-0 analysis snap (pi descriptors), the t>0
    symbolic emission winner (pi, at the pump pass slot), insert_meta /
    word-whole autobind / property-class wholes / maybe_raise_order
    (mint-site sigma/pi stamps).
  * DERIVED ORDER -- ``abstraction_order(row)`` = count(non-NEITHER
    folds); stable across codebook growth and checkpoint reload.
  * PERSISTENCE -- the table rides the ``vocab_extras`` sidecar
    (``ramsification_extras``), never the state_dict.
  * EXPLICIT CONSTRAINTS -- ``apply_definition_constraint`` routes a
    definition update through the row's recorded fold chain
    (``refold_ramsified``), so a high-order update preserves the
    low-order surface form.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")   # sO=3, radix,
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")          # NO mereologyRaise

import Layers  # noqa: E402
import Spaces  # noqa: E402
from Spaces import Codebook  # noqa: E402
from test_basicmodel import _populate_test_config  # noqa: E402

_D = 8

_MODEL = None


def _model():
    """Build the MM_xor radix model ONCE (no mereologyRaise flag)."""
    global _MODEL
    if _MODEL is None:
        import Models
        import Language
        from util import init_config
        init_config(path=_CONFIG, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(_CONFIG)
        Models.TheData.load("xor")
        m.eval()
        _MODEL = m
    return _MODEL


def _batch(m):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def _whole_space(nS=128):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.WholeSpace([nP, _D], [nS, _D], [nS, _D])


def _trained_sigma(D, seed=5, scale=0.5):
    """An invertible sigma fold with perturbed (trained-like) weights --
    the identity-initialized LDU is trivially value-preserving, so any
    provenance-routing test must move the weights off identity first."""
    torch.manual_seed(seed)
    s = Layers.SigmaLayer(D, D, naive=True, invertible=True)
    with torch.no_grad():
        for _n, p in s.layer.named_parameters():
            p.add_(scale * torch.randn_like(p))
    return s


# -- canonical allocation ---------------------------------------------------

def test_created_codebook_carries_table():
    # Every codebook built through create() has the table from birth --
    # width >= 1 even with no config loaded (the bare-instance default).
    cb = Codebook()
    cb.create(4, 6, _D, customVQ=False, monotonic=False)
    assert cb.ramsification is not None
    assert int(cb.ramsification.shape[0]) == 6
    assert int(cb.ramsification_max_order) >= 1
    assert cb.ramsification.dtype == torch.uint8
    assert int(cb.ramsification.sum()) == 0            # all FOLD_NEITHER


def test_no_flag_model_allocates_ps_ws_tables():
    # Build WITHOUT any mereologyRaise flag: PS percept codebook, WS
    # symbol codebook, and the stage-0 analysis store all carry tables
    # sized to the model's subsymbolicOrder (3 in this fixture).
    m = _model()
    assert not bool(getattr(m, "mereology_raise", False))
    so = max(1, int(m.subsymbolicOrder))
    ps_cb = m.perceptualSpace.subspace.what
    assert isinstance(ps_cb, Codebook)
    assert ps_cb.ramsification is not None
    assert int(ps_cb.ramsification_max_order) == so
    for ws in m.wholeSpaces:
        ws_cb = getattr(ws.subspace, "what", None)
        if isinstance(ws_cb, Codebook):
            assert ws_cb.ramsification is not None
            assert int(ws_cb.ramsification_max_order) == so
        an = getattr(ws, "analysis_store", None)
        if isinstance(an, Codebook):
            assert an.ramsification is not None
            assert int(an.ramsification_max_order) == so


# -- live stamping ----------------------------------------------------------

def test_radix_multibyte_percept_stamps_sigma():
    # A multi-byte percept row is PRODUCED by the sigma radix synthesis ->
    # FOLD_SIGMA at pass 0 (order 1); a single-byte atom stays order 0.
    m = _model()
    store = m.perceptualSpace.percept_store
    cb = m.perceptualSpace.subspace.what
    assert cb.ramsification is not None                # degradation guard
    atom = store.insert(b"\x07")
    chunk = store.insert(b"q7z")
    assert cb.abstraction_order(atom) == 0
    assert cb.abstraction_order(chunk) == 1
    assert int(cb.fold_sequence(chunk)[0]) == Codebook.FOLD_SIGMA


def test_forward_stamps_stage0_analysis_pi():
    # The stage-0 unity snap names its analysis descriptors: the selected
    # analysis_store rows carry FOLD_PI at pass slot 0 after a forward.
    m = _model()
    ws0 = m.wholeSpaces[0]
    an = ws0.analysis_store
    assert isinstance(an, Codebook) and an.ramsification is not None
    x = _batch(m)
    with torch.no_grad():
        m.forward(x)
    idx = getattr(ws0, "_stage0_indices", None)
    assert idx is not None, "fixture must exercise the stage-0 unity snap"
    rows = idx.detach().reshape(-1).long().cpu()
    seq = an.ramsification[rows]                       # [n, max_order]
    assert bool((seq[:, 0] == Codebook.FOLD_PI).all())
    for r in rows.tolist():
        assert an.abstraction_order(r) >= 1


def test_forward_stamps_symbolic_emission_winner_at_pass_slot():
    # At t>0 the symbolic iteration emits ONE winner row per stream; the
    # winner is stamped FOLD_PI at ITS pump pass slot.
    m = _model()
    x = _batch(m)
    with torch.no_grad():
        m.forward(x)
    stamped_any = False
    for t, ws in enumerate(m.wholeSpaces):
        if t == 0:
            continue
        em = getattr(ws, "_symbolic_emission", None)
        if em is None:
            continue
        _win, wrow = em
        cb = ws.subspace.what
        assert isinstance(cb, Codebook) and cb.ramsification is not None
        for r in wrow.reshape(-1).tolist():
            assert int(cb.fold_sequence(int(r))[t]) == Codebook.FOLD_PI
            stamped_any = True
    assert stamped_any, ("no t>0 symbolic emission fired -- the fixture "
                         "must exercise the multi-pass pump")


def test_default_autobind_meta_is_order_one():
    # NO flag, NO manual enable: the default insert_meta path stamps the
    # META one sigma fold above its order-0 constituents -- provenance as
    # a normal consequence of processing.
    ws = _whole_space()
    cb = ws.subspace.what
    assert cb.ramsification is not None                # canonical allocation
    ps_pos = ws.ensure_ps_position(7)
    ws_pos = ws.insert_whole(init_vec=torch.randn(_D))
    meta = ws.insert_meta(ps_pos, ws_pos, fused_vec=torch.randn(_D))
    ws_row = ws._ws_pos_to_row[ws_pos]
    meta_row = ws._ws_pos_to_row[meta]
    assert cb.abstraction_order(int(ws_row)) == 0
    assert cb.abstraction_order(int(meta_row)) == 1
    assert int(cb.fold_sequence(int(meta_row))[0]) == Codebook.FOLD_SIGMA


def test_lbg_split_inherits_fold_provenance():
    # copy_fold_provenance: both halves of a split share the parent's
    # fold history (a split refines a region, it does not change order).
    cb = Codebook()
    cb.create(4, 6, _D, customVQ=False, monotonic=False)
    cb.enable_ramsification(2)
    cb.record_fold(2, 0, Codebook.FOLD_SIGMA)
    cb.record_fold(2, 1, Codebook.FOLD_PI)
    cb.copy_fold_provenance(2, 5)
    assert cb.abstraction_order(5) == 2
    assert torch.equal(cb.fold_sequence(5), cb.fold_sequence(2))


# -- derived order: stability across growth ---------------------------------

def test_order_stable_across_growth_and_remove():
    cb = Codebook()
    cb.create(4, 6, _D, customVQ=False, monotonic=False)
    cb.enable_ramsification(2)
    cb.record_fold(2, 0, Codebook.FOLD_SIGMA)
    assert cb.abstraction_order(2) == 1
    # grow_to: appended rows are order 0, stamped row keeps its order.
    cb.grow_to(12)
    assert int(cb.ramsification.shape[0]) == 12
    assert cb.abstraction_order(2) == 1
    assert cb.abstraction_order(11) == 0
    # insert: table follows the appended rows, index-aligned.
    cb.insert(torch.randn(2, _D))
    assert int(cb.ramsification.shape[0]) == int(cb.getW().shape[0])
    assert cb.abstraction_order(2) == 1
    # remove: the SAME row mask keeps the table aligned -- the stamped
    # row shifts from index 2 to index 1 with its order intact.
    cb.remove([0])
    assert int(cb.ramsification.shape[0]) == int(cb.getW().shape[0])
    assert cb.abstraction_order(1) == 1
    assert cb.abstraction_order(2) == 0


def test_record_fold_misalignment_fails_loud():
    cb = Codebook()
    cb.create(4, 6, _D, customVQ=False, monotonic=False)
    try:
        cb.record_fold(999, 0, Codebook.FOLD_SIGMA)
        assert False, "out-of-table row must raise, not clamp"
    except IndexError:
        pass


# -- persistence (vocab_extras sidecar) --------------------------------------

def test_codebook_extras_roundtrip():
    cb = Codebook()
    cb.create(4, 6, _D, customVQ=False, monotonic=False)
    cb.enable_ramsification(3)
    assert cb.ramsification_extras() is None           # nothing stamped
    cb.record_fold(1, 0, Codebook.FOLD_SIGMA)
    cb.record_fold(4, 0, Codebook.FOLD_PI)
    cb.record_fold(4, 2, Codebook.FOLD_SIGMA)
    blob = cb.ramsification_extras()
    assert blob is not None and set(blob["rows"]) == {1, 4}
    cb2 = Codebook()
    cb2.create(4, 6, _D, customVQ=False, monotonic=False)
    cb2.load_ramsification_extras(blob)
    assert cb2.ramsification_max_order == 3
    assert torch.equal(cb2.ramsification, cb.ramsification)
    assert cb2.abstraction_order(4) == 2               # derived, stable


def test_ws_vocab_extras_roundtrip_preserves_orders():
    ws = _whole_space()
    ps_pos = ws.ensure_ps_position(7)
    ws_pos = ws.insert_whole(init_vec=torch.randn(_D))
    meta = ws.insert_meta(ps_pos, ws_pos, fused_vec=torch.randn(_D))
    meta_row = int(ws._ws_pos_to_row[meta])
    blob = ws.vocab_extras()
    assert "ramsification" in blob
    ws2 = _whole_space()
    ws2.load_vocab_extras(blob)
    cb2 = ws2.subspace.what
    assert cb2.abstraction_order(meta_row) == 1
    assert int(cb2.fold_sequence(meta_row)[0]) == Codebook.FOLD_SIGMA


def test_model_blob_carries_ps_ramsification():
    # The model-level envelope carries the PS percept-codebook stamps and
    # restores them (the radix envelope must not skip the restore).
    m = _model()
    store = m.perceptualSpace.percept_store
    cb = m.perceptualSpace.subspace.what
    pid = store.insert(b"w9k")                          # sigma-stamped
    blob = m._collect_vocab_extras()
    assert blob is not None and "ps_ramsification" in blob
    assert cb.abstraction_order(pid) == 1
    with torch.no_grad():
        cb.ramsification.zero_()                        # simulate fresh build
    assert cb.abstraction_order(pid) == 0
    m._restore_vocab_extras(blob)
    assert cb.abstraction_order(pid) == 1


# -- explicit-constraint retraining ------------------------------------------

def test_constraint_updates_high_order_preserves_low():
    # "This word's abstract definition changed": the update routes through
    # the recorded fold chain (refold), so ONLY the high-order row moves
    # and it lands at fold(new_definition) -- invertible back to the new
    # definition -- while the order-0 surface row is untouched.
    D = _D
    sigma = _trained_sigma(D, seed=11)
    cb = Codebook()
    cb.create(4, 4, D, customVQ=False, monotonic=False)
    cb.enable_ramsification(1)
    cb.record_fold(2, 0, Codebook.FOLD_SIGMA)           # row 2 = order 1
    low_before = cb.getW()[0].detach().clone()          # order-0 surface row
    torch.manual_seed(12)
    new_def = (torch.randn(1, 1, D) * 0.6).tanh()       # pre-fold definition
    written = cb.apply_definition_constraint(2, new_def, sigma=sigma)
    W = cb.getW()
    # the high-order row holds the FOLDED definition...
    assert torch.allclose(W[2], written, atol=1e-6)
    assert torch.allclose(
        W[2], sigma.forward(new_def).reshape(-1)[:D], atol=1e-4)
    # ...which unfolds back to the intended definition...
    back = cb.invert_ramsified(W[2:3], 2, sigma=sigma)
    assert torch.allclose(back.reshape(-1)[:D], new_def.reshape(-1), atol=1e-3)
    # ...and the low-order reconstruction is preserved bit-for-bit.
    assert torch.equal(cb.getW()[0], low_before)


def test_ws_constraint_targets_the_right_rung():
    # The WS entry resolves the target layer from the percept's ladder
    # (fold counts), defaulting to the highest-order rung.
    ws = _whole_space()
    cb = ws.subspace.what
    ps_pos = ws.ensure_ps_position(3)
    ws_pos = ws.insert_whole(init_vec=torch.randn(_D))
    meta = ws.insert_meta(ps_pos, ws_pos, fused_vec=torch.randn(_D))
    meta_row = int(ws._ws_pos_to_row[meta])
    ws_row = int(ws._ws_pos_to_row[ws_pos])
    ladder = ws.order_ladder(ps_pos)
    assert (1, meta, meta_row) in ladder
    sigma = _trained_sigma(_D, seed=13)
    low_before = cb.getW()[ws_row].detach().clone()
    torch.manual_seed(14)
    new_def = (torch.randn(1, 1, _D) * 0.6).tanh()
    hit = ws.apply_definition_constraint(ps_pos, new_def, sigma=sigma)
    assert hit == (meta, meta_row)
    assert torch.allclose(
        cb.getW()[meta_row],
        sigma.forward(new_def).reshape(-1)[:_D], atol=1e-4)
    # the order-0 word row (and every other row) is untouched.
    assert torch.equal(cb.getW()[ws_row], low_before)
    # targeting a rung the ladder does not have resolves to None.
    assert ws.apply_definition_constraint(
        ps_pos, new_def, target_order=5, sigma=sigma) is None
