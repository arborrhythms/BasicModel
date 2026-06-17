"""Order-0 mereology word-whole binding (S3), gated behind <mereologyRaise>.

doc/specs/mereological-order-raising.md "order-0 MEREOLOGY": a lexer token's
spell-out pids are the PARTS of the word-as-WHOLE. The gated autobind binds a
token's parts to ONE shared whole (keyed by surface text), so the whole
accumulates > 1 part and ``maybe_raise_order`` fires -- the binding that makes
the (previously dormant) raise live. Flag-off byte-identity is covered by the
full suite; here we assert the gated path's behaviour and that the flag-off
path does NOT word-bind.
"""

import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
from test_basicmodel import _populate_test_config

_D = 8


def _whole_space(nS=128):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.WholeSpace([nP, _D], [nS, _D], [nS, _D])


def _cs_stub(ws):
    """A minimal ConceptualSpace-method host: the autobind methods use only
    ``self.terminalSymbolicSpace_ref`` (+ ``self._autobound_percept_ids`` on
    the flag-off path), so a stub with the bound methods exercises the real
    code without standing up a full model."""
    stub = types.SimpleNamespace()
    stub.terminalSymbolicSpace_ref = ws
    stub.symbolicSpace_ref = None
    stub._maybe_autobind_meta = types.MethodType(
        Spaces.ConceptualSpace._maybe_autobind_meta, stub)
    stub._autobind_word_wholes = types.MethodType(
        Spaces.ConceptualSpace._autobind_word_wholes, stub)
    return stub


# one word "abc" spelled into 3 byte-parts (pids 10,11,12), then null pads.
def _one_word_inputs():
    pid_2d = torch.tensor([[10, 11, 12, 0, 0, 0]], dtype=torch.long)
    word_groups = torch.tensor([[0, 0, 0, -1, -1, -1]], dtype=torch.long)
    vec_tensor = torch.randn(1, 6, _D)
    tokens = [["abc"]]
    return pid_2d, vec_tensor, word_groups, tokens


def test_word_whole_accumulates_parts_and_raises():
    ws = _whole_space()
    ws.subspace.what.enable_ramsification(2)
    ws._mereology_k_many = 2                       # 3 parts > 2 -> raise
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ws)
    # ONE whole, keyed by surface text, with 3 DISTINCT byte-parts under it.
    # (After a raise, ps_children_of_whole double-counts because the new
    # higher-order node's taxonomy references the whole + parts; the part_chain
    # below is the authoritative provenance -- mirrors the existing raise test.)
    assert "abc" in ws._word_whole_ss
    whole = ws._word_whole_ss["abc"]
    parts = set(ws.ps_children_of_whole(whole))
    assert len(parts) == 3
    assert all(ws._pos_kind.get(int(p)) == "ps" for p in parts)
    # the raise fired: a higher-order part subsuming the 3 constituents,
    # order 1, with explicit provenance.
    assert ws.part_chain
    ho = next(iter(ws.part_chain))
    assert len(ws.part_chain[ho]) == 3
    ho_row = ws._ss_pos_to_row[int(ho)]
    assert ws.subspace.what.abstraction_order(int(ho_row)) == 1


def test_same_word_reuses_one_whole_idempotently():
    ws = _whole_space()
    ws.subspace.what.enable_ramsification(2)
    ws._mereology_k_many = 4                         # no raise -> clean counts
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ws)
    whole_first = ws._word_whole_ss["abc"]
    parts_first = sorted(ws.ps_children_of_whole(whole_first))
    # second presentation of the SAME word -> same whole, same parts (the
    # (ps, whole) META edges are idempotent), no churn.
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor.clone(), word_groups, tokens, ws)
    assert ws._word_whole_ss["abc"] == whole_first
    assert sorted(ws.ps_children_of_whole(whole_first)) == parts_first


def test_short_word_below_threshold_does_not_raise():
    ws = _whole_space()
    ws.subspace.what.enable_ramsification(2)
    ws._mereology_k_many = 4                        # 3 parts <= 4 -> no raise
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ws)
    whole = ws._word_whole_ss["abc"]
    assert len(ws.ps_children_of_whole(whole)) == 3
    assert not ws.part_chain                        # singleton-ish; no raise


def test_gate_on_routes_to_word_binding():
    ws = _whole_space()
    ws.subspace.what.enable_ramsification(2)
    ws._mereology_raise = True
    ws._mereology_k_many = 2
    stub = _cs_stub(ws)
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=tokens)
    assert "abc" in getattr(ws, "_word_whole_ss", {})
    assert len(set(ws.ps_children_of_whole(ws._word_whole_ss["abc"]))) == 3


def test_gate_off_does_not_word_bind():
    ws = _whole_space()
    # _mereology_raise unset (default) -> per-pid flag-off path, NO word-whole.
    stub = _cs_stub(ws)
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=tokens)
    assert getattr(ws, "_word_whole_ss", None) is None
