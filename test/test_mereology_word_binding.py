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
from Layers import ATOM, UNIVERSE
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


def _cs_stub(ss):
    """A minimal ConceptualSpace-method host: the autobind methods use only
    ``self.terminalSymbolSpace_ref`` (+ ``self._autobound_percept_ids`` on
    the flag-off path), so a stub with the bound methods exercises the real
    code without standing up a full model."""
    stub = types.SimpleNamespace()
    stub.terminalSymbolSpace_ref = ss
    stub.wholeSpace_ref = None
    stub._maybe_autobind_meta = types.MethodType(
        Spaces.ConceptualSpace._maybe_autobind_meta, stub)
    stub._autobind_word_wholes = types.MethodType(
        Spaces.ConceptualSpace._autobind_word_wholes, stub)
    stub._autobind_cross_tower = types.MethodType(
        Spaces.ConceptualSpace._autobind_cross_tower, stub)
    # The cross-tower path also populates the relation-only CS symbol table,
    # and the orchestrator mints the per-word A/B/C symbols.
    for _m in ("_populate_cs_symbols", "_concept_tables", "new_concept",
               "add_part", "add_whole", "resolve_identities",
               "relate", "reify_concept", "create_word_object_meta",
               "concept_parts", "concept_wholes"):
        setattr(stub, _m, types.MethodType(getattr(Spaces.ConceptualSpace, _m), stub))
    return stub


# one word "abc" spelled into 3 byte-parts (pids 10,11,12), then null pads.
def _one_word_inputs():
    pid_2d = torch.tensor([[10, 11, 12, 0, 0, 0]], dtype=torch.long)
    word_groups = torch.tensor([[0, 0, 0, -1, -1, -1]], dtype=torch.long)
    vec_tensor = torch.randn(1, 6, _D)
    tokens = [["abc"]]
    return pid_2d, vec_tensor, word_groups, tokens


def test_word_whole_accumulates_parts_and_raises():
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_k_many = 2                       # 3 parts > 2 -> raise
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ss)
    # ONE whole, keyed by surface text, with 3 DISTINCT byte-parts under it.
    # (After a raise, ps_children_of_whole double-counts because the new
    # higher-order node's taxonomy references the whole + parts; the part_chain
    # below is the authoritative provenance -- mirrors the existing raise test.)
    assert "abc" in ss._word_whole_ss
    whole = ss._word_whole_ss["abc"]
    parts = set(ss.ps_children_of_whole(whole))
    assert len(parts) == 3
    assert all(ss._pos_kind.get(int(p)) == "ps" for p in parts)
    # the raise fired: a higher-order part subsuming the 3 constituents,
    # order 1, with explicit provenance.
    assert ss.part_chain
    ho = next(iter(ss.part_chain))
    assert len(ss.part_chain[ho]) == 3
    ho_row = ss._ws_pos_to_row[int(ho)]
    assert ss.subspace.what.abstraction_order(int(ho_row)) == 1


def test_same_word_reuses_one_whole_idempotently():
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_k_many = 4                         # no raise -> clean counts
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ss)
    whole_first = ss._word_whole_ss["abc"]
    parts_first = sorted(ss.ps_children_of_whole(whole_first))
    # second presentation of the SAME word -> same whole, same parts (the
    # (ps, whole) META edges are idempotent), no churn.
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor.clone(), word_groups, tokens, ss)
    assert ss._word_whole_ss["abc"] == whole_first
    assert sorted(ss.ps_children_of_whole(whole_first)) == parts_first


def test_short_word_below_threshold_does_not_raise():
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_k_many = 4                        # 3 parts <= 4 -> no raise
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ss)
    whole = ss._word_whole_ss["abc"]
    assert len(ss.ps_children_of_whole(whole)) == 3
    assert not ss.part_chain                        # singleton-ish; no raise


def test_gate_on_routes_to_word_binding():
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_raise = True
    ss._mereology_k_many = 2
    stub = _cs_stub(ss)
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=tokens)
    assert "abc" in getattr(ss, "_word_whole_ss", {})
    assert len(set(ss.ps_children_of_whole(ss._word_whole_ss["abc"]))) == 3


def test_gate_off_does_not_word_bind():
    ss = _whole_space()
    # _mereology_raise unset (default) -> per-pid flag-off path, NO word-whole.
    stub = _cs_stub(ss)
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=tokens)
    assert getattr(ss, "_word_whole_ss", None) is None


def test_gate_on_creates_word_object_meta():
    """The orchestrator mints the per-word A/B/C relation-only symbols on CS
    (Alec 2026-06-17): A = word-symbol (word-parts ⊑ word-whole), B =
    object-symbol (ATOM ⊑ UNIVERSE, to be refined), C = reify(A, B)."""
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_raise = True
    ss._mereology_k_many = 2
    stub = _cs_stub(ss)
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()  # "abc" -> 10,11,12
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=tokens)
    wom = getattr(stub, "_word_obj_meta", {})
    assert "abc" in wom
    A, B, C = wom["abc"]
    # A = word-symbol: parts = the word-parts, whole = the WS word-whole.
    assert set(stub.concept_parts(A)) == {10, 11, 12}
    assert stub.concept_wholes(A) == [ss._word_whole_ss["abc"]]
    # B = object-symbol: maximally unspecified poles, awaiting refinement.
    assert stub.concept_parts(B) == [ATOM] and stub.concept_wholes(B) == [UNIVERSE]
    # C = META: reify A ⊑ B (word≡object).
    assert stub.concept_parts(C) == [("sym", A)]
    assert stub.concept_wholes(C) == [("sym", B)]


def test_meronomy_two_words_keyed_by_surface():
    """Meronomy word grid (Spaces _embed_radix under <synthesis>meronomy>):
    word_groups is a WORD grid ([the=0, sep=-1, cat=1]) and the META key comes
    from the word SURFACE (``word_texts``), not the lexer ``tokens`` (which are
    chunk-indexed / empty in raw serial mode). One A/B/C per word -> 'the','cat'.
    """
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_raise = True
    ss._mereology_k_many = 2
    stub = _cs_stub(ss)
    # "the cat": the=pids 10,11,12 (word 0); sep slot -1; cat=20,21,22 (word 1).
    pid_2d = torch.tensor([[10, 11, 12, 0, 20, 21, 22, 0]], dtype=torch.long)
    word_groups = torch.tensor([[0, 0, 0, -1, 1, 1, 1, -1]], dtype=torch.long)
    vec_tensor = torch.randn(1, 8, _D)
    # raw mode: tokens empty -> the key MUST be sourced from word_texts.
    stub._maybe_autobind_meta(
        pid_2d, vec_tensor, word_groups=word_groups, tokens=None,
        word_texts=[["the", "cat"]])
    wom = getattr(stub, "_word_obj_meta", {})
    assert set(wom.keys()) == {"the", "cat"}
    assert set(stub.concept_parts(wom["the"][0])) == {10, 11, 12}
    assert set(stub.concept_parts(wom["cat"][0])) == {20, 21, 22}


def test_cs_owns_relation_taxonomy_by_reference():
    """S3 relocation (Fix #1; doc/specs "relation-only completion"): CS owns the
    relation taxonomy BY REFERENCE -- its forwarding accessors return the
    terminal WholeSpace's taxonomy, so callers can migrate WS->CS behavior-
    equivalently. The physical position-keyed dicts stay on WS (insert_symbol /
    insert_meta mint codebook rows atomically; a physical move is gated on the
    deferred meta-vector retirement)."""
    ss = _whole_space()
    ss.subspace.what.enable_ramsification(2)
    ss._mereology_k_many = 4                         # 3 parts <= 4 -> no raise
    pid_2d, vec_tensor, word_groups, tokens = _one_word_inputs()  # "abc"
    Spaces.ConceptualSpace._autobind_word_wholes(
        None, pid_2d, vec_tensor, word_groups, tokens, ss)
    whole = ss._word_whole_ss["abc"]
    # a CS host wired to the terminal WS as its relation store (mirrors the
    # Models terminalSymbolSpace_ref / terminalConceptualSpace_ref fan-out).
    cs = types.SimpleNamespace()
    cs.terminalSymbolSpace_ref = ss
    cs.wholeSpace_ref = None
    for _m in ("_relation_store", "taxonomy_children", "taxonomy_parent",
               "taxonomy_parents", "is_meta", "ps_children_of_whole"):
        setattr(cs, _m, types.MethodType(getattr(Spaces.ConceptualSpace, _m), cs))
    # the relation store resolves to the terminal WS, and every read-API
    # accessor forwards byte-for-byte.
    assert cs._relation_store() is ss
    assert sorted(cs.ps_children_of_whole(whole)) == sorted(
        ss.ps_children_of_whole(whole))
    part = ss.ps_children_of_whole(whole)[0]
    meta = ss.taxonomy_parent(part)
    assert cs.taxonomy_parent(part) == meta
    assert cs.taxonomy_parents(part) == ss.taxonomy_parents(part)
    assert cs.is_meta(meta) is True and ss.is_meta(meta) is True
    assert cs.taxonomy_children(meta) == ss.taxonomy_children(meta)
    # unwired CS -> safe empty/false defaults (no crash).
    bare = types.SimpleNamespace(terminalSymbolSpace_ref=None,
                                 wholeSpace_ref=None)
    for _m in ("_relation_store", "taxonomy_children", "is_meta"):
        setattr(bare, _m, types.MethodType(getattr(Spaces.ConceptualSpace, _m), bare))
    assert bare._relation_store() is None
    assert bare.taxonomy_children(0) == [] and bare.is_meta(0) is False
