"""Type-run segmentation of the analysis cut (T2 of
doc/plans/2026-07-10-wholes-are-types-segmentation.md).

A whole is a MAXIMAL CONSTANT-TYPE RUN over the four character types
(LETTER / DIGIT / WHITESPACE / PUNCT); SPACE-type runs (incl. the ``\\0`` pad
sentinel) are discarded. This exercises the module-level ``_LUT_ANALYSIS_TYPE``
byte->type LUT, the vectorised ``_type_run_spans`` cutter, and the
``WholeSpace.stage_analysis_spans`` method that composes them.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # test/ helpers

import torch

from Spaces import (WholeSpace, Codebook, _LUT_ANALYSIS_TYPE, _type_run_spans,
                    _derive_type_lut, _analysis_type_lut,
                    _TYPE_SPACE, _TYPE_LETTER, _TYPE_DIGIT, _TYPE_PUNCT)


def _bytes(s, n=None):
    """A [1, N] byte unity from a python str, optionally \\0-padded to n."""
    b = list(s.encode("ascii"))
    if n is not None:
        b = b + [0] * (n - len(b))
    return torch.tensor([b], dtype=torch.long)


def _types(s, n=None):
    """Per-position TYPE ids [1, N] for a str via the byte->type LUT."""
    return _LUT_ANALYSIS_TYPE[_bytes(s, n)]


def _spans(s, n=None):
    """Non-padded (start, end) span list for a single-row str via the cutter."""
    return _type_run_spans(_types(s, n))[0].tolist()


# -- the byte->type LUT --------------------------------------------------------

def test_lut_type_assignments():
    assert int(_LUT_ANALYSIS_TYPE[0]) == _TYPE_SPACE      # \0 pad sentinel
    assert int(_LUT_ANALYSIS_TYPE[32]) == _TYPE_SPACE     # space
    assert int(_LUT_ANALYSIS_TYPE[9]) == _TYPE_SPACE      # tab
    assert int(_LUT_ANALYSIS_TYPE[ord("A")]) == _TYPE_LETTER
    assert int(_LUT_ANALYSIS_TYPE[ord("z")]) == _TYPE_LETTER
    assert int(_LUT_ANALYSIS_TYPE[ord("5")]) == _TYPE_DIGIT
    assert int(_LUT_ANALYSIS_TYPE[ord("!")]) == _TYPE_PUNCT
    assert int(_LUT_ANALYSIS_TYPE[ord(".")]) == _TYPE_PUNCT


def test_lut_high_bytes_are_letters():
    # Bytes >= 127 (DEL, non-ASCII) keep word-char behavior -> letter type.
    for byte in (127, 128, 200, 255):
        assert int(_LUT_ANALYSIS_TYPE[byte]) == _TYPE_LETTER


# -- the five design-doc examples (single-row cutter) --------------------------

def test_example_words_split_on_space():
    assert _spans("abc def") == [[0, 3], [4, 7]]


def test_example_alnum_splits_by_type():
    # DELTA: was one span (0, 6); now letter-run + digit-run.
    assert _spans("abc123") == [[0, 3], [3, 6]]


def test_example_punct_run_is_one_span():
    # DELTA: was (0,1),(1,2),(2,3),(3,4),(4,5) (punct-per-char); now one span.
    assert _spans("a...b") == [[0, 1], [1, 4], [4, 5]]


def test_example_letters_punct_space_letters():
    assert _spans("hi, there") == [[0, 2], [2, 3], [4, 9]]


def test_example_trailing_pad_never_in_a_span():
    # \0 padding is SPACE type -> discarded; no span reaches into it.
    assert _spans("abc", n=7) == [[0, 3]]
    assert _spans("hi, there", n=16) == [[0, 2], [2, 3], [4, 9]]


# -- edge cases ----------------------------------------------------------------

def test_single_char():
    assert _spans("a") == [[0, 1]]
    assert _spans("7") == [[0, 1]]
    assert _spans("!") == [[0, 1]]


def test_punct_only_run():
    assert _spans("...") == [[0, 3]]        # one punct-whole
    assert _spans("?!") == [[0, 2]]         # one punct-whole (mixed punct)


def test_all_space_row_is_zero_padded():
    out = _type_run_spans(_types("   "))
    assert out.shape == (1, 1, 2)           # K >= 1
    assert out[0].tolist() == [[0, 0]]      # zero-pad row, no runs


def test_empty_and_digit_letter_boundaries():
    # digit-run then letter-run (no space between): two spans by type.
    assert _spans("12ab") == [[0, 2], [2, 4]]
    assert _spans("a1b2") == [[0, 1], [1, 2], [2, 3], [3, 4]]


def test_batch_mixed_lengths_and_padding():
    # Different token counts + trailing \0 pad; zero-pad short rows to K.
    N = 10
    rows = torch.cat([
        _types("abc def", N),        # -> (0,3),(4,7)
        _types("a...b", N),          # -> (0,1),(1,4),(4,5)
        _types("   ", N),            # -> all space -> zero-pad row
    ], 0)
    out = _type_run_spans(rows)
    assert out.shape[0] == 3
    K = out.shape[1]
    assert K >= 3
    r0 = [s for s in out[0].tolist() if s != [0, 0]]
    r1 = [s for s in out[1].tolist() if s != [0, 0]]
    r2 = out[2].tolist()
    assert r0 == [[0, 3], [4, 7]]
    assert r1 == [[0, 1], [1, 4], [4, 5]]
    assert all(s == [0, 0] for s in r2)      # no runs
    # trailing pad never contributes a span
    assert all(e <= 7 for _, e in out[0].tolist())


# -- WholeSpace.stage_analysis_spans (LUT + cutter, mode gate) ------------------
# The method reads only ``self.analysis_mode``, so a SimpleNamespace stands in
# for a full WholeSpace (mirrors the property_spans(None, ...) unbound pattern).

def _stage(mode, s, n=None):
    fake = types.SimpleNamespace(analysis_mode=mode)
    u = _bytes(s, n)
    return WholeSpace.stage_analysis_spans(fake, u)


def test_stage_byte_mode_returns_none():
    for mode in ("byte", "raw", "sentence"):
        assert _stage(mode, "abc def") is None


def test_stage_none_input_returns_none():
    fake = types.SimpleNamespace(analysis_mode="word")
    assert WholeSpace.stage_analysis_spans(fake, None) is None


def test_stage_word_mode_type_runs():
    assert _stage("word", "abc123")[0].tolist() == [[0, 3], [3, 6]]
    assert _stage("word", "hi, there")[0].tolist() == [[0, 2], [2, 3], [4, 9]]


def test_stage_accepts_three_dim_unity():
    # [B, 1, N] unity (the codebook-selection layout) -> row 0 is read.
    fake = types.SimpleNamespace(analysis_mode="word")
    u = _bytes("a...b").unsqueeze(1)         # [1, 1, 5]
    out = WholeSpace.stage_analysis_spans(fake, u)
    assert out[0].tolist() == [[0, 1], [1, 4], [4, 5]]


# == the four TYPE rows ARE a WS-owned subspace's .what (source of truth) ======
# doc/plans/2026-07-10-wholes-are-types-segmentation.md, T2 (Alec 2026-07-10).
# The analysis cut's byte->type LUT is DERIVED from the four tagged rows of the
# frozen ``.what`` codebook of ``WholeSpace.type_subspace`` -- a dedicated
# WS-owned SubSpace (calculations in Spaces, DATA in SubSpaces; "properties are
# WholeSpace.what" made literal). The codebook is a Codebook -- a Tensor --
# whose W is a plain, non-Parameter tensor; the module constant is only the
# no-subspace fallback. These build a LIVE WholeSpace via the same test-config
# harness as test_property_tiling.

from Layers import LETTER, DIGIT, WHITESPACE, PUNCT
from Spaces import Tensor, SubSpace
from test_basicmodel import _populate_test_config
import Models
import torch.nn as nn

_D = 8
_NP = 4
_NS = 64


def _live_ws(analysis="word", nS=_NS):
    """A LIVE WholeSpace owning a frozen ``type_subspace`` for the cut."""
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=_NP, nPercepts=_NP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    Models.TheXMLConfig._data["WholeSpace"]["analysis"] = analysis
    return WholeSpace([_NP, _D], [nS, _D], [nS, _D])


# -- the derivation reproduces the frozen module LUT ---------------------------

def test_derive_type_lut_matches_frozen_module_lut():
    # The derivation, given the canonical four tags, reproduces the frozen
    # constant EXACTLY -- the rows are a drop-in source of truth.
    pk = {0: {WHITESPACE}, 1: {LETTER}, 2: {DIGIT}, 3: {PUNCT}}
    assert torch.equal(_derive_type_lut(pk), _LUT_ANALYSIS_TYPE)


# -- (a) a built model's live WS has the type subspace, exactly four tags ------

def test_live_ws_has_type_subspace_with_four_tagged_rows():
    ws = _live_ws("word")
    sub = ws.type_subspace
    assert isinstance(sub, SubSpace)
    tc = sub.what
    assert isinstance(tc, Codebook) and isinstance(tc, Tensor)
    # The SubSpace adopted the codebook on its .what slot (unmuxed placement).
    assert sub.codebook_slot == "what" and not sub.muxed
    # Exactly four tagged rows, one class each.
    assert tc.property_kind == {0: {WHITESPACE}, 1: {LETTER},
                                2: {DIGIT}, 3: {PUNCT}}
    assert ws.type_property_rows() == {WHITESPACE: 0, LETTER: 1,
                                       DIGIT: 2, PUNCT: 3}
    # Frozen by construction: W is a plain tensor, NOT an nn.Parameter, and the
    # whole subspace exposes no Parameter (so an optimizer can never see it).
    assert torch.is_tensor(tc.getW()) and not isinstance(tc.getW(), nn.Parameter)
    assert list(sub.parameters()) == []


def test_type_subspace_is_idempotent_on_rebuild():
    a = _live_ws("word")
    b = _live_ws("word")
    assert (a.type_subspace.what.property_kind
            == b.type_subspace.what.property_kind)
    # Re-running the build on the SAME space lands the same tags.
    pk_before = dict(a.type_subspace.what.property_kind)
    a._build_type_subspace()
    assert a.type_subspace.what.property_kind == pk_before


def test_byte_mode_builds_no_type_subspace():
    # byte/raw/sentence stage no spans, so they build no type subspace and the
    # cut falls back to the frozen module LUT (byte-identical).
    ws = _live_ws("byte")
    assert ws.type_subspace is None
    assert _analysis_type_lut(ws) is _LUT_ANALYSIS_TYPE
    # ... and no type-codebook key rides the checkpoint blob for byte configs.
    assert "type_property_kinds" not in ws.vocab_extras()


def test_type_subspace_adds_no_state_dict_keys():
    # The frozen type subspace holds a plain (non-Parameter) .what W and only
    # persistent=False buffers, so owning it adds nothing to the WholeSpace
    # state_dict -- checkpoint keys for a word-mode WS match a byte-mode WS
    # exactly on the type-subspace front.
    word = _live_ws("word")
    keys = [k for k in word.state_dict().keys() if "type_subspace" in k]
    assert keys == []


# -- (b) the derived cut is byte-identical to the previous LUT cut -------------

def _cut_from_rows(ws, byte_rows):
    u = torch.tensor(byte_rows, dtype=torch.long)
    return WholeSpace.stage_analysis_spans(ws, u)


def _cut_from_module(byte_rows):
    u = torch.tensor(byte_rows, dtype=torch.long)
    fake = types.SimpleNamespace(analysis_mode="word")
    return WholeSpace.stage_analysis_spans(fake, u)


def test_live_derived_lut_is_byte_identical_to_module_lut():
    ws = _live_ws("word")
    assert torch.equal(_analysis_type_lut(ws), _LUT_ANALYSIS_TYPE)


def test_derived_cut_byte_identical_across_a_spread_of_inputs():
    ws = _live_ws("word")
    N = 12
    samples = [
        list(b"abc123") + [0] * (N - 6),          # letter/digit split
        list(b"hi, there") + [0] * (N - 9),       # letters/punct/space
        list(b"...") + [0] * (N - 3),             # one punct-whole
        list(b"a...b") + [0] * (N - 5),
        [127, 128, 200, 255] + [0] * (N - 4),     # bytes >= 127 -> letter type
        [65, 200, 66, 129] + [0] * (N - 4),       # high bytes glued to letters
        [9, 10, 13, 32] + list(b"ok") + [0] * (N - 6),   # ws set is discarded
        [255] * N,                                # all high bytes -> one run
    ]
    live = _cut_from_rows(ws, samples)
    module = _cut_from_module(samples)
    assert torch.equal(live, module), (live.tolist(), module.tolist())
    # And explicitly: a high-byte run is ONE letter-run whole.
    assert live[4].tolist()[0] == [0, 4]


# -- (c) checkpoint save/load preserves the rows, tags, and the cut ------------

def test_checkpoint_roundtrip_preserves_tags_and_cut():
    a = _live_ws("word")
    extras = a.vocab_extras()
    assert extras["type_property_kinds"] == {
        0: [WHITESPACE], 1: [LETTER], 2: [DIGIT], 3: [PUNCT]}
    b = _live_ws("word")
    # Wipe B's type subspace to prove the LOAD (not the build) restores it.
    b.type_subspace = None
    b._type_lut_cache = None
    b.load_vocab_extras(extras)
    assert b.type_subspace is not None
    assert b.type_subspace.what.property_kind == {
        0: {WHITESPACE}, 1: {LETTER}, 2: {DIGIT}, 3: {PUNCT}}
    assert torch.equal(_analysis_type_lut(b), _LUT_ANALYSIS_TYPE)
    # The cut is unchanged after the roundtrip.
    rows = [list(b"ab 12! xy") + [0]]
    assert torch.equal(_cut_from_rows(b, rows), _cut_from_module(rows))


# -- (d) the type subspace is frozen: no Parameter, untouched by training ------

def test_type_subspace_is_not_a_parameter_and_survives_a_backward():
    ws = _live_ws("word")
    tc = ws.type_subspace.what
    before = tc.getW().clone()
    # It carries no trainable Parameter, so a WholeSpace-wide optimizer step
    # cannot reach it. Simulate the harshest case: run SGD over EVERY Parameter
    # the WholeSpace exposes (the type subspace IS a registered submodule, so
    # ws.parameters() would surface any Parameter it carried) and confirm the
    # type rows are untouched.
    params = list(ws.parameters())
    assert all(p is not tc.getW() for p in params)   # W is not among them
    opt = torch.optim.SGD(params, lr=0.1) if params else None
    if opt is not None:
        opt.zero_grad()
        # A dummy loss over the trainable params; the type subspace is not in
        # the graph, so it receives no gradient and no update.
        loss = sum(p.pow(2).sum() for p in params)
        loss.backward()
        opt.step()
    assert torch.equal(tc.getW(), before)
    assert tc.property_kind == {0: {WHITESPACE}, 1: {LETTER},
                                2: {DIGIT}, 3: {PUNCT}}
