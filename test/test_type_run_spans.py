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
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # test/ helpers

import torch

from Spaces import (WholeSpace, Codebook, _LUT_ANALYSIS_TYPE, _type_run_spans,
                    _derive_type_lut, _analysis_type_lut,
                    _TYPE_SPACE, _TYPE_LETTER, _TYPE_DIGIT, _TYPE_PUNCT)
from property_fixtures import property_reader


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
# The small owner supplies real primitive coefficients for the unbound cut.

def _stage(mode, s, n=None):
    fake = property_reader(analysis_mode=mode)
    u = _bytes(s, n)
    return WholeSpace.stage_analysis_spans(fake, u)


def test_stage_byte_mode_returns_none():
    for mode in ("byte", "raw", "sentence"):
        assert _stage(mode, "abc def") is None


def test_stage_none_input_returns_none():
    fake = property_reader(analysis_mode="word")
    assert WholeSpace.stage_analysis_spans(fake, None) is None


def test_stage_word_mode_type_runs():
    assert _stage("word", "abc123")[0].tolist() == [[0, 3], [3, 6]]
    assert _stage("word", "hi, there")[0].tolist() == [[0, 2], [2, 3], [4, 9]]


def test_stage_accepts_three_dim_unity():
    # [B, 1, N] unity (the codebook-selection layout) -> row 0 is read.
    fake = property_reader(analysis_mode="word")
    u = _bytes("a...b").unsqueeze(1)         # [1, 1, 5]
    out = WholeSpace.stage_analysis_spans(fake, u)
    assert out[0].tolist() == [[0, 1], [1, 4], [4, 5]]


# The live analysis owner reads one learned property basis. Teaching tags name
# its canonical rows; they do not form a second frozen dictionary.

from Layers import LETTER, DIGIT, WHITESPACE, PUNCT
from Spaces import Tensor, SubSpace
from test_basicmodel import _populate_test_config
import Models
import torch.nn as nn

_D = 8
_NP = 4
_NS = 64


def _live_ws(analysis="word", nS=_NS):
    """A live WholeSpace owning the learned primitive-property basis."""
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


# -- the one learned property inventory ---------------------------------------

def test_live_ws_has_one_property_basis_with_canonical_teaching_rows():
    from Spaces import _CANONICAL_PROPERTY_ROWS
    ws = _live_ws("word")
    sub = ws.subspace
    assert isinstance(sub, SubSpace)
    tc = sub.what
    assert isinstance(tc, Codebook) and isinstance(tc, Tensor)
    assert ws.type_subspace is None
    assert tc.property_kind is None
    assert ws.well_known_atoms == {name: row for row, (name, _kind)
                                  in enumerate(_CANONICAL_PROPERTY_ROWS)}
    expected = property_reader().subspace.what.primitive_properties.coefficients()
    assert torch.equal(tc.primitive_properties.coefficients()[:len(expected)], expected)
    assert torch.is_tensor(tc.getW()) and tc.getW().requires_grad
    assert isinstance(tc.primitive_properties.members, nn.Parameter)


def test_property_teaching_rows_are_idempotent_on_rebuild():
    a = _live_ws("word")
    b = _live_ws("word")
    assert a.well_known_atoms == b.well_known_atoms
    names = dict(a.well_known_atoms)
    a.subspace.what.primitive_properties.teach(names["digit"], [ord("2")], [.25])
    coefficients = a.subspace.what.primitive_properties.coefficients().clone()
    a._build_type_subspace()
    assert a.well_known_atoms == names
    assert torch.equal(a.subspace.what.primitive_properties.coefficients(), coefficients)
    assert a.type_subspace is None


def test_byte_mode_owns_properties_without_a_type_dictionary():
    ws = _live_ws("byte")
    assert ws.type_subspace is None
    assert torch.equal(_analysis_type_lut(ws), _LUT_ANALYSIS_TYPE)
    assert not hasattr(ws, "vocab_extras")
    assert any("primitive_properties.members" in key for key in ws.state_dict())


def test_type_subspace_adds_no_state_dict_keys():
    # There is no second type dictionary in a property-only WholeSpace.
    word = _live_ws("word")
    keys = [k for k in word.state_dict().keys() if "type_subspace" in k]
    assert keys == []


# -- native cuts equal the same a-priori primitive definitions ----------------

def _cut_from_rows(ws, byte_rows):
    u = torch.tensor(byte_rows, dtype=torch.long)
    return WholeSpace.stage_analysis_spans(ws, u)


def _cut_from_apriori_properties(byte_rows):
    u = torch.tensor(byte_rows, dtype=torch.long)
    fake = property_reader(analysis_mode="word")
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
    module = _cut_from_apriori_properties(samples)
    assert torch.equal(live, module), (live.tolist(), module.tolist())
    # DEL is a control; the following three bytes share the high-byte property.
    assert [span for span in live[4].tolist() if span[1] > span[0]] == [[0, 1], [1, 4]]


# -- checkpoint save/load preserves learned memberships and their cut --------

def test_checkpoint_roundtrip_preserves_property_definitions_and_cut():
    a = _live_ws("word")
    definitions = a.subspace.what.primitive_properties
    definitions.teach(a.well_known_atoms["digit"], [ord("2")], [.25])
    state = a.state_dict()
    b = _live_ws("word")
    with torch.no_grad():
        b.subspace.what.primitive_properties.members.zero_()
    b.load_state_dict(state, strict=True)
    assert b.type_subspace is None
    assert b.subspace.what.property_kind == a.subspace.what.property_kind
    assert torch.equal(b.subspace.what.primitive_properties.coefficients(),
                       definitions.coefficients())
    rows = [list(b"ab 12! xy") + [0]]
    assert torch.equal(_cut_from_rows(b, rows), _cut_from_rows(a, rows))


def test_property_memberships_are_owned_parameters_and_train():
    ws = _live_ws("word")
    tc = ws.subspace.what
    members = tc.primitive_properties.members
    before = members.detach().clone()
    names = dict(ws.well_known_atoms)
    params = list(ws.parameters())
    assert any(p is members for p in params)
    opt = torch.optim.SGD(params, lr=0.1)
    opt.zero_grad()
    tc.primitive_properties.coefficients().square().sum().backward()
    opt.step()
    assert not torch.equal(members, before)
    assert ws.well_known_atoms == names


# == the digit whole (<digitWholes>, Alec 2026-09-10) ==========================

def _digit_spans(s, n=None):
    t = _types(s, n)
    return _type_run_spans(t, singleton=(t == _TYPE_DIGIT))[0].tolist()


def test_digit_wholes_cut_each_digit():
    assert _digit_spans("12 plus 1") == [[0, 1], [1, 2], [3, 7], [8, 9]]
    assert _digit_spans("ab12cd") == [[0, 2], [2, 3], [3, 4], [4, 6]]
    assert _digit_spans("7") == [[0, 1]]
    assert _digit_spans("abc def") == _spans("abc def")      # no digits: unchanged


def test_stage_digit_wholes_knob():
    fake = property_reader(analysis_mode="word", digit_wholes=True)
    assert WholeSpace.stage_analysis_spans(fake, _bytes("14 plus 1"))[0].tolist() == \
        [[0, 1], [1, 2], [3, 7], [8, 9]]
    fake = property_reader(analysis_mode="word", digit_wholes=False)
    assert WholeSpace.stage_analysis_spans(fake, _bytes("14 plus 1"))[0].tolist() == \
        [[0, 2], [3, 7], [8, 9]]
    assert WholeSpace.stage_analysis_spans(
        property_reader(analysis_mode="word"), _bytes("14 plus 1"))[0].tolist() == \
        [[0, 2], [3, 7], [8, 9]]                                  # default: unchanged


def test_digit_wholes_property_basis_signature(tmp_path):
    from Spaces import _digit_signature_bits, _analysis_property_signature
    from test_wholespace_property_migration import _small_property_model
    ws = _small_property_model(tmp_path).wholeSpace
    lookup, discarded = _analysis_property_signature(ws)
    sig = lookup[_bytes("x12 9")]
    single = (sig & _digit_signature_bits(ws)) != 0
    assert single[0].tolist() == [False, True, True, False, True]
    out = _type_run_spans(sig, discard_mask=discarded, singleton=single)
    assert out[0].tolist() == [[0, 1], [1, 2], [2, 3], [4, 5]]
