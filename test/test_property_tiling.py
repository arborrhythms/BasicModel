"""Char-class property tiling (S6/B1, doc/specs/mereological-order-raising.md
"Analysis = property-tiling"). A property is a BINARY TILING of a byte input
into {has-class}/{not} -- the analysis-side (WholeSpace π) dual of a synthesis
chunk. Selecting a SET of classes ORs them (the analysis-side OR over
.index-selected properties). Wired into Codebook.materialize_property's
content-keyed seam (approach (a)); flag-off (no tag / no input_bytes) keeps the
sinusoidal whole-ranging basis byte-identical.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import torch

import Spaces
from Layers import (PropertyTilingLayer, char_class_region,
                    LETTER, DIGIT, WHITESPACE, PUNCT, WORD)
from Spaces import Codebook, WholeSpace
from Layers import RunStructureLayer
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
    return WholeSpace([nP, _D], [nS, _D], [nS, _D])


# "ab 12!" -> letters, space, digits, punctuation in one row.
_B = torch.tensor([[97, 98, 32, 49, 50, 33]])


def test_char_class_letter():
    assert char_class_region(_B, [LETTER]).tolist() == [[1, 1, -1, -1, -1, -1]]


def test_char_class_digit():
    assert char_class_region(_B, [DIGIT]).tolist() == [[-1, -1, -1, 1, 1, -1]]


def test_char_class_whitespace():
    assert char_class_region(_B, [WHITESPACE]).tolist() == [[-1, -1, 1, -1, -1, -1]]


def test_char_class_punct():
    assert char_class_region(_B, [PUNCT]).tolist() == [[-1, -1, -1, -1, -1, 1]]


def test_char_class_or_union():
    # letters OR digits -> the analysis-side OR (union widens the whole).
    assert char_class_region(_B, [LETTER, DIGIT]).tolist() == [[1, 1, -1, 1, 1, -1]]


def test_complement_is_the_other_whole():
    # the tiling is binary: +1 has-property whole, -1 the complement whole.
    r = char_class_region(_B, [LETTER, DIGIT, WHITESPACE, PUNCT])
    assert r.tolist() == [[1, 1, 1, 1, 1, 1]]      # every byte is in some class
    assert char_class_region(torch.tensor([[0, 7]]), [LETTER]).tolist() == [[-1, -1]]


def test_layer_matches_function():
    layer = PropertyTilingLayer()
    assert torch.equal(layer(_B, [LETTER]), char_class_region(_B, [LETTER]))
    assert torch.equal(layer(_B, [DIGIT, PUNCT]),
                       char_class_region(_B, [DIGIT, PUNCT]))


def test_batched():
    b = torch.tensor([[97, 98, 32], [49, 33, 65]])   # 'ab ' / '1!A'
    out = char_class_region(b, [LETTER])
    assert out.tolist() == [[1, 1, -1], [-1, -1, 1]]


# -- the materialize_property content-keyed seam (approach (a)) ---------------

def _codebook(nVectors=8, nDim=6):
    cb = Codebook()
    cb.create(nInput=4, nVectors=nVectors, nDim=nDim)
    cb.addVectors(nVectors)
    return cb


def test_materialize_property_charclass_when_tagged():
    cb = _codebook()
    cb.set_property_kind(0, LETTER)
    b = torch.tensor([97, 98, 32, 49])               # 'ab 1'
    r = cb.materialize_property(torch.tensor(0), 4, input_bytes=b)
    assert r.tolist() == [1, 1, -1, -1]              # letters a,b


def test_materialize_property_or_union_over_index():
    cb = _codebook()
    cb.set_property_kind(0, LETTER)
    cb.set_property_kind(1, DIGIT)
    b = torch.tensor([97, 98, 32, 49])
    # selecting BOTH tagged rows -> OR-union of their classes (letters OR digits)
    r = cb.materialize_property(torch.tensor([0, 1]), 4, input_bytes=b)
    assert r.tolist() == [1, 1, -1, 1]


def test_materialize_property_sinusoidal_unchanged_without_bytes():
    cb = _codebook()
    base = cb.materialize_property(torch.tensor(0), 4)        # sinusoidal
    cb.set_property_kind(0, LETTER)                            # tag, but...
    same = cb.materialize_property(torch.tensor(0), 4)        # ...no input_bytes
    assert torch.equal(base, same)                            # byte-identical


def test_materialize_property_sinusoidal_when_untagged():
    cb = _codebook()
    b = torch.tensor([97, 98, 32, 49])
    # input_bytes given but row 0 NOT tagged -> sinusoidal path (not {+1,-1}).
    r = cb.materialize_property(torch.tensor(0), 4, input_bytes=b)
    assert r.shape[-1] == 4
    assert not set(r.reshape(-1).tolist()) <= {1.0, -1.0}     # not a char tiling


# -- B-span-cut: property_spans (tiling region -> .where spans) ----------------
# property_spans uses only its args + char_class_region (no WholeSpace state),
# so the unbound method runs with self=None.

_U = torch.tensor([[97, 98, 32, 49, 50, 32, 99, 100]])      # "ab 12 cd"


def test_property_spans_letter_runs():
    spans = WholeSpace.property_spans(None, _U, [LETTER])
    assert spans[0].tolist() == [[0, 2], [6, 8]]             # "ab", "cd"


def test_property_spans_digit_runs():
    spans = WholeSpace.property_spans(None, _U, [DIGIT])
    assert spans[0].tolist() == [[3, 5]]                     # "12"


def test_property_spans_or_union_runs():
    spans = WholeSpace.property_spans(None, _U, [LETTER, DIGIT])
    assert spans[0].tolist() == [[0, 2], [3, 5], [6, 8]]     # letters OR digits


def test_property_spans_feed_run_structure():
    spans = WholeSpace.property_spans(None, _U, [LETTER])
    out = RunStructureLayer()(spans.to(torch.float32))
    assert int(out["n_runs"][0]) == 2                        # two letter runs


def test_property_spans_none_passthrough():
    assert WholeSpace.property_spans(None, None, [LETTER]) is None


# -- A4: record_property_meronomy (run-span instances -> class TYPE + raise) --











# -- live cross-tower binding (reads subspace.where + WS whole spans) ----------







def test_retired_wholespace_taxonomy_writer_is_absent():
    from Spaces import WholeSpace
    assert not hasattr(WholeSpace, "insert_meta")
