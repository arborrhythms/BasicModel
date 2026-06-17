"""Relation-only CS symbol table + taxonomy (Stage 1; doc/specs/mereological-
order-raising.md "The CS symbol table + taxonomy").

A symbol is RELATION-ONLY (no vector): defined by two INDEPENDENT multi-valued
sets Parts(S)/Wholes(S), tying PS part-codes <-> WS whole-codes. Relations
between symbols are reified as symbols. Symbols accumulate parts/wholes, become
actionable when over-collecting, and collapse to an identity tie at one-part-
one-whole. This is the NEW additive home on ConceptualSpace.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
from Layers import WORD
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])


def test_new_symbol_starts_empty():
    cs = _cs()
    s = cs.new_symbol()
    assert cs.symbol_parts(s) == [] and cs.symbol_wholes(s) == []


def test_parts_wholes_are_independent_multivalued():
    cs = _cs()
    A = cs.new_symbol()
    cs.add_part(A, "cat-object")
    cs.add_part(A, "felix-object")              # A accumulates a second part
    cs.add_whole(A, "cat-object-properties")
    assert set(cs.symbol_parts(A)) == {"cat-object", "felix-object"}
    assert cs.symbol_wholes(A) == ["cat-object-properties"]


def test_relate_is_idempotent_per_pair():
    cs = _cs()
    s1 = cs.relate(10, 20)                       # part-code 10 <= whole-code 20
    s2 = cs.relate(10, 20)
    assert s1 == s2                              # minted once
    assert cs.symbol_parts(s1) == [10] and cs.symbol_wholes(s1) == [20]
    assert cs.relate(10, 21) != s1               # different whole -> new symbol


def test_reify_relation_between_symbols():
    cs = _cs()
    A = cs.relate(1, 2)                          # cat-object <= cat-object-props
    B = cs.relate(3, 4)                          # cat-word   <= cat-word-props
    C = cs.reify_relation(A, B)                  # "the object is a word"
    assert cs.symbol_parts(C) == [("sym", A)]
    assert cs.symbol_wholes(C) == [("sym", B)]


def test_identity_at_one_part_one_whole():
    cs = _cs()
    A = cs.relate(1, 2)
    assert cs.symbol_is_identity(A)              # 1 part + 1 whole = identity tie
    cs.add_part(A, 99)
    assert not cs.symbol_is_identity(A)          # 2 parts -> no longer identity


def test_over_collection_is_actionable():
    cs = _cs()
    A = cs.new_symbol()
    for p in range(6):
        cs.add_part(A, p)                        # 6 parts > k_parts
    assert cs.symbol_over_collected(A, k_parts=4)
    B = cs.new_symbol()
    for w in range(6):
        cs.add_whole(B, ("w", w))                # 6 wholes > k_wholes
    assert cs.symbol_over_collected(B, k_wholes=4)
    assert not cs.symbol_over_collected(cs.relate(7, 8))   # 1+1, not over


def test_retire_transient_symbol():
    cs = _cs()
    A = cs.relate(1, 2)
    cs.retire_symbol(A)
    assert cs.symbol_parts(A) == [] and cs.symbol_wholes(A) == []
    # idempotency cache cleared -> re-relating the same pair mints a NEW symbol
    assert cs.relate(1, 2) != A
    cs.retire_symbol(A)                          # idempotent


# -- live population from the .where binding (S2a) -----------------------------

def test_populate_cs_symbols_location_knitting():
    cs = _cs()
    pid_2d = torch.tensor([[65, 66, 67, 68]])        # 4 percepts
    percept_where = torch.tensor([[0, 1, 2, 9]])     # first 3 in (0,3); last at 9
    spans = torch.tensor([[[0, 3], [8, 10]]])        # word-spans (0,3) and (8,10)
    cs._populate_cs_symbols(pid_2d, percept_where, spans)
    parts_sets = [frozenset(cs._sym_parts[s]) for s in cs._sym_parts]
    assert frozenset({65, 66, 67}) in parts_sets     # span (0,3): the 3 covered
    assert frozenset({68}) in parts_sets             # span (8,10): just 68
    # every location-symbol carries the word-whole
    for s in cs._sym_parts:
        assert WORD in cs._sym_wholes[s]


def test_resolve_identities_zeros_out_1to1():
    cs = _cs()
    A = cs.relate(1, 2)                               # 1 part + 1 whole -> identity
    B = cs.new_symbol()
    cs.add_part(B, 10)
    cs.add_part(B, 11)                                # 2 parts -> not 1:1
    cs.add_whole(B, 20)
    resolved = cs.resolve_identities()
    assert A in resolved and B not in resolved
    assert cs.symbol_identity(A) == (1, 2)           # identity recorded
    assert cs.symbol_parts(A) == [] and cs.symbol_wholes(A) == []   # sets vanished
    assert A not in cs.symbols_needing_processing()  # done, no further processing
    assert B in cs.symbols_needing_processing()      # still needs refinement
    assert cs.symbol_identity(B) is None


def test_populate_then_resolve_identities_live_lifecycle():
    cs = _cs()
    pid_2d = torch.tensor([[65, 66, 67, 68]])
    percept_where = torch.tensor([[0, 1, 2, 9]])
    spans = torch.tensor([[[0, 3], [8, 10]]])        # (0,3): 3 parts; (8,10): 1 part
    cs._populate_cs_symbols(pid_2d, percept_where, spans)
    cs.resolve_identities()
    # the single-part location ({68} <= WORD) is a 1:1 identity -> zeroed out
    assert (68, WORD) in [cs.symbol_identity(s) for s in cs._sym_parts]
    # the 3-part location (N:1) stays ACTIVE -- the send-back candidate
    active = cs.symbols_needing_processing()
    assert any(set(cs._sym_parts[s]) == {65, 66, 67} for s in active)


