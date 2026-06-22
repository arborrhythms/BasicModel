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
from Layers import WORD, UNIVERSE, ATOM
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
    s = cs.new_concept()
    assert cs.concept_parts(s) == [] and cs.concept_wholes(s) == []


def test_parts_wholes_are_independent_multivalued():
    cs = _cs()
    A = cs.new_concept()
    cs.add_part(A, "cat-object")
    cs.add_part(A, "felix-object")              # A accumulates a second part
    cs.add_whole(A, "cat-object-properties")
    assert set(cs.concept_parts(A)) == {"cat-object", "felix-object"}
    assert cs.concept_wholes(A) == ["cat-object-properties"]


def test_relate_is_idempotent_per_pair():
    cs = _cs()
    s1 = cs.relate(10, 20)                       # part-code 10 <= whole-code 20
    s2 = cs.relate(10, 20)
    assert s1 == s2                              # minted once
    assert cs.concept_parts(s1) == [10] and cs.concept_wholes(s1) == [20]
    assert cs.relate(10, 21) != s1               # different whole -> new symbol


def test_reify_relation_between_symbols():
    cs = _cs()
    A = cs.relate(1, 2)                          # cat-object <= cat-object-props
    B = cs.relate(3, 4)                          # cat-word   <= cat-word-props
    C = cs.reify_concept(A, B)                  # "the object is a word"
    assert cs.concept_parts(C) == [("sym", A)]
    assert cs.concept_wholes(C) == [("sym", B)]


def test_identity_at_one_part_one_whole():
    cs = _cs()
    A = cs.relate(1, 2)
    assert cs.concept_is_identity(A)              # 1 part + 1 whole = identity tie
    cs.add_part(A, 99)
    assert not cs.concept_is_identity(A)          # 2 parts -> no longer identity


def test_over_collection_is_actionable():
    cs = _cs()
    A = cs.new_concept()
    for p in range(6):
        cs.add_part(A, p)                        # 6 parts > k_parts
    assert cs.concept_over_collected(A, k_parts=4)
    B = cs.new_concept()
    for w in range(6):
        cs.add_whole(B, ("w", w))                # 6 wholes > k_wholes
    assert cs.concept_over_collected(B, k_wholes=4)
    assert not cs.concept_over_collected(cs.relate(7, 8))   # 1+1, not over


def test_retire_transient_symbol():
    cs = _cs()
    A = cs.relate(1, 2)
    cs.retire_concept(A)
    assert cs.concept_parts(A) == [] and cs.concept_wholes(A) == []
    # idempotency cache cleared -> re-relating the same pair mints a NEW symbol
    assert cs.relate(1, 2) != A
    cs.retire_concept(A)                          # idempotent


# -- live population from the .where binding (S2a) -----------------------------

def test_populate_cs_symbols_location_knitting():
    cs = _cs()
    pid_2d = torch.tensor([[65, 66, 67, 68]])        # 4 percepts
    percept_where = torch.tensor([[0, 1, 2, 9]])     # first 3 in (0,3); last at 9
    spans = torch.tensor([[[0, 3], [8, 10]]])        # word-spans (0,3) and (8,10)
    cs._populate_cs_symbols(pid_2d, percept_where, spans)
    parts_sets = [frozenset(cs._concept_parts[s]) for s in cs._concept_parts]
    assert frozenset({65, 66, 67}) in parts_sets     # span (0,3): the 3 covered
    assert frozenset({68}) in parts_sets             # span (8,10): just 68
    # every location-symbol carries the word-whole
    for s in cs._concept_parts:
        assert WORD in cs._concept_wholes[s]


def test_resolve_identities_zeros_out_1to1():
    cs = _cs()
    A = cs.relate(1, 2)                               # 1 part + 1 whole -> identity
    B = cs.new_concept()
    cs.add_part(B, 10)
    cs.add_part(B, 11)                                # 2 parts -> not 1:1
    cs.add_whole(B, 20)
    resolved = cs.resolve_identities()
    assert A in resolved and B not in resolved
    assert cs.symbol_identity(A) == (1, 2)           # identity recorded
    assert cs.concept_parts(A) == [] and cs.concept_wholes(A) == []   # sets vanished
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
    assert (68, WORD) in [cs.symbol_identity(s) for s in cs._concept_parts]
    # the 3-part location (N:1) stays ACTIVE -- the send-back candidate
    active = cs.symbols_needing_processing()
    assert any(set(cs._concept_parts[s]) == {65, 66, 67} for s in active)


# -- word / object / meta creation (Alec 2026-06-17) -------------------------

def test_create_word_object_meta_structure():
    cs = _cs()
    # PS gives the word-parts (pids 65,66,67); WS gives the word-whole (WORD).
    A, B, C = cs.create_word_object_meta([65, 66, 67], WORD)
    # A = WORD-symbol: parts = the word-parts, whole = the word-whole.
    assert set(cs.concept_parts(A)) == {65, 66, 67}
    assert cs.concept_wholes(A) == [WORD]
    # B = OBJECT-symbol: maximally unspecified -- ATOM <= UNIVERSE, to be refined.
    assert cs.concept_parts(B) == [ATOM]
    assert cs.concept_wholes(B) == [UNIVERSE]
    # C = META: reify A <= B (the word≡object binding).
    assert cs.concept_parts(C) == [("sym", A)]
    assert cs.concept_wholes(C) == [("sym", B)]


def test_create_word_object_meta_idempotent_per_key():
    cs = _cs()
    A1, B1, C1 = cs.create_word_object_meta([65, 66], WORD, key="ab")
    A2, B2, C2 = cs.create_word_object_meta([65, 66], WORD, key="ab")
    assert (A1, B1, C1) == (A2, B2, C2)              # same word -> same triple
    # a different surface text mints a fresh triple
    A3, _, _ = cs.create_word_object_meta([67], WORD, key="c")
    assert A3 != A1
    # no key -> always fresh
    A4, _, _ = cs.create_word_object_meta([65, 66], WORD)
    assert A4 != A1


def test_create_word_object_meta_accumulates_word_parts():
    cs = _cs()
    A, _, _ = cs.create_word_object_meta([65], WORD, key="grows")
    # the same word re-presented with a new spelled-out part accrues it onto A.
    A2, _, _ = cs.create_word_object_meta([66], WORD, key="grows")
    assert A2 == A
    assert set(cs.concept_parts(A)) == {65, 66}


def test_object_symbol_is_refinable_not_yet_identity():
    cs = _cs()
    _, B, _ = cs.create_word_object_meta([65, 66, 67], WORD)
    # B starts as a 1-part/1-whole tie (ATOM <= UNIVERSE) -- structurally an
    # identity SHAPE, but it is the unspecified poles awaiting refinement, so it
    # is still in the active set until the lifecycle specializes it.
    assert cs.concept_is_identity(B)                  # one pole each
    assert B in cs.symbols_needing_processing()      # poles not yet resolved


def test_resolve_identities_does_not_collapse_unspecified_object():
    cs = _cs()
    A, B, C = cs.create_word_object_meta([65, 66, 67], WORD)
    resolved = cs.resolve_identities()
    # B (ATOM <= UNIVERSE) is the unspecified object -> NOT collapsed; it stays
    # active for refinement and is not recorded as a resolved identity.
    assert B not in resolved
    assert cs.symbol_identity(B) is None
    assert B in cs.symbols_needing_processing()
    assert cs.concept_parts(B) == [ATOM] and cs.concept_wholes(B) == [UNIVERSE]
    # once refined to a CONCRETE part + whole, it DOES resolve to an identity.
    cs._concept_parts[B] = {65}
    cs._concept_wholes[B] = {WORD}
    assert B in cs.resolve_identities()
    assert cs.symbol_identity(B) == (65, WORD)


# -- over-collection lifecycle: retire-on-trigger driving both towers --------

def test_refine_over_collected_too_many_parts_requests_synthesize():
    cs = _cs()
    A = cs.new_concept()
    for p in range(6):
        cs.add_part(A, p)                 # 6 parts > k_many(4) -> over-collected
    cs.add_whole(A, 100)
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    assert len(reqs) == 1
    assert reqs[0]["sym"] == A and reqs[0]["op"] == "synthesize"
    assert set(reqs[0]["codes"]) == set(range(6))
    # retire-on-trigger: the transient triggering symbol is gone.
    assert cs.concept_parts(A) == [] and A not in cs.symbols_needing_processing()


def test_refine_over_collected_too_many_wholes_requests_analyse():
    cs = _cs()
    B = cs.new_concept()
    cs.add_part(B, 10)
    for w in range(6):
        cs.add_whole(B, ("w", w))         # 6 wholes > k_many -> over-collected
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    assert len(reqs) == 1 and reqs[0]["op"] == "analyse" and reqs[0]["sym"] == B
    assert len(reqs[0]["codes"]) == 6


def test_refine_over_collected_both_sides_emits_both_requests():
    cs = _cs()
    S = cs.new_concept()
    for p in range(5):
        cs.add_part(S, p)
    for w in range(5):
        cs.add_whole(S, ("w", w))
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    ops = sorted(r["op"] for r in reqs)
    assert ops == ["analyse", "synthesize"]
    assert all(r["sym"] == S for r in reqs)
    assert S not in cs.symbols_needing_processing()        # retired once


def test_refine_over_collected_leaves_well_sized_and_identities_alone():
    cs = _cs()
    ident = cs.relate(1, 2)                                 # 1:1 -> identity
    keep = cs.new_concept()
    cs.add_part(keep, 7)
    cs.add_part(keep, 8)                                    # 2 parts <= 4
    cs.add_whole(keep, 9)
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    assert reqs == []                                       # nothing over-collected
    assert cs.symbol_identity(ident) == (1, 2)             # identity resolved
    assert keep in cs.symbols_needing_processing()         # well-sized, untouched


def test_refine_over_collected_is_idempotent():
    cs = _cs()
    A = cs.new_concept()
    for p in range(6):
        cs.add_part(A, p)
    cs.add_whole(A, 100)
    assert len(cs.refine_over_collected(k_parts=4, k_wholes=4)) == 1
    assert cs.refine_over_collected(k_parts=4, k_wholes=4) == []   # already retired


def test_refine_over_collected_applies_synthesis():
    cs = _cs()
    A = cs.new_concept()
    for p in range(6):
        cs.add_part(A, p)
    cs.add_whole(A, 100)
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    H = reqs[0]["result"]                       # the σ-synthesized higher-order
    # H subsumes the over-collected parts as its provenance...
    assert set(cs.concept_parts(H)) == set(range(6))
    # ...is tagged RAISED (so the lifecycle never re-refines it), and is
    # therefore excluded from the active processing set even though it carries
    # > k_many parts.
    assert H in cs._concept_raise_set()
    assert H not in cs.symbols_needing_processing()
    assert not cs.refine_over_collected(k_parts=4, k_wholes=4)  # H not re-raised


def test_synthesize_higher_order_idempotent_per_part_set():
    cs = _cs()
    H1 = cs.synthesize_higher_order([1, 2, 3])
    H2 = cs.synthesize_higher_order([3, 2, 1])   # same set, order-independent
    assert H1 == H2
    assert cs.synthesize_higher_order([1, 2, 4]) != H1   # different set -> new


