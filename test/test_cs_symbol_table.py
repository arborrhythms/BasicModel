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
from Layers import WORD, UNIVERSE, ATOM, NOTHING, EVERYTHING
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
    # C = META-concept (P2 flip, sec 4c): the ORDERED PAIR
    # [whole=word-symbol, part=object-symbol] -- roles are positional slots
    # of an ordered pair, not containment claims.
    assert cs.concept_parts(C) == [("sym", B)]
    assert cs.concept_wholes(C) == [("sym", A)]


def test_meta_pair_is_discretizable_and_persists_resolve():
    """The META is the sec-4c ordered pair: its discrete reading is exact
    ([whole=A, part=B]) and -- per the SINGLETON principle (a sym-sym 1:1
    row is unit-set/pair STRUCTURE, not an id-of-indiscernibles tie) -- it
    survives resolve_identities."""
    cs = _cs()
    A, B, C = cs.create_word_object_meta([65, 66], WORD)
    alloc = cs._concept_allocator
    assert alloc.store_of(C).discretize_row(C) == (("sym", A), ("sym", B))
    resolved = cs.resolve_identities()
    assert C not in resolved
    assert cs.concept_parts(C) == [("sym", B)]       # pair persists
    assert cs.concept_wholes(C) == [("sym", A)]


def test_singleton_concept_is_idempotent_unit_set():
    """The SINGLETON (Alec 2026-07-02): a whole containing exactly one
    symbolic part -- the unit-set {x}, the constructive primitive behind
    if->then ({x} => x) and the recursion vine. Idempotent per symbol;
    persists the lifecycle (no whole -> never the 1:1 identity shape)."""
    cs = _cs()
    x = cs.relate(1, 2)
    S = cs.singleton_concept(x)
    assert cs.singleton_concept(x) == S              # idempotent
    assert cs.concept_parts(S) == [("sym", x)]       # exactly one sym part
    assert cs.concept_wholes(S) == []                # whole side disconnected
    assert cs.singleton_concept(cs.relate(3, 4)) != S
    cs.resolve_identities()
    assert cs.concept_parts(S) == [("sym", x)]       # unit-set persists


def test_singleton_populates_one_part_role_edge():
    """Min-support exemption: the singleton's weighted reading is its ONE
    part-role edge onto the constituent's activation."""
    cs = _cs_sparse_active()
    A, _B, _C = cs.create_word_object_meta([1], 2, key="w")   # order-0 sym
    S = cs.singleton_concept(A)
    assert cs._concept_source_order(S) == 1
    row = cs._csw_rows[(1, S)]
    got = cs.concept_weights(1, row)
    assert got == [(("part", cs._csw_rows[(0, A)]), 1.0)]


def test_meta_word_object_recovers_by_intersection():
    """Typed intersection read-out (Alec 2026-07-02): (word, object) come
    from intersecting the meta's sym constituents with the word-symbol
    class, not from slot order."""
    cs = _cs()
    A, B, C = cs.create_word_object_meta([65, 66], WORD, key="w")
    assert cs.meta_word_object(C) == (A, B)
    assert cs.meta_word_object(A) is None            # not a two-sym meta


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


def _cs_sparse_active(nS=64, order=1):
    """A sparse-active bare CS: stamps mirror the Models.py build stamps
    (P2: the _n_ps_codes/_n_ws_codes source-layout stamps retired with the
    percept families)."""
    cs = _cs(nS=nS)
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    return cs


# -- .when consistency at the tie site (2026-07-02 plan, Task 7) ---------------

def test_populate_cs_symbols_when_mismatch_raises():
    cs = _cs()
    pid = torch.tensor([[1, 2]])
    where = torch.tensor([[0, 1]])
    when = torch.tensor([[7, 8]])                    # DIFFERENT .when: invalid
    spans = torch.tensor([[[0, 2]]])
    try:
        cs._populate_cs_symbols(pid, where, spans, percept_when=when)
        assert False, "mismatched .when must fail loud"
    except ValueError as e:
        assert ".when" in str(e)


def test_populate_cs_symbols_when_uniform_ok():
    cs = _cs()
    pid = torch.tensor([[1, 2]])
    where = torch.tensor([[0, 1]])
    when = torch.tensor([[3, 3]])                    # same .when: fine
    spans = torch.tensor([[[0, 2]]])
    cs._populate_cs_symbols(pid, where, spans, percept_when=when)
    assert len(cs.symbols_needing_processing()) >= 1


# -- closest-links pruning rounds (2026-07-02 plan, Task 8) --------------------

def test_prune_drops_generic_word_class_when_specific_whole_linked():
    cs = _cs()
    c = cs.new_concept()
    cs.add_part(c, 5)
    cs.add_whole(c, WORD)                            # generic word-class
    cs.add_whole(c, 42)                              # specific word-whole
    dropped = cs.prune_concept_links()
    assert (c, "whole", WORD) in dropped
    assert set(cs.concept_wholes(c)) == {42}


def test_prune_drops_constituents_of_linked_raised_symbol():
    cs = _cs()
    H = cs.synthesize_higher_order([7, 8])           # raised: Parts(H)={7,8}
    c = cs.new_concept()
    cs.add_part(c, 7)                                # constituent AND ...
    cs.add_part(c, ("sym", H))                       # ... the raised H itself
    cs.add_whole(c, 42)
    dropped = cs.prune_concept_links()
    assert (c, "part", 7) in dropped
    assert ("sym", H) in cs.concept_parts(c) and 7 not in cs.concept_parts(c)


def test_prune_removes_bias_edge_of_dropped_everything():
    """Raw links are reference-store only post-P2 (no PS/WS edges); the ONE
    physical edge pruning can retire is the EVERYTHING bias of an order>=1
    concept once a tighter whole is linked."""
    cs = _cs_sparse_active()
    A, _B, _C = cs.create_word_object_meta([1], 2, key="w")   # A: order-0 sym
    c = cs.new_concept()
    cs.add_part(c, ("sym", A))                       # order 1
    cs.add_whole(c, EVERYTHING)
    cs._populate_concept_weights(c)
    _p, s1 = cs._sparse_families(1)
    before = s1.nnz
    row = cs._csw_rows[(1, c)]
    assert (("bias", 0), 1.0) in cs.concept_weights(1, row)
    cs.add_whole(c, 42)                              # a tighter whole arrives
    cs.prune_concept_links()
    assert s1.nnz == before - 1                      # the bias edge retired
    assert ("bias", 0) not in [rc for (rc, _w) in cs.concept_weights(1, row)]


def test_prune_drops_raw_links_records_only():
    """Dropping a raw generic whole edits the RECORDS; no tensor edges are
    involved (raw codes never had columns post-P2)."""
    cs = _cs_sparse_active()
    c = cs.new_concept()
    cs.add_part(c, 1)
    cs.add_whole(c, WORD)                            # generic word-class
    cs.add_whole(c, 2)                               # specific whole
    cs._populate_concept_weights(c)                  # order 0: row reserved
    assert cs.concept_weights(0, cs._csw_rows[(0, c)]) == []
    dropped = cs.prune_concept_links()
    assert (c, "whole", WORD) in dropped
    assert set(cs.concept_wholes(c)) == {2}


def test_refine_over_collected_runs_pruning_round():
    cs = _cs()
    c = cs.new_concept()
    cs.add_part(c, 5)
    cs.add_whole(c, WORD)
    cs.add_whole(c, 42)
    cs.refine_over_collected()                       # hook at its tail
    assert set(cs.concept_wholes(c)) == {42}


# -- statement channel: assert_concept_relation (2026-07-02 plan, Task 10) -----

def test_assert_relation_replaces_poles_and_enters_embedding():
    cs = _cs_sparse_active()
    A, B, C = cs.create_word_object_meta([1], 2, key="cat")
    assert set(cs.concept_parts(B)) == {ATOM}
    assert set(cs.concept_wholes(B)) == {UNIVERSE}
    # "a cat has whiskers" (whiskers-object = another concept, id 9):
    cs.assert_concept_relation(B, sym_part=9)
    assert ATOM not in cs.concept_parts(B)           # pole replaced
    # "cats are animals" (animal-object = concept 11):
    cs.assert_concept_relation(B, sym_whole=11)
    assert UNIVERSE not in cs.concept_wholes(B)
    # B's definition now has content -> it holds sparse edges (min-support).
    order = cs._concept_source_order(B)
    row = cs._csw_rows[(order, B)]
    assert cs.concept_weights(order, row) != []


def test_assert_relation_raw_codes_reserve_snap_row_no_edges():
    """Raw-code assertions land in the reference store (P2): the order-0
    concept RESERVES its snap row; no PS/WS columns exist to weight."""
    cs = _cs_sparse_active()
    c = cs.new_concept()
    cs.assert_concept_relation(c, part=3, whole=4, weight=0.5)
    assert 3 in cs.concept_parts(c) and 4 in cs.concept_wholes(c)
    row = cs._csw_rows[(0, c)]                       # snap row reserved
    assert cs.concept_weights(0, row) == []          # no edges at order 0


def test_assert_relation_sym_weight_lands_on_role_edge():
    """A weighted SYM assertion sets the trained value of the role-tagged
    edge (no_grad evidence, not a backprop target)."""
    cs = _cs_sparse_active()
    A, _B, _C = cs.create_word_object_meta([1], 2, key="w")   # order-0 sym
    c = cs.new_concept()
    cs.assert_concept_relation(c, sym_part=A, whole=7, weight=0.5)
    order = cs._concept_source_order(c)
    assert order == 1
    row = cs._csw_rows[(1, c)]
    got = dict(cs.concept_weights(1, row))
    a_row = cs._csw_rows[(0, A)]
    assert got[("part", a_row)] == 0.5               # role edge carries it


# -- the poles as presence-lattice vectors (Alec 2026-07-02) -------------------
# nothing = [0,0,...] (a part contributing zero: NO edge); everything =
# [1,1,...] (a whole = the 1-wide bias role block at order >= 1). Post-P2 a
# fresh zeroth-order object-concept B = (NOTHING, EVERYTHING) RESERVES its
# order-0 codebook row (the snap's maximally-general position; no edges).

def test_pole_rename_aliases_hold():
    from Layers import NOTHING, EVERYTHING
    assert ATOM == NOTHING and UNIVERSE == EVERYTHING


def test_fresh_object_concept_reserves_order0_row_no_edges():
    cs = _cs_sparse_active()
    A, B, C = cs.create_word_object_meta([1, 2], 3, key="cat")
    order = cs._concept_source_order(B)
    assert order == 0                                # newly minted: zeroth order
    row = cs._csw_rows[(0, B)]                       # snap row reserved
    assert cs.concept_weights(0, row) == []          # order 0: no edges
    # A likewise reserves its own order-0 row (a distinct one).
    assert cs._csw_rows[(0, A)] != row


def test_assert_concrete_whole_retires_everything_bias_edge():
    """An order>=1 concept holding the EVERYTHING bias loses that edge when
    a concrete whole replaces the pole (the wide-open object narrows)."""
    cs = _cs_sparse_active()
    A, _B, _C = cs.create_word_object_meta([1, 2], 3, key="cat")
    c = cs.new_concept()
    cs.add_part(c, ("sym", A))                       # order 1
    cs.add_whole(c, EVERYTHING)
    cs._populate_concept_weights(c)
    row = cs._csw_rows[(1, c)]
    assert (("bias", 0), 1.0) in cs.concept_weights(1, row)
    cs.assert_concept_relation(c, whole=5)           # concrete whole arrives
    roles = [rc for (rc, _w) in cs.concept_weights(1, row)]
    assert ("bias", 0) not in roles                  # bias edge retired
    assert 5 in cs.concept_wholes(c)                 # the record landed


def test_prune_drops_everything_when_other_whole_linked():
    cs = _cs_sparse_active()
    c = cs.new_concept()
    cs.add_part(c, 1)
    cs.add_whole(c, EVERYTHING)
    cs.add_whole(c, 2)
    cs._populate_concept_weights(c)
    dropped = cs.prune_concept_links()
    assert (c, "whole", EVERYTHING) in dropped
    assert set(cs.concept_wholes(c)) == {2}


# -- Hebbian word/object tie strengthening (2026-07-02 plan, Task 11) ----------

def test_word_object_tie_strengthens_on_reoccurrence():
    cs = _cs_sparse_active()
    A, B, C = cs.create_word_object_meta([1, 2], 3, key="cat")
    order = cs._concept_source_order(C)
    row = cs._csw_rows[(order, C)]
    before = dict(cs.concept_weights(order, row))
    assert before                                    # C holds sym-ref edges
    A2, B2, C2 = cs.create_word_object_meta([1, 2], 3, key="cat")  # re-mint
    assert (A2, B2, C2) == (A, B, C)
    after = dict(cs.concept_weights(order, row))
    assert any(after[c] > before[c] for c in before)  # Hebbian bump
    assert all(v <= 4.0 for v in after.values())      # clamped




def test_joint_concept_is_bias_bounded_chain():
    """The joint (P2 decision 6) is the ordered Gallistel CHAIN over the
    word-symbols: each link the pair [whole=current, part=rest] bounded by
    the EVERYTHING bias -- a proper hidden unit over role-tagged edges."""
    cs = _cs_sparse_active()
    A1, _B1, _C1 = cs.create_word_object_meta([1], 2, key="w1")
    A2, _B2, _C2 = cs.create_word_object_meta([3], 4, key="w2")
    J = cs.create_joint_concept([A1, A2], key=("w1", "w2"))
    assert cs._concept_source_order(J) == 1
    # The head link is the ordered pair [whole=A1 (current), part=A2 (rest)].
    assert cs.concept_parts(J) == [("sym", A2)]
    assert ("sym", A1) in cs.concept_wholes(J)
    assert EVERYTHING in cs.concept_wholes(J)            # bias-bounded
    row = cs._csw_rows[(1, J)]
    got = dict(cs.concept_weights(1, row))
    roles = list(got)
    assert ("bias", 0) in roles                          # the bias edge
    assert ("whole", cs._csw_rows[(0, A1)]) in roles     # whole = current word
    assert ("part", cs._csw_rows[(0, A2)]) in roles      # part = the rest
    # ORDERED: the reversed sentence is a DIFFERENT chain head.
    assert cs.create_joint_concept([A2, A1], key=("w2", "w1")) != J
    # Idempotent per key; re-occurrence strengthens Hebbianly.
    before = dict(cs.concept_weights(1, row))
    assert cs.create_joint_concept([A1, A2], key=("w1", "w2")) == J
    after = dict(cs.concept_weights(1, row))
    assert any(after[c] > before[c] for c in before)
