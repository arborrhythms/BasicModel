"""Dimensionality-from-participation: recover POS / order from a symbol's
distribution of participation across operator roles (Phase R4).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§7.2 / §8 R4 / §9 D1 / §10. This is the learner that *justifies* role
collapse: POS / construction membership is recovered from stable
participation in operator roles, not declared. The D1 gate is that this
recovers the role structure the transitional grammar declared -- on a
fixture, the transitional grammar's order-variant and per-position role
categories collapse onto exactly the role-collapsed operator roles
(``op_I<n>``), recovered from participation ALONE (names ignored).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _complete():
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    return g


# -- The clean principle, on a synthetic fixture -------------------------

def test_recovers_noun_verb_classes_from_participation_alone():
    """Symbols that occur in the same operator-role slots are recovered as
    one class; different participation -> different class. Names unused."""
    from participation import cluster_by_participation
    profiles = {
        # three "nouns": object of eat, subject of run.
        "cat":  {("eat", 1), ("run", 0)},
        "dog":  {("eat", 1), ("run", 0)},
        "fish": {("eat", 1), ("run", 0)},
        # two "verbs": predicate of say.
        "eat":  {("say", 1)},
        "run":  {("say", 1)},
    }
    cls = cluster_by_participation(profiles)
    assert cls["cat"] == cls["dog"] == cls["fish"]
    assert cls["eat"] == cls["run"]
    assert cls["cat"] != cls["eat"]


def test_threshold_controls_grouping_granularity():
    """Overlapping-but-not-identical signatures group at a looser threshold
    and split at the exact (1.0) threshold."""
    from participation import cluster_by_participation
    profiles = {
        "a": {("op", 0), ("op2", 0)},
        "b": {("op", 0)},                 # subset of a: Jaccard(a,b)=0.5
    }
    assert cluster_by_participation(profiles, threshold=1.0)["a"] != \
        cluster_by_participation(profiles, threshold=1.0)["b"]
    loose = cluster_by_participation(profiles, threshold=0.5)
    assert loose["a"] == loose["b"]


# -- The D1 gate, on the transitional grammar ----------------------------

_ORDER_VARIANT_FAMILIES = [
    ("CONJ_L3", "CONJ_L4", "CONJ_L5"),
    ("CONJ_R3", "CONJ_R4", "CONJ_R5"),
    ("DISJ_L3", "DISJ_L4", "DISJ_L5"),
    ("DISJ_R3", "DISJ_R4", "DISJ_R5"),
    ("NP_EQ3", "NP_EQ4", "NP_EQ5"),
]


def test_order_variants_collapse_to_one_class():
    """The transitional grammar's compact-order variants (CONJ_L3/4/5, ...)
    have identical participation and collapse to a single recovered class
    -- they were never distinct roles."""
    from participation import role_participation, cluster_by_participation
    g = _complete()
    cls = cluster_by_participation(role_participation(g))
    for fam in _ORDER_VARIANT_FAMILIES:
        ids = {cls[s] for s in fam if s in cls}
        assert len(ids) == 1, f"{fam} did not collapse to one class: {ids}"


def test_single_role_categories_recover_the_collapsed_op_role():
    """A transitional category that participates in exactly one operator
    role recovers that role-collapsed role name (op_I<n>). Distinct declared
    categories that fill the SAME role (NP_EQ* and QLEFT_NP3 both isEqual
    input 0) correctly unify -- the role-collapse insight."""
    from participation import single_role_symbols
    sr = single_role_symbols(_complete())
    assert sr["CONJ_L3"] == "conjunction_I1"
    assert sr["CONJ_R3"] == "conjunction_I2"
    assert sr["DISJ_L3"] == "disjunction_I1"
    assert sr["DISJ_R3"] == "disjunction_I2"
    assert sr["NP_EQ3"] == "isEqual_I1"
    assert sr["QLEFT_NP3"] == "isEqual_I1"   # same role as NP_EQ -> unified


def test_recovers_operand_order_from_participation():
    """Left operands recover position 0, right operands position 1 (order is
    read off the participation slot, not declared)."""
    from participation import role_participation
    part = role_participation(_complete())
    assert ("conjunction", 0) in part["CONJ_L3"]
    assert ("conjunction", 1) not in part["CONJ_L3"]
    assert ("conjunction", 1) in part["CONJ_R3"]
    assert ("disjunction", 0) in part["DISJ_L3"]
    assert ("disjunction", 1) in part["DISJ_R3"]


def test_recovered_classes_are_role_pure_d1_gate():
    """D1 gate: every recovered class of single-role categories maps to
    exactly ONE role-collapsed role (cluster purity 1.0), and recovery
    strictly collapses the declared category set."""
    from collections import defaultdict
    from participation import (role_participation, single_role_symbols,
                               cluster_by_participation)
    g = _complete()
    sr = single_role_symbols(g)
    part = role_participation(g)
    cls = cluster_by_participation({s: part[s] for s in sr})
    by_class = defaultdict(set)
    for s, c in cls.items():
        by_class[c].add(sr[s])
    assert all(len(roles) == 1 for roles in by_class.values()), dict(by_class)
    # Collapse actually happened: fewer classes than declared categories.
    assert len(by_class) < len(sr)
