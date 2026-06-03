"""D1 gate: do the participation patterns drive a grammar-recovering collapse?

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§9 D1/D2 + §10.

Role-collapse does NOT replace declared shared POS with another single-label
POS system. It replaces declared shared categories with operator-local
participation categories, and a word may participate in SEVERAL of them ---
those overlaps are expected, not a failure. D1 is therefore not a single-label
POS-recovery test (an earlier version wrongly measured nearest-neighbor POS
agreement and "passed" by staying low). Its real purpose: show the
participation patterns are structured enough to drive a later LEARNED COLLAPSE
into a smaller set of mutually-exclusive syntactic categories where the live
parser needs them --- a collapse that keeps every grammar rule distinguishable
(so the parser retains its choices). The gate is MET when such a compacting,
parser-recovering collapse exists.

  1. On a CLEAN fixture (same-POS symbols interchangeable in operator roles),
     the participation learner recovers the declared POS perfectly.
  2. On the real transitional grammar the role-collapsed OP-ROLES recover
     (order variants collapse to one class).
  3. On the real grammar a participation-guided, conflict-free ``learned_
     collapse`` compacts the symbols into a strictly smaller mutually-exclusive
     category set with ZERO rule-conflicts -- the parser recovers, so the D1
     gate is met under the corrected criterion. (The exact substitutability
     congruence is trivial here -- every symbol is context-unique -- so
     "recovers" means the parser's rule decisions survive the collapse, not
     exact rule regeneration.)
"""

import os
import sys
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


# A clean fixture: each POS has multiple symbols that are INTERCHANGEABLE --
# they fill exactly the same operator roles (the role-collapse premise).
_FIXTURE = {
    "dog":  {("pred", 0), ("obj", 1)}, "cat": {("pred", 0), ("obj", 1)},
    "fish": {("pred", 0), ("obj", 1)},                       # N
    "run":  {("pred", 1)}, "eat": {("pred", 1)},             # V
    "the":  {("det", 0)}, "a": {("det", 0)},                 # DET
    "red":  {("mod", 0)}, "big": {("mod", 0)},               # ADJ
}
_FIXTURE_POS = {
    "dog": "N", "cat": "N", "fish": "N", "run": "V", "eat": "V",
    "the": "DET", "a": "DET", "red": "ADJ", "big": "ADJ",
}


def _is_perfect(recovered, truth):
    """True iff the clustering exactly partitions by POS (each POS is one
    class and each class is one POS)."""
    pos_to_classes = defaultdict(set)
    class_to_pos = defaultdict(set)
    for sym, c in recovered.items():
        pos_to_classes[truth[sym]].add(c)
        class_to_pos[c].add(truth[sym])
    return (all(len(cs) == 1 for cs in pos_to_classes.values())
            and all(len(ps) == 1 for ps in class_to_pos.values())
            and len(set(recovered.values())) == len(set(truth.values())))


def test_clean_fixture_recovers_declared_pos_perfectly():
    """The D1 principle: where same-POS symbols are interchangeable, the
    participation learner recovers the declared POS exactly."""
    from participation import cluster_by_participation
    recovered = cluster_by_participation(_FIXTURE)
    assert _is_perfect(recovered, _FIXTURE_POS), recovered


def test_real_grammar_role_collapsed_op_roles_recover():
    """On the real transitional grammar the role-collapsed OP-ROLES recover:
    the order variants of a per-position role collapse to one class -- the
    structure the role-collapsed grammar actually uses."""
    from Language import Grammar
    from participation import role_participation, cluster_by_participation
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    cls = cluster_by_participation(role_participation(g))
    for fam in (("CONJ_L3", "CONJ_L4", "CONJ_L5"),
                ("DISJ_R3", "DISJ_R4", "DISJ_R5")):
        ids = {cls[s] for s in fam if s in cls}
        assert len(ids) == 1, (fam, ids)


def test_participation_drives_recovering_collapse():
    """D1 (corrected criterion -- the gate is MET).

    The participation patterns are structured enough to drive a learned
    collapse into a strictly smaller mutually-exclusive category set that
    recovers the grammar: every grammar rule stays distinguishable under the
    collapse (zero conflicts), so the live chart parser keeps its
    mutually-exclusive choices. This is the property that justifies promoting
    role-collapse, in place of the earlier (mis-specified) single-label POS
    nearest-neighbor test.
    """
    from Language import Grammar
    from participation import (grammar_rules, learned_collapse,
                               collapse_conflicts)
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    rules = grammar_rules(g)
    collapse = learned_collapse(g)
    n_sym = len(collapse)
    n_cat = len(set(collapse.values()))

    # Mutually-exclusive: each symbol maps to exactly one category (a dict).
    assert all(isinstance(c, int) for c in collapse.values())
    # Compaction: strictly fewer categories than symbols (a real collapse).
    assert n_cat < n_sym, (n_cat, n_sym)
    # Recovery: the parser keeps every rule distinguishable under the collapse.
    assert collapse_conflicts(rules, collapse) == 0, (
        f"collapse introduced parser conflicts: "
        f"{collapse_conflicts(rules, collapse)}")
    # The collapse is a substantial compaction, not a token one.
    assert n_cat <= 0.7 * n_sym, (
        f"weak collapse: {n_cat} categories from {n_sym} symbols")
