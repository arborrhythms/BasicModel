"""Dimensionality-from-participation (Phase R4): recover a symbol's POS /
order from its distribution of participation across operator roles.

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§7.2 / §8 R4 / §9 D1. The role-collapsed grammar declares only operator
roles; POS / construction membership is LEARNED here from which operator
roles a symbol stably participates in. Two symbols that fill the same role
slots are the same category; the slot's position is the operand order. This
is the learner that justifies dropping declared POS categories: on the
transitional grammar it recovers exactly the role-collapsed op-roles the
declared categories spread across (the D1 gate).
"""


def role_name(method, pos):
    """Role-collapsed role name for operator ``method`` operand ``pos``
    (0-based): ``op_I<n>`` with 1-based ``n`` (matching Section 4.1)."""
    return f"{method}_I{int(pos) + 1}"


def role_participation(grammar, direction="compose"):
    """Map each symbol to the frozenset of ``(method, position)`` operator
    role slots it occupies as an operand across the grammar's rules.

    Only operator rules (``method_name`` set) in the chosen ``direction``
    (``"compose"`` upward / ``"generate"`` downward / ``"both"``) contribute;
    bare projections (``method_name is None``) declare no role and are
    skipped. The result is a symbol's recovered participation signature --
    its "dimensionality": which operator roles it can fill.
    """
    if direction == "compose":
        rules = grammar.rules_upward
    elif direction == "generate":
        rules = grammar.rules_downward
    else:
        rules = list(grammar.rules_upward) + list(grammar.rules_downward)
    prof = {}
    for r in rules:
        if getattr(r, "method_name", None) is None:
            continue
        for pos, sym in enumerate(r.rhs_symbols or ()):
            prof.setdefault(sym, set()).add((r.method_name, pos))
    return {s: frozenset(v) for s, v in prof.items()}


def single_role_symbols(grammar, direction="compose"):
    """Symbols that participate in exactly ONE operator role, mapped to that
    role-collapsed role name. These recover unambiguously; distinct declared
    categories that fill the same single role (e.g. ``NP_EQ*`` and
    ``QLEFT_NP3`` both ``isEqual`` input 0) unify onto one role name -- the
    role-collapse insight that the transitional grammar over-declared."""
    part = role_participation(grammar, direction=direction)
    out = {}
    for sym, sig in part.items():
        if len(sig) == 1:
            method, pos = next(iter(sig))
            out[sym] = role_name(method, pos)
    return out


def _jaccard(a, b):
    """Jaccard similarity of two role-slot sets (empty vs empty == 1.0)."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def cluster_by_participation(profiles, threshold=1.0):
    """Cluster items by participation-signature similarity (single-linkage
    union over Jaccard >= ``threshold``).

    ``profiles`` maps item -> iterable of role slots. ``threshold=1.0``
    groups only identical signatures (exact role-equivalence -- e.g. the
    transitional grammar's order variants); a looser threshold groups
    overlapping signatures into broader POS classes. Returns
    ``{item: class_id}`` with stable integer class ids (assigned by first
    appearance of each class root).
    """
    items = list(profiles)
    sig = {it: frozenset(profiles[it]) for it in items}
    parent = {it: it for it in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _jaccard(sig[items[i]], sig[items[j]]) >= threshold:
                union(items[i], items[j])

    root_id, out = {}, {}
    for it in items:
        r = find(it)
        if r not in root_id:
            root_id[r] = len(root_id)
        out[it] = root_id[r]
    return out


# ---------------------------------------------------------------------------
# D1 gate: a learned collapse of the participation patterns into a smaller
# MUTUALLY-EXCLUSIVE category set that RECOVERS the grammar.
#
# Role-collapse does not replace declared shared POS with another single-label
# POS system; it replaces them with operator-local participation categories a
# word may fill SEVERAL of (overlaps are expected). D1 is therefore NOT a
# single-label POS-recovery test. Its point is to show the participation
# patterns are structured enough to DRIVE a learned collapse into the smaller
# mutually-exclusive category set the live chart parser needs --- a collapse
# that keeps every grammar rule distinguishable (so the parser retains its
# choices). On the transitional grammar the exact substitutability congruence
# is trivial (every symbol is context-unique), so "recovers" cannot mean exact
# rule regeneration; it means the parser's rule decisions survive the collapse.
# ---------------------------------------------------------------------------


def grammar_rules(grammar, direction="compose"):
    """Operator-rule signatures ``(lhs, method, rhs_tuple)`` of ``grammar`` in
    the chosen ``direction``; bare projections (no ``method_name``) skipped."""
    if direction == "compose":
        rules = grammar.rules_upward
    elif direction == "generate":
        rules = grammar.rules_downward
    else:
        rules = list(grammar.rules_upward) + list(grammar.rules_downward)
    out = set()
    for r in rules:
        if getattr(r, "method_name", None) is None:
            continue
        out.add((str(r.lhs).strip(), r.method_name,
                 tuple(str(s).strip() for s in (r.rhs_symbols or ()))))
    return out


def collapse_conflicts(rules, collapse):
    """Number of DISTINCT rules that collide onto a shared lifted signature
    ``(category(lhs), method, tuple(category(rhs)))`` under ``collapse`` ---
    i.e. the parser choices the collapse would destroy. Zero means every rule
    stays distinguishable (the collapse recovers the grammar for the parser)."""
    seen, conflicts = {}, 0
    for rule in rules:
        lhs, method, rhs = rule
        sig = (collapse.get(lhs), method, tuple(collapse.get(s) for s in rhs))
        if sig in seen and seen[sig] != rule:
            conflicts += 1
        else:
            seen.setdefault(sig, rule)
    return conflicts


def _combined_signature(rules):
    """Per-symbol combined in/out participation signature over ``rules``:
    operand slots ``("in", method, pos)`` plus result slots ``("out", method)``.
    This is the multi-label participation pattern a learned collapse merges."""
    prof = {}
    for (lhs, method, rhs) in rules:
        for pos, sym in enumerate(rhs):
            prof.setdefault(sym, set()).add(("in", method, pos))
        prof.setdefault(lhs, set()).add(("out", method))
    return {s: frozenset(v) for s, v in prof.items()}


def learned_collapse(grammar, direction="compose"):
    """Participation-guided, parser-preserving category collapse (the D1
    "learned collapse").

    Proposes symbol merges in order of participation-signature similarity ---
    the structure role-collapse exposes --- and accepts a merge only when it
    keeps every grammar rule distinguishable (``collapse_conflicts == 0``), so
    the live parser retains its mutually-exclusive choices. Greedy single-
    linkage under that hard constraint; deterministic (similarity ties broken
    by symbol order). Returns ``{symbol: class_id}`` with stable integer ids
    --- a SMALLER mutually-exclusive category set that recovers the grammar.
    """
    rules = grammar_rules(grammar, direction)
    syms = sorted({s for (lhs, _m, rhs) in rules for s in (lhs, *rhs)})
    sig = _combined_signature(rules)

    def jac(a, b):
        A, B = sig.get(a, frozenset()), sig.get(b, frozenset())
        if not A and not B:
            return 1.0
        union = len(A | B)
        return (len(A & B) / union) if union else 0.0

    cands = sorted(
        ((jac(a, b), a, b)
         for i, a in enumerate(syms) for b in syms[i + 1:] if jac(a, b) > 0.0),
        key=lambda t: (-t[0], t[1], t[2]))

    cls = {s: s for s in syms}
    for _sim, a, b in cands:
        if cls[a] == cls[b]:
            continue
        ra, rb = cls[a], cls[b]
        trial = {k: (ra if v == rb else v) for k, v in cls.items()}
        if collapse_conflicts(rules, trial) == 0:
            cls = trial

    root_id, out = {}, {}
    for s in syms:
        r = cls[s]
        if r not in root_id:
            root_id[r] = len(root_id)
        out[s] = root_id[r]
    return out
