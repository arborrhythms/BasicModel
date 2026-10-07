"""Grammatical categories from the semantic effect of operators (Phase
R4-sem 2).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
+ user steering (2026-06-03). A symbol's grammatical category is determined
by the SEMANTIC EFFECT of the operators it participates with, where the
semantic effect is the operator's vector in the WholeSpace operator
codebook (shaped live by the soft superposition under truth/consequence
supervision -- see ``WholeSpace.shape_operators``). A symbol's *semantic
signature* is the aggregate of its operators' codebook vectors; clustering
the signatures recovers categories.

This refines the structural participation learner (``bin/participation.py``,
which clusters on the raw ``(method, position)`` slot set): two symbols whose
DIFFERENT operators have the SAME semantic effect now unify into one
category, which structural slot-membership alone cannot do.
"""

import torch


def semantic_signature(slots, op_vectors):
    """Aggregate the codebook vectors of the operators a symbol participates
    with into its semantic signature (the mean of the per-slot operator
    vectors). ``slots`` is an iterable of ``(method, position)``; ``op_vectors``
    maps method -> vector. Returns a 1-D tensor, or ``None`` when none of the
    symbol's operators has a codebook vector (e.g. structural-only ops)."""
    vecs = []
    for slot in slots:
        method = slot[0] if isinstance(slot, (tuple, list)) else slot
        v = op_vectors.get(method)
        if v is not None:
            vecs.append(torch.as_tensor(v, dtype=torch.float32).flatten())
    if not vecs:
        return None
    return torch.stack(vecs).mean(dim=0)


def _unit(v):
    return v / (v.norm() + 1e-8)


def recover_semantic_categories(participation, op_vectors, threshold=0.99):
    """Cluster symbols by the SEMANTIC EFFECT of their operators.

    Each symbol's signature is :func:`semantic_signature`; two symbols whose
    (unit) signatures have cosine similarity >= ``threshold`` are the same
    category (single-linkage union). Symbols whose operators are
    semantically similar unify even when their raw operator slots differ.
    Symbols with no operator vector are omitted. Returns ``{symbol: class_id}``
    with stable integer ids.
    """
    sigs = {}
    for sym, slots in participation.items():
        sig = semantic_signature(slots, op_vectors)
        if sig is not None:
            sigs[sym] = _unit(sig)
    items = list(sigs)
    parent = {s: s for s in items}

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
            a, b = sigs[items[i]], sigs[items[j]]
            d = min(int(a.numel()), int(b.numel()))
            cos = float(torch.dot(a[:d], b[:d]))
            if cos >= threshold:
                union(items[i], items[j])

    root_id, out = {}, {}
    for s in items:
        r = find(s)
        if r not in root_id:
            root_id[r] = len(root_id)
        out[s] = root_id[r]
    return out
