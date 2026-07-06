"""T6 -- the typed conceptual definition (snap contract sec 2.4-2.5,
2026-07-06). ``ConceptualSpace.typed_definition(idea)`` decodes an idea by the
signed peel (pursuit of all concepts against the idea -> sparse signed symbol
vector, <= stmCapacity members), frontier-prunes the support through the
relation records, and emits the grammar's compression input:

  head = the minimal covering whole; modifiers = surviving maximal parts;
  exclusions = the negative-polarity members; residual = the difference the
  grammar's transforms (ADV/intersection, ...) must explain.

Constructed through the LIVE APIs (relate / the shared-store row allocation)
so the codebook-row <-> concept-id mapping comes from the system itself.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
import Spaces  # noqa: E402
from Spaces import _concept_alloc_of  # noqa: E402
from test_basicmodel import _populate_test_config  # noqa: E402

_D = 8


def _cs(nS=64, order=3):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    return cs


def _orthonormal_rows(cs, rows):
    """Overwrite the concept dictionary rows at ``rows`` with distinct
    one-hot directions so the peel's support is unambiguous."""
    W = cs.similarity_codebook.getW()
    with torch.no_grad():
        W.zero_()
        for k, r in enumerate(rows):
            W[r, k % _D] = 1.0


def _mint_chain(cs):
    """Mint concepts through the live API: two base concepts B1, B2 and a
    covering whole H (B1 and B2 both have H as a sym-whole). Returns
    (H, B1, B2) cids and their shared-store rows."""
    alloc = _concept_alloc_of(cs)
    H = cs.new_concept()
    B1 = cs.relate(("sym", H), ("sym", H)) if False else cs.new_concept()
    B2 = cs.new_concept()
    # B1, B2 are PARTS of H; H is the WHOLE covering them.
    cs.add_part(H, ("sym", B1))
    cs.add_part(H, ("sym", B2))
    cs.add_whole(B1, ("sym", H))
    cs.add_whole(B2, ("sym", H))
    ly = alloc.layer(0)
    rows = {}
    for cid in (H, B1, B2):
        r = cs._csw_concept_row(cs._concept_source_order(cid), cid)
        rows[cid] = r
    return H, B1, B2, rows


def test_covers_walks_sym_whole_links():
    cs = _cs()
    H, B1, B2, _rows = _mint_chain(cs)
    assert cs._covers(H, B1) and cs._covers(H, B2)
    assert not cs._covers(B1, H)
    assert not cs._covers(B1, B2)


def test_typed_definition_head_modifiers_exclusions():
    cs = _cs()
    H, B1, B2, rows = _mint_chain(cs)
    rH, r1, r2 = rows[H], rows[B1], rows[B2]
    assert None not in (rH, r1, r2), "constructed cids must hold store rows"
    # A fourth, unrelated concept to serve as the EXCLUSION.
    X = cs.new_concept()
    rX = cs._csw_concept_row(cs._concept_source_order(X), X)
    _orthonormal_rows(cs, [rH, r1, r2, rX])
    W = cs.similarity_codebook.getW().detach()
    # idea = head + both parts - 0.6 * excluded concept
    idea = W[rH] + 0.8 * W[r1] + 0.7 * W[r2] - 0.6 * W[rX]
    d = cs.typed_definition(idea)
    assert d is not None
    sup = {r: (c, cid) for (r, c, cid) in d["support"]}
    assert set(sup) == {rH, r1, r2, rX}, d["support"]
    # Head: H covers B1 and B2 -> the minimal covering whole.
    assert d["head"] is not None and d["head"][2] == H, d["head"]
    # Exclusion: the negative-coefficient member, coeff ~ -0.6.
    assert len(d["exclusions"]) == 1 and d["exclusions"][0][0] == rX
    assert abs(d["exclusions"][0][1] + 0.6) < 1e-3
    # Modifiers: B1 and B2 survive (parts of the head are its differentia
    # here -- nothing else covers them).
    mod_cids = {cid for (_r, _c, cid) in d["modifiers"]}
    assert mod_cids == {B1, B2}, d["modifiers"]
    # The support explains the idea: residual ~ 0 on orthonormal rows.
    assert float(d["residual"].norm()) < 1e-4 * (1 + float(idea.norm()))


def test_typed_definition_verbose_not_wrong_without_relations():
    # Sparse relation coverage: concepts with NO whole links -> no coverage
    # walk; the head falls back to the largest inclusion and everything else
    # stays a modifier (verbose, not wrong -- design sec 2.4).
    cs = _cs()
    A = cs.new_concept()
    B = cs.new_concept()
    rA = cs._csw_concept_row(cs._concept_source_order(A), A)
    rB = cs._csw_concept_row(cs._concept_source_order(B), B)
    _orthonormal_rows(cs, [rA, rB])
    W = cs.similarity_codebook.getW().detach()
    idea = 0.9 * W[rA] + 0.5 * W[rB]
    d = cs.typed_definition(idea)
    assert d["head"][0] == rA, "largest inclusion becomes the head"
    assert [m[0] for m in d["modifiers"]] == [rB]
    assert d["exclusions"] == []


def test_typed_definition_respects_stm_capacity_cap():
    # The sec-5 hard cap: the support cannot exceed stm_capacity members.
    cs = _cs()
    cids = [cs.new_concept() for _ in range(6)]
    rows = [cs._csw_concept_row(cs._concept_source_order(c), c) for c in cids]
    _orthonormal_rows(cs, rows)
    W = cs.similarity_codebook.getW().detach()
    idea = sum(W[r] for r in rows)
    object.__setattr__(cs, "stm_capacity", 3)
    d = cs.typed_definition(idea)
    assert len(d["support"]) <= 3, d["support"]
    d_full = cs.typed_definition(idea, max_parts=6)
    assert len(d_full["support"]) == 6


def test_typed_definition_minimal_whole_drops_looser_ancestor():
    # The other frontier: with BOTH a tight cover H and its looser ancestor G
    # (H's whole) in the support, the head is H (the MINIMAL covering whole)
    # and G is dropped as implied ("cat" implies "animal") -- sec 1.3's
    # tightest-cover side.
    cs = _cs()
    H, B1, B2, rows = _mint_chain(cs)
    G = cs.new_concept()                              # the looser whole
    cs.add_whole(H, ("sym", G))                       # H ⊑ G
    rG = cs._csw_concept_row(cs._concept_source_order(G), G)
    rH, r1, r2 = rows[H], rows[B1], rows[B2]
    _orthonormal_rows(cs, [rH, r1, r2, rG])
    W = cs.similarity_codebook.getW().detach()
    idea = W[rH] + 0.8 * W[r1] + 0.7 * W[r2] + 0.5 * W[rG]
    d = cs.typed_definition(idea)
    assert d["head"] is not None and d["head"][2] == H, (
        "the TIGHTEST cover is the head, not the ancestor")
    mod_cids = {cid for (_r, _c, cid) in d["modifiers"]}
    assert G not in mod_cids, "the looser ancestor is implied by the head"
    assert mod_cids == {B1, B2}, d["modifiers"]


def test_typed_definition_none_without_dictionary():
    cs = _cs()
    object.__setattr__(cs, "similarity_codebook", None)
    assert cs.typed_definition(torch.randn(_D)) is None
