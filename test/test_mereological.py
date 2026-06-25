"""The mereological algorithm core (doc/mereological.md): the synthesizer's
join-from-bottom seed + isotonic `.where`-projection (the redistribution that
makes a whole dominate ALL its parts, including cross-boundary), and the
analyzer as its `1-x` mirror."""
import torch

from Mereology import (
    where_containment_edges, join_from_bottom, meet_from_top,
    project_monotone, mereological_synthesize, mereological_analyze,
)

# "abcd" and its contiguous substrings, with their `.where` byte-spans.
_SPANS_KEYS = [('a', (0, 1)), ('b', (1, 2)), ('c', (2, 3)), ('d', (3, 4)),
               ('ab', (0, 2)), ('bc', (1, 3)), ('cd', (2, 4)),
               ('abc', (0, 3)), ('bcd', (1, 4)), ('abcd', (0, 4))]
KEYS = [k for k, _ in _SPANS_KEYS]
SPANS = [s for _, s in _SPANS_KEYS]
IDX = {k: i for i, k in enumerate(KEYS)}


def test_where_poset_is_the_part_spec_incl_cross_boundary():
    """`.where` containment IS the full part spec -- bc inside abcd, etc."""
    eset = set(where_containment_edges(SPANS))
    assert (IDX['abcd'], IDX['bc']) in eset      # cross-boundary part IS an edge
    assert (IDX['abcd'], IDX['ab']) in eset
    assert (IDX['ab'], IDX['a']) in eset
    assert (IDX['a'], IDX['abcd']) not in eset   # the reverse is not an edge
    assert (IDX['ab'], IDX['cd']) not in eset    # disjoint spans -> no edge


def test_join_from_bottom_dominates_its_parts():
    torch.manual_seed(0)
    ab, cd = torch.rand(8), torch.rand(8)
    whole = join_from_bottom(torch.stack([ab, cd]))
    assert (whole >= ab - 1e-6).all() and (whole >= cd - 1e-6).all()
    assert float(whole.min()) >= 0.0 and float(whole.max()) <= 1.0


def test_projection_repairs_every_edge_incl_cross_boundary():
    """A random seed (build-join blind to cross-boundary parts) violates the
    order; the `.where`-isotonic projection redistributes to satisfy ALL edges."""
    torch.manual_seed(0)
    seed = torch.rand(len(KEYS), 8)
    # seed abcd as the join of only its build parts -> blind to bc
    seed[IDX['abcd']] = join_from_bottom(torch.stack([seed[IDX['ab']], seed[IDX['cd']]]))
    edges = where_containment_edges(SPANS)
    before = sum(int((seed[w] < seed[p] - 1e-6).any()) for w, p in edges)
    out = mereological_synthesize(seed, SPANS)
    after = sum(int((out[w] < out[p] - 1e-6).any()) for w, p in edges)
    assert before > 0, "seed should violate the order (else the test is vacuous)"
    assert after == 0, "projection must satisfy every containment edge"
    assert (out[IDX['abcd']] >= out[IDX['bc']] - 1e-5).all()   # cross-boundary repaired
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_projection_is_minimal_and_bounded():
    """Min-norm: untouched-edge coordinates don't move; result stays in [0,1]."""
    torch.manual_seed(3)
    seed = torch.rand(len(KEYS), 8)
    out = project_monotone(seed, where_containment_edges(SPANS))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # an already-feasible chain (a in ab in abc in abcd, all equal) is a fixpoint
    flat = torch.full((len(KEYS), 4), 0.3)
    assert torch.allclose(project_monotone(flat, where_containment_edges(SPANS)), flat)


def test_codebook_substring_poset_and_projection():
    """The codebook-level order is SUBSTRING containment (the radix codebook
    holds strings); projection makes a whole-string code dominate every
    sub-string code, including the cross-boundary one."""
    from Mereology import substring_containment_edges, project_codebook
    strings = ['a', 'b', 'c', 'd', 'ab', 'bc', 'cd', 'abc', 'bcd', 'abcd']
    si = {s: i for i, s in enumerate(strings)}
    b = [s.encode() for s in strings]
    edges = set(substring_containment_edges(b))
    assert (si['abcd'], si['bc']) in edges       # cross-boundary substring IS an edge
    assert (si['abc'], si['bc']) in edges
    assert (si['ab'], si['cd']) not in edges     # disjoint -> not a substring
    torch.manual_seed(0)
    out = project_codebook(torch.rand(len(strings), 8), b)
    assert all((out[w] >= out[p] - 1e-5).all() for w, p in substring_containment_edges(b))
    assert (out[si['abcd']] >= out[si['bc']] - 1e-5).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_analyzer_is_the_complement_mirror():
    torch.manual_seed(1)
    seed = torch.rand(len(KEYS), 8)
    ana = mereological_analyze(seed, SPANS)
    assert torch.allclose(ana, 1.0 - mereological_synthesize(1.0 - seed, SPANS), atol=1e-6)
    # and meet_from_top is the 1-x mirror of join_from_bottom
    parts = torch.rand(3, 8)
    assert torch.allclose(meet_from_top(parts), 1.0 - join_from_bottom(1.0 - parts), atol=1e-6)
