"""Standalone tests for the from-scratch meronomy synthesizer + analyzer
(bin/Meronomy.py): referential binary-tree vector codebooks carrying the
part<=whole order. Synthesis combines the character points; analysis cuts the
gaps between them with boundary dichotomies. Pure algorithm -- no model build.
"""

import torch

from Meronomy import (synthesize, analyze, class_segments, class_dichotomies,
                      char_dichotomies, char_boundary_codes, class_dichotomy_code,
                      word_spans, words, word_bounds, word_wholes, boundary_codes,
                      same_word, synthesize_to_words, word_metas,
                      synthesize_words, synthesize_phrases, all_wholes, tile,
                      intersect_wholes, default_embed_table)
from Mereology import join_from_bottom, meet_from_top, where_containment_edges
from Layers import LETTER, WHITESPACE, DIGIT, PUNCT


# --- synthesizer (on the character points) --------------------------------

def test_class_segments_abc_d():
    assert class_segments("ABC D") == [(LETTER, 0, 3), (WHITESPACE, 3, 4),
                                       (LETTER, 4, 5)]


def test_synthesize_reconstruct_exact():
    for word in ("hello", "the world", "a", "abcd"):
        tree = synthesize(word)
        assert len(tree.roots) == 1
        assert tree.reconstruct(tree.roots[0]) == word


def test_synthesize_whole_dominates_all_parts():
    tree = synthesize("abcd")
    edges = where_containment_edges(tree.where)
    assert edges
    for (w, p) in edges:
        assert tree.dominates(w, p), (tree.label[w], tree.label[p])


def test_synthesize_codes_stay_in_unit_cube():
    codes = synthesize("the world").codes()
    assert bool((codes >= -1e-6).all()) and bool((codes <= 1.0 + 1e-6).all())


def test_referential_decoupled_from_projection():
    tree = synthesize("hello")
    root = tree.roots[0]
    for i in range(len(tree)):
        tree.code[i] = torch.zeros_like(tree.code[i])
    assert tree.reconstruct(root) == "hello"


# --- analyzer (in the gaps between characters) ----------------------------

def test_class_dichotomies_are_between_letters():
    # "ABC D": the two dichotomies sit in the gaps bracketing the space (3, 4),
    # not on any character.
    assert class_dichotomies("ABC D") == [(3, "left-of-space"),
                                          (4, "right-of-space")]


def test_analyze_cut_tree_is_two_dichotomies():
    # is-a-space = left-of-space (gap before ␣) then right-of-space (gap after).
    tree = analyze("ABC D")
    root = tree.roots[0]
    assert tree.tag[root] == "left-of-space" and tree.boundary[root] == 3
    assert not tree.is_atom(root)
    rgt = tree.right[root]
    assert tree.tag[rgt] == "right-of-space" and tree.boundary[rgt] == 4
    assert not tree.is_atom(rgt)


def test_analyze_parts_are_leaves_named_by_analyzer():
    tree = analyze("ABC D")
    parts = [(tree.label[i], tree.tag[i], tree.where[i]) for i in tree.leaves()]
    assert parts == [("ABC", "is-a-letter", (0, 3)),
                     (" ", "is-a-space", (3, 4)),
                     ("D", "is-a-letter", (4, 5))]


def test_analyze_referential_reconstruct():
    # Following the cut tree's children reconstructs the input exactly.
    tree = analyze("ABC D")
    assert tree.reconstruct(tree.roots[0]) == "ABC D"


def test_analyze_region_dominates_its_parts():
    # Each region (internal cut node) dominates its contained parts (part<=whole).
    tree = analyze("hi, 42")
    edges = where_containment_edges(tree.where)
    assert edges
    for (w, p) in edges:
        assert tree.dominates(w, p), (tree.tag[w], tree.label[p])


def test_analyze_multiclass_cuts():
    # letters / punct / space / digit each get their own part leaf.
    tree = analyze("hi, 42")
    tags = [tree.tag[i] for i in tree.leaves()]
    assert tags == ["is-a-letter", "is-a-punctuation", "is-a-space",
                    "is-a-number"]


def test_analyzer_is_de_morgan_dual_of_synthesizer():
    x = default_embed_table(2, 16, seed=7)
    assert torch.allclose(meet_from_top(1.0 - x), 1.0 - join_from_bottom(x),
                          atol=1e-5)


# --- (1) the complete left-of-char / right-of-char basis ------------------

def test_char_dichotomy_basis_is_complete():
    # every interior gap, named by BOTH adjacent characters (not just classes).
    assert char_dichotomies("cat") == [(1, "right-of-c", "left-of-a"),
                                       (2, "right-of-a", "left-of-t")]


def test_class_dichotomy_is_or_of_char_basis():
    # left-of-space (class) == union of left-of-c over the whitespace bytes.
    codes = char_boundary_codes()
    ws = [9, 10, 13, 32]
    expect = join_from_bottom(torch.stack([codes["left-of-%d" % c] for c in ws]))
    assert torch.allclose(class_dichotomy_code(WHITESPACE, "left"), expect,
                          atol=1e-6)


# --- (2) bound the combination to produce only words ----------------------

def test_same_word_gate():
    # "the cat": inside-word gaps True, word-boundary gaps False.
    assert [same_word("the cat", g) for g in range(7)] == [
        False, True, True, False, False, True, True]


def test_synthesize_to_words_produces_only_words():
    tree = synthesize_to_words("the cat sat")
    # the maximal wholes (roots) are exactly the words.
    assert [tree.reconstruct(r) for r in tree.roots] == ["the", "cat", "sat"]
    # and NO whole crosses a word boundary (every span lies within one word).
    spans = word_spans("the cat sat")
    for i in range(len(tree)):
        ws, we = tree.where[i]
        assert any(s <= ws and we <= e for (s, e) in spans), tree.where[i]


def test_word_metas_are_clean_words_with_bounds():
    metas = word_metas("the cat sat")
    assert [m[0] for m in metas] == ["the", "cat", "sat"]
    assert [m[2] for m in metas] == [("start", "left-of-space"),
                                     ("right-of-space", "left-of-space"),
                                     ("right-of-space", "end")]


# --- cutting for only words (separators bracketed out) --------------------

def test_words_drops_separators():
    assert words("ABC D") == ["ABC", "D"]
    assert words("hi, 42") == ["hi", "42"]            # digits stay (number-word)
    assert words("the world!") == ["the", "world"]
    assert words("ab12cd") == ["ab12cd"]              # no separator -> one word
    assert words("   ") == []                          # all separator


def test_word_spans_are_the_content_between_separators():
    assert word_spans("ABC D") == [(0, 3), (4, 5)]
    assert word_spans("  hi  there ") == [(2, 4), (6, 11)]


def test_synthesize_words_one_whole_per_word():
    out = synthesize_words("ABC D")
    assert [span for (span, _) in out] == [(0, 3), (4, 5)]
    # each word-whole reconstructs to its word, with global .where offsets.
    assert out[0][1].reconstruct(out[0][1].roots[0]) == "ABC"
    assert out[1][1].reconstruct(out[1][1].roots[0]) == "D"
    assert out[1][1].where[out[1][1].roots[0]] == (4, 5)


# --- multiple words: the MIDDLE word is bounded on both sides -------------

def test_three_words():
    assert words("the cat sat") == ["the", "cat", "sat"]
    assert word_spans("the cat sat") == [(0, 3), (4, 7), (8, 11)]


def test_middle_word_bounded_on_both_sides():
    # "the cat sat": "cat" is BOTH right-of-space (its left) and left-of-space
    # (its right); the edge words borrow the input boundary on their open side.
    b = word_bounds("the cat sat")
    assert b == [((0, 3), "start", "left-of-space"),
                 ((4, 7), "right-of-space", "left-of-space"),
                 ((8, 11), "right-of-space", "end")]


def test_four_words_two_interior_both_sides():
    sentence = "the quick brown fox"
    assert words(sentence) == ["the", "quick", "brown", "fox"]
    b = word_bounds(sentence)
    # the two interior words are bounded by separators on BOTH sides.
    assert b[1][1] == "right-of-space" and b[1][2] == "left-of-space"
    assert b[2][1] == "right-of-space" and b[2][2] == "left-of-space"
    # only the outer words touch a non-separator edge.
    assert b[0][1] == "start" and b[-1][2] == "end"


def test_synthesize_words_three_global_offsets():
    out = synthesize_words("the cat sat")
    assert [span for (span, _) in out] == [(0, 3), (4, 7), (8, 11)]
    recon = [t.reconstruct(t.roots[0]) for (_, t) in out]
    assert recon == ["the", "cat", "sat"]
    # middle word-whole keeps its GLOBAL span, not a 0-based local one.
    assert out[1][1].where[out[1][1].roots[0]] == (4, 7)


def test_mixed_separators_interior_word():
    # comma + space adjacent around the interior word: it still extracts clean.
    assert words("hi, there, you") == ["hi", "there", "you"]


# --- combining dichotomies in a tree: the wholes of the words -------------

def test_phrase_wholes_of_words():
    # Combine words; the spaces are JOINS, glued back on reconstruction.
    tree = synthesize_phrases("the cat sat")
    leaves = tree.leaves()
    assert [tree.label[i] for i in leaves] == ["the", "cat", "sat"]
    assert tree.reconstruct(tree.roots[0]) == "the cat sat"   # joins restored
    # left-to-right combine ((the cat) sat): each word's phrase wholes.
    wholes = {tree.label[i]: [tree.reconstruct(w) for w in tree.wholes(i)]
              for i in leaves}
    assert wholes["the"] == ["the cat", "the cat sat"]
    assert wholes["cat"] == ["the cat", "the cat sat"]
    assert wholes["sat"] == ["the cat sat"]


def test_phrase_whole_dominates_its_word_parts():
    # Every phrase whole dominates the words it contains (part <= whole).
    tree = synthesize_phrases("the cat sat")
    from Mereology import where_containment_edges
    for (w, p) in where_containment_edges(tree.where):
        assert tree.dominates(w, p), (tree.label[w], tree.label[p])


def test_greedy_union_vs_first_intersection_tiling():
    wholes = all_wholes("the cat sat")
    greedy = tile(wholes, len("the cat sat"), greedy=True)
    first = tile(wholes, len("the cat sat"), greedy=False)
    assert [w[2] for w in greedy] == ["the cat sat"]       # union: one maximal whole
    assert [w[2] for w in first] == ["the", "cat", "sat"]  # intersection: the parts


def test_intersection_recovers_cat():
    wholes = {w[2]: w for w in all_wholes("the cat sat")}
    part = intersect_wholes(wholes["the cat"], wholes["cat sat"])
    assert part is not None
    assert part[0] == (4, 7)                                # spans: [0,7) ∩ [4,11) = cat
    # the meet is dominated by BOTH wholes -- a common part (part <= whole).
    assert bool((part[1] <= wholes["the cat"][1] + 1e-6).all())
    assert bool((part[1] <= wholes["cat sat"][1] + 1e-6).all())
    # and among the single words it is nearest to "cat" (the shared content).
    singles = {k: wholes[k] for k in ("the", "cat", "sat")}
    u = lambda x: x / x.norm().clamp_min(1e-8)
    nearest = max(singles, key=lambda k: float((u(part[1]) * u(singles[k][1])).sum()))
    assert nearest == "cat"


def test_word_whole_is_union_of_its_bounds():
    out = word_wholes("the cat sat")
    # each word's narrowest whole is keyed by its two bounding dichotomies.
    assert [(w, b) for (w, b, _) in out] == [
        ("the", ("start", "left-of-space")),
        ("cat", ("right-of-space", "left-of-space")),
        ("sat", ("right-of-space", "end"))]
    # the whole of "the" IS union(start, left-of-space), and it dominates both.
    codes = boundary_codes()
    the_whole = out[0][2]
    expect = join_from_bottom(torch.stack([codes["start"],
                                           codes["left-of-space"]]))
    assert torch.allclose(the_whole, expect, atol=1e-6)
    assert bool((the_whole >= codes["start"] - 1e-6).all())
    assert bool((the_whole >= codes["left-of-space"] - 1e-6).all())
