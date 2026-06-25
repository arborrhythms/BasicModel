"""Meronomy: the synthesizer and analyzer as referential binary-tree vector
codebooks that carry the part<=whole order.

Both build a binary tree of ``[0,1]`` presence codes; every node stores its two
child codes (the referential part-chain), a ``.where`` byte-span, and -- for the
analyzer -- the ``.boundary`` gap it cuts at. The codes are the vector taxonomy;
the stored children make reconstruction referential (follow the children to
atoms), decoupled from the order the codes carry.

The synthesizer and analyzer are DUAL on the same input but live on different
coordinates:

  SYNTHESIZER -- on the N CHARACTER points. Atoms = bytes;
    ``combine(left_part, right_part)`` builds wholes bottom-up. A whole's seed is
    the dominating join of its parts (``join_from_bottom``: whole >= each build
    part), then ALL codes are projected onto the ``.where`` containment DAG so a
    whole dominates EVERY part, cross-boundary ones (``bc`` inside ``abcd``)
    included.

  ANALYZER -- in the N+1 GAPS between characters. Atoms = boundary DICHOTOMIES
    (``left-of-space`` / ``right-of-space`` / ...), each a binary cut at one gap
    (left | right). Composing dichotomies is a binary CUT TREE -- each internal
    node is a dichotomy at a boundary storing its two child regions; the leaves
    are the parts, coded BY the analyzer (class) that produced them with their
    ``.where``. ``"ABC D"`` cuts at ``left-of-space`` (gap before ␣) then
    ``right-of-space`` (gap after ␣) -> ``ABC`` | ``␣`` | ``D``.

Reconstruction is referential either way: follow the stored children to the
atoms. Reuses ``Mereology`` for the join/meet/projection and
``Layers._CHAR_CLASS_RANGES`` for the class predicates. Pure + deterministic so
the algorithm is testable standalone before it is wired into the forward path.
"""

from __future__ import annotations

import torch

from Mereology import (join_from_bottom, meet_from_top, project_monotone,
                       where_containment_edges)
from Layers import LETTER, DIGIT, WHITESPACE, PUNCT, _CHAR_CLASS_RANGES

# Class-predicate names (the "is-a-X" the analyzer leaves are coded by) and the
# bare class names used inside the boundary-dichotomy names ("left-of-X").
_CLASS_NAME = {LETTER: "is-a-letter", DIGIT: "is-a-number",
               WHITESPACE: "is-a-space", PUNCT: "is-a-punctuation"}
_BARE = {LETTER: "letter", DIGIT: "number",
         WHITESPACE: "space", PUNCT: "punctuation"}
# The separator classes that cut runs out of the letter content. Letters are the
# residual (uncut) class.
_CUT_CLASSES = (WHITESPACE, PUNCT, DIGIT)


def _byte_class(b):
    """Char class of byte ``b`` (LETTER residual for anything unranged)."""
    for cls, ranges in _CHAR_CLASS_RANGES.items():
        for lo, hi in ranges:
            if lo <= b <= hi:
                return cls
    return LETTER


def _as_bytes(seq):
    return list(seq.encode() if isinstance(seq, str) else seq)


def default_embed_table(n, dim, seed=0):
    """Deterministic ``[n, dim]`` ``[0,1]`` presence table (atom seeds).

    Pinned to CPU (the seeded generator is CPU) so the standalone algorithm is
    device-agnostic; wiring supplies the model's own embeddings on its device.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(n, dim, generator=g, device="cpu")


class MeronomyTree:
    """Referential vector codebook: parallel arrays of (code, left, right,
    where, label, tag, boundary). ``left == right == -1`` marks an atom (leaf).
    ``label`` is the content (for reconstruction); ``tag`` is the analyzer /
    dichotomy name; ``boundary`` is the gap an analyzer cut sits at (-1 else)."""

    def __init__(self, dim):
        self.dim = int(dim)
        self.code = []       # list of [D] tensors in [0,1]
        self.left = []       # child id or -1
        self.right = []      # child id or -1
        self.where = []      # (start, end) byte span
        self.label = []      # content str (atom: byte char / part substring)
        self.tag = []        # analyzer/class/dichotomy name (or None)
        self.boundary = []   # gap position an analyzer node cuts at (-1 else)
        self.join = []       # separator glued in BETWEEN children ("" if none)
        self.roots = []      # top-level node ids

    def add(self, code, where, label, left=-1, right=-1, tag=None, boundary=-1,
            join=""):
        i = len(self.code)
        self.code.append(code)
        self.left.append(int(left))
        self.right.append(int(right))
        self.where.append((int(where[0]), int(where[1])))
        self.label.append(label)
        self.tag.append(tag)
        self.boundary.append(int(boundary))
        self.join.append(join)
        return i

    def __len__(self):
        return len(self.code)

    def codes(self):
        return torch.stack(self.code) if self.code else torch.zeros(0, self.dim)

    def is_atom(self, i):
        return self.left[i] < 0 and self.right[i] < 0

    def leaves(self):
        """Leaf node ids in left-to-right ``.where`` order."""
        return sorted((i for i in range(len(self)) if self.is_atom(i)),
                      key=lambda i: self.where[i])

    def reconstruct(self, i):
        """Referential reconstruction: follow stored children to the atoms.

        Returns the concatenated leaf content, gluing each node's ``.join``
        (the dichotomy a phrase combined across) back between its children --
        exact, and independent of the codes (so the order projection that moves
        them cannot corrupt it).
        """
        if self.is_atom(i):
            return self.label[i]
        return (self.reconstruct(self.left[i]) + (self.join[i] or "")
                + self.reconstruct(self.right[i]))

    def wholes(self, i):
        """Ancestor whole node ids of ``i``, innermost-first -- its parts-of
        chain. Each whole CONTAINS ``i`` as a part (the part<=whole order)."""
        parent = {}
        for j in range(len(self)):
            if self.left[j] >= 0:
                parent[self.left[j]] = j
            if self.right[j] >= 0:
                parent[self.right[j]] = j
        out, cur = [], parent.get(i)
        while cur is not None:
            out.append(cur)
            cur = parent.get(cur)
        return out

    def dominates(self, w, p, tol=1e-5):
        """``code(w) >= code(p)`` per coordinate -- the part<=whole order."""
        return bool((self.code[w] >= self.code[p] - tol).all())

    def _writeback(self, projected):
        for i in range(len(self.code)):
            self.code[i] = projected[i]


def _combine_within(tree, leaves):
    """Combine a frontier of node ids into ONE root by balanced-pairwise
    left-to-right joins (each whole = ``join_from_bottom`` of its two parts).
    Returns the root id, or -1 if ``leaves`` is empty. Never combines outside the
    given frontier -- so passing one word's leaves keeps the whole within it."""
    frontier = list(leaves)
    while len(frontier) > 1:
        nxt, k = [], 0
        while k + 1 < len(frontier):
            l, r = frontier[k], frontier[k + 1]
            seed = join_from_bottom(torch.stack([tree.code[l], tree.code[r]]))
            span = (tree.where[l][0], tree.where[r][1])
            nxt.append(tree.add(seed, span, None, left=l, right=r))
            k += 2
        if k < len(frontier):            # carry the odd tail up a level
            nxt.append(frontier[k])
        frontier = nxt
    return frontier[0] if frontier else -1


def synthesize(byte_seq, embed=None, dim=32):
    """Bottom-up binary synthesis on the character points: bytes -> wholes.

    Each whole's seed is ``join_from_bottom`` of its two build parts (so it
    dominates them), then every code is projected onto the ``.where`` containment
    DAG so a whole dominates ALL its parts, cross-boundary included. Merge order
    is balanced-pairwise left-to-right. Returns a :class:`MeronomyTree`.
    """
    bs = _as_bytes(byte_seq)
    if embed is None:
        embed = default_embed_table(256, dim)
    tree = MeronomyTree(embed.shape[1])
    leaves = [tree.add(embed[b].clamp(0.0, 1.0), (pos, pos + 1), chr(b))
              for pos, b in enumerate(bs)]
    if leaves:
        tree.roots = [_combine_within(tree, leaves)]
    if len(tree) > 1:
        tree._writeback(project_monotone(tree.codes(),
                                         where_containment_edges(tree.where)))
    return tree


def class_segments(byte_seq):
    """Maximal same-class runs: list of ``(class_id, start, end)``."""
    bs = _as_bytes(byte_seq)
    segs = []
    i, n = 0, len(bs)
    while i < n:
        c = _byte_class(bs[i])
        j = i + 1
        while j < n and _byte_class(bs[j]) == c:
            j += 1
        segs.append((c, i, j))
        i = j
    return segs


def class_dichotomies(byte_seq):
    """Boundary dichotomies BETWEEN characters: sorted ``(boundary, name)``.

    A dichotomy lives in a GAP (boundary ``i`` is the gap before character ``i``,
    ``0 <= i <= N``), not on a character. For each separator-class run ``[a,b)``:
    ``left-of-X`` is the cut at ``a`` (the gap entering the run); ``right-of-X``
    the cut at ``b`` (the gap leaving it). These are the class-transition gaps the
    analyzer composes into parts. Edge gaps (0 and N) are dropped -- they split
    nothing.
    """
    bs = _as_bytes(byte_seq)
    n = len(bs)
    cls = [_byte_class(b) for b in bs]
    cuts = {}                                   # boundary -> name (first wins)
    for X in _CUT_CLASSES:
        i = 0
        while i < n:
            if cls[i] == X:
                a = i
                while i < n and cls[i] == X:
                    i += 1
                cuts.setdefault(a, "left-of-%s" % _BARE[X])
                cuts.setdefault(i, "right-of-%s" % _BARE[X])
            else:
                i += 1
    return sorted((b, nm) for b, nm in cuts.items() if 0 < b < n)


def char_dichotomies(byte_seq):
    """The COMPLETE relational basis: every interior gap, named by its adjacent
    CHARACTERS (not just classes). Gap ``i`` (between char ``i-1`` and char
    ``i``) is BOTH ``right-of-<char i-1>`` and ``left-of-<char i>``. The
    class dichotomies (``left-of-space``, ...) are a coarsening -- the OR of the
    char dichotomies whose character is in that class (see
    :func:`class_dichotomy_code`). Returns sorted ``(gap, right_name, left_name)``.
    """
    bs = _as_bytes(byte_seq)
    return [(i, "right-of-%s" % chr(bs[i - 1]), "left-of-%s" % chr(bs[i]))
            for i in range(1, len(bs))]


def char_boundary_codes(dim=32, seed=3):
    """Fixed ``[0,1]`` code for every character dichotomy: ``start`` / ``end``
    and ``left-of-c`` / ``right-of-c`` for every byte ``c`` -- the complete
    ``2*256 + 2`` relational basis the analyzer can cut on."""
    names = (["start", "end"]
             + ["left-of-%d" % c for c in range(256)]
             + ["right-of-%d" % c for c in range(256)])
    tbl = default_embed_table(len(names), dim, seed=seed)
    return {nm: tbl[k] for k, nm in enumerate(names)}


def class_dichotomy_code(class_id, side, dim=32, seed=3):
    """A class dichotomy as the UNION (join) of its character dichotomies:
    ``left-of-space = join(left-of-c for c in the whitespace class)``. Confirms
    the class basis is the OR-coarsening of the complete character basis."""
    codes = char_boundary_codes(dim, seed)
    chars = [c for lo, hi in _CHAR_CLASS_RANGES[class_id] for c in range(lo, hi + 1)]
    return join_from_bottom(torch.stack(
        [codes["%s-of-%d" % (side, c)] for c in chars]))


# The separator classes that DELIMIT words: a word is bracketed out by
# left-of-X o right-of-X (the dual of isolating the separator). Letters AND
# digits are word content, so "ab12" stays one word; whitespace/punctuation cut.
_WORD_SEPARATORS = (WHITESPACE, PUNCT)


def word_spans(byte_seq, separators=_WORD_SEPARATORS):
    """Spans of the WORDS: maximal runs delimited by the separator classes.

    The word analyzer is the OR of the separator dichotomies (left-of-X /
    right-of-X for the separator classes) -- it marks every word boundary. A
    word is the content BETWEEN separators (the part to the right of one
    separator and the left of the next, plus the start/end runs); the
    ``[left-of-sep, right-of-sep]`` separator spans are dropped.
    """
    bs = _as_bytes(byte_seq)
    sep = set(separators)
    is_sep = [(_byte_class(b) in sep) for b in bs]
    spans = []
    i, n = 0, len(bs)
    while i < n:
        if is_sep[i]:
            i += 1
            continue
        a = i
        while i < n and not is_sep[i]:
            i += 1
        spans.append((a, i))
    return spans


def words(byte_seq, separators=_WORD_SEPARATORS):
    """The word strings -- :func:`word_spans` materialized to content."""
    bs = _as_bytes(byte_seq)
    return [bytes(bs[s:e]).decode("latin1")
            for (s, e) in word_spans(byte_seq, separators)]


def word_bounds(byte_seq, separators=_WORD_SEPARATORS):
    """Each word with its two bounding dichotomies: ``(span, left, right)``.

    A word is the dual composition ``right-of-sep o left-of-sep`` -- the mirror
    of ``is-a-space = left-of-space o right-of-space``. So a MIDDLE word is
    bounded on BOTH sides: ``left = right-of-(preceding separator)`` and
    ``right = left-of-(following separator)``. The first word's left bound and
    the last word's right bound are the input edges (``"start"`` / ``"end"``).
    """
    bs = _as_bytes(byte_seq)
    n = len(bs)
    out = []
    for (s, e) in word_spans(byte_seq, separators):
        left = ("start" if s == 0
                else "right-of-%s" % _BARE[_byte_class(bs[s - 1])])
        right = ("end" if e == n
                 else "left-of-%s" % _BARE[_byte_class(bs[e])])
        out.append(((s, e), left, right))
    return out


# Every boundary dichotomy that can bound a word -- the input edges plus
# left-of-X / right-of-X for each char class. A word's WHOLE is the union of its
# two bounds (drawn from this fixed atom set), so the codes are consistent
# across inputs.
_BOUND_NAMES = (["start", "end"]
                + ["left-of-%s" % _BARE[c] for c in (LETTER, DIGIT, WHITESPACE, PUNCT)]
                + ["right-of-%s" % _BARE[c] for c in (LETTER, DIGIT, WHITESPACE, PUNCT)])


def boundary_codes(dim=32, seed=2):
    """Fixed ``[0,1]`` code per boundary dichotomy (start / end / left-of-X /
    right-of-X) -- the analyzer atoms a word's whole is unioned from."""
    tbl = default_embed_table(len(_BOUND_NAMES), dim, seed=seed)
    return {nm: tbl[k] for k, nm in enumerate(_BOUND_NAMES)}


def word_wholes(byte_seq, dim=32, separators=_WORD_SEPARATORS, seed=2):
    """Each word with its NARROWEST whole = the UNION of its two bounds.

    The whole of ``"the"`` is a NEW code ``union(start, left-of-space)`` -- it
    comes from the ANALYZER (the bounding dichotomies), distinct from the word's
    PARTS (its bytes, from the synthesizer), so the word-concept sits between
    them. The union is the ``[0,1]`` join, so the whole dominates both bounds.
    Wider wholes are wider bound-pairs (``the cat = union(start, left-of-space)``
    at the next space; the sentence = ``union(start, end)``). Returns a list of
    ``(word, (left_bound, right_bound), whole_code)``.
    """
    codes = boundary_codes(dim, seed)
    bs = _as_bytes(byte_seq)
    out = []
    for (s, e), left, right in word_bounds(byte_seq, separators):
        whole = join_from_bottom(torch.stack([codes[left], codes[right]]))
        out.append((bytes(bs[s:e]).decode("latin1"), (left, right), whole))
    return out


# --- greedy union vs first intersection: the dual matching ----------------
# A whole here is the triple ``((start, end), code, label)``. The codebook of
# wholes is every (word-start gap, word-end gap) region, coded by the UNION
# (join) of its byte content. Greedy/longest match builds the maximal whole;
# first/shortest match (the dual) stops at the minimal part; intersecting two
# overlapping wholes (the meet) recovers their shared part.

def all_wholes(byte_seq, embed=None, dim=32, separators=_WORD_SEPARATORS):
    """Every word/phrase whole: each ``(word-start gap, word-end gap)`` region,
    coded by ``join_from_bottom`` of its byte content. ``"the cat sat"`` yields
    the, the cat, the cat sat, cat, cat sat, sat."""
    bs = _as_bytes(byte_seq)
    if embed is None:
        embed = default_embed_table(256, dim)
    spans = word_spans(byte_seq, separators)
    starts = [s for (s, _) in spans]
    ends = [e for (_, e) in spans]
    out = []
    for a in starts:
        for b in ends:
            if a < b:
                code = join_from_bottom(
                    torch.stack([embed[bs[p]].clamp(0.0, 1.0) for p in range(a, b)]))
                out.append(((a, b), code, bytes(bs[a:b]).decode("latin1")))
    return out


def match_greedy(wholes, start):
    """The LONGEST whole starting at ``start`` -- greedy union (the biggest)."""
    cands = [w for w in wholes if w[0][0] == start]
    return max(cands, key=lambda w: w[0][1], default=None)


def match_first(wholes, start):
    """The SHORTEST whole starting at ``start`` -- first intersection (the
    smallest; stop at the first complete match, the dual of greedy)."""
    cands = [w for w in wholes if w[0][0] == start]
    return min(cands, key=lambda w: w[0][1], default=None)


def tile(wholes, n, greedy=True):
    """Tile ``[0, n)`` by repeatedly matching from the cursor: ``greedy`` =
    longest (union, maximal cover -> the whole) or first = shortest
    (intersection, minimal -> the parts). Gaps (separator positions where no
    whole starts) are jumped over."""
    starts = sorted({w[0][0] for w in wholes})
    pick = match_greedy if greedy else match_first
    out, pos = [], 0
    while pos < n:
        m = pick(wholes, pos)
        if m is None:
            nxt = [s for s in starts if s > pos]
            if not nxt:
                break
            pos = nxt[0]
            continue
        out.append(m)
        pos = m[0][1]
    return out


def intersect_wholes(a, b):
    """The PART shared by two wholes: span intersection + ``meet_from_top`` of
    the codes (the De Morgan dual of the union). Returns ``(span, code)`` or
    ``None`` if the spans are disjoint. ``cat = intersect(the cat, cat sat)``."""
    (as_, ae), (bs_, be) = a[0], b[0]
    s, e = max(as_, bs_), min(ae, be)
    if s >= e:
        return None
    return ((s, e), meet_from_top(torch.stack([a[1], b[1]])))


def synthesize_words(byte_seq, embed=None, dim=32, separators=_WORD_SEPARATORS):
    """One synthesized whole PER WORD -- the convergence.

    The analyzer's separator-cuts bound the synthesizer's combine, so only
    word-wholes remain: synthesis builds the wholes WITHIN the boundaries
    analysis found. Returns a list of ``(span, MeronomyTree)`` with the trees'
    ``.where`` shifted to global byte offsets.
    """
    bs = _as_bytes(byte_seq)
    if embed is None:
        embed = default_embed_table(256, dim)
    out = []
    for (s, e) in word_spans(byte_seq, separators):
        sub = synthesize(bytes(bs[s:e]), embed=embed, dim=dim)
        sub.where = [(a + s, b + s) for (a, b) in sub.where]
        out.append(((s, e), sub))
    return out


def same_word(byte_seq, gap, separators=_WORD_SEPARATORS):
    """The word-bounding GATE: True iff ``gap`` is INSIDE a word -- both adjacent
    characters are word content (no separator). Combining across an inside-word
    gap stays within a word; refusing to combine across a word-boundary gap is
    what keeps the synthesis producing only words."""
    bs = _as_bytes(byte_seq)
    if gap <= 0 or gap >= len(bs):
        return False
    sep = set(separators)
    return (_byte_class(bs[gap - 1]) not in sep
            and _byte_class(bs[gap]) not in sep)


def synthesize_to_words(byte_seq, embed=None, dim=32, separators=_WORD_SEPARATORS):
    """Synthesize with the combination GATED to within-word (:func:`same_word`):
    combine each word's bytes but NEVER across a word-boundary gap, so the
    maximal wholes are exactly the WORDS. ``tree.roots`` are the word-wholes (a
    forest). Bounding the combination to the word level keeps it from forming
    cross-word byte-blobs -- faster learning and clean (accurate) word metas.
    """
    bs = _as_bytes(byte_seq)
    if embed is None:
        embed = default_embed_table(256, dim)
    tree = MeronomyTree(embed.shape[1])
    roots = []
    for (s, e) in word_spans(byte_seq, separators):
        leaves = [tree.add(embed[bs[p]].clamp(0.0, 1.0), (p, p + 1), chr(bs[p]))
                  for p in range(s, e)]
        roots.append(_combine_within(tree, leaves))
    tree.roots = roots
    if len(tree) > 1:
        tree._writeback(project_monotone(tree.codes(),
                                         where_containment_edges(tree.where)))
    return tree


def word_metas(byte_seq, embed=None, dim=32, separators=_WORD_SEPARATORS):
    """The word-level METAS -- where the word-gated synthesizer and the analyzer
    AGREE. Each meta is ``(word, whole_code, (left_bound, right_bound))``: the
    synthesizer's word-bounded whole plus the analyzer's bounds. Because the
    combination is bounded to words, every meta is one clean word (accurate) and
    learning need not explore sub-word or cross-word units.
    """
    tree = synthesize_to_words(byte_seq, embed=embed, dim=dim,
                               separators=separators)
    bounds = word_bounds(byte_seq, separators)
    return [(tree.reconstruct(root), tree.code[root], (lb, rb))
            for root, (_, lb, rb) in zip(tree.roots, bounds)]


def synthesize_phrases(byte_seq, embed=None, dim=32, separators=_WORD_SEPARATORS):
    """Combine the WORDS in a tree -- the separator dichotomies as JOINS.

    The word-cut gives the leaves (words); they are combined left-to-right, each
    phrase-whole spanning the separator gap between its two children. The
    separator is the node's ``.join`` (the dichotomy combined across) -- it
    disappears INTO the whole, the dual of the analyzer bracketing it out, so a
    phrase's ``.where`` spans the gap but the separator is not a separate part.
    Reconstruction is exact (children glued by their joins); ``tree.wholes(leaf)``
    gives each word's phrase wholes. Combine order is left-to-right -- the order
    IS the phrase structure, undetermined by the cuts (where grammar enters).
    """
    bs = _as_bytes(byte_seq)
    if embed is None:
        embed = default_embed_table(256, dim)
    tree = MeronomyTree(embed.shape[1])
    leaves = []
    for (s, e) in word_spans(byte_seq, separators):
        wc = torch.stack([embed[bs[p]].clamp(0.0, 1.0) for p in range(s, e)])
        leaves.append(tree.add(join_from_bottom(wc), (s, e),
                               bytes(bs[s:e]).decode("latin1"), tag="word"))
    if leaves:
        cur = leaves[0]
        for nxt in leaves[1:]:
            gap = bytes(bs[tree.where[cur][1]:tree.where[nxt][0]]).decode("latin1")
            code = join_from_bottom(torch.stack([tree.code[cur], tree.code[nxt]]))
            span = (tree.where[cur][0], tree.where[nxt][1])
            cur = tree.add(code, span, None, left=cur, right=nxt,
                           tag="phrase", join=gap)
        tree.roots = [cur]
        if len(tree) > 1:
            tree._writeback(project_monotone(
                tree.codes(), where_containment_edges(tree.where)))
    return tree


def _build_cut_tree(tree, s, e, cuts, bs, class_embed):
    """Recursively cut region ``[s, e)`` at its interior gap dichotomies.

    Leaves (no interior cut) are parts -- coded by their class, labelled by their
    content, tagged with the ``is-a-X`` analyzer. Internal nodes are dichotomies
    -- a binary cut at the leftmost interior gap, storing the two child regions,
    tagged with the dichotomy name and carrying its ``.boundary`` gap. Seeds the
    region as the dominating join of its children; ``analyze`` projects after.
    """
    interior = [(b, nm) for (b, nm) in cuts if s < b < e]
    if not interior:
        c = _byte_class(bs[s]) if s < e else LETTER
        content = bytes(bs[s:e]).decode("latin1")
        return tree.add(class_embed[c].clamp(0.0, 1.0), (s, e), content,
                        tag=_CLASS_NAME[c])
    b, nm = interior[0]                          # leftmost dichotomy first
    left = _build_cut_tree(tree, s, b, cuts, bs, class_embed)
    right = _build_cut_tree(tree, b, e, cuts, bs, class_embed)
    seed = join_from_bottom(torch.stack([tree.code[left], tree.code[right]]))
    return tree.add(seed, (s, e), None, left=left, right=right,
                    tag=nm, boundary=b)


def analyze(byte_seq, dim=32, class_seed=1):
    """Cut ``byte_seq`` into parts with the boundary-dichotomy cut tree.

    Returns a :class:`MeronomyTree` whose internal nodes are the dichotomies
    (binary cuts in the gaps between characters) and whose leaves are the parts
    (coded by their class, tagged ``is-a-X``, with their ``.where``). Codes are
    projected onto the ``.where`` containment order so each region dominates its
    parts (part <= whole). ``meet_from_top`` is the ``1-x`` De Morgan mirror of
    the synthesizer's join (see :func:`Mereology.meet_from_top`).
    """
    bs = _as_bytes(byte_seq)
    n = len(bs)
    class_embed = default_embed_table(4, dim, seed=class_seed)
    tree = MeronomyTree(dim)
    if n:
        cuts = class_dichotomies(bs)
        root = _build_cut_tree(tree, 0, n, cuts, bs, class_embed)
        tree.roots = [root]
        if len(tree) > 1:
            tree._writeback(project_monotone(
                tree.codes(), where_containment_edges(tree.where)))
    return tree
