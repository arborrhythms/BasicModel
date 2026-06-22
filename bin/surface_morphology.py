"""Pure, table-driven morphological analyzer (modality re-architecture
Phase 7 / Task 7.1; doc/plans/2026-06-03-modality-architecture-plan.md).

``analyze(token) -> (lemma, features)`` generalizes surface_tense's verb
tense/aspect normalization to a single surface token and adds a CORRECTED
lemmatizer that fixes the known -ed / -ing over-fire (``surface_tense._base_of``
strips too aggressively: ``seed -> se``, ``freed -> fre``). Stripping fires
ONLY when the de-doubled residue passes a min-stem-length (>= 3) + stoplist +
irregular gate, so ``seed`` / ``freed`` are not mis-stripped.

``features`` is a small, role-neutral dict -- ``{"tense", "aspect"}`` for a
recognized verb form, ``{}`` for everything else. Non-verb / unknown tokens
pass through as ``(token, {})``.

Pure Python: no torch, no model state, and -- per the hard constraint -- NO
global POS inventory. The only lexical knowledge is the small closed-class
verb tables reused from surface_tense plus a short over-fire stoplist.
"""

import surface_tense as _st

# -ed / -ing surface forms that are NOT verb inflections. The naive strip
# over-fires on them (``seed -> se``, ``freed -> fre``, ``thing -> th``); those
# whose residue is still >= _MIN_STEM (e.g. ``freed -> fre``) slip past the
# length gate, so they are stoplisted explicitly. (``seed -> se`` is already
# caught by the length gate; it is listed for documentation.)
_LEMMA_STOPLIST = {
    # -ed non-inflections
    "freed", "feed", "deed", "weed", "indeed", "speed", "bleed", "breed",
    "creed", "greed", "agreed", "exceed", "succeed", "proceed", "reed",
    "heed", "seed", "embed",
    # -ing non-verbs
    "thing", "king", "ring", "wing", "string", "spring", "ceiling",
    "morning", "evening", "nothing", "something", "everything", "during",
}
_MIN_STEM = 3


def analyze(token):
    """Return ``(lemma, features)`` for a single surface token.

    Recognized verb forms yield ``(lemma, {"tense": ..., "aspect": [...]})``;
    everything else passes through as ``(token, {})``.
    """
    if not isinstance(token, str) or not token:
        return (token, {})

    # 1. Irregular / closed-class verb forms (ran, running, been, is, ...):
    #    reuse surface_tense's irregular-base + finite-tense tables.
    if token in _st._IRREGULAR_BASE:
        lemma = _st._IRREGULAR_BASE[token]
        tense = _st._FINITE_TENSE.get(token, _st._DEFAULT_TENSE)
        aspect = ["PROGRESSIVE"] if _st._is_ing(token) else []
        return (lemma, {"tense": tense, "aspect": aspect})

    # 2. Regular inflection, GATED against the -ed / -ing over-fire: strip
    #    only when the token is not a known non-inflection AND the de-doubled
    #    residue is a plausible stem (len >= _MIN_STEM).
    if token not in _LEMMA_STOPLIST:
        if token.endswith("ing") and len(token) > 3:
            stem = _st._strip_doubled(token[:-3])
            if len(stem) >= _MIN_STEM:
                return (stem, {"tense": _st._DEFAULT_TENSE,
                               "aspect": ["PROGRESSIVE"]})
        if token.endswith("ed") and len(token) > 2:
            stem = _st._strip_doubled(token[:-2])
            if len(stem) >= _MIN_STEM:
                return (stem, {"tense": "PAST", "aspect": []})

    # 3. Plain / unknown token -- role-neutral passthrough.
    return (token, {})


# --------------------------------------------------------------------------
# Generation direction (rewrite() Stage D2, doc/old/2026-06-20-idea-decoder.md
# "rewrite()"): realize(lemma, features) -> surface form. The INVERSE of
# ``analyze``, so the two together are the bidirectional surface realizer (many
# surface forms <-> one lexeme). DUAL ROUTE ("words and rules"):
#   1. a small IRREGULAR lookup (canonical closed-class forms) -- memorized;
#   2. the regular -s / -ed / -ing rules (with CVC doubling / e-drop) -- productive.
# Pure Python, table-driven; reuses surface_tense's irregular/tense knowledge.
# (The productive route is rule-based here; a learned char-seq2seq is the upgrade
# per the SIGMORPHON evidence -- inflection is string transduction.)
# --------------------------------------------------------------------------

# Canonical irregular GENERATION table: (lemma, tense, aspect_key) -> surface.
# aspect_key is "PROGRESSIVE" (V-ing) or "SIMPLE". Person/number is not in the
# feature bundle, so a canonical form is chosen (be+PRESENT -> "is").
_IRREGULAR_GEN = {
    ("run", "PRESENT", "SIMPLE"): "run", ("run", "PAST", "SIMPLE"): "ran",
    ("run", "PRESENT", "PROGRESSIVE"): "running",
    ("be", "PRESENT", "SIMPLE"): "is", ("be", "PAST", "SIMPLE"): "was",
    ("be", "PRESENT", "PROGRESSIVE"): "being",
    ("have", "PRESENT", "SIMPLE"): "has", ("have", "PAST", "SIMPLE"): "had",
    ("do", "PRESENT", "SIMPLE"): "does", ("do", "PAST", "SIMPLE"): "did",
    ("will", "FUTURE", "SIMPLE"): "will",
}


def _double_final(stem):
    """Inverse of ``_strip_doubled``: double a final CVC consonant before a
    vocalic suffix (``run`` -> ``runn``, ``stop`` -> ``stopp``). Skips w/x/y
    and non-CVC endings."""
    if (len(stem) >= 3
            and stem[-1] not in _st._VOWELS and stem[-1] not in "wxy"
            and stem[-2] in _st._VOWELS and stem[-3] not in _st._VOWELS):
        return stem + stem[-1]
    return stem


def _regular_ing(lemma):
    if lemma.endswith("e") and not lemma.endswith("ee") and len(lemma) > 2:
        return lemma[:-1] + "ing"            # make -> making
    return _double_final(lemma) + "ing"      # run -> running, walk -> walking


def _regular_ed(lemma):
    if lemma.endswith("e"):
        return lemma + "d"                   # like -> liked
    return _double_final(lemma) + "ed"       # stop -> stopped, walk -> walked


def realize(lemma, features=None):
    """``(lemma, features) -> surface form`` -- the inverse of ``analyze``.

    ``features`` = ``{"tense": ..., "aspect": [...]}`` (as ``analyze`` returns).
    Irregular lookup wins; otherwise the regular rule for the tense/aspect fires.
    Unknown/empty features -> the lemma (PRESENT simple)."""
    if not isinstance(lemma, str) or not lemma:
        return lemma
    features = features or {}
    tense = features.get("tense", _st._DEFAULT_TENSE)
    aspect = features.get("aspect", []) or []
    ak = "PROGRESSIVE" if "PROGRESSIVE" in aspect else "SIMPLE"
    surf = _IRREGULAR_GEN.get((lemma, tense, ak))
    if surf is not None:
        return surf
    if ak == "PROGRESSIVE":
        return _regular_ing(lemma)
    if tense == "PAST":
        return _regular_ed(lemma)
    if tense == "FUTURE":
        return "will " + lemma               # periphrastic future
    return lemma                             # PRESENT simple
