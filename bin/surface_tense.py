"""Pure, table-driven surface tense/aspect normalizer (Phase 4, Task 4.1).

doc/plans/2026-06-03-contextual-bind-preposition-when.md "Operation 3:
tense / aspect". ``normalize_surface(tokens)`` turns an ordered verb-group
surface (a list of auxiliaries + the head verb) into the abstract triple

    (tense, aspect_chain, base_verb)

that the unary ``.when`` ops (``TenseLayer`` / ``AspectLayer``) consume:

  * ``tense``        one of PRESENT / PAST / FUTURE -- read from the FIRST
                     finite element of the group (``had`` / ``did`` / ``will``
                     ...); PRESENT is the default for ``is`` / ``has`` /
                     ``does`` / a bare present head.
  * ``aspect_chain`` the aspects in SURFACE order, OUTERMOST first. Each
                     auxiliary classifies the form of the token it governs:
                     ``have`` + past-participle -> PERFECT; ``be`` + V-ing ->
                     PROGRESSIVE; ``do`` is tense-support only (SIMPLE, no
                     aspect); ``will`` is FUTURE tense with no aspect.
  * ``base_verb``    the lemma of the head verb (last token), via an irregular
                     table + a suffix-strip (``-ing`` / ``-ed``) fallback with
                     consonant de-doubling (``running`` -> ``run``).

This module is deliberately PURE Python: no torch, no model state. It is the
analyzer half; the synthesis half lives in the ops. There is NO global POS
inventory -- the small closed-class auxiliary tables below are the only
lexical knowledge, exactly as the spec requires.
"""

# Finite forms that are inherently PAST (or FUTURE for ``will``). Anything not
# listed -- ``is`` / ``has`` / ``does`` / a bare present head -- defaults to
# PRESENT. Tense is read from the FIRST finite element of the group.
_FINITE_TENSE = {
    "ran": "PAST", "had": "PAST", "did": "PAST", "was": "PAST",
    "were": "PAST", "will": "FUTURE",
    # bare present finites map to the default below, but listed for clarity:
    # "is"/"are"/"am"/"has"/"have"/"do"/"does" -> PRESENT (default).
}
_DEFAULT_TENSE = "PRESENT"

# Closed-class auxiliary lemmas, keyed by surface form -> family. The family
# tells us which aspect the auxiliary contributes (or that it is tense-support
# / a modal). The head verb is whatever follows the last auxiliary.
_HAVE_AUX = {"have", "has", "had"}            # + past participle -> PERFECT
_BE_AUX = {"be", "is", "am", "are", "was",    # + V-ing          -> PROGRESSIVE
           "were", "been", "being"}
_DO_AUX = {"do", "does", "did"}               # tense-support only (SIMPLE)
_MODAL = {"will"}                             # FUTURE tense, no aspect

# Irregular base (lemma) lookups (the fixtures require ``ran`` -> ``run`` and
# ``been`` -> ``be``). ``running`` is also listed for directness, though the
# de-doubling suffix strip below would derive it too.
_IRREGULAR_BASE = {
    "ran": "run", "run": "run", "running": "run",
    "been": "be", "being": "be", "is": "be", "am": "be", "are": "be",
    "was": "be", "were": "be", "be": "be",
    "had": "have", "has": "have", "have": "have",
    "did": "do", "does": "do", "do": "do",
    "will": "will",
}

# Consonants whose final letter is doubled before ``-ing`` / ``-ed`` (CVC
# doubling, e.g. run -> running, stop -> stopped). Used by the de-doubling
# strip so ``runn`` -> ``run`` rather than the naive ``runn``.
_VOWELS = set("aeiou")


def _strip_doubled(stem):
    """De-double a trailing CVC consonant: ``runn`` -> ``run``.

    Only collapses a doubled FINAL consonant (``...XX`` where ``X`` is a
    non-vowel), which is the orthographic gemination English adds before a
    vocalic suffix. Leaves genuine doubles in other positions alone.
    """
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        return stem[:-1]
    return stem


def _base_of(token):
    """Lemma of a single surface verb form.

    Order: irregular table first (covers ``ran`` / ``been`` / ``running`` /
    auxiliaries), then a generic suffix strip: ``-ing`` with CVC de-doubling
    (``running`` -> ``runn`` -> ``run``), then ``-ed`` (``walked`` -> ``walk``;
    de-doubled like ``stopped`` -> ``stop``). Falls back to the token itself.
    """
    if token in _IRREGULAR_BASE:
        return _IRREGULAR_BASE[token]
    if token.endswith("ing") and len(token) > 3:
        return _strip_doubled(token[:-3])
    if token.endswith("ed") and len(token) > 2:
        return _strip_doubled(token[:-2])
    return token


def _is_ing(token):
    """True for a present-participle (V-ing) surface form."""
    return token.endswith("ing")


def _is_non_progressive(token):
    """True for any non-V-ing surface form (named for what it tests).

    The spec's PERFECT trigger is ``have + past-participle``. We keep no
    participle inventory: a have-aux followed by any non-V-ing token is read as
    governing a participle (the de-doubling lemma table handles the base verb)."""
    return not _is_ing(token)


def normalize_surface(tokens):
    """``(tense, aspect_chain, base_verb)`` for an ordered verb group.

    See module docstring. ``tokens`` is the surface verb group in order
    (auxiliaries then head). Tense is the first finite element's tense
    (PRESENT default); the aspect chain is built OUTERMOST-first by walking
    the auxiliaries; the base verb is the head (last token) lemmatized.
    """
    if not tokens:
        return (_DEFAULT_TENSE, [], "")

    # Tense: first finite element. The first token IS the finite element of a
    # verb group (auxiliaries front the group; a bare head is itself finite).
    tense = _FINITE_TENSE.get(tokens[0], _DEFAULT_TENSE)

    # Aspect chain: walk auxiliaries left -> right. Each auxiliary classifies
    # the NEXT token's form and contributes its aspect (OUTERMOST first).
    aspect_chain = []
    i = 0
    while i < len(tokens) - 1:          # last token is the head, never an aux here
        aux = tokens[i]
        nxt = tokens[i + 1]
        if aux in _HAVE_AUX and _is_non_progressive(nxt):
            aspect_chain.append("PERFECT")          # have + participle
        elif aux in _BE_AUX and _is_ing(nxt):
            aspect_chain.append("PROGRESSIVE")       # be + V-ing
        elif aux in _DO_AUX:
            pass                                     # tense-support: SIMPLE (no aspect)
        elif aux in _MODAL:
            pass  # MODAL(will, X) hook noted, not built -- FUTURE tense, no aspect
        # else: not a recognized auxiliary -> contributes no aspect.
        i += 1

    base_verb = _base_of(tokens[-1])
    return (tense, aspect_chain, base_verb)
