"""Single source of truth for the converged architecture's fixed per-space_role
shape. .where/.when are no longer config options: every space's spatial/
temporal widths come from canonical_shape(section).

Per the 2026-06-06 dim-convention unification: the INTERIOR space_roles all carry
the SAME band so the dimensional formula is uniform:
``nDim = nWhat + nWhere + nWhen``. The only difference between interior space_roles
is whether the band slots are actively muxed (carry per-event where/when
values) or ride along as inert padding — the bookkeeping is identical. This
SUPERSEDES the earlier convention that gave WS/SymbolSpace ``(0, 0)`` and
demuxed at the CS->WS boundary (now a no-op identity reshape), which
simplifies the constructor chain and makes ``space[i].nOutputDim ==
space[i+1].nInputDim`` directly comparable for handoff validation.

2026-07-09 multi-rung pass: the band is (nWhere=4, nWhen=4) — `.when` is the
4-dim 2-rung start ladder (WhenStartDurationEncoding; slots [-4..-1]) and
`.where` is now the SAME 2-rung ladder shape over the byte START
(WhereEncoding; slots [-8..-5]; LF = <wherePeriod> range, HF =
wherePeriod/<whereRungRatio> sub-byte resolution). The endpoint-sum bracket is
retired from the muxed band (the analyzer's EndpointSumWhere keeps it).

The TWO principled exceptions stay ``(0, 0)``: there is no positional
encoding BEFORE the input or AFTER the output. OutputSpace (the terminal
prediction / answer) is one — a scalar/answer has no .where/.when to mux,
and the loss would otherwise slice empty where/when segments and NaN."""

_CANONICAL_SHAPE = {
    "InputSpace":      (4, 4),
    "PartSpace": (4, 4),
    # ModalSpace is the demuxed perceptual-space_role composite (Spaces.ModalSpace):
    # it routes what/where/when through sub-PartSpaces and shares the
    # perceptual shape. No live config currently enables demuxed mode.
    "ModalSpace":      (4, 4),
    "ConceptualSpace": (4, 4),
    "WholeSpace":   (4, 4),
    # Exception: the terminal output carries no positional encoding -- the
    # answer has no .where/.when (see module docstring).
    "OutputSpace":     (0, 0),
    "SymbolSpace":       (4, 4),
}
# 2026-06-04: no space_role's codebook is mandatory. A config opts into a codebook
# explicitly via <codebook>quantize</codebook>; any space_role may resolve to
# <codebook>none</codebook>. This restores compatibility with the pre-modality
# exact-XOR reconstruction smoke test, where PartSpace / ConceptualSpace
# / WholeSpace are full-width INVERTIBLE PASSTHROUGHS (no VQ snap) so the
# forward<->reverse chain round-trips exactly and the butterfly pi computes
# XOR with cross-slot reach. Reverts the modality re-architecture's
# mandatory-PS/WS-codebook constraint.
MANDATORY_CODEBOOK_SPACE_ROLES = set()


def canonical_shape(section):
    """(nWhere, nWhen) for a space section. Raises on an unknown section so a
    new space_role cannot silently default to a wrong shape."""
    try:
        return _CANONICAL_SHAPE[section]
    except KeyError:
        raise ValueError(f"canonical_shape: unknown section {section!r}")
