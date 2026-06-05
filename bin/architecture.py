"""Single source of truth for the converged architecture's fixed per-tier
shape. .where/.when are no longer config options: every space's spatial/
temporal widths come from canonical_shape(section).

Per the modality re-architecture (doc/plans/2026-06-03-modality-architecture
-design.md), .where/.when are properties of occurrences/events, not of
symbols. They mux into the muxed event at IS->PS->CS (each carries
where=2/when=2) and demux at CS->SS, so a symbol qua symbol (SymbolicSpace)
and the output tier (OutputSpace) carry NEITHER; WordSpace likewise carries
neither. This SUPERSEDES the earlier SS-promotion convergence: SS is *not* a
where/when carrier; CS is."""

_CANONICAL_SHAPE = {
    "InputSpace":      (2, 2),
    "PerceptualSpace": (2, 2),
    # ModalSpace is the demuxed perceptual-tier composite (Spaces.ModalSpace):
    # it routes what/where/when through sub-PerceptualSpaces and shares the
    # perceptual shape. No live config currently enables demuxed mode.
    "ModalSpace":      (2, 2),
    "ConceptualSpace": (2, 2),
    "SymbolicSpace":   (0, 0),
    "OutputSpace":     (0, 0),
    "WordSpace":       (0, 0),
}
# 2026-06-04: no tier's codebook is mandatory. A config opts into a codebook
# explicitly via <codebook>quantize</codebook>; any tier may resolve to
# <codebook>none</codebook>. This restores compatibility with the pre-modality
# exact-XOR reconstruction smoke test, where PerceptualSpace / ConceptualSpace
# / SymbolicSpace are full-width INVERTIBLE PASSTHROUGHS (no VQ snap) so the
# forward<->reverse chain round-trips exactly and the butterfly pi computes
# XOR with cross-slot reach. Reverts the modality re-architecture's
# mandatory-PS/SS-codebook constraint.
MANDATORY_CODEBOOK_TIERS = set()


def canonical_shape(section):
    """(nWhere, nWhen) for a space section. Raises on an unknown section so a
    new tier cannot silently default to a wrong shape."""
    try:
        return _CANONICAL_SHAPE[section]
    except KeyError:
        raise ValueError(f"canonical_shape: unknown section {section!r}")
