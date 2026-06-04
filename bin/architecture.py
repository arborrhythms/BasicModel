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
MANDATORY_CODEBOOK_TIERS = {"PerceptualSpace", "SymbolicSpace"}


def canonical_shape(section):
    """(nWhere, nWhen) for a space section. Raises on an unknown section so a
    new tier cannot silently default to a wrong shape."""
    try:
        return _CANONICAL_SHAPE[section]
    except KeyError:
        raise ValueError(f"canonical_shape: unknown section {section!r}")
