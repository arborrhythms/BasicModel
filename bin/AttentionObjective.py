"""Validation and lane tolerance for live attention."""
import math


def tolerate_heterogeneity(lanes, tolerance):
    """Remove the configured overlap, preserving the two meaning lanes."""
    if not math.isfinite(tolerance) or not 0. <= tolerance <= 1.:
        raise ValueError('hetTolerance must be finite and between zero and one')
    if lanes.shape[-1] != 2:
        raise ValueError('heterogeneity requires paired meaning lanes')
    if tolerance == 1.:
        return lanes
    return lanes - (1. - tolerance) * lanes.amin(-1, keepdim=True)
