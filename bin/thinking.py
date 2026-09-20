"""Truth interval values retained for evidence readers.

The frame kernel, Testimony/addressees and five-operation imitation policy
were retired. BasicModel.run_selected_thought is the ordinary controller.
"""
from dataclasses import dataclass, field

TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"
MIXED = "mixed"
CONFLICTING = "conflicting"
BOUNDED_UNKNOWN = "bounded_unknown"

# -- Truth intervals (spec §1.3/§5.2) -----------------------------------------

@dataclass
class TruthInterval:
    """Signed truth interval ``[lower, upper] ⊆ [-1, 1]`` + trust + provenance."""

    lower: float = 0.0
    upper: float = 0.0
    trust: float = 0.0
    provenance: list = field(default_factory=list)

    @property
    def luminosity(self) -> float:
        """Distance from unknownness (§1.2; the §15 sketch's max-abs)."""
        return max(abs(self.lower), abs(self.upper))

    def status(self, tau: float = 0.5) -> str:
        """Classify against a determination bar ``tau``: one-sided luminous ⇒
        true/false; two-sided-strong ⇒ conflicting; a luminous straddle ⇒
        mixed; else unknown."""
        if self.luminosity == 0:
            return UNKNOWN
        if self.lower < 0 < self.upper and self.lower <= -tau and self.upper >= tau:
            return CONFLICTING
        if self.luminosity < tau:
            return UNKNOWN
        if self.lower > 0.0:
            return TRUE
        if self.upper < 0.0:
            return FALSE
        return MIXED

    @classmethod
    def from_evidence(cls, evidence):
        """Build from ``[(signed_value, trust, provenance), …]``; empty ⇒
        ``[0, 0]`` at trust 0 (effectively unknown)."""
        ev = [e for e in (evidence or []) if e is not None]
        if not ev:
            return cls()
        vals = [float(v) for (v, _t, _p) in ev]
        return cls(lower=min(vals), upper=max(vals),
                   trust=max(abs(float(t)) for (_v, t, _p) in ev),
                   provenance=[p for (_v, _t, p) in ev])
