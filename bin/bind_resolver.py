"""Parse-time contextual-BIND ranking. Ranks accessible participants from the
current parse's left-context by: (1) constructional licensing -- want =>
subject-control (prefer subject NP), persuade => object-control (prefer
object NP); (2) locality -- more recent (higher position) wins; (3) learned
participation -- additive score hook (default 0.0). Pure ranking over small
records so it is unit-testable without a live parse. This is the resolution
*core* and the licensing refinement path; the live fold (Task 2.4) uses its
locality branch as a vectorized nearest-left pick when lemmas are unavailable."""
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Participant:
    id: int
    vec: torch.Tensor          # constructed NP content, e.g. NP1 = INTERSECT(Alice, tired)
    role: str                  # 'subject' / 'object' / 'other'
    position: int              # surface order index (higher = more recent)
    participation: float = 0.0 # learned score hook

_LICENSE_ROLE = {"subject_control": "subject", "object_control": "object"}

def _score(part: Participant, licensing: Optional[str]) -> tuple:
    pref = _LICENSE_ROLE.get(licensing or "")
    licensed = 1.0 if (pref is not None and part.role == pref) else 0.0
    return (licensed, part.participation, float(part.position))

def rank_candidates(participants, licensing=None):
    """Return (ranked_best_first, chosen_index_into_original) or ([], None)."""
    if not participants:
        return [], None
    indexed = sorted(enumerate(participants),
                     key=lambda iv: _score(iv[1], licensing), reverse=True)
    return [p for _i, p in indexed], indexed[0][0]

def resolve_bind(participants, licensing=None):
    """Return (vec, chosen_index) or (None, None) when nothing is accessible."""
    _ranked, chosen = rank_candidates(participants, licensing=licensing)
    return (None, None) if chosen is None else (participants[chosen].vec, chosen)
