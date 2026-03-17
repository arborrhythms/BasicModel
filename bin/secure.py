"""Input security for BasicModel.

Prompt injection detection and input guard. Self-contained — no external
dependencies beyond stdlib.

WikiOracle's bin/security.py re-exports these and adds output safety
filters (detect_identifiability, detect_asymmetric_claim) from truth.py.
"""

from __future__ import annotations

import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Input Guard — prompt injection detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?prior\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"new\s+instructions?\s*:",
        r"system\s+prompt\s*:",
        r"act\s+as\s+(if\s+)?you\s+are",
        r"pretend\s+(that\s+)?you\s+are",
        r"override\s+(your\s+)?instructions",
        r"forget\s+(all\s+)?(your\s+)?instructions",
        r"reveal\s+(your\s+)?system\s+prompt",
        r"output\s+(your\s+)?system\s+(message|prompt)",
        r"what\s+is\s+your\s+system\s+prompt",
    ]
]

# Base64-encoded instruction blocks (common injection vector)
_BASE64_BLOCK = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# Excessive control characters (non-printable, non-whitespace)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]{3,}")


def detect_injection(content: str) -> Optional[str]:
    """Detect common prompt injection patterns.

    Returns a reason string if injection is detected, or None if clean.
    """
    if not isinstance(content, str) or not content.strip():
        return None

    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(content)
        if m:
            return f"prompt injection pattern: '{m.group(0)}'"

    # Check for suspiciously long base64 blocks (possible encoded instructions)
    if _BASE64_BLOCK.search(content):
        b64_matches = _BASE64_BLOCK.findall(content)
        total_b64 = sum(len(m) for m in b64_matches)
        if total_b64 > len(content) * 0.5 and total_b64 > 100:
            return "suspicious base64-encoded content"

    if _CONTROL_CHARS.search(content):
        return "excessive control characters"

    return None


# Configurable guard: when disabled, logs but does not block
_GUARD_ENABLED = os.getenv("BASICMODEL_INPUT_GUARD",
                           os.getenv("WIKIORACLE_INPUT_GUARD", "true")).lower() in ("true", "1", "yes")


def guard_input(content: str) -> Optional[str]:
    """Check input for injection. Returns reason if blocked, None if allowed.

    When BASICMODEL_INPUT_GUARD / WIKIORACLE_INPUT_GUARD is false, detection
    still runs but the function returns None (log-only mode).
    """
    reason = detect_injection(content)
    guard_input.last_detection = reason
    if reason and _GUARD_ENABLED:
        return reason
    return None

guard_input.last_detection = None
