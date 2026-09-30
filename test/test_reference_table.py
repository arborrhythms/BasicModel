"""Stage 6 of doc/plans/MeronomyPlan.md: the word/object binding table.

MeronomySpec §6 (rev 2026-06-11) / §10.8 / §10.9: full rows only,
word-keyed (deref indexed; ref = unindexed object-side search),
append-only and gate-licensed; symbols are atomic (quasi-orthogonal
zero-banded codes, "approximately the index"); mint-time dominance
makes search work (`A ⊑ σ(A,B)`); ⊥ extents cached as
definable-but-empty and ⊤ saturation detectable.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch



_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from References import symbol_code
from Layers import PiLayer2, SigmaLayer2, Ops

D = 4


# ---------------------------------------------------------------------------
# Full rows only; append-only; gate-licensed.
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# deref indexed / ref unindexed — the API audit.
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Symbol codes: atomic, zero-banded, approximately the index.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_symbol_code_shape_and_zero_bands():
    c = symbol_code(0, n_what=4, n_where=2, n_when=2)
    assert c.shape == (8,), "MM_20M idiom: 4+2+2 = 8"
    assert (c[4:] == 0).all(), "zero-band signature: symbols are not in space"
    assert abs(c[:4].norm().item() - 1.0) < 1e-6


def test_symbol_code_deterministic_identity():
    a1 = symbol_code(3, n_what=8)
    a2 = symbol_code(3, n_what=8)
    b = symbol_code(4, n_what=8)
    assert torch.equal(a1, a2), "the code IS (approximately) the index"
    assert not torch.allclose(a1, b), "distinct indices, distinct codes"


def test_symbol_codes_are_mereologically_inert():
    # Quasi-orthogonal signed codes: pairwise incomparable under the
    # dominance order (seed-pinned configuration), so symbols carry no
    # size relations -- atoms, outside the meronomy.
    codes = [symbol_code(i, n_what=8)[:8] for i in range(12)]
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            assert not bool(Ops.partOf(codes[i], codes[j])), (
                f"symbol {i} ⊑ symbol {j}: codes must not be size-related")


# ---------------------------------------------------------------------------
# Mint-time dominance is what makes ref-search work (§10.9).
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Gauge orientation at bind; evaluate-before-cache degeneracies.
# ---------------------------------------------------------------------------
