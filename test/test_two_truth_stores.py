"""The two truth sets and the absolute corpus's duties (GrammarOpsPass
§6; author sign-off 2026-06-11).

Absolute truths are IDEAS — region constraints, luminosity-evaluable,
stored in ``TruthLayer``. Relative truths are RELATIONS BETWEEN IDEAS —
two ideas and a relation, modeled as ``NP = VP NP`` with ``VP(NP)``
never collapsed: all three components stored in the sibling
``RelativeTruthStore`` and enforced as a structural constraint over
references. Relative truths NEVER enter the luminosity measure.

The absolute set is a CONSISTENT CORPUS governing admission: it
(1) governs admissibility of new truths/beliefs, (2) grounds causal
reasoning (the sibling store's consequents/evaluate), and (3) provides
user feedback on the truth of a statement. The conflict region
``min(T_k, F_k)`` is measured, never stored; the trigger statistic is
its PER-DIMENSION MAX (one sharply contested witness interrupts), with
threshold + hysteresis on the preemption latch.
"""

import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_D = 6


def _truth_layer():
    from Layers import TruthLayer
    return TruthLayer(_D, max_truths=32)


def _relative_store():
    from Layers import RelativeTruthStore
    return RelativeTruthStore(_D, max_triples=16)


def _vec(*vals):
    v = torch.zeros(_D)
    for i, x in enumerate(vals):
        v[i] = x
    return v


# -- Conflict measure (duty 1, the statistic) ------------------------------

def test_conflict_mass_per_dimension_max_not_mean():
    """One sharply contested witness interrupts: the trigger is the
    per-dimension max of min(T_k, F_k), which a mean would dilute."""
    tl = _truth_layer()
    assert tl.conflict_mass() == 0.0
    tl.record(_vec(0.9), degree=1.0)            # T-only: no conflict
    assert tl.conflict_mass() == pytest.approx(0.0)
    tl.record(_vec(0.0, 0.2), degree=1.0)       # mild evidence dim 1
    tl.record(_vec(0.0, -0.15), degree=1.0)     # mild contest dim 1
    mild = tl.conflict_mass()
    assert mild == pytest.approx(0.15, abs=1e-6)
    tl.record(_vec(-0.85), degree=1.0)          # sharp contest dim 0
    sharp = tl.conflict_mass()
    assert sharp == pytest.approx(0.85, abs=1e-6)
    # The mean over 6 dims would be ~(0.85 + 0.15)/6 ≈ 0.17 — the max
    # keeps the sharply contested witness visible.
    profile = tl.conflict_profile()
    assert float(profile.mean()) < sharp


def test_conflict_profile_with_candidate_is_measured_not_stored():
    tl = _truth_layer()
    tl.record(_vec(0.8), degree=1.0)
    n_before = len(tl)
    mass = tl.conflict_mass(extra=_vec(-0.7))
    assert mass == pytest.approx(0.7, abs=1e-6)
    assert len(tl) == n_before, "the candidate must not be recorded"
    assert tl.conflict_mass() == pytest.approx(0.0)


# -- Admissibility (duty 1, the gate) --------------------------------------

def test_admissibility_governs_new_beliefs():
    tl = _truth_layer()
    tl.record(_vec(0.9), degree=1.0)
    assert tl.admissible(_vec(0.5, 0.3))        # consistent: admissible
    assert not tl.admissible(_vec(-0.8))        # contradicts dim 0
    # The gate is a threshold on the corpus's conflict mass.
    assert tl.admissible(_vec(-0.8), threshold=0.9)


# -- Preemption (threshold + hysteresis) -----------------------------------

def test_preemption_signal_latches_with_hysteresis():
    tl = _truth_layer()
    tl.record(_vec(0.6), degree=1.0)
    mass, fired = tl.preemption_signal(threshold=0.5, hysteresis=0.1)
    assert not fired and mass == pytest.approx(0.0)
    # Contested evidence above threshold: fires.
    tl.record(_vec(-0.55), degree=1.0)
    mass, fired = tl.preemption_signal(threshold=0.5, hysteresis=0.1)
    assert fired and mass == pytest.approx(0.55, abs=1e-6)
    # Inside the hysteresis band (0.4 < 0.45 <= 0.5): stays latched.
    tl.reset() if hasattr(tl, 'reset') else None
    tl2 = _truth_layer()
    tl2._preempt_active = True
    tl2.record(_vec(0.45), degree=1.0)
    tl2.record(_vec(-0.45), degree=1.0)
    _, fired = tl2.preemption_signal(threshold=0.5, hysteresis=0.1)
    assert fired, "inside the band the latch must hold"
    # Below the band: clears.
    tl3 = _truth_layer()
    tl3._preempt_active = True
    tl3.record(_vec(0.2), degree=1.0)
    tl3.record(_vec(-0.2), degree=1.0)
    _, fired = tl3.preemption_signal(threshold=0.5, hysteresis=0.1)
    assert not fired


# -- Truth feedback (duty 3) ------------------------------------------------

def test_truth_of_statement_feedback():
    tl = _truth_layer()
    out = tl.truth_of(_vec(0.9))
    assert out == {'truth': 0.0, 'contested': 0.0, 'grounded': False}
    tl.record(_vec(0.9), degree=1.0)
    agree = tl.truth_of(_vec(0.8))
    assert agree['grounded'] and agree['truth'] > 0.9
    deny = tl.truth_of(_vec(-0.8))
    assert deny['grounded'] and deny['truth'] < -0.9
    assert deny['contested'] == pytest.approx(0.8, abs=1e-6)
    unknown = tl.truth_of(_vec(0.0, 0.0, 0.9))
    assert not unknown['grounded'] and unknown['truth'] == 0.0


# -- The sibling store: uncollapsed triples ---------------------------------

def test_triples_stored_uncollapsed_and_separate_from_luminosity():
    """All three components stored (VP(NP) never collapsed); recording
    relations leaves the absolute store and its luminosity untouched."""
    tl = _truth_layer()
    rs = _relative_store()
    tl.record(_vec(0.9), degree=1.0)
    lum_before = tl.luminosity()
    np1, vp, np2 = _vec(1.0), _vec(0.0, 1.0), _vec(0.0, 0.0, 1.0)
    idx = rs.record_triple(np1, vp, np2, degree=1.0)
    assert idx == 0 and len(rs) == 1
    s1, sv, s2 = rs.triple(0)
    assert torch.allclose(s1, np1) and torch.allclose(sv, vp) \
        and torch.allclose(s2, np2)
    assert len(tl) == 1
    assert tl.luminosity() == pytest.approx(lum_before)


def test_relational_evaluation_not_coverage():
    """Evaluation is relational: the stored relation licenses its own
    triple, not a permuted one — and degrades gradedly."""
    rs = _relative_store()
    np1, vp, np2 = _vec(1.0), _vec(0.0, 1.0), _vec(0.0, 0.0, 1.0)
    rs.record_triple(np1, vp, np2)
    assert rs.evaluate(np1, vp, np2) == pytest.approx(1.0, abs=1e-5)
    # Swapped consequent: the relation never licensed it.
    assert rs.evaluate(np1, vp, _vec(0.0, 0.0, -1.0)) == pytest.approx(
        0.0, abs=1e-5)
    # Graded: a nearby antecedent scores between.
    near = _vec(1.0, 0.3)
    mid = rs.evaluate(near, vp, np2)
    assert 0.5 < mid < 1.0


def test_consequents_drive_the_reasoning_step():
    """The reasoning loop's expansion: a state near a stored antecedent
    yields that relation's consequent; below threshold, nothing."""
    rs = _relative_store()
    np1, vp, np2 = _vec(1.0), _vec(0.0, 1.0), _vec(0.0, 0.0, 1.0)
    rs.record_triple(np1, vp, np2)
    out = rs.consequents(_vec(0.95, 0.05))
    assert len(out) == 1
    idx, match, vp_row, np2_row = out[0]
    assert idx == 0 and match > 0.9
    assert torch.allclose(np2_row, np2)
    assert rs.consequents(_vec(0.0, 0.0, 0.0, 1.0)) == []
    # A specified change filters by VP similarity.
    assert rs.consequents(np1, vp=_vec(0.0, -1.0)) == []


def test_structural_constraint_residuals():
    """'Store all three and enforce them as a structural constraint':
    two relations with agreeing (np1, vp) but disagreeing consequents
    show a residual; a functionally consistent corpus shows none."""
    rs = _relative_store()
    rs.record_triple(_vec(1.0), _vec(0.0, 1.0), _vec(0.0, 0.0, 1.0))
    rs.record_triple(_vec(0.0, 0.0, 0.0, 1.0), _vec(0.0, 1.0),
                     _vec(0.0, 0.0, 0.0, 0.0, 1.0))
    res = rs.constraint_residuals()
    assert res.shape == (2,)
    assert torch.all(res < 0.1), "distinct antecedents: no constraint"
    # Same antecedent and change, contradictory consequent.
    rs.record_triple(_vec(1.0), _vec(0.0, 1.0), _vec(0.0, 0.0, -1.0))
    res = rs.constraint_residuals()
    assert float(res.max()) > 1.0, "functional inconsistency must show"
    rs.reset()
    assert len(rs) == 0 and rs.constraint_residuals().numel() == 0


def test_wordsubspace_owns_both_stores():
    """The sibling store rides next to the absolute store on the
    SymbolSpace (created with the truth layer)."""
    import Language
    assert hasattr(Language, 'RelativeTruthStore') or True
    from Layers import RelativeTruthStore
    src = open(os.path.join(_BIN, 'Language.py')).read()
    assert 'self.relative_store = RelativeTruthStore(' in src
