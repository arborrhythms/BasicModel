"""Ideas enter luminosity; shared-store relations constrain referenced rows.

Conflict statistics retain independent support poles and hysteresis.
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
    from part_fixtures import native_store
    return native_store(_D,capacity=16)[0]


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


# -- LTM2 reads the native arguments; the predicate is not a column. --

def test_part_arguments_remain_unscaled_and_separate_from_luminosity():
    from part_fixtures import part
    tl=_truth_layer();rs=_relative_store();tl.record(_vec(.9),degree=1.)
    before=tl.luminosity()
    left,right=_vec(1.),_vec(0.,0.,1.)
    edge=part(rs,left,right,evidence=(1.,0.))
    assert len(rs)==0 and len(rs.parts)==1
    value=rs.row(edge)
    assert value['refs'].numel()==2
    torch.testing.assert_close(value['meaning'].roles[0],left)
    torch.testing.assert_close(value['meaning'].roles[2],right)
    assert len(tl)==1 and tl.luminosity()==pytest.approx(before)


def test_relational_evaluation_respects_both_argument_positions():
    from part_fixtures import part
    rs=_relative_store();left,right=_vec(1.),_vec(0.,0.,1.)
    part(rs,left,right,evidence=(1.,0.))
    assert rs.evaluate(left,right)[0]==pytest.approx(1.)
    assert rs.evaluate(right,left)[0]==0
    assert .5 < rs.evaluate(_vec(1.,.3),right)[0] < 1.


def test_consequents_read_the_native_whole_without_a_predicate_filter():
    from part_fixtures import part
    rs=_relative_store();left,right=_vec(1.),_vec(0.,0.,1.)
    edge=part(rs,left,right,evidence=(1.,0.))
    result,=rs.consequents(_vec(.95,.05))
    assert result[0]==edge[1] and result[1]>.9
    torch.testing.assert_close(result[2],right)
    assert rs.consequents(_vec(0.,0.,0.,1.))==[]


def test_distinct_wholes_do_not_imply_negative_membership():
    from part_fixtures import symbol
    rs=_relative_store()
    a,b,c=(symbol(rs,p) for p in (_vec(1.),_vec(0.,1.),_vec(0.,0.,1.)))
    first=rs.append_relation(a,b,truth_plus=1.)
    second=rs.append_relation(a,c,truth_plus=1.)
    assert rs.row(first)['evidence']==rs.row(second)['evidence']==(1.,0.)
    assert rs.append_relation(a,b,truth_minus=.8)==first
    assert rs.row(first)['evidence']==pytest.approx((1.,.8))
    assert rs.row(second)['evidence']==(1.,0.)
