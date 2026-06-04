"""Contextual BIND (Phase 2).

doc/plans/2026-06-03-contextual-bind-preposition-when.md "Operation 2:
contextual BIND". A binary C-tier marker op that, at parse time, resolves
LIFT(BIND, VP) into LIFT(resolved_ref, VP): the missing NP is filled from
an accessible participant already built in the current parse. Resolution
is (1) constructional licensing (want => subject-control, persuade =>
object-control), (2) locality (nearest-left), (3) learned participation.
Hard rule: no global POS inventory -- ``bind`` is a role-only operator.
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from bind_resolver import Participant, rank_candidates, resolve_bind


def _p(id, role, position, participation=0.0):
    return Participant(id=id, vec=torch.full((4,), float(id)), role=role,
                       position=position, participation=participation)


def test_subject_control_picks_subject():
    _ranked, chosen = rank_candidates([_p(1, "subject", 0)], licensing="subject_control")
    assert chosen == 0


def test_object_control_picks_object_not_nearer_subject():
    parts = [_p(1, "subject", 0), _p(2, "object", 1)]
    _ranked, chosen = rank_candidates(parts, licensing="object_control")
    assert parts[chosen].role == "object"


def test_subject_control_does_not_grab_nearer_object():
    parts = [_p(1, "subject", 0), _p(2, "object", 1)]
    _ranked, chosen = rank_candidates(parts, licensing="subject_control")
    assert parts[chosen].role == "subject"


def test_unknown_licensing_falls_back_to_locality():
    _ranked, chosen = rank_candidates([_p(1, "other", 0), _p(2, "other", 5)], licensing=None)
    assert chosen == 1                      # most recent wins


def test_no_participants_is_unresolved():
    assert rank_candidates([], "subject_control") == ([], None)
    assert resolve_bind([], "subject_control") == (None, None)


# --- Task 2.2: ContextualBindLayer ----------------------------------------
from Language import ContextualBindLayer


def test_resolves_nearest_left_from_slab():
    layer = ContextualBindLayer()
    alice = torch.tensor([1.,0,0,0]); bind_m = torch.tensor([0.,1,0,0]); run = torch.tensor([0.,0,1,0])
    slab = torch.stack([alice, bind_m, run]).unsqueeze(0)        # [1, 3, 4]
    layer.set_bind_context(slab=slab)
    left, right = slab[:, :-1, :], slab[:, 1:, :]               # pairs as the fold passes them
    out = layer.compose(left, right)                            # [1, 2, 4]
    assert torch.allclose(out[:, 1, :], alice)                  # pair (BIND, run) -> nearest-left Alice


def test_resolves_object_control_from_participants():
    layer = ContextualBindLayer()
    alice = Participant(1, torch.zeros(1,1,4), "subject", 0)
    bob   = Participant(2, torch.ones(1,1,4),  "object",  1)
    layer.set_bind_context(participants=[alice, bob], licensing="object_control")
    out = layer.compose(torch.randn(1,1,4), torch.randn(1,1,4))
    assert torch.allclose(out, bob.vec.expand_as(out), atol=1e-6)


def test_no_context_passes_marker_through():
    layer = ContextualBindLayer()
    m = torch.randn(1,1,4)
    assert torch.allclose(layer.compose(m, torch.randn(1,1,4)), m, atol=1e-6)


def test_clear_bind_context_reverts_to_passthrough():
    layer = ContextualBindLayer()
    layer.set_bind_context(slab=torch.randn(1, 3, 4))
    assert layer._bind_context is not None
    layer.clear_bind_context()                                  # symmetric reset of set_bind_context
    assert layer._bind_context is None
    m = torch.randn(1, 1, 4)                                    # resolution reverts to marker passthrough
    assert torch.allclose(layer.compose(m, torch.randn(1, 1, 4)), m, atol=1e-6)


def test_class_contract():
    assert (ContextualBindLayer.rule_name == "bind" and ContextualBindLayer.arity == 2
            and ContextualBindLayer.tier == "C")


# --- Task 2.4: live-fold stash --------------------------------------------
def test_reduction_layer_stashes_live_slab_on_bind():
    from Language import (BinaryStructuredReductionLayer, ContextualBindLayer,
                          _BinaryGrammarOpAdapter)
    bind = ContextualBindLayer()
    layer = BinaryStructuredReductionLayer(d_model=4, ops=[_BinaryGrammarOpAdapter(bind)])
    alice = torch.tensor([1.,0,0,0]); bind_m = torch.tensor([0.,1,0,0]); run = torch.tensor([0.,0,1,0])
    x = torch.stack([alice, bind_m, run]).unsqueeze(0)         # [1, 3, 4]
    layer.forward(x)                                           # triggers the stash
    assert bind._bind_context is not None and bind._bind_context['slab'] is x
    out = bind.compose(x[:, :-1, :], x[:, 1:, :])
    assert torch.allclose(out[:, 1, :], alice)


if __name__ == "__main__":
    unittest.main()
