"""Explicit ownership for isolated tests of the composed gain and basis map."""
from MeaningKernels import MeaningAction, MeaningBasis


def action_with_basis(width, **kwargs):
    action = MeaningAction(width, **kwargs)
    # The model owns this in VerbLayer. A standalone numerical fixture owns
    # both modules so dtype/device moves and checkpoint tests include both.
    del action.object_basis
    action.object_basis = MeaningBasis(width, reflections=kwargs.get('reflections', 4))
    return action
