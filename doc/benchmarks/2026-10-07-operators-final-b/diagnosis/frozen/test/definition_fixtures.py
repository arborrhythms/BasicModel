"""Definition owner for isolated conceptual-inventory mechanism fixtures."""
from Layers import TernaryTruthStore


def with_definitions(cs):
    object.__setattr__(cs, '_definition_store', TernaryTruthStore(cs.nDim, capacity=1024))
    return cs
