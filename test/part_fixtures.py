"""Native concept seats and Part edges for isolated storage certificates."""
from types import SimpleNamespace

import torch

from ClauseRow import attach_clause_index
from Layers import LTM0
from Spaces import ConceptualSpace
from test_basicmodel import _populate_test_config


def native_store(width=4, capacity=32, points=()):
    _populate_test_config(inputDim=width, perceptDim=width, conceptDim=width,
        symbolDim=width, wordDim=width, outputDim=width, nInput=4, nPercepts=4,
        nConcepts=512, nSymbols=512, nWords=512, nOutput=512, nWhere=0, nWhen=0)
    owner = ConceptualSpace([4, width], [512, width], [512, width])
    with torch.no_grad():
        owner.similarity_codebook.W.sigmoid_()
    store = LTM0(width, capacity=capacity)
    object.__setattr__(store, '_fixture_owner', owner)
    attach_clause_index(SimpleNamespace(), store, owner)
    symbols = tuple(symbol(store, point) for point in points)
    return store, symbols


def symbol(store, point, *, types=()):
    owner = store._concept_owner()
    identity = owner.new_concept()
    row = owner._csw_concept_row(0, identity)
    with torch.no_grad():
        owner.similarity_codebook.W[row].copy_(point)
    owner._concept_allocator.symbol_types[identity] = {'concept', *types}
    return identity


def part(store, left, right, *, evidence=(1., 0.), types=()):
    """Supply native arguments explicitly; their identity is never inferred."""
    a, b = (symbol(store, point, types=types) for point in (left, right))
    return store.append_relation(a, b, truth_plus=evidence[0], truth_minus=evidence[1])
