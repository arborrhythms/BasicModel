"""Numerical memory tests retain their supplied observations in an idea store."""
from types import SimpleNamespace
import torch
from Memory import LTM0


def attach(pool, width, namespace, *, capacity=64):
    store = LTM0(width, capacity=capacity, percept_width=0, relation_capacity=128)
    owner = SimpleNamespace(mtm=pool, _closed_clause_store=lambda:store,
        _concept_allocator=SimpleNamespace(symbol_types={}), _csw_row_of=lambda concept:None,
        similarity_codebook=SimpleNamespace(W=torch.zeros(1, width), mereology=None))
    pool._owner = lambda:owner
    return owner, store


def put(store, address, value):
    store.append_idea(value.detach(), address=address[-1], time=False)
    return address
