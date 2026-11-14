"""Numerical memory tests retain their supplied observations in an idea store."""
from types import SimpleNamespace
import torch
from Memory import STM0


def working_symbol_store(width, capacity, *, content_width, episodic_concepts,
                         relation_capacity, percept_width):
    """A time-less embedding fixture has working, not durable, memory."""
    return STM0(width, capacity, percept_width=percept_width,
                relation_capacity=relation_capacity)


def attach(pool, width, namespace, *, capacity=64):
    store = STM0(width, capacity=capacity, percept_width=0, relation_capacity=128)
    owner = SimpleNamespace(mtm=pool, _closed_clause_store=lambda:store,
        _concept_allocator=SimpleNamespace(symbol_types={}), _csw_row_of=lambda concept:None,
        similarity_codebook=SimpleNamespace(W=torch.zeros(1, width), mereology=None))
    pool._owner = lambda:owner
    return owner, store


def put(store, address, value):
    from Meaning import ConceptualMeaning
    store.append_meaning(ConceptualMeaning.from_description(value.detach()), name=address[-1])
    return address
