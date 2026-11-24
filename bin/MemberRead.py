"""The shared memory read: entry address plus a concept within that idea."""
from contextvars import ContextVar
from functools import wraps

import torch


_percept_reads = ContextVar('member_percept_reads', default=None)


def share_percept_reads(function):
    """Gather each percept once while constructing one read-only menu.

    A menu can contain many collections with the same members. Its scope
    ends before any selected link writes memory or an optimizer changes a
    code. No cache survives into a different trial, close or autograd mode.
    """
    @wraps(function)
    def shared(owner, *args, **kwargs):
        token = _percept_reads.set((owner, {}))
        try:
            return function(owner, *args, **kwargs)
        finally:
            _percept_reads.reset(token)
    return shared


def read(owner, address, concept=None):
    if concept is None and address[:1] == ('bp',):
        concept = address
    if isinstance(concept, tuple) and concept[:1] == ('bp',):
        from torch.nn.functional import pad
        scope = _percept_reads.get()
        values = scope[1] if scope is not None and scope[0] is owner else {}
        key = (concept[1], torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        if key not in values:
            values[key] = owner.basic_percepts.code(concept[1])
        value = values[key]
        return pad(value, (0, owner.similarity_codebook.W.shape[-1]-value.numel()))
    if concept is not None:
        alloc = owner._concept_allocator
        if 'word' in alloc.symbol_types.get(concept, ()):
            # An addressed historical member has no active presentation stream.
            concept = owner.interpret.referent(concept, batch=None)
        row = owner._concept_row_of(concept)
        if row is not None:
            return owner.similarity_codebook.lookup_rows([row])[0]
        if concept is not None:
            raise ValueError(f'unavailable memory member {address!r}, concept {concept!r}')
    if address[:1] == ('sym',):
        row = owner._concept_row_of(address[1])
        if row is not None:
            return owner.similarity_codebook.lookup_rows([row])[0]
    stm = getattr(owner, 'stm', None)
    for group in getattr(stm, '_thought_stores', ()):
        for shared in group:
            # Local relation numbers are not native symbol identities.
            # Only an addressed occurrence can select a thought store.
            if isinstance(shared.location(address), int):
                return shared.point_of_row(address)
    strata = getattr(stm, 'strata', None)
    if strata is not None:
        for shared in strata.recent_stores:
            if isinstance(shared.location(address), int):
                return shared.point_of_row(address)
        point = strata.trace_point(address)
        if point is not None:
            return point
    store = owner._closed_clause_store()
    pool = getattr(owner, 'mtm', None)
    if pool is not None:
        store = pool.idea_store(address, store)
    location = None if store is None else store.location(address)
    if isinstance(location, int):
        value = store.point_of_row(address)
        if value is None:
            raise ValueError(f'unavailable memory member content {address!r}')
        return value
    if isinstance(location, tuple):
        # A relative event has no cached result. Its state is the current
        # union of the ideas/concepts in the retained binary relation.
        from Memory import OCCURRENCE, RELATION
        pending, seen, values = [location[1]], set(), []
        while pending:
            relation = pending.pop()
            if relation in seen:
                continue
            seen.add(relation)
            for kind, member in store.parts.references(relation):
                if kind == RELATION:
                    pending.append(member)
                else:
                    reference = ('ltm', store.address_domain, member) if kind == OCCURRENCE else ('sym', member)
                    values.append(read(owner, reference))
        if values:
            return torch.stack(values).amax(0)
    raise ValueError(f'unavailable memory member {address!r}, concept {concept!r}')
