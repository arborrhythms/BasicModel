"""Promotion records the member path in the existing binary relation store."""
from Memory import OCCURRENCE, SYMBOL, RELATION


def member(store, address):
    """Keep a stored event's relation tag and its own percept bands."""
    index = store.location(address)
    if isinstance(index, int):
        return (OCCURRENCE, address[-1]), None, None
    if isinstance(index, tuple) and index[:1] == ('sym',) and store.parts.contains(index[1]):
        return (RELATION, index[1]), None, None
    raise ValueError(f'promoted path has unavailable member {address!r}')


def membership(owner, address, concept, *, evidence=(1., 0.), when=None, where=None):
    """The common occurrence-membership writer, including null resolution."""
    store, parts = owner._closed_clause_store(), owner._concept_allocator.parts
    reference, member_when, member_where = member(store, address)
    when = member_when if when is None else when
    where = member_where if where is None else where
    return parts.add(reference, (SYMBOL, concept),
        truth_plus=evidence[0], truth_minus=evidence[1], witness=True)


def write(owner, entry, concept):
    store, parts = owner._closed_clause_store(), owner._concept_allocator.parts
    previous = None
    for address in entry.occurrences:
        reference, when, where = member(store, address)
        membership(owner, address, concept, when=when, where=where)
        if previous is not None:
            parts.add(previous, reference, truth_plus=1., witness=True)
        previous = reference
