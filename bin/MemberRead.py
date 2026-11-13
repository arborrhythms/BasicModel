"""The shared memory read: entry address plus a concept within that idea."""


def read(owner, address, concept=None):
    if concept is None and address[:1] == ('bp',):
        concept = address
    if isinstance(concept, tuple) and concept[:1] == ('bp',):
        from torch.nn.functional import pad
        value = owner.basic_percepts.code(concept[1])
        return pad(value, (0, owner.similarity_codebook.W.shape[-1]-value.numel()))
    if concept is not None:
        alloc = owner._concept_allocator
        if 'word' in alloc.symbol_types.get(concept, ()):
            concept = owner.interpret.referent(concept)
        row = owner._csw_row_of(concept)
        if row is not None:
            return owner.similarity_codebook.lookup_rows([row])[0]
        if concept is not None:
            raise ValueError(f'unavailable memory member {address!r}, concept {concept!r}')
    if address[:1] == ('sym',):
        row = owner._csw_row_of(address[1])
        if row is not None:
            return owner.similarity_codebook.lookup_rows([row])[0]
    stm = getattr(owner, 'stm', None)
    for group in getattr(stm, '_thought_stores', ()):
        for shared in group:
            # Local relation numbers are not native symbol identities.
            # Only an addressed occurrence can select a thought store.
            if isinstance(shared.location(address), int):
                return shared.point_of_row(address[-1])
    strata = getattr(stm, 'strata', None)
    if strata is not None:
        for shared in (*strata.recent_stores, *strata.banks):
            if isinstance(shared.location(address), int):
                return shared.point_of_row(address[-1])
    store = owner._closed_clause_store()
    location = None if store is None else store.location(address)
    if isinstance(location, int):
        value = store.point_of_row(address[-1])
        if value is None:
            raise ValueError(f'unavailable memory member content {address!r}')
        return value
    if isinstance(location, tuple):
        # A relative event has no cached result. Its state is the current
        # union of the ideas/concepts in the retained binary relation.
        from Memory import OCCURRENCE, RELATION
        import torch
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
