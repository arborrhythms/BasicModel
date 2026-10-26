"""Read only the surviving semantic chains, under the caller's work bound."""
from LTM1 import OCCURRENCE, SYMBOL, RELATION


def envelope(store, reference):
    kind, value = reference
    return ('ltm', store.address_domain, value) if kind == OCCURRENCE else ('sym', value)


def point(store, reference):
    kind, value = reference
    if kind == OCCURRENCE:
        row = store.index_of_row(value)
        return None if row is None else store.slots[row].detach()
    if kind == SYMBOL:
        return store._concept_point(value)
    return None


def head_idea(store, row):
    seen = set()
    while store.parts.contains(row) and row not in seen:
        seen.add(row)
        left, right = store.parts.references(row)
        value = point(store, left)
        if value is not None:
            return value
        if left[0] == RELATION:
            row = left[1]
        elif right[0] == RELATION:
            row = right[1]
        else:
            return point(store, right)
    return None


def components(store, roots, *, max_records, work, scanned=0):
    pending = list(roots)
    visited, result, incomplete = set(), {}, []
    while pending:
        reference = pending.pop()
        if reference in visited:
            continue
        visited.add(reference)
        if reference[0] != 'sym' or not store.parts.contains(reference[1]):
            continue  # an idea has no finer stored structure
        if scanned >= max_records or not work.consume('record'):
            incomplete.append('chain_read_limit'); break
        scanned += 1
        operands = tuple(envelope(store, ref) for ref in store.parts.references(reference[1]))
        result[reference] = dict(kind='chain', operands=operands,
                                meaning=store.part_row(reference[1])['meaning'])
        pending.extend(operands)
    return result, scanned, tuple(incomplete)
