"""Concepts on store rows, for tests that write the concept store by hand.

A kind reads symbols, each holding a concept. A test that writes an entry onto a bare store row first records a
concept on that row, so that the entry's symbol has a constituent.
"""


def readable(cs, *rows):
    """Each of ``rows`` holds a concept: a row that holds none is recorded
    as holding concept ``1_000_000 + row``, a record of the store alone (no
    allocation), as the hand-written fixtures took the row to be a concept.
    Returns the rows."""
    store = cs._concept_allocator.layer(0)
    for row in rows:
        row = int(row)
        if store._tensor_row_keys.get(row) is None:
            key = ('concept', 1_000_000 + row)
            store._tensor_rows[key] = row
            store._tensor_row_keys[row] = key
    return rows


def symbol_columns(cs, reader, *rows, lane=0):
    """The store's columns by which ``reader`` reads each of ``rows``
    (through :meth:`ConceptualSpace.things_reading`, making a symbol where
    none holds the row's concept)."""
    store = cs._concept_allocator.layer(0)
    readable(cs, *rows)
    return [store.column_of_thing(thing, lane) for row in rows
            for thing in cs.things_reading(reader, row)]


def store_columns(store, *rows, lane=0):
    """On a bare store (no space): each of ``rows`` holds a concept
    (recorded as ``1_000_000 + row`` where it holds none) and a symbol
    holds it, found or made; returns those symbols' columns in lane block
    ``lane``."""
    columns = []
    for row in rows:
        row = int(row)
        key = store._tensor_row_keys.get(row)
        if key is None:
            key = ('concept', 1_000_000 + row)
            store._tensor_rows[key] = row
            store._tensor_row_keys[row] = key
        concept = key[1] if isinstance(key, tuple) else key
        table = store.symbol_table
        symbol = table.fit(int(concept))
        if symbol is None:
            symbol = table.make(('test', int(concept)), int(concept))
        columns.append(store.column_of_thing(('sym', symbol), lane))
    return columns
