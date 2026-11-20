"""Symbols above order zero: no vector, no place; two unsigned rows of a Pi.

A symbol is made when MTM nominates it, and it brings one concept: its Pi
row starts with one member, that concept, at one. That is how it starts and
not a limit: a row may read any number of concepts, and any number of
concepts may read one symbol. A symbol has its own index; it shares none
with a concept. Its value in a reading is the Pi's product of its members'
lanes, one value on each lane (for, against); it holds no code, place or
time.
"""
from OrderZero import SparsePiLayer3


class SymbolTable:
    """The symbols, their index and the Pi that gives each its two lanes.

    ``index`` maps a symbol's handle (what nominated it) to its index;
    ``handles`` and ``brought`` are by index: the handle, and the concept
    the symbol's making brought, recorded then and never inferred. The Pi
    is a :class:`SparsePiLayer3` whose rows are the symbols and whose members are
    the keys ``('concept', cid)``. Room is fixed at construction.
    """

    ENTRIES_PER_ROW = 8

    def __init__(self, capacity):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError('the symbol table needs room for a symbol')
        self.capacity = capacity
        self.index, self.handles, self.brought = {}, [], []
        self.pi = SparsePiLayer3(capacity, capacity * self.ENTRIES_PER_ROW)

    def __len__(self):
        return len(self.handles)

    def __contains__(self, handle):
        return handle in self.index

    def make(self, handle, concept):
        """The symbol of ``handle``, made if it has none: one Pi row over the
        concept it brings, at one. A handle made before keeps its symbol."""
        found = self.index.get(handle)
        if found is not None:
            return found
        if len(self.handles) >= self.capacity:
            raise OverflowError('the symbol table has no room for another symbol')
        symbol = self.pi.append((('concept', int(concept)),))
        if symbol != len(self.handles):
            raise RuntimeError('the symbol Pi and the symbol index disagree')
        self.index[handle] = symbol
        self.handles.append(handle)
        self.brought.append(int(concept))
        return symbol

    def of(self, handle):
        """The index of ``handle``'s symbol, or ``None``."""
        return self.index.get(handle)

    def members(self, symbol):
        """The concepts the symbol's Pi row reads."""
        return [key[1] for key in self.pi.row_members(int(symbol))]

    def reading(self, concept):
        """The symbols whose Pi rows read ``concept``."""
        return self.pi.rows_reading(('concept', int(concept)))

    def read(self, lanes, concepts, symbols=None):
        """The two lanes of ``symbols`` (every symbol by default),
        ``[len(symbols), ..., 2]``, from ``lanes[k]``, the two lanes of
        ``concepts[k]`` (``[len(concepts), ..., 2]``). A symbol whose member
        is not among ``concepts`` reads nothing."""
        symbols = range(len(self.handles)) if symbols is None else symbols
        keys = [('concept', int(concept)) for concept in concepts]
        return self.pi.forward(lanes, list(map(int, symbols)), keys)

    def state(self):
        """A plain record: each symbol's handle, the concept it brought and
        its members. The Pi's rows are written again from it."""
        return dict(version=1, symbols=[(handle, brought, tuple(self.members(i)))
                                        for i, (handle, brought) in enumerate(zip(self.handles, self.brought))])

    def restore(self, state):
        if state.get('version') != 1:
            raise ValueError('unsupported symbol table checkpoint')
        symbols = state.get('symbols', ())
        if len(symbols) > self.capacity:
            raise ValueError('the checkpoint holds more symbols than the constructed room')
        self.index, self.handles, self.brought = {}, [], []
        self.pi.clear()
        for handle, brought, members in symbols:
            handle = tuple(handle) if isinstance(handle, list) else handle
            symbol = self.pi.append(tuple(('concept', int(m)) for m in members))
            self.index[handle] = symbol
            self.handles.append(handle)
            self.brought.append(int(brought))
