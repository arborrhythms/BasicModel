"""Symbols above order zero: no vector, no place; two unsigned rows of a Pi.

A symbol is made when MTM nominates it, or where a kind needs a member that
no symbol holds, and it brings one concept: its Pi
row starts with one member, that concept, at one. That is how it starts and
not a limit: a row may read any number of concepts, and any number of
concepts may read one symbol. A symbol has its own index; it shares none
with a concept. Its value in a reading is the Pi's product of its members'
lanes, one value on each lane (for, against); it holds no code, place or
time. A symbol's index is a column of the concept store
(``ConceptualAttentionLayer``): a kind above order zero reads symbols, and a
symbol is read by its row's fold, the product, as a basic percept is over
its parts. A symbol's constituents are concepts of one order, and its row
is a cone over coordinates of that order's conceptual space
(:mod:`SymbolCone`): a concept falls in it where its code holds what the
constituents share and not what their nearest neighbours add.
"""
from Layers import SparsePiLDU


class SymbolTable:
    """The symbols, their index and the Pi that gives each its two lanes.

    ``index`` maps a symbol's handle (what nominated it) to its index;
    ``handles`` and ``brought`` are by index: the handle, and the concept
    the symbol's making brought, recorded then and never inferred. The Pi
    is a :class:`SparsePiLDU` (rule 78) whose outputs are the symbols and
    whose inputs are the concepts they read, each by a member id; the
    concept a symbol brought is its slot's diagonal, so the Pi has a way
    back (:meth:`back`). Room is fixed at construction.
    """

    ENTRIES_PER_ROW = 8

    def __init__(self, capacity):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError('the symbol table needs room for a symbol')
        self.capacity = capacity
        self.index, self.handles, self.brought = {}, [], []
        self.revision = 0
        room = capacity * self.ENTRIES_PER_ROW
        self.pi = SparsePiLDU(room, capacity, room)
        self.member_of, self.concepts = {}, []

    def __len__(self):
        return len(self.handles)

    def __contains__(self, handle):
        return handle in self.index

    #: The nearest neighbours whose added coordinates are a cone's against side.
    NEIGHBOURS = 3

    def make(self, handle, concept, *others, owner=None):
        """The symbol of ``handle``: one that already fits (:meth:`fit`:
        every one of ``concept`` and ``others`` is among its constituents),
        or else one made: a Pi row over them at one, which brings
        ``concept``. A handle made before keeps its symbol; a handle whose
        concepts a symbol already holds names that symbol."""
        found = self.index.get(handle)
        if found is None:
            found = self.fit(concept, *others, owner=owner)
        if found is not None:
            self.index.setdefault(handle, found)
            return found
        if len(self.handles) >= self.capacity:
            raise OverflowError('the symbol table has no room for another symbol')
        concepts = tuple(dict.fromkeys(int(c) for c in (concept, *others)))
        symbol = self._append(concepts)
        self.index[handle] = symbol
        self.handles.append(handle)
        self.brought.append(int(concept))
        self.revision += 1
        return symbol

    def fit(self, concept, *others, owner=None):
        """The fit of a symbol: one whose cone holds ``concept`` (and each of
        ``others``): a constituent is held; with ``owner``, so is a concept
        of the constituents' order whose code falls in the cone (read one
        code at a time). Of those, the one with the fewest constituents (the
        first made among equals), or ``None``."""
        concepts = (int(concept), *map(int, others))
        found = set(self.reading(concepts[0]))
        for other in concepts[1:]:
            found &= set(self.reading(other))
        if owner is not None:
            for symbol in range(len(self.handles)):
                if symbol not in found and all(self.holds(owner, symbol, c) for c in concepts):
                    found.add(symbol)
        if not found:
            return None
        return min(found, key=lambda symbol: (len(self.members(symbol)), symbol))

    def cone(self, owner, symbol):
        """The cone of ``symbol``, ``(for, against)`` over coordinates of its
        constituents' order (:mod:`SymbolCone`)."""
        from SymbolCone import for_side, against_side, codes_of, neighbours
        members = self.members(symbol)
        codes = codes_of(owner, members)
        weights = for_side(codes)
        return weights, against_side(weights, neighbours(owner, members, weights, self.NEIGHBOURS))

    def holds(self, owner, symbol, concept):
        """Whether ``concept`` falls in ``symbol``'s cone: a constituent, or
        a concept of the constituents' order whose code it holds."""
        members = self.members(symbol)
        if int(concept) in members:
            return True
        from SymbolCone import holds, codes_of, same_order
        if not members or not same_order(owner, members[0], concept):
            return False
        weights, against = self.cone(owner, symbol)
        return bool(holds(weights, against, codes_of(owner, [concept]))[0])

    def of(self, handle):
        """The index of ``handle``'s symbol, or ``None``."""
        return self.index.get(handle)

    def _member(self, concept):
        """The Pi's input of ``concept``, made on first use."""
        found = self.member_of.get(int(concept))
        if found is None:
            if len(self.concepts) >= self.pi.nInput:
                raise OverflowError('the symbol Pi has no room for another member')
            found = self.member_of[int(concept)] = len(self.concepts)
            self.concepts.append(int(concept))
        return found

    def _append(self, concepts):
        """The next symbol, over ``concepts`` at one; the first is the one
        it brings, its slot's diagonal where that concept has none yet."""
        symbol = len(self.handles)
        for concept in concepts:
            self.pi.add(symbol, self._member(concept), 1.)
        return symbol

    def members(self, symbol):
        """The concepts the symbol's Pi row reads."""
        return [self.concepts[i] for i, _, _ in self.pi.entries(int(symbol))]

    def reading(self, concept):
        """The symbols whose Pi rows read ``concept``."""
        member = self.member_of.get(int(concept))
        return [] if member is None else self.pi.rows_reading(member)

    def _inputs(self, concepts):
        return [self.member_of.get(int(concept), -1) for concept in concepts]

    def read(self, lanes, concepts, symbols=None):
        """The two lanes of ``symbols`` (every symbol by default),
        ``[len(symbols), ..., 2]``, from ``lanes[k]``, the two lanes of
        ``concepts[k]`` (``[len(concepts), ..., 2]``). A symbol whose member
        is not among ``concepts`` reads nothing."""
        symbols = range(len(self.handles)) if symbols is None else symbols
        return self.pi(lanes, self._inputs(concepts), list(map(int, symbols)))

    def back(self, lanes, concepts):
        """The way back (rule 78): the two lanes of ``concepts``
        ``[len(concepts), ..., 2]`` from every symbol's, ``lanes``
        ``[len(self), ..., 2]``; exact for a concept its symbol brought and
        no symbol whose output is not given reaches. A concept no symbol
        reads comes back at one, the empty product's."""
        full = lanes.new_ones(self.pi.nOutput, *lanes.shape[1:])
        full[:len(lanes)] = lanes
        return self.pi.reverse(full, self._inputs(concepts))

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
        self.member_of, self.concepts = {}, []
        for handle, brought, members in symbols:
            handle = tuple(handle) if isinstance(handle, list) else handle
            symbol = self._append(tuple(dict.fromkeys(int(m) for m in members)))
            self.index[handle] = symbol
            self.handles.append(handle)
            self.brought.append(int(brought))
        self.revision += 1
