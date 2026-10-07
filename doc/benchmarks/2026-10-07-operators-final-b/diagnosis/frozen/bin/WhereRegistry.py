"""A model's fixed address ranges for input locations and stored percepts.

These addresses are coordinates, never keys into a content codebook. The
registry is model-local so constructing another model cannot move a slice.
"""
from types import MappingProxyType

import torch


class WhereRegistry:
    def __init__(self, capacities):
        end, slices = 0, {}
        for name, capacity in capacities:
            capacity = int(capacity)
            if capacity < 0 or name in slices:
                raise ValueError('where ranges require unique names and nonnegative capacities')
            slices[name] = (end, end + capacity)
            end += capacity
        self.slices = MappingProxyType(slices)
        self.capacity = end
        from Spaces import WhereEncoding
        self.encoding = WhereEncoding(nWhere=4, nWhen=4).set_capacity(end)

    def encode(self, name, indices):
        """Located percepts carry these bands; integers are decoded on demand."""
        intervals = self.intervals(name, indices)
        bands = self.encoding.encode(intervals[..., 0])
        return torch.where((indices >= 0)[..., None], bands, 0.)

    def __deepcopy__(self, memo):
        from copy import deepcopy
        copied = type(self)((name, end - start)
                            for name, (start, end) in self.slices.items())
        memo[id(self)] = copied
        copied.encoding = deepcopy(self.encoding, memo)
        return copied

    def intervals(self, name, indices):
        start, end = self.slices[name]
        valid = indices >= 0
        torch._assert_async((~valid | (indices < end - start)).all(),
                            'occurrence exceeds its configured where-space capacity')
        positions = indices.to(torch.int64) + start
        result = torch.stack((positions, positions + 1), -1)
        return torch.where(valid[..., None], result, -1)

def install_where_registry(model):
    parts = model.perceptualSpace.subspace.what
    wholes = []
    seen = set()
    for space in model.wholeSpaces:
        basis = space.subspace.what
        if id(basis) not in seen:
            seen.add(id(basis))
            wholes.append(basis)
    symbols = model.symbolSpace.subspace.what

    def capacity(basis):
        return int(getattr(basis, 'lexicon_capacity', getattr(basis, 'nVectors', 0)))

    from util import TheXMLConfig
    # The lexer retains raw byte coordinates even when its downstream event
    # has fewer token/percept slots. Promotion lets a later slot begin much
    # further into those same bytes. Reserve the input's existing byte bound;
    # neither vocabulary promotion nor a batch may resize this address space.
    input_extent = max(int(model.inputSpace.outputShape[0]),
                       int(getattr(model.inputSpace.data, 'inputLength', 0) or 0))
    if model.word_brackets and not bool(TheXMLConfig.get('architecture.serialObjectMeta', default=False)):
        # A mixing serial InputSpace declares word slots, whereas occurrence
        # coordinates are byte starts. Convert the existing unit/atom bounds
        # to byte addresses; do not compare a byte offset with a word count.
        # The fixed residual-byte layout supplies the per-word bound.
        word_capacity = int(model.perceptualSpace.outputShape[0])
        part_capacity = int(TheXMLConfig.get('architecture.serialResidualPartCapacity', default=16))
        input_extent = max(input_extent, word_capacity * part_capacity)
    registry = WhereRegistry((
        ('input', input_extent),
        ('parts', capacity(parts)),
        ('wholes', sum(capacity(basis) for basis in wholes)),
        ('symbols', 2 * max(capacity(symbols), int(model.conceptualSpaces[0].nVectors))),
    ))
    parts.where_offset = registry.slices['parts'][0]
    offset = registry.slices['wholes'][0]
    for basis in wholes:
        basis.where_offset = offset
        offset += capacity(basis)
    symbols.where_offset = registry.slices['symbols'][0]
    for basis in (parts, *wholes, symbols):
        if callable(getattr(basis, 'freeze_capacity', None)) and not getattr(basis, '_capacity_frozen', False):
            basis.freeze_capacity('perceptual nVectors')
    import weakref
    for space in (model.inputSpace, model.perceptualSpace, *model.wholeSpaces, model.symbolSpace):
        object.__setattr__(space, '_address_model', weakref.ref(model))
    object.__setattr__(model, 'where_registry', registry)
    object.__setattr__(model.symbolSpace, 'where_registry', registry)
    # One owner; compatibility carriers refer to the same ladders even when
    # their content-only layout has no muxed tail. Their nWhere/nWhen stays 0.
    from Spaces import Space, SubSpace, WhenEncoding
    model.where_encoding = registry.encoding
    ltm_capacity = int(TheXMLConfig.space('SymbolSpace', 'ltmCapacity', default=1024) or 1024)
    addresses = getattr(getattr(model.inputSpace, 'data', None), 'source_addresses', {})
    document_bound = max((int(row.get('sentence', 0)) + 2 for rows in addresses.values()
                          for row in rows), default=1)
    model.when_encoding = WhenEncoding(n_when=4).set_capacity(max(ltm_capacity, document_bound))
    store = getattr(model.symbolSpace, 'ltm_store', None)
    if store is not None:
        object.__setattr__(store, '_address_encoding', model.when_encoding)
        reference = weakref.ref(model)
        object.__setattr__(store, '_timestamp_source', lambda: float(reference().when_time))
    for module in tuple(model.modules()):
        if isinstance(module, Space):
            sub = module.subspace
            for name, enc in (('whereEncoding', model.where_encoding),
                              ('whenEncoding', model.when_encoding)):
                module._owned_encoders._modules.pop(name, None)
                object.__setattr__(sub, name, enc)
        if isinstance(module, SubSpace):
            for name, enc in (('whereEncoding', model.where_encoding),
                              ('whenEncoding', model.when_encoding)):
                module._modules.pop(name, None)
                object.__setattr__(module, name, enc)
        if callable(getattr(module, '_when_encoding', None)):
            object.__setattr__(module, '_shared_when_encoding', model.when_encoding)
    # SymbolSubSpace is a nonregistered compatibility carrier.
    for sub in (model.symbolSpace.subspace,):
        object.__setattr__(sub, 'whereEncoding', model.where_encoding)
        object.__setattr__(sub, 'whenEncoding', model.when_encoding)
    object.__setattr__(model.perceptualSpace, 'where_registry', registry)
    for space in model.wholeSpaces:
        object.__setattr__(space, 'where_registry', registry)
        if getattr(space, '_symbol_where', None) is not None:
            rows = torch.arange(space._symbol_where.shape[0])
            space._symbol_where = registry.encoding.encode(rows + space.subspace.what.where_offset)
    return registry
