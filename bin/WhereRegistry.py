"""A model's fixed address ranges for input, codebooks and live conceptual slots.

These addresses are coordinates, never keys into a content codebook. The
registry is model-local so constructing another model cannot move a slice.
"""
from functools import cached_property
from types import MappingProxyType

import torch
from torch.nn import functional as F


class WhereRegistry:
    """Named ranges of addresses, each a box of places in the space of the
    ``.where``.

    A range is declared by its name and either a count, a line on the first
    axis, or a shape ``(w, h[, d])``, a box whose addresses run along x,
    then y, then z; a third member, the box's origin, may be given. Its
    origin is otherwise ``(its first address, 0, 0)``, so for a range
    declared by a count every place is ``(address, 0, 0)``. No two boxes
    share a place (the domains of two modalities do not overlap).
    """
    def __init__(self, capacities):
        from architecture import AXIS_WIRES, WHERE_AXES, WHEN_WIRES
        end, slices, boxes = 0, {}, {}
        for entry in capacities:
            name, extent = entry[0], entry[1]
            shape = tuple(int(n) for n in extent) if isinstance(extent, (tuple, list)) else (int(extent),)
            if len(shape) > WHERE_AXES or min(shape, default=0) < 0 or name in slices:
                raise ValueError('where ranges require unique names and nonnegative capacities')
            shape = shape + (1,) * (WHERE_AXES - len(shape))
            capacity = 1
            for n in shape:
                capacity *= n
            origin = (tuple(int(n) for n in entry[2]) if len(entry) > 2
                      else (end,) + (0,) * (WHERE_AXES - 1))
            if len(origin) != WHERE_AXES or min(origin) < 0:
                raise ValueError('a box has a place of whole numbers for its origin')
            slices[name] = (end, end + capacity)
            boxes[name] = (origin, shape)
            end += capacity
        live = [(name, o, sh) for name, (o, sh) in boxes.items() if min(sh) > 0]
        for i, (a, oa, sa) in enumerate(live):
            for b, ob, sb in live[i + 1:]:
                if all(x < y + m and y < x + n for x, n, y, m in zip(oa, sa, ob, sb)):
                    raise ValueError(f'where ranges {a!r} and {b!r} share a place')
        self.slices = MappingProxyType(slices)
        self.boxes = MappingProxyType(boxes)
        self.capacity = end
        from Spaces import WhereEncoding
        self.encoding = WhereEncoding(nWhere=AXIS_WIRES * WHERE_AXES, nWhen=WHEN_WIRES,
                                      axes=WHERE_AXES).set_capacity(end)
        self.axes = tuple((a * AXIS_WIRES, (a + 1) * AXIS_WIRES) for a in range(WHERE_AXES))

    def _box_table(self, device):
        """Each live range's first address, origin and shape, ``[R, 1 + 2 * axes]``."""
        rows = [(lo, *self.boxes[name][0], *self.boxes[name][1])
                for name, (lo, hi) in self.slices.items() if hi > lo]
        return torch.tensor(rows, dtype=torch.long, device=device).reshape(len(rows), -1)

    def place(self, addresses):
        """The place ``[..., axes]`` of each address; an address below zero
        or beyond the registry has none (``-1`` on every axis)."""
        addresses = torch.as_tensor(addresses, dtype=torch.long)
        table = self._box_table(addresses.device)
        A = len(self.axes)
        if not len(table):
            return addresses.new_full((*addresses.shape, A), -1)
        lo, origin, shape = table[:, 0], table[:, 1:1 + A], table[:, 1 + A:]
        size = shape.prod(-1)
        local = addresses[..., None] - lo
        inside = (local >= 0) & (local < size)
        which = inside.long().argmax(-1)
        local = local.gather(-1, which[..., None])[..., 0]
        place, stride = [], torch.ones_like(local)
        for a in range(A):
            n = shape[which, a]
            place.append(origin[which, a] + torch.div(local, stride, rounding_mode='floor') % n)
            stride = stride * n
        place = torch.stack(place, -1)
        return torch.where(inside.any(-1)[..., None], place, -1)

    def address(self, places):
        """The address of each place ``[..., axes]``; a place in no range
        has none (``-1``)."""
        places = torch.as_tensor(places, dtype=torch.long)
        table = self._box_table(places.device)
        A = len(self.axes)
        if not len(table):
            return places.new_full(places.shape[:-1], -1)
        lo, origin, shape = table[:, 0], table[:, 1:1 + A], table[:, 1 + A:]
        local = places[..., None, :] - origin
        inside = ((local >= 0) & (local < shape)).all(-1)
        stride = torch.cumprod(F.pad(shape[:, :-1], (1, 0), value=1), -1)
        address = lo + (local * stride).sum(-1)
        which = inside.long().argmax(-1)
        found = address.gather(-1, which[..., None])[..., 0]
        return torch.where(inside.any(-1), found, -1)

    @cached_property
    def grid(self):
        """The fixed differentiable grid: one row of the ``.where`` wires
        of each address's place, built on its first reader.

        Pure address arithmetic uses only the encoding. A model reading the
        wires shares one gradient leaf outside its optimizer.
        """
        from util import TheDevice
        with torch.inference_mode(False), torch.no_grad():
            addresses = torch.arange(max(1, self.capacity), dtype=torch.long)
            grid = self.encoding.encode_place(self.place(addresses).clamp_min(0)).float()
            return torch.nn.Parameter(grid.to(TheDevice.get()), requires_grad=True)

    def band(self, addresses):
        """The rows of the where wires at registry ``addresses`` (any shape);
        an address below zero makes no location claim and reads nothing."""
        if self.grid.device != addresses.device:
            self.grid.data = self.grid.data.to(addresses.device)
        rows = self.grid.index_select(0, addresses.clamp_min(0).reshape(-1).to(torch.long))
        rows = rows.reshape(*addresses.shape, self.grid.shape[-1])
        return torch.where((addresses >= 0)[..., None], rows, rows.new_zeros(()))

    def encode(self, name, indices):
        """Located percepts carry these bands, read from the where wires;
        integers are decoded on demand by the encoding."""
        return self.band(self.intervals(name, indices)[..., 0])

    def decode(self, wires):
        """The address of each row of ``.where`` wires ``[..., wires]``: each
        axis decoded to its number, the place's address (``-1`` for a place
        in no range)."""
        return self.address(self.encoding.decode_place(wires))

    def __deepcopy__(self, memo):
        from copy import deepcopy
        copied = type(self)((name, self.boxes[name][1], self.boxes[name][0])
                            for name in self.slices)
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

def text_extent(model):
    """How many byte addresses the text slice holds."""
    from util import TheXMLConfig
    # The lexer retains raw byte coordinates even when its downstream event
    # has fewer token/percept slots. Promotion lets a later slot begin much
    # further into those same bytes. Reserve the input's existing byte bound;
    # neither vocabulary promotion nor a batch may resize this address space.
    input_extent = max(int(model.inputSpace.outputShape[0]),
                       int(getattr(model.inputSpace.data, 'inputLength', 0) or 0))
    if model.word_brackets and not bool(TheXMLConfig.get('architecture.serialObjectMeta')):
        # A mixing serial InputSpace declares word slots, whereas occurrence
        # coordinates are byte starts. Convert the existing unit/atom bounds
        # to byte addresses; do not compare a byte offset with a word count.
        # The fixed residual-byte layout supplies the per-word bound.
        word_capacity = int(model.perceptualSpace.outputShape[0])
        part_capacity = int(TheXMLConfig.get('architecture.serialResidualPartCapacity'))
        input_extent = max(input_extent, word_capacity * part_capacity)
    return input_extent


def install_where_registry(model):
    parts = model.perceptualSpace.subspace.what
    whole = model.wholeSpace.subspace.what

    def capacity(basis):
        return int(getattr(basis, 'lexicon_capacity', getattr(basis, 'nVectors', 0)))

    input_extent = text_extent(model)
    # The numeric input is a box of its own shape: h rows of w numbers lie
    # with x the column and y the row, its addresses row by row; n places
    # of one number each lie on the first axis.
    numeric_extent = 0
    if model.inputSpace.model_type == 'numeric':
        rows, *columns = (int(n) for n in model.inputSpace.inputShape)
        width = columns[0] if columns else 1
        numeric_extent = (width, rows) if width > 1 else rows
    registry = WhereRegistry((
        ('text', input_extent),
        ('numeric', numeric_extent),
        ('scalar', 1),
        ('parts', capacity(parts)),
        ('wholes', capacity(whole)),
        # The current conceptual eight-space has locations of its own.
        # The C stages use the same slot geometry, independently of which
        # codebook rows or composed ideas currently occupy those slots.
        ('conceptual', max(int(space.stm.capacity) for space in model.conceptualSpaces)),
    ))
    parts.where_offset = registry.slices['parts'][0]
    whole.where_offset = registry.slices['wholes'][0]
    # A symbol has no dimensions and no address: it has no slice here.
    for basis in (parts, whole):
        if callable(getattr(basis, 'freeze_capacity', None)) and not getattr(basis, '_capacity_frozen', False):
            basis.freeze_capacity('perceptual nVectors')
    import weakref
    for space in (model.inputSpace, model.perceptualSpace, model.wholeSpace, model.symbolSpace):
        object.__setattr__(space, '_address_model', weakref.ref(model))
    object.__setattr__(model, 'where_registry', registry)
    object.__setattr__(model.symbolSpace, 'where_registry', registry)
    for space in model.conceptualSpaces:
        object.__setattr__(space, 'where_registry', registry)
    # One owner; compatibility carriers refer to the same ladders even when
    # their content-only layout has no muxed tail. Their nWhere/nWhen stays 0.
    from Spaces import Space, SubSpace
    model.where_encoding = registry.encoding
    # The one encoder of model time; every space's when band is its code.
    from RowTime import encoder
    model.when_encoding = encoder()
    model._sentence_ticks = {}
    model._open_sentence_slot = None
    model._episode_counts = dict(written=0, anchors=0, relation_only_closings=0)
    for space in model.conceptualSpaces:
        object.__setattr__(space.stm, '_time_owner', weakref.ref(model))
    store = getattr(model.symbolSpace, 'ltm_store', None)
    if store is not None:
        object.__setattr__(store, '_address_encoding', model.when_encoding)
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
    object.__setattr__(model.wholeSpace, 'where_registry', registry)
    return registry
