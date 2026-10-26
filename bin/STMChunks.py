"""The shared, fixed-capacity working set carried by the sentence fork.

Only the buffer holds working values. Stack position is metadata over its
physical chunks; journals retain grammar provenance after a chunk is freed.
All updates are functional, including reads, so a discarded trial changes
neither the retained working set nor the MTM pool.
"""
import torch


BUFFER, DEPTH, ORDERS, GRAMMAR, ROWS, ACTIVATIONS = range(6)
REFS, IDENTITY, TIMELINE, STACK, READ, WHERE, WHEN = range(6, 13)
SYMBOL, IDENTITY_REF, ACTION, RELATION, OCCURRENCE = range(1, 6)
FIELDS = ('buffer', 'depth', 'orders', 'grammar_orders', 'concept_rows',
          'concept_activations', 'chunk_refs', 'chunk_identity', 'chunk_timeline',
          'chunk_stack', 'chunk_read', 'chunk_where', 'chunk_when')


def empty(batch, capacity, width, *, device=None, dtype=torch.float32):
    """Construction only: B streams × stmCapacity chunks × conceptual width.

    Each chunk has two tagged references and two four-coordinate percept
    bands. Neither a word count nor a journal length can enlarge this set.
    """
    shape = (int(batch), int(capacity))
    if min(shape) < 1 or int(width) < 0:
        raise ValueError('STM construction requires positive streams and capacity')
    def longs(extra=(), fill=0):
        return torch.full((*shape, *extra), fill, dtype=torch.long, device=device)
    def floats(extra=()):
        return torch.zeros((*shape, *extra), dtype=dtype, device=device)
    return (floats((int(width),)), torch.zeros(shape[0], dtype=torch.long, device=device),
            longs(fill=-1), longs(fill=-1), longs(fill=-1), floats(),
            longs((2, 2)), longs(fill=-1), longs(), longs(fill=-1), longs(),
            floats((4,)), floats((4,)))


def gather(value, indices):
    index = indices.reshape(*indices.shape, *([1] * (value.ndim-2)))
    return value.gather(1, index.expand(*indices.shape, *value.shape[2:]))


def at(state, position=0):
    """Physical index for a newest-first stack position; -1 when absent."""
    position = torch.as_tensor(position, device=state[STACK].device)
    if position.ndim == 0:
        position = position.expand(state[STACK].shape[0])
    match = state[STACK] == position[:, None]
    return torch.where(match.any(-1), match.long().argmax(-1), -1)


def stack_indices(state):
    positions = torch.arange(state[STACK].shape[1], device=state[STACK].device)
    match = state[STACK][:, None, :] == positions[None, :, None]
    return torch.where(match.any(-1), match.long().argmax(-1), -1)


def stack_view(state):
    """A read-only positional view for grammar and legacy stack consumers."""
    indices = stack_indices(state)
    result = []
    for field, value in enumerate(state[:6]):
        if field == DEPTH:
            result.append(value)
            continue
        member = gather(value, indices.clamp_min(0))
        mask = (indices >= 0).reshape(*indices.shape, *([1]*(value.ndim-2)))
        result.append(torch.where(mask, member, -1 if field in (ORDERS, GRAMMAR, ROWS) else 0))
    return tuple(result)


def write(value, indices, member, gate):
    positions = torch.arange(value.shape[1], device=value.device)[None]
    mask = gate.reshape(-1, 1) & (indices[:, None] >= 0) & (positions == indices[:, None])
    return torch.where(mask.reshape(*mask.shape, *([1]*(value.ndim-2))), member[:, None], value)


def clear(state, mask):
    result = list(state)
    for field, value in enumerate(state):
        if field == DEPTH:
            continue
        selected = mask.reshape(*mask.shape, *([1]*(value.ndim-2)))
        result[field] = torch.where(selected, -1 if field in (ORDERS, GRAMMAR, ROWS, IDENTITY, STACK) else 0, value)
    result[DEPTH] = (result[STACK] >= 0).sum(-1)
    return tuple(result)


def allocate_chunk(state, gate):
    """Return (cleared state, index); all-stack streams refuse with -1."""
    off = state[STACK] < 0
    free = off & (state[READ] == 0)
    oldest = state[READ].masked_fill(~off, torch.iinfo(torch.long).max).argmin(-1)
    index = torch.where(free.any(-1), free.long().argmax(-1), oldest)
    valid = gate.reshape(-1).bool() & off.any(-1)
    index = torch.where(valid, index, -1)
    mask = torch.arange(off.shape[1], device=off.device)[None] == index[:, None]
    return clear(state, mask), index


def read(state, indices, gate=None, *, tick=None):
    """Refresh only actual reads; enumerating an offered bind calls no read."""
    if indices.ndim == 1:
        indices = indices[:, None]
    if gate is None:
        gate = torch.ones_like(indices, dtype=torch.bool)
    elif gate.ndim == 1:
        gate = gate[:, None]
    tick = state[READ].amax(-1)+1 if tick is None else torch.as_tensor(tick, device=indices.device).expand(indices.shape[0])
    positions = torch.arange(state[STACK].shape[1], device=indices.device)[None, :, None]
    touched = ((positions == indices[:, None]) & gate[:, None] & (indices[:, None] >= 0)).any(-1)
    result = list(state)
    result[READ] = torch.where(touched, tick[:, None], state[READ])
    return tuple(result)


def deposit(state, value, gate, *, order=None, grammar_order=None, row=None,
            activation=None, refs=None, identity=None, timeline=None,
            where=None, when=None, stack=False, tick=None):
    """Allocate one arrival; remembered and prepended entries stay off stack."""
    before = state
    state, index = allocate_chunk(state, gate)
    active = index >= 0
    result = list(state)
    b = value.shape[0]
    defaults = {BUFFER:value, ORDERS:order, GRAMMAR:grammar_order, ROWS:row,
                ACTIVATIONS:activation, REFS:refs, IDENTITY:identity,
                TIMELINE:timeline, WHERE:where, WHEN:when}
    for field, member in defaults.items():
        if member is None:
            continue
        member = torch.as_tensor(member, device=state[field].device, dtype=state[field].dtype)
        member = member.expand(b, *state[field].shape[2:])
        result[field] = write(state[field], index, member, active)
    if stack:
        result[STACK] = torch.where(active[:, None] & (state[STACK] >= 0), state[STACK]+1, state[STACK])
        result[STACK] = write(result[STACK], index, torch.zeros_like(index), active)
        result[DEPTH] = before[DEPTH] + active.long()
    next_tick = before[READ].amax(-1)+1 if tick is None else tick
    return read(tuple(result), index, active, tick=next_tick), index


def resolve_top_reference(state, value, row, order, activation, gate):
    index = at(state)
    gate = gate.reshape(-1).bool() & (index >= 0) & (row.reshape(-1) >= 0)
    result = list(state)
    for field, member in ((BUFFER,value), (ROWS,row), (ORDERS,order), (ACTIVATIONS,activation)):
        result[field] = write(state[field], index, member.reshape(value.shape[0], *state[field].shape[2:]), gate)
    return tuple(result)


def soft_reset_rows(state, gate):
    """Drop only unfinished sentence operands; keep recalled/recent chunks."""
    return clear(state, gate.reshape(-1, 1).bool() & (state[STACK] >= 0))


def keep_close(state, gate, *, refs, value=None, when=None):
    """Retain the kept top as an off-stack recent entry, freeing its peers."""
    top = at(state)
    active = gate.reshape(-1).bool() & (top >= 0)
    positions = torch.arange(state[STACK].shape[1], device=top.device)[None]
    result = list(clear(state, active[:, None] & (state[STACK] >= 0) & (positions != top[:, None])))
    for field, member in ((STACK,torch.full_like(top,-1)), (REFS,refs), (BUFFER,value), (WHEN,when)):
        if member is not None:
            result[field] = write(result[field], top, member, active)
    result[DEPTH] = (result[STACK] >= 0).sum(-1)
    return read(tuple(result), top, active)


def refresh(state, indices, gate=None):
    """Read the current value of a bound off-stack identity/timeline."""
    identities = gather(state[IDENTITY], indices.clamp_min(0))
    timelines = gather(state[TIMELINE], indices.clamp_min(0))
    match = ((identities[..., None] == state[IDENTITY][:, None])
             & (timelines[..., None] == state[TIMELINE][:, None])
             & (state[STACK][:, None] < 0) & (identities[..., None] >= 0))
    field = match & ((state[REFS][:,None,:,0,0] == SYMBOL)
                     | (state[REFS][:,None,:,0,0] == IDENTITY_REF))
    match = torch.where(field.any(-1)[...,None],field,match)
    sources = match.long().argmax(-1)
    available = match.any(-1) & (indices >= 0)
    if gate is not None:
        available &= gate
    values = torch.where(available[..., None], gather(state[BUFFER], sources),
                         gather(state[BUFFER], indices.clamp_min(0)))
    state = read(state, torch.where(available, sources, -1), available)
    return read(state, indices, gate), values


def bind_menu(state, mention, *, exclusions=None):
    """Fixed K+1 field variants: dummy first, then eligible off-stack chunks."""
    batch, positions, width = mention.shape
    own = mention[:, :, None]
    candidates = state[BUFFER][:, None].expand(batch, positions, -1, width)
    available = ((state[STACK] < 0) & (state[IDENTITY] >= 0))[:, None].expand(batch, positions, -1)
    if exclusions is not None:
        x, y = exclusions[:, 0], exclusions[:, 1]
        ax, ay = own.index_select(-1, x) >= .3, own.index_select(-1, y) >= .3
        bx, by = candidates.index_select(-1, x) >= .3, candidates.index_select(-1, y) >= .3
        available = available & ~((ax & ~bx & by & ~ay) | (ay & ~by & bx & ~ax)).any(-1)
    dummy = torch.zeros(batch, positions, 1, dtype=torch.long, device=mention.device)
    chunks = torch.arange(state[STACK].shape[1], device=mention.device)[None, None].expand(batch, positions, -1)
    return (torch.cat((torch.zeros_like(own), candidates), 2),
            torch.cat((dummy, state[IDENTITY][:, None].expand_as(chunks)), 2),
            torch.cat((dummy-1, chunks), 2),
            torch.cat((torch.ones_like(dummy,dtype=torch.bool), available), 2))


def apply_choice(state, choice, *, operand_actions=None):
    """Write the grammatical head's physical chunk and free only a consumed operand."""
    active = choice.applied.bool()
    bind = (torch.zeros_like(active) if choice.stack_arity is None else
            (choice.kind == 1) & (choice.stack_arity == 1))
    binary = active & (choice.kind == 1) & ~bind
    unary = active & ((choice.kind == 2) | bind)
    newest, older = at(state), at(state, 1)
    chosen = at(state, choice.position)
    head = torch.ones_like(newest) if choice.head_role is None else choice.head_role
    target = torch.where(binary, torch.where(head == 2, newest, older),
                         torch.where(bind, newest, chosen))
    other = torch.where(head == 2, older, newest)
    valid = (binary | unary) & (target >= 0)
    positions = torch.arange(state[STACK].shape[1], device=newest.device)[None]
    state = read(state, torch.stack((newest, torch.where(binary, older, chosen)), -1),
                 torch.stack((active, binary | unary), -1))
    result = list(clear(state, binary[:, None] & (positions == other[:, None])))
    result[STACK] = torch.where(binary[:, None] & (result[STACK] >= 2), result[STACK]-1, result[STACK])
    result[STACK] = write(result[STACK], target, torch.zeros_like(target), binary)
    result[DEPTH] = state[DEPTH]-binary.long()
    result[BUFFER] = write(result[BUFFER], target, choice.candidate, valid)
    previous = gather(state[GRAMMAR], target.clamp_min(0)[:, None])[:, 0]
    pair = torch.stack((gather(state[GRAMMAR], newest.clamp_min(0)[:, None])[:, 0],
                        gather(state[GRAMMAR], older.clamp_min(0)[:, None])[:, 0]), -1)
    grammar = torch.where(binary, torch.where((pair >= 0).all(-1), pair.amax(-1)+1, -1),
                          torch.where(previous >= 0, previous+1, -1))
    for field, member in ((GRAMMAR,grammar), (ORDERS,choice.order),
                           (ROWS,torch.full_like(newest,-1)), (ACTIVATIONS,choice.candidate.new_zeros(len(newest)))):
        if member is not None:
            result[field] = write(result[field], target, member, valid)
    # SYMBOL/IDENTITY and reference-free compressed values are resolved.
    # ACTION records remain only until both operands can be compressed.
    unresolved = (state[REFS][..., 0] == ACTION).any(-1)
    unresolved = gather(unresolved, torch.stack((older, newest), -1).clamp_min(0)).any(-1)
    references = state[REFS].new_zeros(len(newest), 2, 2)
    if operand_actions is not None:
        references = torch.stack((torch.full_like(operand_actions,ACTION), operand_actions), -1)
        references = torch.where(unresolved[:, None, None], references, 0)
    result[REFS] = write(result[REFS], target, references, binary)
    if choice.identity is not None:
        result[IDENTITY] = write(result[IDENTITY], target, choice.identity, bind & valid)
    if choice.chunk is not None:
        selected = choice.chunk.clamp_min(0)
        timeline = gather(state[TIMELINE], selected[:, None])[:, 0]
        result[TIMELINE] = write(result[TIMELINE], target, timeline, bind & valid & (choice.chunk >= 0))
        result = list(read(tuple(result), choice.chunk, bind & valid & (choice.chunk >= 0)))
    # Transpose exchanges stack positions, never the chunks' values or ages.
    transpose = active & (choice.kind == 3)
    result[STACK] = torch.where(transpose[:, None] & (state[STACK] >= 0) & (state[STACK] < 2),
                                1-state[STACK], result[STACK])
    return tuple(result)
