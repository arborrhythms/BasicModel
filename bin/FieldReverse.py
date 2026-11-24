"""Order-zero evidence to native activity, without a position-by-vocabulary slab."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NativeActivity:
    """Sufficient statistics of sparse native activity for both readouts."""
    rows: torch.Tensor
    strength: torch.Tensor
    percepts: torch.Tensor
    separators: torch.Tensor | None = None
    # The pasted amount of each byte at each place ``[B, position, 256]``,
    # graded: what the reconstruction's two costs read (:func:`cost`).
    amounts: torch.Tensor | None = None


class NativeDecode(nn.Module):
    """Fixed native spellings and position addresses, filled on admission."""
    def __init__(self, capacity, positions, chunk_width):
        super().__init__()
        if min(capacity, positions, chunk_width) <= 0:
            raise ValueError('native decode requires positive construction bounds')
        self.register_buffer('spellings', torch.zeros(capacity, chunk_width, dtype=torch.uint8), persistent=False)
        self.register_buffer('lengths', torch.zeros(capacity, dtype=torch.long), persistent=False)
        self.register_buffer('positions', torch.arange(positions + 1), persistent=False)
        self._table = None
        self._admitted = 0

    @torch.no_grad()
    def sync(self, table):
        if self._table is not table:
            self._table, self._admitted = table, 0
            self.lengths.zero_()
            self.spellings.zero_()
        for row in range(self._admitted, len(table)):
            raw = table[row]
            if row >= len(self.lengths) or len(raw) > self.spellings.shape[1]:
                raise OverflowError('native spelling exceeds decoder construction capacity')
            self.spellings[row, :len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=self.spellings.device)
            self.lengths[row] = len(raw)
        self._admitted = len(table)

    @torch.no_grad()
    def forward(self, rows, strength):
        B, W = rows.shape
        if W >= len(self.positions):
            raise OverflowError('native activity exceeds decoder position capacity')
        if not bool(torch.isfinite(strength).all()):
            raise RuntimeError('radix activity contains NaN/Inf')
        pos = self.positions[:W + 1].expand(B, -1)
        length = torch.where(strength > 0, self.lengths[rows], 0)
        # The old greedy walk follows p -> p + max(1, spelling_length).
        # Pointer doubling finds the positions reachable from zero in log W
        # tensor steps, with no host loop over batches, bytes or native rows.
        jump = torch.cat(((pos[:, :W] + length.clamp_min(1)).clamp_max(W), pos[:, W:]), 1)
        origin = pos.clone()
        for _ in range(max(1, math.ceil(math.log2(W + 1)))):
            origin = origin.scatter_reduce(1, jump, origin, reduce='amin', include_self=True)
            jump = jump.gather(1, jump)
        starts = torch.where(origin[:, :W] == 0, pos[:, :W], 0).cummax(1).values
        offset = pos[:, :W] - starts
        selected = rows.gather(1, starts)
        valid = offset < length.gather(1, starts)
        return torch.where(valid, self.spellings[selected, offset.clamp_max(self.spellings.shape[1] - 1)], 0).long()


def assemble_parts(parts, starts, width):
    """Extend each assembly with its most-overlapping compatible next occurrence.

    Parts are ordered witnesses [rows, occurrences, bytes], padded with -1.
    Starts are relative to the predicted percept. A gap or a conflict can
    leave an incomplete spelling. Empty bytes are never filled from the input.
    """
    count, occurrences, length = parts.shape
    assembled = parts.new_full((count, width), -1)
    if not occurrences or not width or not length:
        return assembled
    offset = torch.arange(length, device=parts.device)
    position = starts[..., None] + offset
    valid = parts >= 0
    stop = starts + valid.sum(-1)
    frontier = starts.new_zeros(count)
    place = torch.arange(width, device=parts.device)
    batch = torch.arange(count, device=parts.device)
    for _ in range(occurrences):
        old = assembled[batch[:, None, None], position.clamp(0, width - 1)]
        shared = valid & (position < frontier[:, None, None])
        agrees = (~shared | (old == parts)).all(-1)
        grows = (starts >= 0) & (starts <= frontier[:, None]) & (stop > frontier[:, None]) & (stop <= width)
        eligible = grows & agrees
        overlap = torch.where(eligible, frontier[:, None] - starts, -1)
        next_part = overlap.argmax(1)
        end = torch.where(eligible.any(1), stop[batch, next_part], frontier)
        inside = (place >= frontier[:, None]) & (place < end[:, None])
        at = (place - starts[batch, next_part, None]).clamp(0, length - 1)
        assembled = torch.where(inside, parts[batch, next_part].gather(1, at), assembled)
        frontier = end
    return assembled


def concatenate_parts(parts, width):
    """Decode ordered native parts, preserving repeated elements and byte zero."""
    valid = parts >= 0
    lengths = valid.long().sum(-1)
    starts = lengths.cumsum(1) - lengths
    positions = starts[..., None] + torch.arange(parts.shape[-1], device=parts.device)
    target = torch.where(valid, positions.clamp_max(width), width).flatten(1)
    result = parts.new_full((parts.shape[0], width + 1), -1)
    return result.scatter_reduce(1, target, parts.flatten(1), 'amax')[:, :width]


def _spelling_definition(basic, row):
    root = basic.form_root(row)
    if root in getattr(basic.native, 'part_groups', {}):
        children, end = [], 0
        for leaf in basic._leaves()(root):
            element = basic._part_text(leaf)
            if element is None:
                raise ValueError('a spelling element has no character translation')
            children.append((element, end))
            end += len(element)
        return end, None, children
    raw = basic.form_of(row)
    if raw is not None:
        identity = getattr(basic.native, 'identity', None)
        children = () if identity is None else identity.nested.get(raw, ())
        if not children:
            return len(raw), raw, None
        return len(raw), None, tuple((part, lo) for part, lo, hi in children
                                     if part != raw and 0 <= lo < hi <= len(raw) and hi - lo == len(part))
    run = basic.runs.get(int(row))
    if run is None:
        texts = [basic._part_text(part) for part in basic.occurrences[int(row)][0]]
        run = None if not texts or any(text is None for text in texts) else b''.join(texts)
    return 0 if run is None else len(run), run, None


def _assemble_definitions(definitions, width, device):
    occurrences = max(map(len, definitions))
    length = max((len(part) for group in definitions for part, _ in group), default=0)
    parts = torch.full((len(definitions), occurrences, length), -1, dtype=torch.long, device='cpu')
    starts = torch.full((len(definitions), occurrences), -1, dtype=torch.long, device='cpu')
    contiguous = True
    for row, group in enumerate(definitions):
        end = 0
        for i, (part, start) in enumerate(group):
            contiguous &= start == end and start + len(part) <= width
            end = start + len(part)
            parts[row, i, :len(part)] = torch.tensor(list(part), device='cpu')
            starts[row, i] = start
    parts = parts.to(device)
    if contiguous:
        return concatenate_parts(parts, width)
    # Old checkpoints may carry overlapping, explicitly located witnesses.
    return assemble_parts(parts, starts.to(device), width)


def spelling(basic, row):
    """Host inspection of the bytes reconstructed from a basic percept's definition."""
    width, raw, children = _spelling_definition(basic, row)
    if children is None:
        return raw
    result = _assemble_definitions([children], width, 'cpu')[0]
    return bytes(result[result >= 0].tolist())


def codebook(basic):
    """Each predicted percept's byte projection, shared until its definition changes."""
    identity = getattr(basic.native, 'identity', None)
    key = (basic.revision, len(basic.index), None if identity is None else identity.revision,
           basic.value_codes.device)
    held = basic.__dict__.get('_reconstruction_codebook')
    if held is not None and held[0] == key:
        return held[1]
    rows, definitions = [], []
    for row in range(len(basic.index)):
        if row in basic.place_rows or row in basic.conjunctions or basic.is_value(row):
            continue
        width, raw, children = _spelling_definition(basic, row)
        if width:
            rows.append(row)
            definitions.append((width, raw, children))
    width = max((length for length, _, _ in definitions), default=1)
    table = torch.full((len(rows), width), -1, dtype=torch.long, device='cpu')
    nested, seats = [], []
    for n, (_, raw, children) in enumerate(definitions):
        if children is not None:
            seats.append(n)
            nested.append(children)
        else:
            table[n, :len(raw)] = torch.tensor(list(raw), device='cpu')
    device = basic.value_codes.device
    table = table.to(device)
    if nested:
        table.index_copy_(0, torch.tensor(seats, device=device), _assemble_definitions(nested, width, device))
    basic.__dict__['_reconstruction_codebook'] = (key, (rows, table))
    return rows, table


def byte_rows(radix):
    """Each byte's row of the percept store ``[256]`` (``-1`` where none)."""
    if hasattr(radix, 'byte_rows'):
        return radix.byte_rows()
    return torch.tensor([-1 if (row := radix.get_id(bytes([v]))) is None else row for v in range(256)])


#: The window hash of :func:`forms_in`: a polynomial in the bytes, modulo a
#: number small enough that ``h * base`` stays inside an int64.
HASH_BASE, HASH_MODULUS = 257, (1 << 53) - 111


def store_forms(radix, device=None):
    """The percept store's forms of two bytes or more: their rows ``[F]``,
    spellings ``[F, L]`` (``-1`` past a form's length), and for each length
    the forms of that length ``(length, window hashes sorted, their
    forms)``. Kept until a form is added."""
    table = radix.inverse_table
    device = torch.device('cpu') if device is None else torch.device(device)
    held = radix.__dict__.get('_multi_byte_forms')
    if held is not None and held[0] is table and held[1:3] == (len(table), device):
        return held[3]
    found = [(row, raw) for row, raw in enumerate(table) if len(raw) > 1]
    forms = index_spellings(found, device)
    radix.__dict__['_multi_byte_forms'] = (table, len(table), device, forms)
    return forms


def index_spellings(found, device):
    """Index (address, bytes) pairs by length, preserving every hash collision."""
    length = max((len(raw) for _, raw in found), default=1)
    spellings = torch.full((len(found), length), -1, dtype=torch.long, device='cpu')
    by_length = {}
    for n, (_, raw) in enumerate(found):
        spellings[n, :len(raw)] = torch.tensor(list(raw), device='cpu')
        h = 0
        for byte in raw:
            h = (h * HASH_BASE + byte + 1) % HASH_MODULUS
        by_length.setdefault(len(raw), {}).setdefault(h, []).append(n)
    groups = []
    for size, by_hash in sorted(by_length.items()):
        # Most hashes have just one candidate. Keep colliding candidates
        # in additional tables so every exact occurrence survives the lookup.
        buckets = []
        for key, candidates in sorted(by_hash.items()):
            for tie, n in enumerate(candidates):
                if tie == len(buckets):
                    buckets.append(([], []))
                buckets[tie][0].append(key)
                buckets[tie][1].append(n)
        for keys, indices in buckets:
            groups.append((size, torch.tensor(keys, dtype=torch.long, device=device),
                           torch.tensor(indices, dtype=torch.long, device=device)))
    return (torch.tensor([row for row, _ in found], dtype=torch.long, device=device),
             spellings.to(device), groups)


def _window_hits(raw, valid, spellings, groups):
    """Yield each length's candidate rows and exact matches at their starts."""
    B, width = raw.shape
    h = raw.new_zeros(B, width, dtype=torch.long)
    ok = torch.ones_like(valid)
    done = 0
    for size, keys, indices in groups:
        if size > width:
            break
        for length in range(done + 1, size + 1):
            stop = width + 1 - length
            h = (h[:, :stop] * HASH_BASE + raw[:, length - 1:] + 1) % HASH_MODULUS
            ok = ok[:, :stop] & valid[:, length - 1:]
        done = size
        at = torch.searchsorted(keys, h.contiguous()).clamp_max(len(keys) - 1)
        rows = indices[at]
        same = (raw.unfold(1, size, 1) == spellings[rows, :size]).all(-1)
        match = ok & (keys[at] == h) & same
        yield size, rows, match


def form_presence(raw, valid, spellings, groups):
    """Find every attested run in the visible bytes with fixed-shape device reads."""
    present = raw.new_zeros(len(spellings), dtype=torch.int32)
    for _, rows, match in _window_hits(raw, valid, spellings, groups):
        present = present.scatter_reduce(0, rows.flatten(), match.flatten().int(), reduce='amax')
    return present.bool()


def forms_in(radix, raw, valid):
    """The store's forms of two bytes or more present in the bytes ``raw``
    ``[B, P]`` where ``valid``, anywhere in the batch: their rows ``[F]`` and
    spellings ``[F, L]`` (``-1`` past a form's length). Each window of each
    form length is hashed and looked up among the forms of that length; a
    hit is checked byte by byte."""
    rows, spellings, groups = store_forms(radix, raw.device)
    device = raw.device
    if not len(rows):
        return rows.to(device), spellings[:, :0].to(device)
    spellings = spellings.to(device)
    present = form_presence(raw.long(), valid, spellings, groups)
    kept = present.nonzero()[:, 0]
    spelled = spellings[kept]
    width = int((spelled >= 0).sum(-1).max()) if len(kept) else 0
    return rows.to(device)[kept], spelled[:, :width]


def sentence_forms(radix, raw, valid, *, with_cover=False):
    """Observed forms, optionally with their multiplicity at every target byte."""
    rows, spellings, groups = store_forms(radix, raw.device)
    # At each length and hash tie a visible window names at most one row.
    # This bounds the gather even when the inventory contains many more rows
    # than there are possible substrings in this presentation.
    capacity = min(len(rows), sum(min(len(keys), raw.shape[0] * (raw.shape[1] - size + 1))
                                  for size, keys, _ in groups if size <= raw.shape[1]))
    if not capacity:
        bank = rows[:0], spellings[:0, :0], rows[:0].bool()
        return (*bank, raw.new_empty(len(raw), raw.shape[1], 0)) if with_cover else bank
    hits = _window_hits(raw, valid, spellings, groups)
    if with_cover:
        hits = tuple(hits)
    found = raw.new_zeros(len(rows), dtype=torch.int32)
    for _, candidates, match in hits:
        found.scatter_reduce_(0, candidates.flatten(), match.flatten().int(), reduce='amax')
    found = found.bool()
    # Preserve inventory order for the summation and put unused seats last.
    order = torch.arange(len(rows), device=raw.device)
    selected = torch.topk(torch.where(found, order, order + len(rows)),
                          capacity, largest=False, sorted=True).indices
    length = max(size for size, _, _ in groups if size <= raw.shape[1])
    bank = rows[selected], spellings[selected, :length], found[selected]
    if not with_cover:
        return bank
    # A matching window adds one to each cell of its extent. Accumulate
    # starts and ends, so overlapping matches of the same form count twice.
    seats = torch.zeros_like(order).scatter_(0, selected, order[:capacity])
    stride = raw.shape[1] + 1
    edges = raw.new_zeros(len(raw), capacity * stride, dtype=torch.int32)
    for size, candidates, match in hits:
        offset = torch.arange(candidates.shape[1], device=raw.device)
        start = seats[candidates] * stride + offset
        edges.scatter_add_(1, start, match.int())
        edges.scatter_add_(1, start + size, -match.int())
    cover = edges.reshape(len(raw), capacity, stride).long().cumsum(-1)[..., :-1].transpose(1, 2)
    return *bank, cover


def _form_cover(padded, spellings, positions):
    """Gather one block of forms, then fold consecutive bytes in source order."""
    count, size = spellings.shape
    batch, _, width = padded.shape
    present = (spellings >= 0).T[..., None]
    # One indexed read for the shorter runs that fit the workspace.
    # For a single very long run use the same product one offset at a time.
    if batch * count * size * width <= 1 << 22:
        taken = padded.index_select(1, spellings.clamp_min(0).T.reshape(-1))
        taken = torch.where(present, taken.reshape(batch, size, count, width), 1.)
        presence = taken[:, 0, :, :positions]
        for offset in range(1, size):
            presence = presence * taken[:, offset, :, offset:offset + positions]
    else:
        presence = padded.new_ones(batch, count, positions)
        for offset in range(size):
            values = padded[:, :, offset:offset + positions].index_select(1, spellings[:, offset].clamp_min(0))
            presence = presence * torch.where(present[offset], values, 1.)
    cover = presence.clone()
    for offset in range(1, min(size, positions)):
        cover[..., offset:] += presence[..., :positions - offset] * (spellings[:, offset, None] >= 0)
    return cover


def content(amounts, byte_codes, spellings, form_codes):
    """The content at each place of graded bytes ``amounts`` ``[B, P, 256]``
    (§19.135, 4): the ``.what`` codes ``byte_codes`` ``[256, what]`` of the
    bytes there, and the code ``form_codes`` ``[F, what]`` of each form
    ``spellings`` ``[F, L]`` present there, at each place of its extent. A
    form's presence at a start is the product of the amounts of its bytes at
    their places, so it is graded and takes a gradient; on bytes read
    (amounts of zero and one) it is one where the form is spelled and zero
    elsewhere. Returns ``[B, P, what]``."""
    out = amounts @ byte_codes
    Fn, L = spellings.shape
    if not Fn:
        return out
    B, P, _ = amounts.shape
    # Places last so an offset reads its own contiguous slice. Keep the
    # vocabulary summation in its original order, even across work blocks.
    padded = F.pad(amounts, (0, 0, 0, L)).transpose(1, 2).contiguous()     # [B, 256, P + L]
    step = max(1, (1 << 22) // max(1, B * L * (P + L)))
    if step >= Fn:
        cover = _form_cover(padded, spellings, P)
    else:
        from torch.utils.checkpoint import checkpoint
        blocks = []
        for forms in spellings.split(step):
            if padded.requires_grad and torch.is_grad_enabled():
                blocks.append(checkpoint(_form_cover, padded, forms, P,
                                         use_reentrant=False, preserve_rng_state=False))
            else:
                blocks.append(_form_cover(padded, forms, P))
        cover = torch.cat(blocks, 1)
    return out + cover.transpose(1, 2) @ form_codes


def answer_pyramid_cost(radix, amounts, emitted, text):
    """Compare the realized and requested percepts at their native byte places."""
    from Layers import Error
    target = emitted.new_tensor(list(text.encode('utf8')), dtype=torch.long)[None]
    width = max(amounts.shape[1], emitted.shape[1], target.shape[1], 1)
    target_places = torch.arange(width, device=emitted.device)[None] < target.shape[1]
    target = F.pad(target, (0, width - target.shape[1]))
    emitted = F.pad(emitted.long(), (0, width - emitted.shape[1]))
    rows, forms = forms_in(radix, torch.cat((emitted, target)),
                           torch.cat((emitted != 0, target_places)))
    byte_ids = byte_rows(radix).to(emitted.device)
    codes = radix._basis.lookup_rows(byte_ids.clamp_min(0)).detach().to(amounts)
    codes = codes * (byte_ids >= 0)[:, None]
    form_codes = radix._basis.lookup_rows(rows).detach().to(amounts)
    emitted_amounts = F.pad(amounts, (0, 0, 0, width - amounts.shape[1]))
    wanted_amounts = F.one_hot(target, 256).to(amounts) * target_places[..., None]
    put = content(emitted_amounts, codes, forms, form_codes)
    wanted = content(wanted_amounts, codes, forms, form_codes)
    # A previously unseen byte still occupies a place in the target. Its
    # primitive identity can be compared without admitting it to the store.
    missing = byte_ids < 0
    put = torch.cat((put, emitted_amounts[..., missing]), -1)
    wanted = torch.cat((wanted, wanted_amounts[..., missing]), -1)
    shared, wanted_mass = Error.pyramid_match(put, wanted)
    mass = put.clamp_min(0).flatten(1).sum(-1) + wanted_mass
    # Both extra and missing percepts cost. One unit of smoothing keeps
    # silence defined and gives unwanted speech a gradient toward silence.
    return (mass - 2 * shared).clamp_min(0) / (1 + mass)


def where_match(wires, grid, axes, sharpness):
    """Each leaf's match ``[..., addresses]`` of its ``.where`` wires against
    the grid's rows, axis by axis (each a slice of the wires), by distance:
    a softmax of the summed distances. One axis today."""
    distance = sum((wires[..., None, lo:hi] - grid[:, lo:hi]).norm(dim=-1) for lo, hi in axes)
    return torch.softmax(-sharpness * distance, -1)


def drawn(score):
    """The index of the greatest of ``score`` on its last axis, a draw among
    equals."""
    tied = score == score.amax(-1, keepdim=True)
    return (tied * torch.rand(score.shape, device=score.device)).argmax(-1)


def reverse(model, sub, field):
    """The input's reconstruction leaf by leaf (rule 79; §19.133): each
    reading's own concept goes back through the order-zero Sigma and the
    first Pi to the basic percept it recreates; the percept is taken apart;
    its ``.what`` is matched against the percept codebook and its ``.where``
    wires against the grid; the matched percept is pasted at the matched
    place. Both matches are graded, so the cost's gradient reaches what
    reads them; the decoded bytes are the greatest of each, a draw among
    equals."""
    from util import TheXMLConfig
    if field.wires is None:
        raise ValueError("a field is reconstructed from its readings' own .where wires")
    cs, ps = model._concept_owner(), model.perceptualSpace
    basic, radix, registry = cs.basic_percepts, ps.percept_store, model.where_registry
    decoder = radix.activity_decoder
    B, E = field.where.shape[:2]
    # Each leaf's concepts on the order-zero Sigma's outputs, each lane
    # apart (a sign is two lanes, up and back): for, and against negated.
    sigma = cs.install_order_zero()
    from SentenceField import inventory
    rows_at = inventory(cs, field)
    value = torch.stack((field.evidence[..., 0], -field.evidence[..., 1]), -1)
    if rows_at.ndim == 1:
        rows_at = rows_at[:, None]
    # Accumulate every lane of each stream independently. A spare row
    # absorbs padding and higher orders without reading their indices back.
    address = torch.where((rows_at >= 0) & (rows_at < sigma.nOutput), rows_at, sigma.nOutput)
    address = address[..., None, None].expand_as(value)
    concepts = value.new_zeros(sigma.nOutput + 1, B, E, 2).scatter_add(0, address, value)[:-1]
    # The Sigma's way back, then the first Pi's: the recreated .what.
    rows = list(range(len(basic.index)))
    held = sigma.reverse(concepts, rows).clamp_min(0).sum(-1)                # [R, B, E]
    kept = basic.kept_percept(rows)[0][:, 0].to(held)                        # [R, width]
    what = torch.einsum('rbe,rw->bew', held, kept)
    # A leaf is present as its greatest row is, through the chart (tanh).
    presence = torch.tanh(held.amax(0)) if len(rows) else value.new_zeros(B, E)
    value = value[..., 0]
    # The .what against the percept codebook: each leaf's graded amount of
    # each byte at each offset of the percept it recreates.
    spelled, table = codebook(basic)
    table = table.to(value.device)
    P, L = table.shape
    position = torch.arange(L, device=value.device)
    if P:
        codes = kept.index_select(0, torch.tensor(spelled, device=kept.device))
        score = F.cosine_similarity(what[..., None, :], codes, dim=-1)        # [B, E, P]
        w = torch.softmax(float(TheXMLConfig.get('architecture.reconstructionWhatSharpness')) * score, -1)
        spelt = F.one_hot(table.clamp_min(0), 256).to(w) * (table >= 0)[..., None].to(w)   # [P, L, 256]
        offsets = torch.einsum('bep,plv->belv', w * presence[..., None], spelt)
        choice = drawn(score.detach())
    else:
        offsets = value.new_zeros(B, E, L, 256)
        choice = torch.zeros(B, E, dtype=torch.long, device=value.device)
    # The .where wires against the grid; a reading with no wires has no place.
    t0, t1 = registry.slices['text']
    grid = registry.code.to(value.device).float()
    wires = field.wires.to(value.device).float()
    q = where_match(wires, grid, registry.axes, float(TheXMLConfig.get('architecture.reconstructionWhereSharpness')))
    q = (q[..., t0:t1] * wires.ne(0).any(-1)[..., None]).to(value)          # [B, E, A]
    place = drawn(q.detach())
    A = t1 - t0
    # Paste: byte i of a leaf's percept at its place plus i.
    width = A + L
    amounts = value.new_zeros(B, width, 256)
    when_code = ps.subspace.whenEncoding.encode(field.when[..., 0].long()).to(value) if ps.subspace.nWhen else None
    when = None if when_code is None else value.new_zeros(B, width, when_code.shape[-1])
    for i in range(L):
        amounts[:, i:i + A] += torch.einsum('bea,bev->bav', q, offsets[:, :, i])
        if when is not None:
            when[:, i:i + A] += torch.einsum('bea,be,bed->bad', q, offsets[:, :, i].sum(-1), when_code)
    # The decoded bytes: each leaf's greatest percept at its greatest place.
    rows_of = byte_rows(radix).to(value.device)
    with torch.no_grad():
        bytes_at = (table.index_select(0, choice.reshape(-1)).reshape(B, E, L) if P else
                    table.new_full((B, E, L), -1))
        ids_at = torch.where(bytes_at >= 0, rows_of[bytes_at.clamp_min(0)], -1)
        values = (presence * w.gather(-1, choice[..., None])[..., 0] if P else presence) * q.gather(-1, place[..., None])[..., 0]
        values = values[..., None] * (ids_at >= 0)
        at = (place[..., None] + position).reshape(B, -1)
        strength = values.new_zeros(B, width).scatter_reduce(1, at, values.reshape(B, -1), 'amax', include_self=True)
        best = values.reshape(B, -1) == strength.gather(1, at)
        live = best & (values.reshape(B, -1) > 0)
        selected = torch.zeros(B, width, dtype=torch.long, device=value.device).scatter(
            1, torch.where(live, at, width - 1), torch.where(live, ids_at.reshape(B, -1).clamp_min(0), 0))
        positions = torch.arange(1, width + 1, device=value.device)
        emitted_end = torch.where((strength > 0).any(0), positions, 0).amax().clamp_min(1)
        # The buffer holds the sentence's extent, so that a percept not put
        # back is missing at its places, and whatever is pasted beyond it.
        ends = torch.where((field.where[..., 0] >= t0) & (field.where[..., 1] <= t1), field.where[..., 1] - t0, 0)
        extent = emitted_end.maximum(ends.amax()) if ends.numel() else emitted_end
        span = int(extent)
    if span >= len(decoder.positions):
        raise OverflowError('a reconstructed sentence exceeds decoder position capacity')
    amounts, strength, selected = amounts[:, :span], strength[:, :span], selected[:, :span]
    # The pasted code: each byte's prototype by its amount.
    prototypes = radix._basis.lookup_rows(rows_of.clamp_min(0)).to(value) * (rows_of >= 0)[:, None].to(value)
    code = amounts @ prototypes
    muxed = int(ps.subspace.nWhat + ps.subspace.nWhere + ps.subspace.nWhen)
    code = F.pad(code, (0, muxed - code.shape[-1]))
    where_indices = ps._radix_where_indices
    if where_indices and where_indices[0] >= radix.dim:
        # A byte pasted at a place is at that place: the grid's row there.
        code[..., where_indices] = grid[t0:t0 + span].to(code) * amounts.sum(-1)[..., None]
    if when is not None:
        code[..., -ps.subspace.nWhen:] = when[:, :span]
    recovered = sub.carrier_like()
    positions = decoder.positions[:span][None]
    lit = strength > 0
    first = torch.where(lit, positions, span).amin(-1)
    last = torch.where(lit, positions, -1).amax(-1)
    separators = (positions > first[:, None]) & (positions < last[:, None]) & ~lit
    recovered._radix_activity = NativeActivity(selected, strength, code, separators, amounts)
    return recovered


def word_texts(surface, where):
    """Read decoded bytes at the supplied field brackets, independent of input."""
    values = surface[:, 0].detach().long().cpu().tolist()
    return tuple(tuple(bytes(row[lo:hi]).decode('utf8', 'replace')
                       for lo, hi in spans if hi > lo)
                 for row, spans in zip(values, where.cpu().tolist()))
