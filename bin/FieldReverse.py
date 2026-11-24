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


def spelling(basic, row):
    """The bytes a basic percept spells: its stored form, its run, or its
    parts' own texts in order; ``None`` where it spells none."""
    raw = basic.form_of(row)
    if raw is None:
        raw = basic.runs.get(int(row))
    if raw is None:
        texts = [basic._part_text(part) for part in basic.occurrences[int(row)][0]]
        raw = None if not texts or any(text is None for text in texts) else b''.join(texts)
    return raw


def codebook(basic):
    """The percept codebook's spelled rows: each basic percept that spells
    bytes (no place, no conjunction, no value), with each of its bytes
    ``[P, L]`` (``-1`` past its length). Kept until a row is added."""
    key = (basic.revision, len(basic.index))
    held = basic.__dict__.get('_reconstruction_codebook')
    if held is not None and held[0] == key:
        return held[1]
    rows, spelled = [], []
    for row in range(len(basic.index)):
        if row in basic.place_rows or row in basic.conjunctions or basic.is_value(row):
            continue
        raw = spelling(basic, row)
        if raw:
            rows.append(row)
            spelled.append(list(raw))
    length = max([len(raw) for raw in spelled] + [1])
    table = torch.full((len(spelled), length), -1, dtype=torch.long)
    for n, raw in enumerate(spelled):
        table[n, :len(raw)] = torch.tensor(raw)
    basic.__dict__['_reconstruction_codebook'] = (key, (rows, table))
    return rows, table


def byte_rows(radix):
    """Each byte's row of the percept store ``[256]`` (``-1`` where none)."""
    return torch.tensor([-1 if (row := radix.get_id(bytes([v]))) is None else row for v in range(256)])


#: The window hash of :func:`forms_in`: a polynomial in the bytes, modulo a
#: number small enough that ``h * base`` stays inside an int64.
HASH_BASE, HASH_MODULUS = 257, (1 << 53) - 111


def store_forms(radix):
    """The percept store's forms of two bytes or more: their rows ``[F]``,
    spellings ``[F, L]`` (``-1`` past a form's length), and for each length
    the forms of that length ``(length, window hashes sorted, their
    forms)``. Kept until a form is added."""
    key = len(radix.inverse_table)
    held = radix.__dict__.get('_multi_byte_forms')
    if held is not None and held[0] == key:
        return held[1]
    found = [(row, raw) for row, raw in enumerate(radix.inverse_table) if len(raw) > 1]
    length = max([len(raw) for _, raw in found] + [1])
    spellings = torch.full((len(found), length), -1, dtype=torch.long)
    hashes = []
    for n, (_, raw) in enumerate(found):
        spellings[n, :len(raw)] = torch.tensor(list(raw))
        h = 0
        for byte in raw:
            h = (h * HASH_BASE + byte + 1) % HASH_MODULUS
        hashes.append(h)
    hashes = torch.tensor(hashes, dtype=torch.long)
    lengths = (spellings >= 0).sum(-1)
    groups = []
    for size in sorted({len(raw) for _, raw in found}):
        of = (lengths == size).nonzero()[:, 0]
        keys, order = hashes[of].sort()
        groups.append((size, keys, of[order]))
    forms = (torch.tensor([row for row, _ in found], dtype=torch.long), spellings, groups)
    radix.__dict__['_multi_byte_forms'] = (key, forms)
    return forms


def forms_in(radix, raw, valid):
    """The store's forms of two bytes or more present in the bytes ``raw``
    ``[B, P]`` where ``valid``, anywhere in the batch: their rows ``[F]`` and
    spellings ``[F, L]`` (``-1`` past a form's length). Each window of each
    form length is hashed and looked up among the forms of that length; a
    hit is checked byte by byte."""
    rows, spellings, groups = store_forms(radix)
    device = raw.device
    if not len(rows):
        return rows.to(device), spellings[:, :0].to(device)
    spellings = spellings.to(device)
    B, P = raw.shape
    L = spellings.shape[1]
    padded = F.pad(raw.long(), (0, L), value=-1)
    ok_at = F.pad(valid, (0, L))
    h = torch.zeros(B, P, dtype=torch.long, device=device)
    ok = torch.ones(B, P, dtype=torch.bool, device=device)
    present = torch.zeros(len(rows), dtype=torch.bool, device=device)
    done = 0
    for size, keys, of in groups:
        for length in range(done + 1, size + 1):
            h = (h * HASH_BASE + padded[:, length - 1:length - 1 + P] + 1) % HASH_MODULUS
            ok = ok & ok_at[:, length - 1:length - 1 + P]
        done = size
        keys, of = keys.to(device), of.to(device)
        at = torch.searchsorted(keys, h).clamp_max(len(keys) - 1)
        hit = (ok & (keys[at] == h)).nonzero()
        if not len(hit):
            continue
        b, s = hit.unbind(-1)
        form = of[at[b, s]]
        window = padded[b[:, None], s[:, None] + torch.arange(size, device=device)]
        present[form[(window == spellings[form, :size]).all(-1)]] = True
    kept = present.nonzero()[:, 0]
    spelled = spellings[kept]
    width = int((spelled >= 0).sum(-1).max()) if len(kept) else 0
    return rows.to(device)[kept], spelled[:, :width]


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
    # Places last, so that each form's byte at each offset is one row.
    padded = F.pad(amounts, (0, 0, 0, L)).transpose(1, 2).contiguous()     # [B, 256, P + L]
    spelled = (spellings >= 0).T[:, :, None]                                # [L, F, 1]
    taken = padded.index_select(1, spellings.clamp_min(0).T.reshape(-1)).reshape(B, L, Fn, P + L)
    taken = torch.where(spelled, taken, 1.)
    presence = taken[:, 0, :, :P]
    for i in range(1, L):
        presence = presence * taken[:, i, :, i:i + P]                     # [B, F, P]
    # A form present at a start is present at each place of its extent.
    cover = presence.clone()
    for i in range(1, min(L, P)):
        cover[..., i:] += presence[..., :P - i] * spelled[i]
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
        used = (strength > 0).any(0).nonzero()
        # The buffer holds the sentence's extent, so that a percept not put
        # back is missing at its places, and whatever is pasted beyond it.
        ends = torch.where((field.where[..., 0] >= t0) & (field.where[..., 1] <= t1), field.where[..., 1] - t0, 0)
        span = max(int(used.max()) + 1 if len(used) else 1, int(ends.max()) if ends.numel() else 0)
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
