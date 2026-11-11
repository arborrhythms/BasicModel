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


def reverse(model, sub, field):
    """A basic percept puts its parts at its own field bracket, never an input trace."""
    cs, ps = model._concept_owner(), model.perceptualSpace
    radix, primitive = ps.percept_store, model.wholeSpace.subspace.what.primitive_properties
    decoder = radix.activity_decoder
    ids = field.concept_ids
    store = cs._concept_allocator.layer(0)
    by_id = {(key[1] if isinstance(key, tuple) else key): row for row, key in store._tensor_row_keys.items()}
    rows = torch.tensor([by_id.get(int(cid), -1) for cid in ids.reshape(-1).tolist()],
                        device=ids.device, dtype=torch.long).reshape(ids.shape)
    columns, attributed, _ = cs.cs_percept_attribution(field.evidence,
        observed=field.evidence, inventory_rows=rows)
    B, E = field.where.shape[:2]
    width = max(1, int(field.where[..., 1].max()))
    if width >= len(decoder.positions):
        raise OverflowError('field bracket exceeds decoder position capacity')
    extents = (field.where[..., 1] - field.where[..., 0]).clamp_min(0)
    span = int(extents.max()) if E else 0
    offsets = decoder.positions[:span]
    positions = field.where[..., :1] + offsets
    valid = offsets < extents[..., None]
    indices = positions.clamp(0, width - 1).reshape(B, -1)

    def place(values, *, shift=None):
        if shift is not None:
            at = (field.where[..., 0] + shift).clamp(0, width - 1)
            return values.new_zeros(B, width).scatter_reduce(1, at, values, reduce='amax', include_self=True)
        values = (values[..., None] * valid).reshape(B, -1)
        return values.new_zeros(B, width).scatter_reduce(1, indices, values, reduce='amax', include_self=True)

    properties = [[attributed.new_zeros(B, width) for _ in range(2)] for _ in primitive.members]
    parts = {}
    # identityByParts is optional on native PartSpace configurations.
    identity = getattr(radix, 'identity', None)
    atoms = set(identity.atoms.values()) if identity is not None else set()
    for edge, col in enumerate(columns.tolist()):
        support = attributed[edge, ..., 0]
        group, wholes = cs.basic_percepts.occurrences[col // 2]
        for whole in wholes:
            properties[whole][col % 2] = torch.maximum(properties[whole][col % 2], place(support))
        if col % 2 == 0:
            if group and any(pid in atoms for pid in group):
                # Boundary-pair/length atoms name an admitted native word;
                # their diagnostic spellings are not surface bytes.
                raw = identity.read_parts(group)
                group = (identity.word_rows[raw],)
            length = sum(len(radix.bytes_for(pid)) for pid in group)
            fitted = support * (extents == length)
            shift = 0
            for pid in group:
                for byte in radix.bytes_for(pid):
                    row = radix.get_id(bytes([byte]))
                    if row is None:
                        raise ValueError('an admitted native form has an unadmitted byte')
                    parts.setdefault(row, []).append((fitted, shift))
                    shift += 1
    properties = (torch.stack([torch.stack(p, -1) for p in properties], -2) if properties else
                  attributed.new_zeros(B, width, 0, 2))
    # Compute one primitive column at a time. Even a full 8192-position field
    # never allocates an activity axis of len(radix) (or a dense byte alphabet).
    weights = primitive.coefficients()
    known = primitive.observed | (primitive.members.detach() != 0)
    pairs = torch.stack((weights, (1 - weights) * known), -1).clamp_min(0)
    pairs = pairs / pairs.sum(1, keepdim=True).clamp_min(1e-12)
    byte_rows = {pid: byte for byte in range(256) if (pid := radix.get_id(bytes([byte]))) is not None}
    strength = attributed.new_zeros(B, width)
    selected = torch.zeros(B, width, device=strength.device, dtype=torch.long)
    code = attributed.new_zeros(B, width, radix.dim)
    for pid in sorted(parts):
        value = attributed.new_zeros(B, width)
        for support, shift in parts.get(pid, ()):
            value = torch.maximum(value, place(support, shift=shift))
        prototype = radix._basis.lookup_rows(torch.tensor([pid], device=value.device))[0].to(value)
        code = code + value[..., None] * prototype
        selected = torch.where(value > strength, pid, selected)
        strength = torch.maximum(strength, value)
    # A basic percept's ordered identity and property extension are two
    # descriptions of the same support. Property attribution supplies gaps;
    # summing it into competing identities loses order and counts support
    # repeatedly when several properties describe the same part.
    identity_present = strength > 0
    for pid, byte in sorted(byte_rows.items()):
        value = (properties * pairs[:, byte][None, None]).sum((-1, -2))
        value = torch.where(identity_present, 0., value)
        prototype = radix._basis.lookup_rows(torch.tensor([pid], device=value.device))[0].to(value)
        code = code + value[..., None] * prototype
        selected = torch.where(value > strength, pid, selected)
        strength = torch.maximum(strength, value)
    muxed = int(ps.subspace.nWhat + ps.subspace.nWhere + ps.subspace.nWhen)
    code = F.pad(code, (0, muxed - code.shape[-1]))
    where_indices = ps._radix_where_indices
    if where_indices and where_indices[0] >= radix.dim:
        locations = decoder.positions[:width].expand(B, -1)
        stamp = model.where_registry.encode('input', locations)
        code[..., where_indices] = stamp.to(code) * strength[..., None]
    if ps.subspace.nWhen:
        times = place(field.when[..., 0].to(device=code.device, dtype=torch.long))
        code[..., -ps.subspace.nWhen:] = ps.subspace.whenEncoding.encode(times).to(code) * strength[..., None]
    recovered = sub.carrier_like()
    covered = torch.zeros(B, width, device=code.device, dtype=torch.long).scatter_add(
        1, indices, valid.reshape(B, -1).long()).bool()
    begin = torch.where(extents > 0, field.where[..., 0], width).amin(-1)
    end = field.where[..., 1].amax(-1)
    positions = decoder.positions[:width][None]
    separators = ((positions >= begin[:, None]) & (positions < end[:, None]) &
                  ~covered & (strength > 0).any(-1)[:, None])
    recovered._radix_activity = NativeActivity(selected, strength, code, separators)
    return recovered


def word_texts(surface, where):
    """Read decoded bytes at the supplied field brackets, independent of input."""
    values = surface[:, 0].detach().long().cpu().tolist()
    return tuple(tuple(bytes(row[lo:hi]).decode('utf8', 'replace')
                       for lo, hi in spans if hi > lo)
                 for row, spans in zip(values, where.cpu().tolist()))
