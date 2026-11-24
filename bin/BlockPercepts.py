"""Sentence-local reads of promoted Pi definitions while pages discover new parts."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockParts:
    rows: torch.Tensor
    available: torch.Tensor       # [stream, new part], earliest opening byte
    starts: torch.Tensor          # sorted sentence starts per stream, padded at the right
    runs: tuple                   # admitted names used by the existing holder links

    def openings(self, rows):
        at = torch.searchsorted(self.rows, rows.contiguous()).clamp_max(len(self.rows) - 1)
        return torch.where(self.rows[at] == rows, self.available[:, at], 0)


@dataclass(frozen=True)
class RowRevisions:
    rows: tuple
    indices: torch.Tensor
    starts: torch.Tensor
    codes: torch.Tensor           # [changed row, stream, sentence, lane, coordinate]

    def at(self, offsets):
        sentence = torch.searchsorted(self.starts, offsets.contiguous(), right=True) - 1
        streams = torch.arange(offsets.shape[0], device=offsets.device)[:, None]
        return self.codes[:, streams, sentence.clamp_min(0)]


def _definition(basic, row, form, block):
    """Fold the member references and nearest visible holders at each opening."""
    from WordIdentity import text_of
    identity = basic._identity()
    iw = basic.identity_width
    parts, wholes = basic.occurrences[row]
    device = block.rows.device
    name = text_of(form)

    def active(refs):
        ids = torch.tensor(refs, device=device, dtype=torch.long)
        starts = block.openings(ids)[:, None]
        return (starts != torch.iinfo(starts.dtype).max) & (block.starts[..., None] >= starts)

    # A nested form supplies the coordinates of the atoms it contains.
    # Preserve native rows which are already separate members of this Pi.
    nested = {identity.word_rows[child]: identity.words[child]
              for child, _, _ in identity.nested.get(form, ())}
    members = tuple(dict.fromkeys(ref for part in parts for ref in
                    ([identity.atoms[a] for a in nested[part]] if part in nested else [part])))
    ids = torch.tensor(members, device=device, dtype=torch.long)
    basis = basic.native._basis.lookup_rows(ids)[..., :iw]
    lower = torch.where(active(members)[..., None], basis, 0.).amax(-2)
    atoms = set(identity.words[form])
    if not basic.property_width:
        upper = lower[..., :0]
    else:
        upper = (basic._stored_boundary_codes(wholes, whole=True, row=row).amin(0)
                 if wholes else lower.new_ones(basic.property_width))
        if basic._tightest(parts) is not None:
            by_row = identity._atom_rows
            held = set()
            for part in parts:
                source = by_row[part]
                held.update(identity.part_atoms(source) if isinstance(source, bytes) else source)
            ordered = sorted(held)
            refs = [identity.atoms[atom] for atom in ordered]
            catalog = torch.stack([identity.catalog_code(atom) for atom in ordered]).to(lower)
            tight = torch.where(active(refs)[..., None], catalog, 0.).amax(-2)
            upper = torch.minimum(upper, tight)
        upper = upper.expand(*lower.shape[:-1], -1)
    opposed = torch.zeros_like(lower)
    holders = sorted(identity.holders.get(form, ()))
    if holders:
        # The nearest holder can differ per page even for an established part.
        holder_ids = [identity.atoms[('part', text_of(h))] for h in holders]
        lengths = ids.new_tensor(list(map(len, holders)))
        distance = torch.where(active(holder_ids), lengths, torch.iinfo(torch.long).max)
        nearest = (distance == distance.amin(-1, keepdim=True)) & (distance == lengths)
        for i, holder in enumerate(holders):
            extra = [identity.atoms[a] for a in identity.words[holder] if a not in atoms and name not in a[1]]
            if extra:
                values = basic.native._basis.lookup_rows(ids.new_tensor(extra))[..., :iw]
                present = active(extra) & nearest[..., i, None]
                opposed = torch.maximum(opposed, torch.where(present[..., None], values, 0.).amax(-2))
    positive = torch.cat((lower, upper), -1)
    negative = torch.cat((torch.minimum(opposed, 1 - lower.clamp(0, 1)), 1 - upper.clamp(0, 1)), -1)
    return torch.stack((positive, negative), -2).clamp(0., 1.)


@torch.no_grad()
def revisions(basic, rows, block):
    """Materialize only the affected, named Pi rows for this block's openings."""
    if block is None:
        return None
    identity = basic._identity()
    indices, codes = [], []
    changed = {}
    def definition(row):
        if row in changed:
            return changed[row]
        form = basic.form_of(row)
        if form is not None and form in identity.words:
            holders = identity.holders.get(form, ())
            if any(run in form or form in run or any(run in holder for holder in holders)
                   for run in block.runs):
                changed[row] = _definition(basic, row, form, block)
                return changed[row]
        changed[row] = None
        return None

    # A promoted conjunction contributes its other members to the against
    # lane. The conjunction graph is acyclic by construction (members are older).
    def positive(row):
        members = basic.conjunctions.get(row)
        if members is None:
            value = definition(row)
            return None if value is None else value[..., 0, :basic.identity_width]
        values = [positive(member) for member in members]
        if all(v is None for v in values):
            return None
        template = next(v for v in values if v is not None)
        values = [basic.code(member)[:basic.identity_width].expand_as(template) if v is None else v
                  for member, v in zip(members, values)]
        draw = torch.rand(basic.width, device='cpu', generator=torch.Generator().manual_seed(row))
        return (torch.stack(values).amax(0) + basic.INCREMENT * draw[:basic.identity_width].to(template)).clamp_max(1.)

    for i, row in enumerate(rows):
        if row in basic.place_rows or row in basic.conjunctions or basic.occurrences[row] is None:
            continue
        lanes = definition(row)
        holders = [(whole, member) for whole, members in basic.conjunctions.items() if row in members
                   for member in members if member != row]
        if holders:
            contributions = {member: positive(member) for _, member in holders}
            if lanes is not None or any(v is not None for v in contributions.values()):
                if lanes is None:
                    # The row's own named-holder evidence still applies when
                    # only the conjunction's other input changed.
                    form = basic.form_of(row)
                    if form in identity.words:
                        lanes = _definition(basic, row, form, block)
                    else:
                        pos = basic.code(row).expand(*block.starts.shape, -1)
                        neg = torch.cat((torch.zeros_like(pos[..., :basic.identity_width]),
                                         1 - pos[..., basic.identity_width:].clamp(0, 1)), -1)
                        lanes = torch.stack((pos, neg), -2)
                lanes = lanes.clone()
                lower = lanes[..., 0, :basic.identity_width]
                values = []
                for whole, member in holders:
                    v = contributions[member]
                    if v is None:
                        v = basic.code(member)[:basic.identity_width].expand_as(lower)
                    values.append(torch.where(basic.promoted[whole], v, 0.))
                opposing = torch.minimum(torch.stack(values).amax(0), 1 - lower.clamp(0, 1))
                lanes[..., 1, :basic.identity_width] = torch.maximum(lanes[..., 1, :basic.identity_width], opposing)
        if lanes is not None:
            indices.append(i)
            codes.append(lanes)
    if not indices:
        return None
    return RowRevisions(tuple(rows[i] for i in indices), block.rows.new_tensor(indices), block.starts, torch.stack(codes))
