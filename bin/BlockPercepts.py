"""Sentence-local reads of promoted Pi definitions while pages discover new parts."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockParts:
    rows: torch.Tensor
    available: torch.Tensor       # [stream, new part], earliest opening byte
    starts: torch.Tensor          # sorted sentence starts per stream, padded at the right
    runs: tuple                   # admitted names used by the existing holder links

    repairs: 'RowRevisions | None' = None

    def openings(self, rows):
        if not len(self.rows):
            return rows.new_zeros(self.starts.shape[0], len(rows))
        at = torch.searchsorted(self.rows, rows.contiguous()).clamp_max(len(self.rows) - 1)
        return torch.where(self.rows[at] == rows, self.available[:, at], 0)

    def values(self, rows, codes, start, offsets=None):
        if self.repairs is None:
            return codes
        changes = self.repairs
        at = torch.searchsorted(changes.indices, rows.contiguous()).clamp_max(len(changes.indices) - 1)
        delta = changes.codes if offsets is None else changes.at(offsets)
        selected = delta[at, ..., start:start + codes.shape[-1]].permute(1, 2, 0, 3)
        return codes + torch.where((changes.indices[at] == rows)[:, None], selected, 0.).to(codes)


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
    lower = torch.where(active(members)[..., None], block.values(ids, basis, 0), 0.).amax(-2)
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
            catalog = block.values(ids.new_tensor(refs), catalog, iw)
            tight = torch.where(active(refs)[..., None], catalog, 0.).amax(-2)
            upper = torch.minimum(upper, tight)
        upper = upper.expand(*lower.shape[:-1], -1)
    opposed = torch.zeros_like(lower)
    holders = sorted(identity.holders.get(form, ()))
    if holders:
        # The nearest holder can differ per page even for an established part.
        holder_ids = [identity.word_rows[h] for h in holders]
        lengths = ids.new_tensor(list(map(len, holders)))
        distance = torch.where(active(holder_ids), lengths, torch.iinfo(torch.long).max)
        nearest = (distance == distance.amin(-1, keepdim=True)) & (distance == lengths)
        for i, holder in enumerate(holders):
            extra = [identity.atoms[a] for a in identity.words[holder] if a not in atoms and name not in a[1]]
            if extra:
                other = ids.new_tensor(extra)
                values = block.values(other, basic.native._basis.lookup_rows(other)[..., :iw], 0)
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


@torch.no_grad()
def repaired_parts(identity, block, born, observed, repairs):
    """Re-evaluate containment repairs on each page's actual admission prefix.

    A later admission can select a different coordinate even when it repairs
    exactly the same part. The signed difference from the consolidated atom
    is needed on both the occurrence and the promoted Pi sides.
    """
    from WordIdentity import bytes_of, text_of, distinguishing_coordinates
    device = block.rows.device
    atoms = tuple(identity.atoms)
    words = identity._ordered
    seat = {a: i for i, a in enumerate(atoms)}
    changed = sorted({a for a, *_ in repairs}, key=identity.atoms.__getitem__)
    indices = block.rows.new_tensor([identity.atoms[a] for a in changed])
    atom_ids = block.rows.new_tensor([identity.atoms[a] for a in atoms])
    word_ids = block.rows.new_tensor([identity.word_rows[w] for w in words])
    end = torch.iinfo(torch.long).max
    zero = torch.zeros_like(block.starts[:, 0])
    offsets = (torch.stack(list(observed.values()), 1).sort(1).values if observed
               else block.starts[:, :0])
    distinct = offsets != end
    distinct[:, 1:] = distinct[:, 1:] & (offsets[:, 1:] != offsets[:, :-1])
    rounds = distinct.sum(1).amax()

    def rank_of(at):
        return torch.where(at == end, end, ((offsets <= at[:, None]) & distinct).sum(1))

    rank = {row: rank_of(starts) for row, starts in born}

    def ranks(addresses):
        values = [rank.get(row, zero) for row in addresses]
        return (torch.stack(values, 1) if values else block.starts[:, :0])[:, None]

    word_rank = ranks(identity.word_rows[w] for w in words)
    atom_rank = ranks(identity.atoms[a] for a in atoms)
    word_visible = block.starts[..., None] >= block.openings(word_ids)[:, None]
    atom_visible = block.starts[..., None] >= block.openings(atom_ids)[:, None]
    basis = torch.stack([identity.atom_code(a) for a in atoms])
    if identity.property_width:
        basis = torch.cat((basis, torch.stack([identity.catalog_code(a) for a in atoms])), -1)
    basis = basis.to(device).bool()
    initial = basis.clone()
    use = identity.usage.to(device).clone()
    bit_changes = []
    for atom, gained, witnesses, containment in repairs:
        gained = gained.to(device)
        initial[seat[atom]] &= ~gained
        if containment:
            use -= gained.long()
        else:
            limits = [observed[text_of(w)] for w in witnesses if text_of(w) in observed]
            bit_changes.append((seat[atom], gained, rank_of(torch.stack(limits).amax(0) if limits else zero)))
    state = initial.expand(*block.starts.shape, *initial.shape).clone()
    edges = [(w, seat[a]) for w, word in enumerate(words) for a in identity.words[word]]
    word_at, atom_at = block.rows.new_tensor(edges).T
    lengths = block.rows.new_tensor(list(map(len, words)))
    targets = {}
    for atom, _, _, containment in repairs:
        if containment and atom not in targets:
            inner = bytes_of(atom[1])
            holders = identity.holders[inner]
            holds = block.rows.new_tensor([w in holders for w in words], dtype=torch.bool)
            extras = [(identity.order[h], [seat[a] for a in identity.words[h]
                       if a not in identity.words[inner] and atom[1] not in a[1]]) for h in sorted(holders)]
            targets[atom] = identity.order[inner], holds, identity._kept(inner, device=device), extras

    for step in range(1, offsets.shape[1] + 1):
        live = word_visible & (word_rank <= step) & (step <= rounds)
        present = atom_visible & (atom_rank <= step) & (step <= rounds)
        for member, gained, first in bit_changes:
            reached = (first[:, None] <= step) & present[..., member]
            state[..., member, :] |= reached[..., None] & gained
        for atom, (i, holders, retained, extras) in targets.items():
            current = state & present[..., None]
            shape = (*live.shape, identity.total)
            codes = torch.zeros(shape, dtype=torch.int32, device=device)
            chunk = max(1, (1 << 22) // (block.starts.numel() * identity.total))
            for group, parts in zip(word_at.split(chunk), atom_at.split(chunk)):
                index = group[:, None].expand(*block.starts.shape, len(group), identity.total)
                codes.scatter_reduce_(-2, index, current[..., parts, :].int(), 'amax')
            codes = codes.bool()
            own = codes[..., i:i+1, :]
            accidents = (~(own & ~codes).any(-1) & live & live[..., i, None] & ~holders)
            accidents[..., i] = False
            added = (live[..., None] & holders[:, None] & ~codes).sum(-2)
            nearest = torch.where(live & holders, lengths, end).amin(-1)
            opposed = torch.zeros_like(own[..., 0, :])
            for h, other in extras:
                if other:
                    extra = current[..., other, :identity.width].any(-2)
                    opposed[..., :identity.width] |= (live[..., h] & (lengths[h] == nearest))[..., None] & extra
            given, outstanding = distinguishing_coordinates(~(own | codes), retained.expand_as(opposed),
                                                             opposed, added, use, accidents)
            # The shared rule leaves the part untouched if any outer is inseparable.
            given = given & ~outstanding.any(-1, keepdim=True)
            state[..., seat[atom], :] |= given
            use = use + given.long()
    places = [seat[a] for a in changed]
    delta = (state[..., places, :].float() - basis[places].float()).permute(2, 0, 1, 3)
    return RowRevisions(tuple(identity.atoms[a] for a in changed), indices, block.starts, delta)
