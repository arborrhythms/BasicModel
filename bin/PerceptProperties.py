"""Learned unions of byte primitives and folds over observed extents."""
import torch
from torch import nn

from ConceptEvidence import union


def uniform_spans(batch, positions, slots, *, device):
    """Disjoint contiguous input regions, with empty padding after short inputs."""
    count = min(int(positions), int(slots))
    result = torch.zeros(int(batch), int(slots), 2, device=device, dtype=torch.long)
    if count:
        cuts = torch.div(torch.arange(count + 1, device=device) * int(positions),
                         count, rounding_mode='floor')
        result[:, :count] = torch.stack((cuts[:-1], cuts[1:]), -1)
    return result


def counts_in_spans(byte_ids, spans, observed=None):
    """Scatter observed byte counts into each half-open extent."""
    if byte_ids.ndim != 2 or spans.ndim != 3 or spans.shape[-1] != 2:
        raise ValueError('byte extents require bytes [B,L] and spans [B,E,2]')
    B, L = byte_ids.shape
    spans = spans.to(byte_ids.device).clamp(0, L)
    # Pages have many padding bytes and few extents. For binary observations,
    # integer multiplicities fit exactly in float32 through this length. Count
    # directly into the requested histograms, including reversed extents.
    if (spans.shape[1] <= 64 and L <= 2**24 and torch.get_default_dtype() == torch.float32
            and (observed is None or observed.dtype == torch.bool)):
        start, end = spans.unbind(-1)
        positions = torch.arange(L, device=byte_ids.device)
        inside = ((positions >= torch.minimum(start, end)[..., None])
                  & (positions < torch.maximum(start, end)[..., None]))
        if observed is not None:
            inside = inside & observed.to(device=byte_ids.device)[:, None]
        addresses = byte_ids.long().clamp(0, 255)[:, None].expand_as(inside)
        output = torch.zeros(B, spans.shape[1], 256, device=byte_ids.device)
        output.scatter_add_(2, addresses, inside.to(output.dtype))
        return torch.where((end < start)[..., None], -output, output)
    # Keep the scan axis contiguous. The old [B,L,256] one-hot needed a
    # large int64 temporary and a strided cumsum for every byte channel.
    # Each (batch, position) writes one distinct cell, including repeats.
    counts = torch.zeros(B, 256, L, dtype=torch.get_default_dtype(), device=byte_ids.device)
    weights = counts.new_ones(B, 1, L) if observed is None else observed[:, None].to(counts)
    counts = counts.scatter(1, byte_ids.long().clamp(0, 255)[:, None], weights)
    prefix = torch.nn.functional.pad(counts.cumsum(-1), (1, 0))
    lo = spans[..., 0][:, None].expand(-1, 256, -1)
    hi = spans[..., 1][:, None].expand(-1, 256, -1)
    return (prefix.gather(2, hi) - prefix.gather(2, lo)).transpose(1, 2)


class _PropertySupportRead(torch.autograd.Function):
    """Min/max over observed bytes without retaining the broadcast product.

    Autograd's reduction saves every event/property/byte value. Recompute
    that bounded workspace in backward instead, including neutral-byte ties
    so the gradient is exactly the dense amin/amax gradient.
    """
    @staticmethod
    def forward(ctx, weights, present, conjunctive):
        chunks = []
        for start in range(0, weights.shape[0], 256):
            values = torch.where(present.unsqueeze(-2), weights[start:start + 256],
                                 1. if conjunctive else 0.)
            chunks.append(values.amin(-1) if conjunctive else values.amax(-1))
        result = (torch.cat(chunks, -1) if chunks else
                  weights.new_empty((*present.shape[:-1], 0)))
        ctx.save_for_backward(weights, present, result)
        ctx.conjunctive = conjunctive
        return result

    @staticmethod
    def backward(ctx, gradient):
        weights, present, result = ctx.saved_tensors
        chunks = []
        for start in range(0, weights.shape[0], 256):
            values = torch.where(present.unsqueeze(-2), weights[start:start + 256],
                                 1. if ctx.conjunctive else 0.)
            ties = values == result[..., start:start + 256, None]
            share = gradient[..., start:start + 256, None] / ties.sum(-1, keepdim=True)
            credited = torch.where(present.unsqueeze(-2), share * ties, 0.)
            if present.ndim > 1:
                credited = credited.sum(tuple(range(present.ndim - 1)))
            chunks.append(credited)
        return (torch.cat(chunks, 0) if chunks else torch.zeros_like(weights)), None, None


#: The prefix that draws the catalog a second time for the wholes half.
CATALOG_PREFIX = b'\x00whole-basis/'


def catalog_coordinates(part, width, ones):
    """A catalog part's coordinates in the wholes half: the part drawn a
    second time, apart from its coordinates in the parts half."""
    from WordIdentity import bit_stream
    return bit_stream(CATALOG_PREFIX + bytes(part), int(width))[:int(ones)]


#: The row of the table of wholes for the class of all characters.
TOP = -1


class PrimitiveProperties(nn.Module):
    """One bounded coefficient per property and byte, with no tag reader.

    A byte read is the sparse evaluation of a union over one-hot primitives.
    Dense primitive mixtures use that same union. Teaching uses the delta
    rule for squared error on observed primitives; ordinary gradients can
    subsequently refine the same coefficients.
    """

    def __init__(self, rows, *, property_width=0, ones=3, device=None, dtype=None):
        super().__init__()
        self.property_width, self.ones = int(property_width), int(ones)
        if self.property_width < 0 or self.ones < 1 or (self.property_width and self.ones > self.property_width):
            raise ValueError('property codes require propertyDim >= propertyOnes > 0')
        self.members = nn.Parameter(torch.zeros(int(rows), 256, device=device, dtype=dtype))
        # A coefficient of zero can mean either a witnessed exclusion or an
        # unwritten primitive. Only the former supplies counterevidence.
        self.register_buffer('observed', torch.zeros(int(rows), 256, device=device, dtype=torch.bool))
        self.register_buffer('fixed_rows', torch.zeros(int(rows), device=device, dtype=torch.bool))
        self.register_buffer('fixed_members', torch.zeros(int(rows), 256, device=device, dtype=dtype))
        # Every byte is a part of the catalog, drawn a second time in the
        # wholes half's own coordinates (sigma-pi rule 52); a property's code
        # is the OR of its member bytes' codes, and all bytes OR'ed are the top.
        bits = torch.zeros(256, self.property_width, device=device, dtype=dtype)
        if self.property_width:
            for byte in range(256):
                bits[byte, list(catalog_coordinates(bytes([byte]), self.property_width, self.ones))] = 1
        if self.property_width and not bool(bits.amax(0).eq(1).all()):
            raise ValueError('the catalog must cover the wholes half: its OR is the class of all characters')
        self.register_buffer('byte_bits', bits, persistent=bool(self.property_width))
        self._byte_columns = tuple(tuple(bits[:, i].nonzero().flatten().cpu().tolist())
                                   for i in range(self.property_width))
        # A row given by other rows of the table: row -> those rows. Its code
        # is the AND of theirs, and it holds of a byte as far as all of them do.
        self.given_by = {}
        self._revision = 0
        self._revision_versions = self._member_versions()

    def _member_versions(self):
        return tuple((id(value), value._version) for value in
                     (self.members, self.fixed_rows, self.fixed_members))

    @property
    def revision(self):
        """Observe teaching, optimizer writes and projection at the eager barrier."""
        versions = self._member_versions()
        if versions != self._revision_versions:
            self._revision += 1
            self._revision_versions = versions
        return self._revision

    def codes(self, rows=None):
        """Each row's code in the wholes half. A row given by catalog parts
        is the OR of its member bytes' codes, weighted, without a
        rows-by-bytes-by-width workspace; a row given by other rows is the
        AND of their codes."""
        if self.given_by:
            every = self._union_codes(self.coefficients())
            every = self._given_codes(every)
            return every if rows is None else every.index_select(0, rows)
        return self._union_codes(self.coefficients(rows))

    def _union_codes(self, weights):
        if not self.property_width:
            return weights[:, :0]
        return torch.stack([weights[:, list(ids)].amax(-1) if ids else
                            weights.sum(-1) * 0 for ids in self._byte_columns], -1)

    def _given_codes(self, values):
        """Replace each row given by other rows with the AND (the least) of
        theirs, the rows it is given by first."""
        done, out = {}, list(values.unbind(0))
        def resolve(row, seen=()):
            if row in done:
                return done[row]
            if row in seen:
                raise ValueError('a row of the table of wholes is given by itself')
            if row in self.given_by:
                done[row] = torch.stack([resolve(other, (*seen, row)) for other in self.given_by[row]]).amin(0)
            else:
                done[row] = out[row]
            return done[row]
        for row in self.given_by:
            out[row] = resolve(row)
        return torch.stack(out)

    @torch.no_grad()
    def define_by_rows(self, row, rows):
        """Give a row of the table by other rows of it (\"capital letter\" by
        \"letter\" and \"capital\"): its code is the AND of theirs, so it lies
        at or below each."""
        rows = tuple(sorted(dict.fromkeys(map(int, rows))))
        if int(row) in rows or not rows or not all(0 <= r < len(self.members) for r in rows):
            raise ValueError('a row of the table is given by two or more other rows of it')
        self.given_by[int(row)] = rows
        self.members[row].copy_(self.coefficients()[list(rows)].amin(0))
        self.observed[row] = self.observed[list(rows)].all(0)
        self._revision += 1

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(given_by=tuple(self.given_by.items())))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        self.given_by = {int(k): tuple(v) for k, v in _unpack_state(state).get('given_by', ())}

    def top(self):
        """The code of the class of all characters, the row ``TOP`` of the
        table of wholes: every byte belongs to it, so it is the OR of the
        whole catalog, all ones."""
        return self.byte_bits.amax(0)

    def directly_inside(self):
        """Each whole of the table with the wholes directly inside it.

        The table is the class of all characters (``TOP``) and every row that
        holds of some byte. One whole lies inside another when it holds of no
        byte more than the other does and of some byte less; it lies directly
        inside where no whole lies between. Wholes over one set of bytes are
        one level, inside neither."""
        revision = self.revision
        cached = getattr(self, '_inside', None)
        if cached is not None and cached[0] == revision:
            return cached[1]
        members = self.coefficients().detach()
        sets = {TOP: torch.ones_like(members[0]) if len(members) else members.new_ones(256)}
        sets.update({row: members[row] for row in range(len(members)) if bool(members[row].gt(0).any())})
        below = {a: [b for b in sets if b != a and bool((sets[b] <= sets[a]).all())
                     and bool((sets[b] < sets[a]).any())] for a in sets}
        result = {a: tuple(b for b in inner if not any(b in below[c] for c in inner))
                  for a, inner in below.items()}
        self._inside = (revision, result)
        return result

    def fill(self, raw):
        """Soft meet of the types holding over a part's own byte multiset."""
        counts = self.members.new_zeros(256)
        if raw:
            ids = torch.tensor(list(raw), device=counts.device, dtype=torch.long)
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
        presence = self.on_counts(counts, conjunctive=True)
        # Only a rows-by-property_width workspace; no primitive broadcast.
        return (1 - presence[:, None] * (1 - self.codes())).amin(0)

    @torch.no_grad()
    def define_word(self, row):
        """An immutable whole over exactly the reader's letter predicate."""
        from Meronomy import _letter
        value=torch.tensor([_letter(i) for i in range(256)],device=self.members.device,dtype=self.members.dtype)
        self.fixed_rows[row]=True
        self.fixed_members[row].copy_(value)
        self.members[row].copy_(value)
        self.observed[row]=True

    def coefficients(self, rows=None):
        if self.given_by:
            every = self._coefficients()
            done = {}
            def resolve(row):
                if row not in done:
                    done[row] = (torch.stack([resolve(o) for o in self.given_by[row]]).amin(0)
                                 if row in self.given_by else every[row])
                return done[row]
            every = torch.stack([resolve(r) if r in self.given_by else every[r] for r in range(len(every))])
            return every if rows is None else every.index_select(0, rows)
        return self._coefficients(rows)

    def _coefficients(self, rows=None):
        # Projection constrains storage after the optimizer step. The read
        # uses an STE at the rails: clamp's zero boundary derivative would
        # freeze an a-priori 0/1 definition against all later evidence.
        members = self.members if rows is None else self.members.index_select(0, rows)
        fixed=self.fixed_rows if rows is None else self.fixed_rows.index_select(0,rows)
        value=self.fixed_members if rows is None else self.fixed_members.index_select(0,rows)
        return torch.where(fixed[:,None],value,members + (members.clamp(0, 1) - members).detach())

    def forward(self, byte_ids, observed=None, *, rows=None):
        ids = byte_ids.to(device=self.members.device, dtype=torch.long).clamp(0, 255)
        result = self.coefficients(rows).t()[ids]
        if observed is not None:
            result = result * observed.to(result).unsqueeze(-1)
        return result

    def from_primitives(self, presences):
        """Union of weighted primitive presences, including diffuse inputs."""
        return union(torch.minimum(presences.unsqueeze(-2), self.coefficients()), dim=-1)

    def complement(self, byte_ids, observed):
        ids = byte_ids.to(device=self.members.device, dtype=torch.long).clamp(0, 255)
        known = (self.observed | (self.members.detach() != 0)).t()[ids]
        return (1 - self(byte_ids)) * known * observed.to(self.members).unsqueeze(-1)

    def evidence_on_counts(self, counts, *, rows=None):
        """Two observed poles of a property on each native run.

        A run witnesses a property or its complement only on known primitives;
        unseen bytes and padding cannot create a closed-world negative.
        """
        present = counts.to(self.members).unsqueeze(-2) > 0
        observed = self.observed if rows is None else self.observed.index_select(0, rows)
        members = self.members if rows is None else self.members.index_select(0, rows)
        known = observed | (members.detach() != 0)
        complete = ((~present | known).all(-1)
                    & (counts.sum(-1, keepdim=True) > 0))
        positive = self.on_counts(counts, rows=rows)
        negative = self.on_counts(counts, complement=True, rows=rows)
        return torch.stack((positive, negative), -1) * complete[..., None]

    def on_counts(self, counts, *, conjunctive=True, complement=False, rows=None):
        """Read a byte multiset by intersection or union of its properties.

        Counts keep the ordered witness separate while allowing each live
        run to use a bounded 256-column tensor in compiled code.
        """
        weights = self.coefficients(rows)
        if complement:
            weights = 1 - weights
        # Counts identify the observed support. Repeating a primitive does
        # not change its membership or make a long run less of a whole.
        result = _PropertySupportRead.apply(weights, counts.to(weights) > 0, conjunctive)
        return result * (counts.sum(-1, keepdim=True) > 0).to(result)

    def reverse(self, evidence):
        """Distribute property support over its learned primitive members.

        This is normalized attribution through a many-to-one read. Located
        conceptual evidence supplies its scope; radix activity decodes the
        attributed primitive rows as a best-effort reconstruction.
        """
        weights = self.coefficients()
        # Descent follows existing memberships. Unlike forward teaching,
        # it has no observed primitive on which to write a missing member.
        weights = torch.where(weights > 0, weights, 0.)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return evidence @ weights

    def reverse_poles(self, evidence):
        """Attribute either observed pole through the same primitive basis."""
        weights = self.coefficients()
        known = self.observed | (self.members.detach() != 0)
        pairs = torch.stack((weights, (1 - weights) * known), -1).clamp_min(0)
        pairs = pairs / pairs.sum(1, keepdim=True).clamp_min(1e-12)
        return torch.einsum('...rp,rvp->...v', evidence, pairs)

    @torch.no_grad()
    def teach(self, row, byte_ids, targets, *, steps=32, rate=.5):
        """Fit observed primitive memberships by the squared-error delta rule."""
        ids = torch.as_tensor(byte_ids, device=self.members.device, dtype=torch.long)
        targets = torch.as_tensor(targets, device=self.members.device, dtype=self.members.dtype)
        if ids.ndim != 1 or targets.shape != ids.shape:
            raise ValueError('property teaching requires paired byte observations and memberships')
        if ids.unique().numel() != ids.numel():
            raise ValueError('aggregate repeated primitive observations before teaching')
        if bool(self.fixed_rows[row]):
            if not torch.equal(targets,self.fixed_members[row,ids]):
                raise ValueError('a fixed property cannot be taught conflicting memberships')
            return 0., 0.
        self.observed[row, ids] = True
        before = (self.members[row, ids] - targets).square().mean()
        for _ in range(int(steps)):
            self.members[row, ids] += float(rate) * (targets - self.members[row, ids])
        self.project()
        after = (self.members[row, ids] - targets).square().mean()
        return float(before), float(after)

    @torch.no_grad()
    def project(self, optimizer=None):
        self.members.clamp_(0, 1)
        self.members.copy_(torch.where(self.fixed_rows[:,None],self.fixed_members,self.members))
        if optimizer is not None:
            velocity = optimizer.state.get(self.members, {}).get('momentum_buffer')
            if velocity is not None:
                current = self.members[:velocity.shape[0]]
                # SGD subtracts velocity: negative pushes out at the upper
                # bound, positive pushes out at the lower bound.
                outward = ((current >= 1) & (velocity < 0)) | ((current <= 0) & (velocity > 0))
                velocity.masked_fill_(outward, 0.)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old property checkpoints contain only tags. Keep the a-priori
        # learned initialization until their structural intake teaches tags.
        key = prefix + 'members'
        if key not in state_dict:
            state_dict[key] = self.members.detach().clone()
            # Dropped dictionary checkpoints have neither column. Preserve
            # the complete a-priori examples, including witnessed exclusions.
            state_dict.setdefault(prefix + 'observed', self.observed.detach().clone())
        elif prefix + 'observed' not in state_dict:
            state_dict[prefix + 'observed'] = state_dict[key] != 0
        for name in ('fixed_rows','fixed_members'):
            state_dict.setdefault(prefix+name,getattr(self,name).clone())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
