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
    counts = torch.nn.functional.one_hot(byte_ids.long().clamp(0, 255), 256).to(torch.get_default_dtype())
    if observed is not None:
        counts = counts * observed[..., None].to(counts)
    prefix = torch.cat((counts.new_zeros(B, 1, 256), counts.cumsum(1)), 1)
    lo = spans[..., :1].expand(-1, -1, 256)
    hi = spans[..., 1:].expand(-1, -1, 256)
    return prefix.gather(1, hi) - prefix.gather(1, lo)


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


class PrimitiveProperties(nn.Module):
    """One bounded coefficient per property and byte, with no tag reader.

    A byte read is the sparse evaluation of a union over one-hot primitives.
    Dense primitive mixtures use that same union. Teaching uses the delta
    rule for squared error on observed primitives; ordinary gradients can
    subsequently refine the same coefficients.
    """

    def __init__(self, rows, *, device=None, dtype=None):
        super().__init__()
        self.members = nn.Parameter(torch.zeros(int(rows), 256, device=device, dtype=dtype))
        # A coefficient of zero can mean either a witnessed exclusion or an
        # unwritten primitive. Only the former supplies counterevidence.
        self.register_buffer('observed', torch.zeros(int(rows), 256, device=device, dtype=torch.bool))
        self.register_buffer('fixed_rows', torch.zeros(int(rows), device=device, dtype=torch.bool))
        self.register_buffer('fixed_members', torch.zeros(int(rows), 256, device=device, dtype=dtype))

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
    def project(self):
        self.members.clamp_(0, 1)
        self.members.copy_(torch.where(self.fixed_rows[:,None],self.fixed_members,self.members))

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
