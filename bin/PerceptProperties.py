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
    positions = torch.arange(L, device=byte_ids.device)[None, None]
    spans = spans.to(byte_ids.device)
    mask = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
    if observed is not None:
        mask = mask & observed[:, None].to(device=mask.device, dtype=torch.bool)
    counts = torch.zeros(B, spans.shape[1], 256, device=byte_ids.device,
                         dtype=torch.get_default_dtype())
    return counts.scatter_add(2, byte_ids.long().clamp(0, 255)[:, None].expand_as(mask),
                              mask.to(counts))


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

    def coefficients(self):
        # Projection constrains storage after the optimizer step. The read
        # uses an STE at the rails: clamp's zero boundary derivative would
        # freeze an a-priori 0/1 definition against all later evidence.
        return self.members + (self.members.clamp(0, 1) - self.members).detach()

    def forward(self, byte_ids, observed=None):
        ids = byte_ids.to(device=self.members.device, dtype=torch.long).clamp(0, 255)
        result = self.coefficients().t()[ids]
        if observed is not None:
            result = result * observed.to(result).unsqueeze(-1)
        return result

    def from_primitives(self, presences):
        """Union of weighted primitive presences, including diffuse inputs."""
        return union(torch.minimum(presences.unsqueeze(-2), self.coefficients()), dim=-1)

    def complement(self, byte_ids, observed):
        return (1 - self(byte_ids)) * observed.to(self.members).unsqueeze(-1)

    def on_counts(self, counts, *, conjunctive=True, complement=False):
        """Read a byte multiset by intersection or union of its properties.

        Counts keep the ordered witness separate while allowing each live
        run to use a bounded 256-column tensor in compiled code.
        """
        weights = self.coefficients()
        if complement:
            weights = 1 - weights
        # Counts identify the observed support. Repeating a primitive does
        # not change its membership or make a long run less of a whole.
        values = torch.where(counts.to(weights).unsqueeze(-2) > 0, weights,
                             1. if conjunctive else 0.)
        result = values.amin(-1) if conjunctive else values.amax(-1)
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

    @torch.no_grad()
    def teach(self, row, byte_ids, targets, *, steps=32, rate=.5):
        """Fit observed primitive memberships by the squared-error delta rule."""
        ids = torch.as_tensor(byte_ids, device=self.members.device, dtype=torch.long)
        targets = torch.as_tensor(targets, device=self.members.device, dtype=self.members.dtype)
        if ids.ndim != 1 or targets.shape != ids.shape:
            raise ValueError('property teaching requires paired byte observations and memberships')
        if ids.unique().numel() != ids.numel():
            raise ValueError('aggregate repeated primitive observations before teaching')
        before = (self.members[row, ids] - targets).square().mean()
        for _ in range(int(steps)):
            self.members[row, ids] += float(rate) * (targets - self.members[row, ids])
        self.project()
        after = (self.members[row, ids] - targets).square().mean()
        return float(before), float(after)

    @torch.no_grad()
    def project(self):
        self.members.clamp_(0, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old property checkpoints contain only tags. Keep the a-priori
        # learned initialization until their structural intake teaches tags.
        key = prefix + 'members'
        if key not in state_dict:
            state_dict[key] = self.members.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
