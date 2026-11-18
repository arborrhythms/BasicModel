"""Leave one spatial extent out of the omission-trained concept forecast."""
import torch
from torch.nn import functional as F

import ConceptSpread
from FieldProjection import inventory, close
from SentenceField import inventory as field_rows


def contained(outer, inner):
    """For each query interval, identify the nonempty readings it contains."""
    lo, hi = inner.unbind(-1)
    return ((hi > lo)[:, None] & (outer[..., 0, None] <= lo[:, None]) &
            (hi[:, None] <= outer[..., 1, None]))


def difference(actual, forecast):
    residual = actual - forecast
    return torch.stack((residual.clamp_min(0).sum((-1, -2)),
                        (-residual).clamp_min(0).sum((-1, -2))), -1)


class LookForecast:
    """A path's located observations and most recent surprise at each word extent."""
    @torch.no_grad()
    def __init__(self, owner, opened, spans):
        self.owner = owner
        self.order_zero_width = owner._order_caps()[0]
        self.rows = inventory(owner)
        self.basic_count = sum(row < self.order_zero_width for row in self.rows)
        self.spans = spans
        allocated = spans.new_tensor(self.rows)
        self.index = spans.new_full((owner.nVectors + 1,), -1)
        self.index[allocated] = torch.arange(len(allocated), device=spans.device)
        # All order-zero addresses are forecast, including currently silent rows.
        self.address = torch.cat((allocated[:self.basic_count], torch.arange(
            self.order_zero_width, self.order_zero_width + len(allocated) - self.basic_count, device=spans.device)))
        self.width = self.order_zero_width + len(allocated) - self.basic_count
        self.locations = opened.where
        evidence = self._located(opened)
        self.observations = evidence[:, :, :self.basic_count]
        actual = self.read(evidence, self.locations, self.spans)
        self.latest = difference(actual, self.forecast(self.spans))

    def _located(self, field):
        """Put each native reading at its allocated row, preserving both lanes."""
        rows = field_rows(self.owner, field)
        values = field.evidence.detach().permute(1, 2, 0, 3)
        result = values.new_zeros(*values.shape[:2], len(self.rows), 2)
        if rows.ndim == 1:
            rows = rows[:, None].expand(-1, len(values))
        for batch, column in enumerate(rows.T):
            indices = self.index[column]
            valid = indices >= 0
            result[batch, :, indices[valid]] = values[batch, :, valid]
        return result

    @staticmethod
    def _pool(values, mask):
        # The observed knowing is the greatest reading on each lane.
        return torch.where(mask[..., None, None], values[:, None], 0.).amax(2)

    @torch.no_grad()
    def forecast(self, spans):
        """Forecast each nonempty query from disjoint readings only."""
        live = spans[..., 1] > spans[..., 0]
        present = live.nonzero()
        result = self.observations.new_zeros(*live.shape, self.width, 2)
        if not len(present):
            return result
        lo, hi = self.locations.unbind(-1)
        outside = ((hi > lo)[:, None] & ((hi[:, None] <= spans[..., 0, None]) |
                                           (lo[:, None] >= spans[..., 1, None])))
        knowing = self._pool(self.observations, outside)[live]
        inputs = knowing.new_zeros(len(knowing), self.order_zero_width, 2)
        inputs[:, self.address[:self.basic_count]] = knowing
        prediction = ConceptSpread.expected(self.owner, inputs)
        full = F.pad(prediction, (0, 0, 0, self.width - self.order_zero_width))
        folded = close(self.owner, self.rows, full[:, self.address].transpose(0, 1))
        full[:, self.address] = folded.transpose(0, 1)
        result[live] = full
        return result

    def read(self, evidence, locations, spans):
        """Read both lanes within each span at the same addresses as a forecast."""
        values = self._pool(evidence, contained(spans, locations))
        result = values.new_zeros(*values.shape[:2], self.width, 2)
        result[:, :, self.address] = values
        return result

    @torch.no_grad()
    def observe(self, field, spans, forecast):
        evidence = self._located(field)
        actual = self.read(evidence, field.where, spans)
        pair = difference(actual, forecast)
        covered = contained(spans, self.spans)
        self.latest = torch.where(covered[:, 0, :, None], pair, self.latest)
        self.locations = torch.cat((self.locations, field.where), 1)
        self.observations = torch.cat((self.observations, evidence[:, :, :self.basic_count]), 1)
        return pair

    def for_brackets(self, intervals):
        return contained(intervals, self.spans).to(self.latest) @ self.latest
