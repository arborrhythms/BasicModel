"""At each look, what is expected from the readings outside its extent: the
vertical spread on the structure (:mod:`ConceptSpread`)."""
import torch
from torch.nn import functional as F

import ConceptSpread
from FieldProjection import inventory
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

    def _outside(self, spans):
        live = spans[..., 1] > spans[..., 0]
        lo, hi = self.locations.unbind(-1)
        outside = ((hi > lo)[:, None] & ((hi[:, None] <= spans[..., 0, None]) |
                                           (lo[:, None] >= spans[..., 1, None])))
        knowing = self._pool(self.observations, outside)[live]
        inputs = knowing.new_zeros(len(knowing), self.order_zero_width, 2)
        inputs[:, self.address[:self.basic_count]] = knowing
        return live, inputs

    def _compact(self, value):
        """The inventory's values ``[n, V, 2]`` at this forecast's positions."""
        allocated = self.address.new_tensor(self.rows)
        return torch.cat((value[:, :self.order_zero_width], value[:, allocated[self.basic_count:]]), 1)

    @torch.no_grad()
    def forecast(self, spans):
        """Forecast each nonempty query from disjoint readings only."""
        live, inputs = self._outside(spans)
        result = self.observations.new_zeros(*live.shape, self.width, 2)
        if not bool(live.any()):
            return result
        result[live] = self._compact(ConceptSpread.expected(self.owner, inputs))
        return result

    def inventory_rows(self):
        """The inventory row of each forecast position."""
        allocated = self.address.new_tensor(self.rows)
        return torch.cat((torch.arange(self.order_zero_width, device=allocated.device),
                          allocated[self.basic_count:]))

    def stage_cost(self, field, spans):
        """The concepts' stage at a look, ``[B]``: what each concept expected
        from the readings outside the look's extent, with the gradient of
        its own entries alone, against what the look brought."""
        live, inputs = self._outside(spans)
        error = self.observations.new_zeros(live.shape[0])
        if not bool(live.any()):
            return error
        evidence = self._located(field)
        actual = self.read(evidence, field.where, spans)[live]
        arrived = actual.new_zeros(len(actual), int(self.owner.nVectors), 2)
        arrived[:, self.inventory_rows()] = actual
        cost = ConceptSpread.look_cost(self.owner, inputs, arrived)
        batch = live.nonzero()[:, 0]
        return error.index_add(0, batch, cost)

    def percept_stage_cost(self, perception, spans):
        """The basic percepts' stage at a look, ``[B]``: what each basic
        percept expected from the readings outside the look's extent, with
        the gradient of its own entries of the first Pi alone, against the
        basic percepts the look brought (``perception``, the look's read)."""
        live, inputs = self._outside(spans)
        error = self.observations.new_zeros(live.shape[0])
        if perception is None or not bool(live.any()):
            return error
        held = [(i, key[1]) for i, key in enumerate(perception.keys) if key[0] == 'bp']
        if not held:
            return error
        width = max(row for _, row in held) + 1
        values = perception.values.detach().amax(2)
        arrived = values.new_zeros(values.shape[1], width, 2)
        for i, row in held:
            arrived[:, row] = values[i]
        batch = live.nonzero()[:, 0]
        cost = ConceptSpread.percept_cost(self.owner, inputs, arrived.index_select(0, batch))
        return error.index_add(0, batch, cost)

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
        live = self.spans[..., 1] > self.spans[..., 0]
        same_word = live & (self.spans == spans).all(-1)
        self.latest = torch.where(same_word[..., None], pair, self.latest)
        self.locations = torch.cat((self.locations, field.where), 1)
        self.observations = torch.cat((self.observations, evidence[:, :, :self.basic_count]), 1)
        return pair

    def for_brackets(self, intervals):
        return contained(intervals, self.spans).to(self.latest) @ self.latest
