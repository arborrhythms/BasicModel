"""At each look, what is expected from the readings outside its extent: the
vertical spread on the structure (:mod:`ConceptSpread`). A look's extent is
the readings it holds, ``[B, readings]``; what it brings against what was
expected is one gain field (:func:`Expectation.gain_field`), whose excess
and omission are the surprise ``attend`` reads."""
import torch
from torch.nn import functional as F

import ConceptSpread
from FieldProjection import inventory
from SentenceField import inventory as field_rows


class LookForecast:
    """A path's observations, each at the reading it was read at, and the
    most recent surprise."""
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
        evidence = self._located(opened)
        readings = evidence.shape[1]
        self.slots = torch.arange(readings, device=spans.device)
        self.observations = evidence[:, :, :self.basic_count]
        everything = torch.zeros(len(evidence), readings, dtype=torch.bool, device=spans.device)
        everything[:, :spans.shape[1]] = spans[..., 1] > spans[..., 0]
        self.latest = self.surprise(self.read(evidence, everything), self.forecast(everything), everything)

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

    def _extent(self, extent):
        """``extent`` ``[B, readings]`` as wide as the observations' readings."""
        width = int(self.slots.max()) + 1
        return F.pad(extent, (0, max(0, width - extent.shape[1])))[:, :width]

    def _outside(self, extent):
        """Each stream's look (``live``, ``[B]``) and what the readings
        outside its extent observed, at the order-zero addresses."""
        extent = self._extent(extent)
        live = extent.any(-1)
        outside = ~extent.gather(1, self.slots[None].expand(len(extent), -1))
        knowing = self._pool(self.observations, outside[:, None])[:, 0][live]
        inputs = knowing.new_zeros(len(knowing), self.order_zero_width, 2)
        inputs[:, self.address[:self.basic_count]] = knowing
        return live, inputs

    def _compact(self, value):
        """The inventory's values ``[n, V, 2]`` at this forecast's positions."""
        allocated = self.address.new_tensor(self.rows)
        return torch.cat((value[:, :self.order_zero_width], value[:, allocated[self.basic_count:]]), 1)

    @torch.no_grad()
    def forecast(self, extent):
        """What each stream's look is expected to hold, ``[B, width, 2]``,
        from the readings outside its extent only."""
        live, inputs = self._outside(extent)
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

    def stage_cost(self, field, extent):
        """The concepts' stage at a look, ``[B]``: what each concept expected
        from the readings outside the look's extent, with the gradient of
        its own entries alone, against what the look brought."""
        live, inputs = self._outside(extent)
        error = self.observations.new_zeros(live.shape[0])
        if not bool(live.any()):
            return error
        evidence = self._located(field)
        actual = self.read(evidence, extent)[live]
        arrived = actual.new_zeros(len(actual), int(self.owner.nVectors), 2)
        arrived[:, self.inventory_rows()] = actual
        cost = ConceptSpread.look_cost(self.owner, inputs, arrived)
        batch = live.nonzero()[:, 0]
        return error.index_add(0, batch, cost)

    def percept_stage_cost(self, perception, extent):
        """The basic percepts' stage at a look, ``[B]``: what each basic
        percept expected from the readings outside the look's extent, with
        the gradient of its own entries of the first Pi alone, against the
        basic percepts the look brought (``perception``, the look's read)."""
        live, inputs = self._outside(extent)
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

    def read(self, evidence, extent):
        """Both lanes over the readings of each stream's ``extent``, at the
        same addresses as a forecast, ``[B, width, 2]``."""
        width = evidence.shape[1]
        extent = F.pad(extent, (0, max(0, width - extent.shape[1])))[:, :width]
        values = self._pool(evidence, extent[:, None])[:, 0]
        result = values.new_zeros(values.shape[0], self.width, 2)
        result[:, self.address] = values
        return result

    @staticmethod
    def surprise(actual, forecast, extent):
        """The look's excess and omission, ``[B, 2]``: one gain field with
        the look's activity as its attention and its forecast as its
        expected."""
        from Expectation import gain_field
        attention = extent.any(-1).to(actual)[:, None]
        field = gain_field(actual.clamp(0., 1.), forecast.clamp(0., 1.), attention=attention)
        return torch.stack((field.excess.sum((-1, -2)), field.omission.sum((-1, -2))), -1)

    @torch.no_grad()
    def observe(self, field, extent, forecast):
        evidence = self._located(field)
        pair = self.surprise(self.read(evidence, extent), forecast, extent)
        live = extent.any(-1)
        self.latest = torch.where(live[:, None], pair, self.latest)
        self.slots = torch.cat((self.slots, torch.arange(evidence.shape[1], device=self.slots.device)))
        self.observations = torch.cat((self.observations, torch.where(
            F.pad(extent, (0, max(0, evidence.shape[1] - extent.shape[1])))[:, :evidence.shape[1], None, None],
            evidence[:, :, :self.basic_count], 0.)), 1)
        return pair
