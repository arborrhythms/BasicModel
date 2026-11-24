"""At each look, what is expected from the readings outside its extent: the
vertical spread on the structure (:mod:`ConceptSpread`). A look's extent is
the readings it holds, ``[B, readings]``; what it brings against what was
expected is one gain field (:func:`Expectation.gain_field`), whose excess
and omission are the surprise ``attend`` reads."""
import torch
from torch.nn import functional as F

import ConceptSpread
from CandidateGrowth import read_streams
from FieldProjection import inventory
from SentenceField import inventory as field_rows


def locate_evidence(values, rows, width):
    """Copy each stream's readings to its addressed inventory, skipping empty seats."""
    batch, readings, slots, lanes = values.shape
    # Each assignment reads the last source slot at that address. Slot zero
    # of the padded values supplies silence for an absent inventory row.
    source = torch.arange(1, slots + 1, device=rows.device).expand_as(rows)
    at = rows.new_zeros(batch, width + 1).scatter_reduce(
        1, torch.where(rows >= 0, rows, width), source, 'amax')
    gather = at[:, None, :width, None].expand(batch, readings, width, lanes)
    return F.pad(values, (0, 0, 1, 0)).gather(2, gather)


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
        self.higher_rows = allocated[self.basic_count:]
        self.forecast_rows = torch.cat((torch.arange(self.order_zero_width, device=spans.device),
                                        self.higher_rows))
        self.index = spans.new_full((owner.nVectors + 1,), -1)
        self.index[allocated] = torch.arange(len(allocated), device=spans.device)
        # All order-zero addresses are forecast, including currently silent rows.
        self.address = torch.cat((allocated[:self.basic_count], torch.arange(
            self.order_zero_width, self.order_zero_width + len(allocated) - self.basic_count, device=spans.device)))
        self.width = self.order_zero_width + len(allocated) - self.basic_count
        evidence = self._located(opened)
        readings = evidence.shape[1]
        self.observations = evidence[:, :, :self.basic_count]
        everything = torch.zeros(len(evidence), readings, dtype=torch.bool, device=spans.device)
        width = min(readings, spans.shape[1])
        everything[:, :width] = (spans[..., 1] > spans[..., 0])[:, :width]
        self.latest = self.surprise(self.read(evidence, everything), self.forecast(everything), everything)

    def _located(self, field):
        """Put each native reading at its allocated row, preserving both lanes."""
        rows = field_rows(self.owner, field)
        values = field.evidence.detach().permute(1, 2, 0, 3)
        if rows.ndim == 1:
            rows = rows[:, None].expand(-1, len(values))
        return locate_evidence(values, self.index[rows.T], len(self.rows))

    @staticmethod
    def _pool(values, mask):
        # The observed knowing is the greatest reading on each lane.
        return torch.where(mask[..., None, None], values[:, None], 0.).amax(2)

    def _extent(self, extent):
        """``extent`` ``[B, readings]`` as wide as the observations' readings."""
        width = self.observations.shape[1]
        return F.pad(extent, (0, max(0, width - extent.shape[1])))[:, :width]

    def _outside(self, extent, looked=None):
        """Each stream's look (``live``, ``[B]``: ``looked``, or where its
        extent holds a reading) and what the readings outside its extent
        observed, at the order-zero addresses. A look whose extent holds
        nothing is expected from everything observed."""
        extent = self._extent(extent)
        live = extent.any(-1) if looked is None else looked.to(extent.device)
        outside = ~extent
        knowing = self._pool(self.observations, outside[:, None])[:, 0][live]
        inputs = knowing.new_zeros(len(knowing), self.order_zero_width, 2)
        inputs[:, self.address[:self.basic_count]] = knowing
        return live, inputs

    def _compact(self, value):
        """The inventory's values ``[n, V, 2]`` at this forecast's positions."""
        return torch.cat((value[:, :self.order_zero_width], value[:, self.higher_rows]), 1)

    @torch.no_grad()
    def forecast(self, extent, looked=None):
        """What each stream's look is expected to hold, ``[B, width, 2]``,
        from the readings outside its extent only."""
        live, inputs = self._outside(extent, looked)
        result = self.observations.new_zeros(*live.shape, self.width, 2)
        if not len(inputs):
            return result
        result[live] = self._compact(read_streams(
            self.owner, lambda: ConceptSpread.expected(self.owner, inputs), axis=0, live=live))
        return result

    def inventory_rows(self):
        """The inventory row of each forecast position."""
        return self.forecast_rows

    def stage_costs(self, field, perception, extent, looked=None):
        """Score both teaching stages against the same observations outside this look."""
        live, inputs = self._outside(extent, looked)
        empty = self.observations.new_zeros(live.shape[0])
        if not len(inputs):
            return empty, empty
        batch = live.nonzero()[:, 0]
        evidence = self._located(field)
        actual = self.read(evidence, extent).index_select(0, batch)
        arrived = actual.new_zeros(len(actual), int(self.owner.nVectors), 2)
        arrived[:, self.inventory_rows()] = actual
        cost = read_streams(self.owner, lambda: ConceptSpread.look_cost(self.owner, inputs, arrived),
                            axis=0, live=live)
        concepts = empty.index_add(0, batch, cost)

        # The final occurrence supplies a repeated percept address, as in the
        # sequential assignment. Stage the two index arrays in one transfer.
        held = {} if perception is None else {
            key[1]: i for i, key in enumerate(perception.keys) if key[0] == 'bp'}
        if not held:
            return concepts, empty
        values = perception.values.detach().amax(2)
        rows, sources = torch.tensor(list(held.items()), device=values.device, dtype=torch.long).T
        observed = values.new_zeros(len(batch), max(held) + 1, 2)
        observed[:, rows] = values.index_select(0, sources).index_select(1, batch).transpose(0, 1)
        cost = read_streams(self.owner, lambda: ConceptSpread.percept_cost(self.owner, inputs, observed),
                            axis=0, live=live)
        return concepts, empty.index_add(0, batch, cost)

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
    def surprise(actual, forecast, extent, looked=None):
        """The look's excess and omission, ``[B, 2]``: one gain field with
        the look's activity as its attention and its forecast as its
        expected."""
        from Expectation import gain_field
        attention = (extent.any(-1) if looked is None else looked).to(actual)[:, None]
        field = gain_field(actual.clamp(0., 1.), forecast.clamp(0., 1.), attention=attention)
        return torch.stack((field.excess.sum((-1, -2)), field.omission.sum((-1, -2))), -1)

    @torch.no_grad()
    def observe(self, field, extent, forecast, looked=None):
        evidence = self._located(field)
        pair = self.surprise(self.read(evidence, extent), forecast, extent, looked)
        live = extent.any(-1) if looked is None else looked
        self.latest = torch.where(live[:, None], pair, self.latest)
        # Every forecast selects entire reading positions, then takes the
        # greatest evidence on each lane. Maxima at the same position can
        # be folded as they arrive, once the current forecast has been graded.
        width = evidence.shape[1]
        observed = torch.where(
            F.pad(extent, (0, max(0, width - extent.shape[1])))[:, :width, None, None],
            evidence[:, :, :self.basic_count], 0.)
        old = self.observations
        if old.shape[1] < width:
            old = F.pad(old, (0, 0, 0, 0, 0, width - old.shape[1]), value=-torch.inf)
        if width < old.shape[1]:
            observed = F.pad(observed, (0, 0, 0, 0, 0, old.shape[1] - width), value=-torch.inf)
        self.observations = torch.maximum(old, observed)
        return pair


class OccurrenceCredit:
    """Predict basic percept activity from the query, then grade the observed region."""

    def __init__(self, owner, bank):
        from Salience import rows_read
        self.owner, self.bank = owner, bank
        self.terms = []
        # Keep the open sensory reading independent of the request that will
        # select a region. Channels are the basic percepts' two evidence lanes.
        with torch.no_grad():
            self.observed = rows_read(owner.basic_percepts, bank).permute(1, 2, 0, 3).flatten(2)

    def fork(self):
        # The sensory target belongs to the shared opening. Each path grades
        # its own future requests against the fixations it actually made.
        from copy import copy
        other = copy(self)
        other.terms = []
        return other

    def forecast(self, request):
        """The same descending content request and native place query used to look."""
        from RequestRead import spatial_match
        from Salience import concepts_asked, rows_asked
        rows = self.owner.nVectors
        concepts = concepts_asked(self.owner, request[:rows])
        what = rows_asked(self.owner.order_zero, self.bank.rows, concepts).transpose(0, 1).flatten(1)
        where = spatial_match(self.bank, request[rows:, :, 0].T, self.bank.span[..., 0])
        return what.clamp(0., 1.)[:, None] * where.sigmoid()[..., None]

    def observe(self, anticipated, index):
        from Layers import Error
        bank = self.bank
        from BasicPercepts import reading_scope
        if not bank.where.shape[1] or not anticipated.shape[-1]:
            return
        batch = torch.arange(len(index), device=index.device)
        scoped = reading_scope(bank, bank.where.shape[1])[batch, index.clamp_min(0)]
        seen = (index[:, None] >= 0) & scoped & (bank.activity[..., 0] > 0)
        live = seen.any(1)
        if not bool(live.any()):
            return
        # The fixation limits *both* sides. Its action probability is absent
        # from this cost. Overlapping events at a byte share the same place.
        starts = torch.where(seen, bank.span[..., 0], 0)
        width = int(starts.max()) + 1
        addresses = starts[..., None].expand_as(anticipated)
        def located(values):
            grid = values.new_zeros(len(values), width, values.shape[-1])
            return grid.scatter_reduce(1, addresses, values * seen[..., None], 'amax')
        put, wanted = located(anticipated), located(self.observed)
        shared, mass = Error.pyramid_match(put, wanted)
        error = (put.sum((1, 2)) + mass - 2 * shared).clamp_min(0.)
        self.terms.append((error, mass, live))

    def for_rows(self, rows):
        from copy import copy
        selected = copy(self)
        selected.terms = [(error, mass, live & rows) for error, mass, live in self.terms]
        return selected

    def register(self, registry):
        for error, mass, live in self.terms:
            registry.error('attention.next_observation', error, mass, mask=live,
                           category='expectation', objective='attention')
