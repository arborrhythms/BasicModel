"""The first layer: basic percepts as rows of a Pi, concepts as a dense Sigma.

A basic percept is one row of the first Pi over its parts and wholes, on two
lanes. Its value for a reading is the product of its members' memberships,
each to the power of its weight, on that lane alone. A row is admitted only
when MTM nominates it, and each row brings one concept (one column of the
Sigma's dense matrix).

For a reading the active percepts' vectors are combined on the gaps: for each
coordinate and lane, ``x = 1 - prod_p (1 - v_p)^{b_p}``, with ``b_p`` the Pi's
value for percept ``p`` on that lane. The reading's own place and time follow
the content coordinates, at the degree that the reading is present. The
concepts read the for vector less the against vector, ``y = tanh(x U)``.
"""
import torch
from torch import nn

from Layers import EPS_LOG, membership_log, signed_lanes, SigmaLayer3


class PerceptPi(nn.Module):
    """The square, sparse Pi of the first layer, stored as compressed rows.

    Row ``r`` is a basic percept. Its entries are its members, addressed by a
    member id that names one key of the percept tower (a part run or a
    whole). Every row also has its own reading on the diagonal; a basic
    percept is read only through its members, so its own reading is one and
    adds nothing to the product. The bias is zero. The storage is allocated
    at construction (``rows`` rows and ``entries`` entries) and grows into
    it; a column index is kept beside the rows.
    """

    def __init__(self, rows, entries):
        super().__init__()
        rows, entries = int(rows), int(entries)
        if rows < 1 or entries < 1:
            raise ValueError('the first Pi needs room for rows and entries')
        # The rows are written again from the basic percepts' occurrences
        # when a model is loaded, so the storage is not saved.
        self.register_buffer('crow', torch.zeros(rows + 1, dtype=torch.long), persistent=False)
        self.register_buffer('col', torch.full((entries,), -1, dtype=torch.long), persistent=False)
        self.register_buffer('value', torch.zeros(2, entries), persistent=False)
        self.register_buffer('ccol', torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer('crow_of', torch.zeros(0, dtype=torch.long), persistent=False)
        self.n_rows = 0
        self.members = {}
        self.keys = []

    @property
    def nnz(self):
        return int(self.crow[self.n_rows])

    def member(self, key):
        """The id of a key of the percept tower, made on first use."""
        found = self.members.get(key)
        if found is None:
            found = self.members[key] = len(self.keys)
            self.keys.append(key)
        return found

    def append(self, keys, weight=1.0):
        """Write the next row over its members at ``weight`` on both lanes."""
        ids = list(dict.fromkeys(self.member(key) for key in keys))
        row, start = self.n_rows, self.nnz
        if row + 1 >= len(self.crow):
            raise OverflowError('the first Pi has no room for another row')
        if start + len(ids) > len(self.col):
            raise OverflowError('the first Pi has no room for another entry')
        end = start + len(ids)
        if ids:
            self.col[start:end] = torch.tensor(ids, dtype=torch.long, device=self.col.device)
            self.value[:, start:end] = float(weight)
        self.crow[row + 1] = end
        self.n_rows = row + 1
        self._index_columns()
        return row

    def row_members(self, row):
        start, end = int(self.crow[row]), int(self.crow[row + 1])
        return [self.keys[int(member)] for member in self.col[start:end].tolist()]

    def rows_reading(self, key):
        """The rows that have ``key`` as an entry, by the column index."""
        member = self.members.get(key)
        if member is None:
            return []
        start, end = int(self.ccol[member]), int(self.ccol[member + 1])
        return self.crow_of[start:end].tolist()

    def _index_columns(self):
        nnz, count = self.nnz, len(self.keys)
        cols = self.col[:nnz]
        order = torch.argsort(cols, stable=True)
        rows = torch.repeat_interleave(
            torch.arange(self.n_rows, device=cols.device),
            self.crow[1:self.n_rows + 1] - self.crow[:self.n_rows])
        self.crow_of = rows[order]
        counts = torch.bincount(cols, minlength=count) if nnz else cols.new_zeros(count)
        self.ccol = torch.cat((cols.new_zeros(1), counts.cumsum(0)))

    def forward(self, memberships, rows, keys):
        """The value of each of ``rows`` on two lanes.

        ``memberships[k]`` is the membership of ``keys[k]``, with its two
        lanes on the last axis. A row whose member is not among ``keys``, or
        that has no member, reads nothing. A member at or below the floor
        annihilates the row, which then reads zero.
        """
        position = {key: index for index, key in enumerate(keys)}
        local, member, weight = [], [], []
        missing = set()
        for slot, row in enumerate(rows):
            start, end = int(self.crow[row]), int(self.crow[row + 1])
            if start == end:
                missing.add(slot)
            for entry in range(start, end):
                key = self.keys[int(self.col[entry])]
                if key not in position:
                    missing.add(slot)
                    continue
                local.append(slot)
                member.append(position[key])
                weight.append(entry)
        shape = (len(rows), *memberships.shape[1:])
        if not local:
            return memberships.new_zeros(shape)
        device = memberships.device
        local = torch.tensor(local, dtype=torch.long, device=device)
        member = torch.tensor(member, dtype=torch.long, device=device)
        weight = self.value[:, torch.tensor(weight, dtype=torch.long, device=self.value.device)]
        weight = weight.t().to(memberships).reshape(-1, *([1] * (memberships.ndim - 2)), 2)
        gathered = memberships.index_select(0, member)
        logs = membership_log(gathered) * weight
        summed = memberships.new_zeros(shape).index_add(0, local, logs)
        floor = (gathered <= EPS_LOG).to(memberships.dtype)
        absent = memberships.new_zeros(shape).index_add(0, local, floor) > 0
        value = torch.where(absent, torch.zeros_like(summed), torch.exp(summed))
        if missing:
            keep = torch.ones(len(rows), dtype=torch.bool, device=device)
            keep[list(missing)] = False
            value = value * keep.reshape(-1, *([1] * (value.ndim - 1)))
        return value

    def clear(self):
        self.crow.zero_()
        self.col.fill_(-1)
        self.value.zero_()
        self.n_rows, self.members, self.keys = 0, {}, []
        self._index_columns()


def read_word(evidence, mask, reading, place, *, content_width):
    """The order-zero concept of a word at each of its locations.

    ``evidence`` ``[B, N, K]`` holds the signed match of each of the word's
    basic percept's members (its run of parts and its wholes) at each
    location, and ``mask`` which are its members there. The basic percept's
    value on each lane is the Pi of its members: the product of their
    memberships on that lane, zero where one is at or below the floor; the
    run of parts stands at its presence on both lanes.
    ``reading`` ``[B, 1, D + C]`` is the concept's column of the order-zero
    Sigma (``D`` wide) and the basic percept's vector (``C`` wide);
    ``place`` ``[B, N, D - C]`` is each location's own place and time.
    """
    width = reading.shape[-1] - content_width
    column, code = reading[..., :width], reading[..., width:]
    lanes = torch.stack((evidence.clamp(0, 1), (-evidence).clamp(0, 1)), -1)
    # The first slot is the run of parts: the percept's presence, the same
    # on both lanes. The wholes keep their own lanes; with none held, the
    # against lane reads nothing.
    run = evidence[..., :1].clamp(0, 1)[..., None].expand(*evidence.shape[:-1], 1, 2)
    lanes = torch.cat((run, lanes[..., 1:, :]), -2)
    held = mask[..., None]
    logs = torch.where(held, membership_log(lanes), 0.).sum(-2)
    absent = (held & (lanes <= EPS_LOG)).any(-2) | ~held.any(-2)
    presence = torch.where(absent, 0., torch.exp(logs))
    whole = mask[..., 1:].any(-1)
    presence = torch.stack((presence[..., 0], torch.where(whole, presence[..., 1], 0.)), -1)
    gaps = torch.log((1.0 - code).clamp_min(EPS_LOG))
    content = 1.0 - torch.exp(presence[..., :, None] * gaps[..., None, :])
    coordinates = presence[..., :, None] * place[..., None, :]
    combined = torch.cat((content, coordinates), -1)
    return torch.tanh(((combined[..., 0, :] - combined[..., 1, :]) * column).sum(-1))


def combine(presence, codes, *, where=None, when=None):
    """The combined vector of each reading, on two lanes.

    ``presence`` is ``[P, ..., 2]``, each percept's Pi value on the two
    lanes; ``codes`` is ``[P, W]``, each percept's vector in ``[0, 1]``. The
    content coordinates are ``1 - prod_p (1 - v_p)^{b_p}``. ``where`` and
    ``when``, ``[..., D]`` in ``[0, 1]``, are the reading's own place and
    time; they are taken at the reading's presence, the greatest of its
    percepts' values on the lane. The result is ``[..., 2, W + D]``.
    """
    gaps = torch.log((1.0 - codes).clamp_min(EPS_LOG)).to(presence)
    content = 1.0 - torch.exp(torch.einsum('p...l,pw->...lw', presence, gaps))
    parts = [content]
    present = presence.amax(0) if len(presence) else presence.new_zeros(presence.shape[1:])
    for coordinates in (where, when):
        if coordinates is not None:
            parts.append(present[..., :, None] * coordinates.to(presence)[..., None, :])
    return torch.cat(parts, -1)


class OrderZeroSigma(SigmaLayer3):
    """The Sigma of order zero: ``U`` dense, the perceptual width by the
    concepts of order zero. A column starts at zero and is written when its
    concept is made, from its members' vectors."""

    def __init__(self, width, concepts, *, content_width):
        super().__init__(int(width), int(concepts), scale=0.0)
        self.content_width = int(content_width)
        # Of the rest: the reading's own place, then its own time.
        self.where_width = self.when_width = (int(width) - self.content_width) // 2

    @torch.no_grad()
    def start_column(self, column, vectors, signs=None):
        """A column from its members: place and time weights at zero, then
        each member added (:meth:`add_member`). A member with sign -1 is a
        Not entry."""
        vectors = torch.as_tensor(vectors).to(self.weight).reshape(-1, self.content_width)
        signs = [1.] * len(vectors) if signs is None else [float(x) for x in signs]
        self.weight[:, column] = 0.
        for vector, sign in zip(vectors, signs):
            self.add_member(column, vector, sign)

    @torch.no_grad()
    def add_member(self, column, vector, sign=1.):
        """Add ``sign * v / |v|^2`` to the column's content, so that the
        member's own row reads one before the squash; nothing is
        normalized afterwards."""
        vector = torch.as_tensor(vector).to(self.weight).reshape(self.content_width)
        size = vector.square().sum()
        if size > 0:
            self.weight[:self.content_width, column] += float(sign) * vector / size

    def concepts(self, combined, columns):
        """Each column's signed value for each reading of ``combined``,
        ``[..., 2, width]``; a column of -1 reads zero."""
        columns = torch.as_tensor(columns, dtype=torch.long, device=self.weight.device)
        valid = columns >= 0
        weight = self.weight.index_select(1, columns.clamp_min(0)) * valid
        return torch.tanh((combined[..., 0, :] - combined[..., 1, :]) @ weight)
