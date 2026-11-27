"""The first layer: basic percepts as rows of a Pi, concepts as a sparse Sigma.

A basic percept is one row of the first Pi over its parts and wholes. Its
value for a reading is the product of its members' memberships, each to the
power of its weight. Evidence at order zero is positive (item 1.4): a row is
read for, to a degree, and nothing is read against it; the second lane of a
row's value is kept and is empty. A row is admitted only when MTM nominates
it, and it brings a concept, which starts with one entry, on that row, at
one. The Pi alone reads the percepts' codes.

The concepts of order zero are a ``SparseSigmaLDU`` over the rows' values.
There is no Not between the first Pi and this Sigma. What tells against a
concept is another row found at the same place, by an entry below zero that
the Sigma learns on a link MTM writes at zero when that row is nominated
(``ConceptualSpace.link_holders``). Any number of concepts may read one row.
"""
import torch
from torch import nn

from Layers import EPS_LOG, membership_log


class SparsePiLayer3(nn.Module):
    """The first Pi: the sparse Pi of basic percepts, stored as compressed
    rows. It is read only by the basic percepts (a look's table); the
    symbols' Pi is a ``Layers.SparsePiLDU``.

    Row ``r`` is a basic percept and its members are keys of the percept
    tower (a part run or a whole). Its entries are its members, addressed by a member id that names one
    key, joined by the product. Every row also has its own reading on the diagonal; a basic
    percept is read only through its members, so its own reading is one and
    adds nothing to the product. The bias is zero. The storage is allocated
    at construction (``rows`` rows and ``entries`` entries) and grows into
    it; a column index is kept beside the rows.
    """

    def __init__(self, rows, entries):
        super().__init__()
        rows, entries = int(rows), int(entries)
        if rows < 1 or entries < 1:
            raise ValueError('the sparse Pi needs room for rows and entries')
        # The rows are written again from the basic percepts' occurrences
        # when a model is loaded, so the storage is not saved.
        self.register_buffer('crow', torch.zeros(rows + 1, dtype=torch.long), persistent=False)
        self.register_buffer('col', torch.full((entries,), -1, dtype=torch.long), persistent=False)
        self.register_buffer('value', torch.zeros(2, entries), persistent=False)
        self.register_buffer('ccol', torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer('crow_of', torch.zeros(0, dtype=torch.long), persistent=False)
        self.n_rows = 0
        self._nnz = 0
        self.members = {}
        self.keys = []

    @property
    def nnz(self):
        return self._nnz

    def member(self, key):
        """The id of a key of the percept tower, made on first use."""
        found = self.members.get(key)
        if found is None:
            found = self.members[key] = len(self.keys)
            self.keys.append(key)
        return found

    @staticmethod
    def _row_ids(member, keys):
        return list(dict.fromkeys(map(member, keys)))

    def append(self, keys, weight=1.0, weights=None):
        """Write the next row over its members at ``weight`` on both lanes
        (or each at its own of ``weights``)."""
        ids = self._row_ids(self.member, keys)
        row, start = self.n_rows, self.nnz
        if row + 1 >= len(self.crow):
            raise OverflowError('the first Pi has no room for another row')
        if start + len(ids) > len(self.col):
            raise OverflowError('the first Pi has no room for another entry')
        end = start + len(ids)
        if end > start:
            self.col[start:end] = torch.tensor(ids, dtype=torch.long, device=self.col.device)
            self.value[:, start:end] = float(weight)
            if weights is not None:
                self.value[:, start:end] = torch.tensor(
                    [float(w) for w in weights][:len(ids)], dtype=self.value.dtype, device=self.value.device)
        self.crow[row + 1] = end
        self.n_rows = row + 1
        self._nnz = end
        # The column index is made again when it is next read.
        self._columns_stale = True
        return row

    def refresh_members(self, rows):
        """Refresh an entire changed membership topology in the existing CSR
        storage: ``rows`` is each row's member keys."""
        # Compare one snapshot. An unchanged topology keeps its current weights.
        count = self.n_rows
        snapshot = torch.cat((self.crow[:count + 1], self.col[:self.nnz])).tolist()
        ends, entries = snapshot[:count + 1], snapshot[count + 1:]
        if len(rows) == count:
            for row, keys in enumerate(rows):
                if [self.keys[i] for i in entries[ends[row]:ends[row + 1]]] != list(dict.fromkeys(keys)):
                    break
            else:
                return

        # Use the same encounter order as append. A repeated key has one
        # entry on a row.
        members, columns, ends = {}, [], [0]
        def member(key):
            return members.setdefault(key, len(members))
        for keys in rows:
            columns.extend(self._row_ids(member, keys))
            ends.append(len(columns))
        if len(ends) > len(self.crow) or len(columns) > len(self.col):
            raise OverflowError('the first Pi has no room for the refreshed topology')
        topology = self.col.new_tensor(ends + columns)
        self.clear()
        self.members, self.keys = members, list(members)
        self.n_rows, self._nnz = len(rows), len(columns)
        self.crow[:len(ends)].copy_(topology[:len(ends)])
        self.col[:self.nnz].copy_(topology[len(ends):])
        self.value[:, :self.nnz].fill_(1.)
        self._columns_stale = True

    def row_members(self, row):
        """A row's entries."""
        start, end = int(self.crow[row]), int(self.crow[row + 1])
        return [self.keys[int(member)] for member in self.col[start:end].tolist()]

    def rows_reading(self, key):
        """The rows that have ``key`` as an entry, by the column index."""
        member = self.members.get(key)
        if member is None:
            return []
        if self.__dict__.get('_columns_stale'):
            self._index_columns()
        start, end = int(self.ccol[member]), int(self.ccol[member + 1])
        return self.crow_of[start:end].tolist()

    def _index_columns(self):
        self._columns_stale = False
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
            entries = list(range(start, end))
            if not entries:
                missing.add(slot)
            for entry in entries:
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
        self._nnz = 0
        self._index_columns()


def read_word(evidence, mask, value):
    """The order-zero concept of a word at each of its locations: the Pi of
    its row, the Not, the Sigma.

    ``evidence`` ``[B, N, K]`` holds the signed match of each of the word's
    basic percept's members (its run of parts and its wholes) at each
    location, and ``mask`` which are its members there. The Pi: the basic
    percept's value on each lane is the product of its members' memberships
    on that lane, zero where one is at or below the floor; the run of parts
    stands at its presence on both lanes. The Not: the row's two options,
    its value and its not-value. The Sigma: ``value`` ``[B, 1, 1]`` is the
    learned number of one concept's entry on the row, the concept the row
    brought (the word's own concept): its side of zero is the option it
    reads, its size the Sigma's weight. Other concepts that read the row
    are read by the order-zero field, not here.
    """
    from Layers import NotLayer3, _EntryRead
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
    options = NotLayer3.options(presence)
    return torch.tanh(_EntryRead.apply(value[..., 0].to(options).expand_as(options[..., 0]), options))
