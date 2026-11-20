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
concepts read the for vector and the against vector as two inputs, each with
weights of its own, ``y = tanh(x_for U + x_against A)``; ``A`` starts as
``-U``, where a concept reads the for vector less the against vector.
"""
import torch
from torch import nn

from Layers import EPS_LOG, membership_log, signed_lanes, SigmaLayer3


class PerceptPi(nn.Module):
    """The square, sparse Pi of the first layer, stored as compressed rows.

    Row ``r`` is a basic percept. Its entries are its members, addressed by a
    member id that names one key of the percept tower (a part run or a
    whole). Its for entries are joined by the product; its against entries
    (the parts that the stored forms nearest to holding it add) are joined
    by the union, any one present being enough. Every row also has its own reading on the diagonal; a basic
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
        self.register_buffer('against', torch.zeros(entries, dtype=torch.bool), persistent=False)
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

    def append(self, keys, weight=1.0, against=()):
        """Write the next row over its members at ``weight`` on both lanes,
        then its against entries."""
        ids = list(dict.fromkeys(self.member(key) for key in keys))
        opposed = [i for i in dict.fromkeys(self.member(key) for key in against) if i not in ids]
        row, start = self.n_rows, self.nnz
        if row + 1 >= len(self.crow):
            raise OverflowError('the first Pi has no room for another row')
        if start + len(ids) + len(opposed) > len(self.col):
            raise OverflowError('the first Pi has no room for another entry')
        end = start + len(ids) + len(opposed)
        if end > start:
            self.col[start:end] = torch.tensor(ids + opposed, dtype=torch.long, device=self.col.device)
            self.value[:, start:end] = float(weight)
            self.against[start:end] = False
            self.against[start + len(ids):end] = True
        self.crow[row + 1] = end
        self.n_rows = row + 1
        self._index_columns()
        return row

    def row_members(self, row):
        """A row's for entries."""
        start, end = int(self.crow[row]), int(self.crow[row + 1])
        return [self.keys[int(member)] for member, opposed in
                zip(self.col[start:end].tolist(), self.against[start:end].tolist()) if not opposed]

    def row_against(self, row):
        """A row's against entries."""
        start, end = int(self.crow[row]), int(self.crow[row + 1])
        return [self.keys[int(member)] for member, opposed in
                zip(self.col[start:end].tolist(), self.against[start:end].tolist()) if opposed]

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
            entries = [e for e in range(start, end) if not bool(self.against[e])]
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

    def opposed(self, memberships, rows, keys):
        """Each of ``rows``' against entries joined by the union on the for
        lane of their memberships, ``1 - prod (1 - m)``: any one present
        tells against the row. A row with none, or whose entries are not
        among ``keys``, reads zero. ``[len(rows), ...]``."""
        position = {key: index for index, key in enumerate(keys)}
        result = []
        for row in rows:
            found = [position[key] for key in self.row_against(row) if key in position]
            if not found:
                result.append(memberships.new_zeros(memberships.shape[1:-1]))
                continue
            index = torch.tensor(found, dtype=torch.long, device=memberships.device)
            present = memberships.index_select(0, index)[..., 0].clamp(0, 1)
            result.append(1 - (1 - present).prod(0))
        if not result:
            return memberships.new_zeros(0, *memberships.shape[1:-1])
        return torch.stack(result)

    def clear(self):
        self.crow.zero_()
        self.col.fill_(-1)
        self.value.zero_()
        self.against.zero_()
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
    ``reading`` ``[B, 1, 2D + 2C]`` is the concept's two columns of the
    order-zero Sigma, its for weights and its against weights (``D`` wide
    each), and the basic percept's read pair, what tells for it and what
    against it (``C`` wide each); ``place`` ``[B, N, D - C]`` is each
    location's own place. The percept enters at its net, as
    :func:`combine` has it, and each lane is read by its own column.
    """
    width = (reading.shape[-1] - 2 * content_width) // 2
    column = reading[..., :2 * width].unflatten(-1, (2, width))
    pair = reading[..., 2 * width:].unflatten(-1, (2, content_width))
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
    gaps = torch.log((1.0 - pair).clamp_min(EPS_LOG))
    net = presence[..., 0] - presence[..., 1]
    up, down = net.clamp_min(0)[..., None, None], (-net).clamp_min(0)[..., None, None]
    content = 1.0 - torch.exp(up * gaps + down * gaps.flip(-2))
    coordinates = presence[..., :, None] * place[..., None, :]
    combined = torch.cat((content, coordinates), -1)
    # x_for U + x_against A, as OrderZeroSigma reads it: the one input of
    # for less against, then what the against channel adds where A is not -U.
    net = ((combined[..., 0, :] - combined[..., 1, :]) * column[..., 0, :]).sum(-1)
    return torch.tanh(net + (combined[..., 1, :] * (column[..., 1, :] + column[..., 0, :])).sum(-1))


def combine(presence, codes, *, where=None):
    """The combined vector of each reading, on two lanes.

    ``presence`` is ``[P, ..., 2]``, each percept's Pi value on the two
    lanes; ``codes`` is ``[P, 2, W]``, what each percept's vector tells for
    it and what tells against it (``BasicPercepts.read_pair``), in
    ``[0, 1]``. A percept enters once, at its net, for less against: at a
    net above zero what tells for it joins the for lane and what tells
    against it the against lane; below zero the reverse; a percept read
    both for and against equally adds nothing. Each lane's content is
    ``1 - prod_p (1 - v_p)^{n_p}``. ``where``, ``[..., D]`` in ``[0, 1]``,
    is the reading's own place, taken at the reading's presence, the
    greatest of its percepts' values on the lane. A reading's time is not
    read. The result is ``[..., 2, W + D]``.
    """
    gaps = torch.log((1.0 - codes).clamp_min(EPS_LOG)).to(presence)
    net = presence[..., 0] - presence[..., 1]
    up, down = net.clamp_min(0), (-net).clamp_min(0)
    toward = torch.einsum('p...,pw->...w', up, gaps[:, 0]) + torch.einsum('p...,pw->...w', down, gaps[:, 1])
    away = torch.einsum('p...,pw->...w', up, gaps[:, 1]) + torch.einsum('p...,pw->...w', down, gaps[:, 0])
    content = 1.0 - torch.exp(torch.stack((toward, away), -2))
    parts = [content]
    present = presence.amax(0) if len(presence) else presence.new_zeros(presence.shape[1:])
    if where is not None:
        parts.append(present[..., :, None] * where.to(presence)[..., None, :])
    return torch.cat(parts, -1)


class OrderZeroSigma(SigmaLayer3):
    """The Sigma of order zero, dense, the perceptual width by the concepts
    of order zero, on two channels (rule 59): ``weight`` reads the for lane
    of the combined vector and ``against`` its against lane, each with
    weights of its own, ``y = tanh(x_for U + x_against A)``. A concept's
    value stays one signed number. Where ``A = -U`` a concept reads the for
    lane less the against lane, one input with one signed weight (rule 3),
    and every column is written so: a column starts at zero and is written
    when its concept is made, from its members' vectors, the same share
    added to ``U`` and taken from ``A``. What a concept then does with the
    two channels is learned."""

    def __init__(self, width, concepts, *, content_width):
        super().__init__(int(width), int(concepts), scale=0.0)
        self.against = nn.Parameter(-self.weight.detach().clone())
        self.content_width = int(content_width)
        # The rest is the reading's own place; its time is not read.
        self.where_width = int(width) - self.content_width

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A Sigma saved on one input of for less against has no against
        # weights: they are its for weights negated.
        if prefix + 'weight' in state_dict and prefix + 'against' not in state_dict:
            state_dict[prefix + 'against'] = -state_dict[prefix + 'weight']
        # A Sigma saved when it read the reading's time has those rows last,
        # after the content and the place: they are dropped.
        for name in ('weight', 'against'):
            saved = state_dict.get(prefix + name)
            if saved is not None and saved.ndim == 2 and saved.shape[0] > self.nInput:
                state_dict[prefix + name] = saved[:self.nInput]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def channels(self):
        """The for weights and the against weights, ``[2, width, concepts]``."""
        return torch.stack((self.weight, self.against))

    @torch.no_grad()
    def start_column(self, column, vectors, signs=None, against=None):
        """A column from its members: place weights at zero, then
        each member added (:meth:`add_member`). A member with sign -1 is a
        Not entry; ``against`` holds what tells against each member."""
        vectors = torch.as_tensor(vectors).to(self.weight).reshape(-1, self.content_width)
        signs = [1.] * len(vectors) if signs is None else [float(x) for x in signs]
        against = ([None] * len(vectors) if against is None else
                   torch.as_tensor(against).to(self.weight).reshape(-1, self.content_width))
        self.weight[:, column] = 0.
        self.against[:, column] = 0.
        for vector, sign, opposed in zip(vectors, signs, against):
            self.add_member(column, vector, sign, against=opposed)

    @staticmethod
    def member_column(positive, negative=None):
        """A member's share of a column (rule 54; Alec, §19.82, as amended in
        §23.3c.3): the mean of its sides, what tells for it over its own
        squared length and what tells against it over its own,
        ``(P / |P|^2 - N / |N|^2) / 2``. Each side is divided by its own
        count of contributions, so all that tells against takes the reading
        to zero however few its coordinates, and the member's own row, read
        with its own against lane, reads one (rule 21). A member with nothing
        against it gives its for side alone, ``P / |P|^2``."""
        def share(vector):
            size = vector.square().sum()
            return vector / size if size > 0 else torch.zeros_like(vector)
        if negative is None or not bool(negative.square().sum() > 0):
            return share(positive)
        return (share(positive) - share(negative)) / 2

    @torch.no_grad()
    def add_share(self, column, share):
        """Add ``share`` to a column's for weights and take it from its
        against weights, as one input of for less against reads it."""
        share = torch.as_tensor(share).to(self.weight)
        self.weight[:self.content_width, column] += share
        self.against[:self.content_width, column] -= share

    @torch.no_grad()
    def add_member(self, column, vector, sign=1., *, against=None):
        """Add ``sign`` times the member's share (:meth:`member_column`) to
        the column's content (:meth:`add_share`), so that the member's own
        row reads one before the squash; nothing is normalized afterwards.
        ``vector`` is what tells for the member, ``against`` what tells
        against it."""
        vector = torch.as_tensor(vector).to(self.weight).reshape(self.content_width)
        if against is not None:
            against = torch.as_tensor(against).to(self.weight).reshape(self.content_width)
        if vector.square().sum() > 0:
            self.add_share(column, float(sign) * self.member_column(vector, against))

    @staticmethod
    def _read(combined, weight, against):
        """``x_for U + x_against A``, written as rule 3's one input of for
        less against, ``(x_for - x_against) U``, and what the against
        channel adds where it has left ``-U``, ``x_against (A + U)``: the
        same sum, and at a column's start exactly the one-input read."""
        return ((combined[..., 0, :] - combined[..., 1, :]) @ weight
                + combined[..., 1, :] @ (against + weight))

    def forward(self, x):
        """The concepts of a combined vector ``[..., 2, width]``."""
        return torch.tanh(self._read(x, self.weight, self.against))

    def concepts(self, combined, columns):
        """Each column's signed value for each reading of ``combined``,
        ``[..., 2, width]``; a column of -1 reads zero."""
        columns = torch.as_tensor(columns, dtype=torch.long, device=self.weight.device)
        valid = columns >= 0
        index = columns.clamp_min(0)
        weight = self.weight.index_select(1, index) * valid
        against = self.against.index_select(1, index) * valid
        return torch.tanh(self._read(combined, weight, against))

    def reverse(self, y):
        """The input's two lanes by least squares through the for weights
        (:meth:`SigmaLayer3.reverse`), exact where the against weights are
        the for weights negated."""
        return super().reverse(y)
