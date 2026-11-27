"""Item 1.4, §10.3: the measure of positive evidence at order zero.

Before item 1.4 a basic percept's row had two lanes, for and against, the
Not of order zero made them one signed value, for less against, and a
concept of order zero was a Sigma over those values. Item 1.4 gives a row
its for lane alone and lets a concept say what tells against it by a weight
below zero on another percept's row. The two are the same where every row's
against lane is a fixed weighting of the rows' for lanes: with ``a = H f``
at every reading, a concept ``v (f - a)`` is ``v (I - H) f``, signed weights
on positive rows. That was measured before the removal (the plan's §11);
since it, a row's second lane is empty and this module certifies that.

This module measures it on a model: it records what the
order-zero Sigma is given (:func:`record`), and for each row whose against
lane is ever lit finds the weighting (:func:`transfer`); and it records,
occurrence by occurrence, which way a row was told against and whether a
row read for was there to carry it (:func:`record_events`). It changes
nothing in the model.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Readings:
    """What the order-zero Sigma was given: each distinct reading once.

    ``lanes`` is ``[N, P, 2]`` over the percept rows ``rows`` (a row the
    reading did not hold reads nothing); ``count`` how often each was given.
    """
    rows: list = field(default_factory=list)
    seen: dict = field(default_factory=dict)
    calls: int = 0

    def add(self, lanes, inputs):
        """One call's ``lanes`` ``[P, B, E, 2]`` over the percepts ``inputs``."""
        self.calls += 1
        if not lanes.numel():
            return
        lanes = lanes.detach().to('cpu', torch.float64)
        flat = lanes.reshape(lanes.shape[0], -1, 2).transpose(0, 1)          # [B * E, P, 2]
        names = [int(row) for row in inputs]
        for reading in flat.unique(dim=0) if len(names) else flat[:0]:
            lit = reading.amax(-1) > 0
            key = tuple((names[p], *reading[p].tolist()) for p in lit.nonzero().flatten().tolist())
            if key:
                self.seen[key] = self.seen.get(key, 0) + 1
        for name in names:
            if name not in self.rows:
                self.rows.append(name)

    def table(self):
        """``[N, P, 2]`` over ``sorted(rows)``, and each reading's count."""
        rows = sorted(self.rows)
        at = {row: p for p, row in enumerate(rows)}
        lanes = torch.zeros(len(self.seen), len(rows), 2, dtype=torch.float64)
        for n, key in enumerate(self.seen):
            for row, positive, negative in key:
                lanes[n, at[row]] = torch.tensor((positive, negative), dtype=torch.float64)
        return rows, lanes, torch.tensor(list(self.seen.values()))


@contextmanager
def record(owner):
    """Record every reading given to ``owner``'s order-zero Sigma."""
    readings = Readings()
    read = owner.order_zero_field

    def recorded(lanes, inputs, rows):
        readings.add(lanes, inputs)
        return read(lanes, inputs, rows)
    object.__setattr__(owner, 'order_zero_field', recorded)
    try:
        yield readings
    finally:
        object.__delattr__(owner, 'order_zero_field')


@dataclass
class Carried:
    """How one row's against lane is carried by the for lanes."""
    row: int
    lit: int                 # readings where the against lane is above nothing
    carriers: tuple          # the rows whose for lanes never exceed it
    union: float             # the greatest error of their union
    weights: dict            # the fitted weighting, row -> weight
    fit: float               # its greatest error on every reading
    uncarried: int           # lit readings where no other row is lit for


def _fit(basis, target):
    return torch.linalg.lstsq(basis, target[:, None], driver='gelsd').solution[:, 0]


def transfer(readings, *, tolerance=1e-6):
    """Each row whose against lane is lit, as a weighting of the for lanes.

    Two measures for a row ``p``, each the greatest error over readings:

    - ``union``: the against lane as the union of the for lanes of its
      carriers, the other rows whose for lane never exceeds it (the form the
      first Pi joins against entries by);
    - ``fit``: as the best fixed weighting of every row's for lane (what a
      Sigma can hold), its own included, since a row is also told against
      where its own parts are found outside its whole. A fit through
      fewer distinct readings than rows shows only that those readings do
      not refute it.

    Returns ``(rows, H, report)`` with ``H`` ``[P, P]`` the fitted weighting
    (``a = H f``) and ``report`` a :class:`Carried` for each row.
    """
    rows, lanes, _count = readings.table() if isinstance(readings, Readings) else readings
    positive, negative = lanes[..., 0], lanes[..., 1]
    N, P = positive.shape
    H = torch.zeros(P, P, dtype=torch.float64)
    report = []
    for p in range(P):
        target = negative[:, p]
        if not bool((target > tolerance).any()):
            continue
        others = [q for q in range(P) if q != p]
        carriers = [q for q in others if bool((positive[:, q] <= target + tolerance).all())
                    and bool((positive[:, q] > tolerance).any())]
        union = 1 - (1 - positive[:, carriers]).prod(-1) if carriers else torch.zeros_like(target)
        weights = _fit(positive, target)
        H[p] = weights
        fit = (positive @ weights - target).abs().max()
        lit = target > tolerance
        alone = lit & ~(positive[:, others] > tolerance).any(-1)
        report.append(Carried(rows[p], int(lit.sum()), tuple(rows[q] for q in carriers),
                              float((union - target).abs().max()),
                              {rows[q]: round(float(w), 6) for q, w in enumerate(weights) if abs(float(w)) > tolerance},
                              float(fit), int(alone.sum())))
    return rows, H, report


def positive_weights(sigma, rows, H):
    """The order-zero Sigma's weights over positive rows: ``W (I - H)``,
    ``[concepts, P]`` over the percept rows ``rows``, from its entries as
    built (each ``v`` on a row's signed value)."""
    count = sigma.nnz
    at = {row: p for p, row in enumerate(rows)}
    W = torch.zeros(sigma.nOutput, len(rows), dtype=torch.float64)
    weights = sigma.w(occupied_only=True).detach().to('cpu', torch.float64)
    for concept, percept, weight in zip(sigma.entry_row[:count].tolist(),
                                        sigma.entry_input[:count].tolist(), weights.tolist()):
        if percept in at:
            W[concept, at[percept]] += weight
    return W, W @ (torch.eye(len(rows), dtype=torch.float64) - H)


def concept_error(sigma, readings, H=None):
    """The greatest difference, over every reading and concept, between a
    concept's sum as built (its entries on the rows' signed values) and its
    sum over positive rows alone by the weights of :func:`positive_weights`."""
    rows, lanes, _count = readings.table() if isinstance(readings, Readings) else readings
    if H is None:
        rows, H, _report = transfer((rows, lanes, _count))
    signed, positive = positive_weights(sigma, rows, H)
    built = (lanes[..., 0] - lanes[..., 1]).clamp(-1, 1) @ signed.t()
    return float((built - lanes[..., 0] @ positive.t()).abs().max()) if built.numel() else 0.


def summary(readings, sigma=None):
    """One model's measurement, as a dict that prints."""
    table = readings.table()
    rows, H, report = transfer(table)
    result = dict(calls=readings.calls, readings=len(table[1]), rows=len(rows),
                  against_rows=len(report),
                  union=max((c.union for c in report), default=0.),
                  fit=max((c.fit for c in report), default=0.),
                  uncarried=sum(c.uncarried for c in report))
    if sigma is not None:
        result['concepts'] = concept_error(sigma, table, H)
    return result, report


@dataclass
class Events:
    """What tells against a row, occurrence by occurrence, and whether a
    positive row is there to carry it.

    Before the removal a row was told against in four ways at the first Pi
    (the plan's §11.3 has their counts). After it, nothing is: ``counts`` is
    ``[carried, partly, uncarried]`` over every attended occurrence where a
    row's second lane is above nothing (another row is read for there to
    the degree of the against, to a lesser degree, or none is), and all
    three stay at zero.
    """
    counts: list = field(default_factory=lambda: [0, 0, 0])
    looks: int = 0

    def add(self, basic, bank, percepts, *, tolerance=1e-6):
        events = percepts.events.detach()
        if not events.numel() or not bank.rows:
            return
        self.looks += 1
        rows = [key[1] for key in percepts.keys]
        positive, negative = events[..., 0], events[..., 1]
        for r, b, a in (negative > tolerance).nonzero().tolist():
            if rows[r] in basic.conjunctions:
                continue
            others = torch.cat((positive[:r, b, a], positive[r + 1:, b, a]))
            best = float(others.max()) if others.numel() else 0.
            told = float(negative[r, b, a])
            self.counts[0 if best >= told - tolerance else 1 if best > tolerance else 2] += 1

    def table(self):
        return dict(zip(('carried', 'partly', 'uncarried'), self.counts))


@contextmanager
def record_events(basic):
    """Record what tells against each row at each occurrence of every look
    ``basic`` reads (:class:`Events`)."""
    found = Events()
    read = basic.read_bank

    def recorded(bank, *args, **kwargs):
        percepts = read(bank, *args, **kwargs)
        with torch.no_grad():
            found.add(basic, bank, percepts)
        return percepts
    object.__setattr__(basic, 'read_bank', recorded)
    try:
        yield found
    finally:
        object.__delattr__(basic, 'read_bank')
