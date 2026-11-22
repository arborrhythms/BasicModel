"""A sparse Sigma that grows and is pruned by its penalty, against the full
matrix (sigma-pi §21, rule 67; certificate 58). In tests only.

Both forms read ``y = tanh(V s)``: ``s`` ``[R]`` is each row's signed value
(the Not's value, for less against; zero where the row is absent) and
``V[c, r]`` the learned number of concept ``c``'s entry on row ``r`` (its
size the Sigma's weight, its side of zero the option it reads, as
``SparseSigmaLDU.value``; ``v s`` is ``|v|`` times the chosen option).
Both are taught by the same steps on the same readings: the squared error
on the concepts an error reaches, a plain gradient step of rate ``lr``,
then the proximal step of the penalty, ``v <- sign(v) max(|v| - lr lam, 0)``
on every entry but the protected ones (a concept's own row and its declared
members), which are never thinned or removed.

Under that step an entry of the full matrix that stands at zero leaves zero
exactly when the gradient gathered on it over the step's readings exceeds
``lam`` in size, and an entry returns to zero exactly when the penalty
brings it there. The sparse form keeps only its entries and, for the pairs
of a present row and a concept an error reaches, the gradient its missing
entry would have had, gathered over the step's readings in a table of fixed
size that drops the weakest; it writes the entry when what has gathered
exceeds ``lam`` and removes an entry the penalty brings to zero.
"""
import torch


def soft(value, threshold):
    """The proximal step of an L1 penalty: shrink towards zero by ``threshold``."""
    return torch.sign(value) * (value.abs() - threshold).clamp_min(0)


def concept_error(S, V, T, M):
    """The error reaching each concept in each reading, ``[N, C]``: of the
    mean squared error over the readings, through the tanh."""
    Y = torch.tanh(S @ V.t())
    return M * (Y - T) * (1 - Y * Y) / S.shape[0]


class Full:
    """Every concept by every row, dense."""

    def __init__(self, concepts, rows, start, protected, lam, lr):
        self.V = torch.zeros(concepts, rows, dtype=torch.float64)
        for (c, r), v in start.items():
            self.V[c, r] = v
        self.free = torch.ones_like(self.V, dtype=torch.bool)
        for c, r in protected:
            self.free[c, r] = False
        self.lam, self.lr = float(lam), float(lr)

    def step(self, S, T, M):
        D = concept_error(S, self.V, T, M)
        V = self.V - self.lr * (D.t() @ S)
        self.V = torch.where(self.free, soft(V, self.lr * self.lam), V)

    def read(self, S):
        return torch.tanh(S @ self.V.t())

    def entries(self):
        return {(int(c), int(r)): float(self.V[c, r]) for c, r in self.V.nonzero().tolist()}


class Sparse:
    """Its entries alone, and a table of fixed size of the gradients its
    missing entries would have had."""

    def __init__(self, concepts, rows, start, protected, lam, lr, table):
        self.concepts, self.rows = int(concepts), int(rows)
        self.values = dict(start)
        self.protected = set(protected)
        self.lam, self.lr, self.table = float(lam), float(lr), int(table)
        self.written = self.removed = self.dropped = self.needed = 0

    def dense(self):
        V = torch.zeros(self.concepts, self.rows, dtype=torch.float64)
        for (c, r), v in self.values.items():
            V[c, r] = v
        return V

    def read(self, S):
        return torch.tanh(S @ self.dense().t())

    def _gather(self, gathered, key, amount):
        if key in gathered:
            gathered[key] += amount
            return
        if len(gathered) >= self.table:
            weakest = min(gathered, key=lambda k: abs(gathered[k]))
            if abs(gathered[weakest]) >= abs(amount):
                self.dropped += 1
                return
            del gathered[weakest]
            self.dropped += 1
        gathered[key] = amount

    def step(self, S, T, M):
        # The forward reads the entries alone; the error is the full form's
        # error on the same values.
        D = concept_error(S, self.dense(), T, M)
        # The stored entries' gradients.
        grads = {key: 0. for key in self.values}
        gathered, seen = {}, set()
        for n in range(S.shape[0]):
            present = S[n].nonzero().flatten().tolist()
            reached = D[n].nonzero().flatten().tolist()
            for c in reached:
                d = float(D[n, c])
                for r in present:
                    amount = d * float(S[n, r])
                    if (c, r) in grads:
                        grads[(c, r)] += amount
                    else:
                        seen.add((c, r))
                        self._gather(gathered, (c, r), amount)
        self.needed = max(self.needed, len(seen))
        threshold = self.lr * self.lam
        for key, g in grads.items():
            v = self.values[key] - self.lr * g
            if key in self.protected:
                self.values[key] = v
                continue
            v = float(soft(torch.tensor(v, dtype=torch.float64), threshold))
            if v == 0.:
                del self.values[key]
                self.removed += 1
            else:
                self.values[key] = v
        # A missing entry is written when what has gathered exceeds the
        # penalty: it leaves zero by the same step as the full matrix's.
        for key, g in gathered.items():
            if abs(g) > self.lam:
                self.values[key] = float(soft(torch.tensor(-self.lr * g, dtype=torch.float64), threshold))
                self.written += 1

    def entries(self):
        return dict(self.values)


def run(S, T, M, *, concepts, start, protected, lam, lr, steps, table):
    """The two forms taught by the same steps on the same readings
    (``S`` ``[N, R]``, targets ``T`` and masks ``M`` ``[N, C]``)."""
    S, T, M = (x.to(torch.float64) for x in (S, T, M))
    full = Full(concepts, S.shape[1], start, protected, lam, lr)
    sparse = Sparse(concepts, S.shape[1], start, protected, lam, lr, table)
    for _ in range(steps):
        full.step(S, T, M)
        sparse.step(S, T, M)
    return full, sparse


def compare(full, sparse, S, *, small, effect):
    """What certificate 58 reads: the entries of each form, the largest
    entry of the full matrix that the sparse form lacks, the greatest
    difference of their weights, and the entries of the full matrix that
    are small and whose removal changes a concept's reading by more than
    ``effect``, with whether the sparse form holds each."""
    S = S.to(torch.float64)
    held_full, held_sparse = full.entries(), sparse.entries()
    lacks = [abs(v) for k, v in held_full.items() if k not in held_sparse]
    extra = [k for k in held_sparse if k not in held_full]
    keys = set(held_full) | set(held_sparse)
    difference = max((abs(held_full.get(k, 0.) - held_sparse.get(k, 0.)) for k in keys), default=0.)
    base = full.read(S)
    largest = max((abs(v) for v in held_full.values()), default=0.)
    telling = []
    for (c, r), v in held_full.items():
        if abs(v) >= small * largest:
            continue
        V = full.V.clone()
        V[c, r] = 0.
        change = float((torch.tanh(S @ V.t()) - base).abs().max())
        if change > effect:
            telling.append(dict(entry=(c, r), weight=v, change=change, sparse=(c, r) in held_sparse))
    return dict(full=len(held_full), sparse=len(held_sparse), lacks=max(lacks, default=0.),
                lacking=len(lacks), extra=len(extra), difference=difference,
                small_telling=telling, written=sparse.written, removed=sparse.removed,
                dropped=sparse.dropped, needed=sparse.needed, table=sparse.table)


# Certificate 23's learning cases. Rows: base percepts, then conjunction rows
# (present when all their members are), then the outcome's own row, which
# is absent in every reading: the outcome's concept starts on it at one and
# must read the cues' rows to be expected from them.

def case_readings(cases, rows):
    """``(S, T, M, start, protected)`` for one taught concept, the last."""
    R = len(rows) + 1
    S = torch.zeros(len(cases), R)
    for n, present in enumerate(cases):
        for r in present:
            S[n, r] = 1.
    T = torch.zeros(len(cases), R)
    M = torch.zeros(len(cases), R)
    T[:, R - 1] = torch.tensor(list(cases.values()))
    M[:, R - 1] = 1.
    start = {(c, c): 1. for c in range(R)}
    return S, T, M, start, set(start)


def certificate_23_cases():
    """Negative patterning, conditioned inhibition, and XOR with a row for
    each pair, as certificate 23 states them."""
    A, B, X, AB, AX = range(5)
    rows = [(), (), (), (A, B), (A, X)]
    words = 4
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    xor_rows = [()] * words + [p for p in pairs]
    return {
        'negative patterning': ({(A,): 1., (B,): 1., (A, B, AB): 0.}, rows),
        'conditioned inhibition': ({(A,): 1., (A, X, AX): 0., (B,): 1.}, rows),
        'xor with a row for the pair': ({(0, 2, 4): 0., (0, 3, 5): 1., (1, 2, 6): 1., (1, 3, 7): 0.},
                                        xor_rows),
    }


def omission_readings(fields, own, declared):
    """The lesson's readings with the reconstruction's error, in the form of
    omission: from each distinct reading of the order-zero rows (a field's
    rows, each at its strongest place), one word's
    row hidden at a time; the target of every concept whose row is in the
    reading is its value on the whole reading (each concept on its own row
    at one), so the concept of the hidden word must be read from the rows
    left. ``fields``: the percept fields the order-zero Sigma read; ``own``:
    concept index -> its own row (the basic percept whose nomination
    brought it); ``declared``: the declared members ``(concept, row)``.
    Returns ``(S, T, M, start, protected, rows, concepts)`` on compact
    indices."""
    from Layers import NotLayer3
    row_of_concept = dict(own)
    rows = sorted(set(row_of_concept.values()) | {r for _, r in declared})
    concepts = sorted(row_of_concept)
    rpos = {r: i for i, r in enumerate(rows)}
    cpos = {c: i for i, c in enumerate(concepts)}
    distinct = set()
    for field in fields:
        if not len(field.keys):
            continue
        signed = NotLayer3.value(field.values.detach().to(torch.float64))  # [P, B, E]
        # A field's places (E) each hold one word; the reading of the
        # sentence is each row at its strongest place, for each stream (B).
        strongest = signed.abs().argmax(-1, keepdim=True)
        flat = signed.gather(-1, strongest).squeeze(-1)  # [P, B]
        for column in range(flat.shape[1]):
            reading = tuple(sorted((rpos[int(k[1])], round(float(flat[p, column]), 9))
                                   for p, k in enumerate(field.keys)
                                   if k[0] == 'bp' and int(k[1]) in rpos and float(flat[p, column]) != 0.))
            if len(reading) >= 2:
                distinct.add(reading)
    S, T, M = [], [], []
    for reading in sorted(distinct):
        whole = torch.zeros(len(rows), dtype=torch.float64)
        for r, v in reading:
            whole[r] = v
        target = torch.zeros(len(concepts), dtype=torch.float64)
        mask = torch.zeros(len(concepts), dtype=torch.float64)
        for c, r in row_of_concept.items():
            if whole[rpos[r]] != 0:
                target[cpos[c]] = torch.tanh(whole[rpos[r]])
                mask[cpos[c]] = 1.
        for hidden, _ in reading:
            s = whole.clone()
            s[hidden] = 0.
            S.append(s), T.append(target), M.append(mask)
    if not S:
        return None
    start = {(cpos[c], rpos[r]): 1. for c, r in row_of_concept.items()}
    protected = set(start) | {(cpos[c], rpos[r]) for c, r in declared if c in cpos}
    return (torch.stack(S), torch.stack(T), torch.stack(M), start, protected, rows, concepts)
