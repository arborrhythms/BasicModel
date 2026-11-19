"""Compare attended rows, fields and all three matrix gradients with §64."""
import torch
from pytest import MonkeyPatch
import Spaces
from Spaces import _concept_alloc_of
from Layers import SparseLayer
from basic_fixtures import feature
from test_cs_sparse_weights import _cs, _mint_row
from test_concept_memberships import binary_features

def refbind(self, inventory_rows, *, paired=True):
    """Gather a differentiable sparse view into a transient attended field.

    Inventory addresses stay in the persistent definition. Missing source
    concepts contribute zero evidence; the standing poles keep their rails.
    """
    addresses = [int(r) for r in inventory_rows]
    lookup = {r: i for i, r in enumerate(addresses) if r >= 0}
    size = len(addresses)
    view = SparseLayer(2 * (size + 1) if paired else self.nInput,
                       size, nonlinear=False, device=self._device())
    selected, rows, cols, missing = [], [], [], []
    for pos, (row, col) in enumerate(zip(self._rows, self._cols)):
        target = lookup.get(row)
        if target is None:
            continue
        absent = False
        if paired:
            source, pole = col % (self.nOutput + 1), col // (self.nOutput + 1)
            source = size if source == self.nOutput else lookup.get(source)
            if source is None:
                # Both poles of an unattended source are unknown. Keep
                # its exponent and mask membership on both fold passes.
                absent = True
                source, pole = size, 1
            col = source + pole * (size + 1)
        selected.append(pos)
        rows.append(target)
        cols.append(col)
        missing.append(absent)
    view._rows, view._cols = rows, cols
    view._index = {pair: i for i, pair in enumerate(zip(rows, cols))}
    if self.values is not None:
        index = torch.tensor(selected, device=self.values.device, dtype=torch.long)
        object.__setattr__(view, 'values', self.values.index_select(0, index))
    view._inventory_edges = selected
    view._missing_sources = torch.tensor(missing, dtype=torch.bool)
    view._paired_sources = paired
    view._feature_sources = getattr(self, '_feature_sources', False)
    # Pair original sources, not the attended addresses: several missing
    # sources can share the view's silent bias slot without being opposites.
    source_columns = getattr(self, '_fold_source_columns', self._cols)
    view._fold_source_columns = tuple(source_columns[pos] for pos in selected)
    view._fold_source_offset = getattr(self, '_fold_source_offset', self.nInput // 2)
    return view

def ref_bind_attended_concepts(self, required, matrix):
    """Gather every definition reachable from active native percept poles."""
    store = _concept_alloc_of(self).layer()
    active = (required.amax((2, 3)) > 0 if required.shape[2] else
              required.new_zeros(required.shape[:2], dtype=torch.bool))
    dependencies, standing = [], set()
    for parent in (store, store.conjunctive):
        if parent is store.conjunctive and not self.conceptual_pi:
            continue
        for (target, column), pos in parent._index.items():
            # Zero-weight candidate edges still need their inputs to
            # receive learning credit; the fold keeps their value zero.
            source = column % (store.nOutput + 1)
            if source < store.nOutput:
                dependencies.append((target, source))
            elif column == store.nOutput:
                standing.add(target)
    orders = max(2, int(self._symbolic_order)) + 1
    by_batch = []
    retired = _concept_alloc_of(self).retired
    for batch in range(required.shape[1]):
        reached = standing | {row for edge, row in enumerate(matrix._rows)
            if bool(active[edge, batch])}
        while True:
            parents = {target for target, source in dependencies if source in reached}
            if parents <= reached:
                break
            reached |= parents
        groups = [[] for _ in range(orders)]
        for row in sorted(reached):
            if self.concept_id_at_row(row) in retired:
                continue
            order = self._row_order(row)
            if order < orders:
                groups[order].append(row)
        by_batch.append(groups)
    caps = [max([min(self._order_caps()[0], int(self.outputShape[0])) if k == 0 else 0]
                + [len(groups[k]) for groups in by_batch]) for k in range(orders)]
    addresses = [[row for k, group in enumerate(groups)
                  for row in (group + [-1] * (caps[k] - len(group)))]
                 for groups in by_batch]
    rows = torch.tensor(addresses, dtype=torch.long, device=required.device).t()
    ids = [[self.concept_id_at_row(row) if row >= 0 else None for row in batch]
           for batch in addresses]
    ids = torch.tensor([[cid if cid is not None else -1 for cid in batch] for batch in ids],
                       dtype=torch.long, device=required.device).t()
    if rows.shape[1] == 1 or torch.equal(rows, rows[:, :1].expand_as(rows)):
        rows, ids = rows[:, 0], ids[:, 0]
    object.__setattr__(self, '_cs_field_caps', tuple(caps))
    object.__setattr__(self, '_cs_field_rows', rows)
    object.__setattr__(self, '_cs_field_concept_ids', ids)

def ref_compose_order0(self, field, *, inventory_rows=None, publish=True):
    """Intersect and union predicates of the shared subject before encoding.

    Native percept poles have already been pooled inside the bracket.
    Dependencies are evaluated once in topological order, preserving
    sparse identity; an extent's both corner is only a diagnosis.
    """
    store = self._concept_field_store(inventory_rows)
    n0, B, E, _ = field.shape
    S = store.nOutput
    working = torch.cat((field, field.new_zeros(S - n0, B, E, 2)))
    raw = working.clone()
    deps = {r: set() for r in range(n0)}
    for matrix in (store, store.conjunctive):
        if matrix is store.conjunctive and not self.conceptual_pi:
            continue
        for (r, c), edge in matrix._index.items():
            source = c % (S + 1)
            if (r < n0 and source < n0 and float(matrix.values[edge].detach()) > 0
                    and not bool(matrix._missing_sources[edge])):
                deps[r].add(source)
    pi_rows = self._concept_part_rows(store.conjunctive, field.device)
    pending, done = set(range(n0)), set()
    while pending:
        ready = sorted(r for r in pending if deps[r] <= done)
        if not ready:
            raise ValueError('cyclic order-0 concept definitions')
        for row in ready:
            positive, negative = working.unbind(-1)
            source = torch.cat((positive.reshape(S, B * E), field.new_ones(1, B * E),
                                negative.reshape(S, B * E), field.new_zeros(1, B * E)))
            own = field[row:row+1].reshape(1, B * E, 2)
            own_mask = store.feature_defined[row:row+1]
            if self.conceptual_pi and bool(pi_rows[row]):
                own = torch.stack((
                    store.conjunctive.fold_presence(source, conjunctive=True, start=row, end=row+1,
                        own=own[..., 0], own_mask=own_mask, own_known=own.any(-1)),
                    store.conjunctive.fold_presence(source, conjunctive=True, dual=True, start=row, end=row+1,
                        own=own[..., 1], own_mask=own_mask, own_known=own.any(-1))), -1)
                own_mask = torch.ones_like(own_mask)
            result = torch.stack((
                store.fold_presence(source, start=row, end=row+1,
                    own=own[..., 0], own_mask=own_mask, own_known=own.any(-1)),
                store.fold_presence(source, dual=True, start=row, end=row+1,
                    own=own[..., 1], own_mask=own_mask, own_known=own.any(-1))), -1).reshape(1, B, E, 2)
            inventory = (self._field_inventory_rows() if inventory_rows is None else inventory_rows)
            symbol = self.concept_id_at_row(int(inventory[row]))
            if symbol is not None and any(role == 'candidate_literal' for role, _
                    in _concept_alloc_of(self).store_of(symbol).constituents(symbol)):
                # A located literal's field support initializes its form;
                # that context cannot assert a different native sequence.
                result = result * (field[row:row+1, ..., :1] > 0)
            index = torch.tensor([row], device=field.device)
            raw = raw.index_copy(0, index, result)
            working = working.index_copy(0, index, result)
            done.add(row)
            pending.remove(row)
    if publish:
        object.__setattr__(self, '_cs_order0_raw', raw[:n0].detach())
    return working[:n0]

def chain_fixture(*, pi=False, negated=False):
    """Order-0 rows with native parts, then sigma and pi edges between them."""
    cs = _cs(nS=64, order=1)
    rows = []
    for i in range(6):
        row = _mint_row(cs, 0, 10000 + i)
        feature(cs, row, 'ps', 1000 + i, 1. if i % 2 == 0 else .6)
        rows.append(row)
    feature(cs, rows[1], 'ps', (1002, 1003), .5)
    composite = _mint_row(cs, 0, 10100)
    cs.add_concept_edge(composite, rows[0], .8)
    cs.add_concept_edge(composite, rows[1], .4)
    deeper = _mint_row(cs, 0, 10101)
    cs.add_concept_edge(deeper, composite, .9)
    cs.add_concept_edge(deeper, rows[2], .3)
    if negated:
        cs.add_concept_edge(deeper, rows[3], .5, negated=True)
    if pi:
        object.__setattr__(cs, 'conceptual_pi', True)
        cs.add_concept_edge(composite, rows[4], .7, conjunctive=True)
        cs.add_concept_edge(composite, rows[5], .2, conjunctive=True)
    ids = torch.tensor([[1000, 1001, 1002, 1003, 1000, 1001, 1002, 1003],
                        [1004, 1005, 1002, 1003, 1004, 1005, 1000, 1001]])
    starts = torch.arange(8)
    positions = torch.stack((starts, starts + 1), -1)[None].expand(2, -1, -1)
    raw = torch.full((2, 8), 65)
    extents = torch.tensor([[[0, 4], [4, 8]], [[0, 8], [2, 6]]])
    return cs, (ids, positions, None, raw, positions), extents


def literal_fixture():
    cs = _cs()
    feature(cs, 0, 'ps', (7, 8), 1.)
    feature(cs, 1, 'ps', 9, 1.)
    raw = torch.full((1, 8), 65)
    ids = torch.tensor([[7, 8, 9, 9, 9, 9, 9, 9]])
    starts = torch.arange(8)
    positions = torch.stack((starts, starts + 1), -1)[None]
    return cs, (ids, positions, None, raw, positions), torch.tensor([[[0, 4], [4, 8]]])


def matrices(cs):
    store = Spaces._concept_alloc_of(cs).layer(0)
    return [m for m in (store, store.conjunctive, store.features) if m.values is not None]


def grads_of(cs, run):
    for m in matrices(cs):
        m.values.grad = None
    out = run()
    out.sum().backward()
    return [None if m.values.grad is None else m.values.grad.clone() for m in matrices(cs)]


def assert_same_read(cs, percepts, extents):
    def read():
        return cs.cs_read_memberships(percepts, extents)
    with MonkeyPatch.context() as patch:
        patch.setattr(SparseLayer, 'bind', refbind)
        patch.setattr(Spaces.ConceptualSpace, '_bind_attended_concepts', ref_bind_attended_concepts)
        patch.setattr(Spaces.ConceptualSpace, '_compose_order0', ref_compose_order0)
        old = read().detach().clone()
        raw = cs._cs_order0_raw.clone()
        caps, rows, ids = cs._cs_field_caps, cs._cs_field_rows.clone(), cs._cs_field_concept_ids.clone()
        old_grads = grads_of(cs, read)
    new = read()
    assert torch.equal(old, new)
    assert torch.equal(raw, cs._cs_order0_raw)
    assert cs._cs_field_caps == caps
    assert torch.equal(rows, cs._cs_field_rows) and torch.equal(ids, cs._cs_field_concept_ids)
    for expected, actual in zip(old_grads, grads_of(cs, read)):
        assert (expected is None and actual is None) or torch.equal(expected, actual)
    return new

def test_bind_selects_the_same_topology_paired_and_unpaired():
    cs, percepts, extents = chain_fixture(pi=True, negated=True)
    store = Spaces._concept_alloc_of(cs).layer(0)
    inventory = cs._field_inventory_rows()
    rows = torch.cat((inventory[:5], torch.tensor([-1, 3, 3, 70])))  # padding, a repeat, out of range
    for matrix in (store, store.conjunctive, store.features):
        for paired in (True, False):
            if matrix is store.features and paired:
                continue
            old = refbind(matrix, rows.tolist(), paired=paired)
            new = matrix.bind(rows, paired=paired)
            assert new._rows == old._rows and new._cols == old._cols
            assert new._index == old._index and new.nnz == old.nnz
            assert new.nInput == old.nInput and new.nOutput == old.nOutput
            assert list(new._inventory_edges) == old._inventory_edges
            assert torch.equal(new._missing_sources.cpu(), old._missing_sources)
            if matrix.values is not None:
                assert torch.equal(new.values, old.values)


def test_property_features_read_identically():
    cs = _cs()
    native, extents = binary_features(cs, torch.tensor([[49, 49], [48, 48], [48, 49], [49, 48]]))
    assert_same_read(cs, native, extents)


def test_literal_groups_read_identically():
    cs, percepts, extents = literal_fixture()
    assert_same_read(cs, percepts, extents)


def test_sigma_chains_with_per_batch_inventories_read_identically():
    cs, percepts, extents = chain_fixture()
    field = assert_same_read(cs, percepts, extents)
    assert field.shape[1] == 2 and cs._cs_field_rows.ndim == 2
    # The composed rows were reached through the closure and folded at a
    # later level than their parts: their field is not the pooled input.
    deeper = cs._csw_concept_row(0, 10101)
    positions = (cs._cs_field_rows == deeper).nonzero()
    assert len(positions), 'the two-level composite must be in the attended field'
    assert any(bool(field[int(p), int(b)].any()) for p, b in positions.tolist())


def test_pi_and_negated_definitions_read_identically():
    cs, percepts, extents = chain_fixture(pi=True, negated=True)
    assert_same_read(cs, percepts, extents)
    with torch.no_grad():
        for m in matrices(cs):
            m.values.mul_(.5)
    assert_same_read(cs, percepts, extents)


def test_cyclic_definitions_still_fail_loud():
    cs, percepts, extents = chain_fixture()
    store = Spaces._concept_alloc_of(cs).layer(0)
    inventory = cs._field_inventory_rows()
    a, b = int(inventory[0]), int(inventory[1])
    cs.add_concept_edge(a, b, .5)
    cs.add_concept_edge(b, a, .5)
    try:
        cs.cs_read_memberships(percepts, extents)
    except ValueError as error:
        assert 'cyclic' in str(error)
    else:
        raise AssertionError('a cycle must raise')


def test_unwritten_edge_sees_completed_earlier_sibling_for_learning():
    cs = _cs()
    a, b, first, second = [_mint_row(cs, 0, 9010+i) for i in range(4)]
    feature(cs, a, 'ps', 7, 1.)
    feature(cs, b, 'ps', 8, 1.)
    cs.add_concept_edge(first, a, .8)
    cs.add_concept_edge(second, b, .5)
    cs.add_concept_edge(second, first, 0.)
    inventory = torch.tensor([a, b, first, second])
    field = torch.zeros(4, 1, 1, 2)
    field[0, 0, 0, 0] = .6
    def gradient(fold):
        matrix = _concept_alloc_of(cs).layer()
        result = fold(cs, field, inventory_rows=inventory)
        return torch.autograd.grad(result[3, 0, 0, 0], matrix.values)[0]
    old = gradient(ref_compose_order0)
    assert old[-1] > 0
    assert torch.equal(gradient(Spaces.ConceptualSpace._compose_order0), old)
