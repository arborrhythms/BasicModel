"""§56.2: only journalled operations introduce local identity handles.

The packed journal has one row per operation; SHIFT and transpose actions do
not acquire a transaction by aliasing row zero. Capacities are constructed
once from the fixture's actions, using the existing native owner/STM fixtures.
"""
from dataclasses import replace

import pytest
import torch

from Memory import STMChunks as chunks
from ChunkJournal import IDENTITY, LEAVES, RECORD_WIDTH, RESOLVED, SOURCE, capture, empty_scope
from chunk_fixtures import Walk, choice, context


def _packed(*, owner=None, identity=None, metadata=True):
    if owner is None:
        from test_cs_sparse_weights import _cs
        owner = _cs()
    if identity is None:
        identity = int(owner.mtm.transient_ids[0])
    state = chunks.empty(1, 8, 8)
    scope = empty_scope(state[0])
    if metadata:
        scope[0, 0, IDENTITY] = identity
        scope[0, 0, SOURCE] = 1
        scope[0, 0, RESOLVED] = 1
        scope[0, 0, LEAVES] = 1
    records = torch.full((1, 1, RECORD_WIDTH), -1, dtype=torch.long)
    records[0, 0, 2:8] = torch.tensor([identity, -1, -1, 37, -1, 1])
    records[0, 0, 8:12] = torch.tensor([1, 1, 1, 0])
    values = torch.zeros(1, 1, 3 * 8)
    return owner, identity, state, scope, records, values


def _capture(packed, actions, columns, *, positions=(0,)):
    owner, _, state, scope, records, values = packed
    return capture(owner, state, scope, records, values, batch=0,
        positions=torch.tensor(positions, dtype=torch.long),
        columns=torch.tensor(columns), actions=torch.tensor(actions))


@pytest.mark.parametrize('kind', (1, 2))
def test_real_operation_at_action_zero_keeps_its_origin(kind):
    result = _capture(_packed(), [(kind, 0, -1)], [0])
    assert result['identity_origins'].tolist() == [[3, 0, 1]]
    assert result['identity_actions'][0].tolist() == [1, 0, 0, 37, -1, 1]
    assert result['timeline_actions'][0].tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize('transpose_column', (-1, 0))
def test_nonoperations_have_zero_events_and_timeline(transpose_column):
    result = _capture(_packed(),
        [(2, 0, -1), (0, -1, 0), (3, -1, -1)], [0, -1, transpose_column])
    assert result['identity_origins'].tolist() == [[3, 0, 1]]
    assert result['identity_actions'][0].tolist() == [1, 0, 0, 37, -1, 1]
    assert result['timeline_actions'][0].tolist() == [1, 1, 1, 0]
    assert torch.count_nonzero(result['identity_actions'][1:]).item() == 0
    assert torch.count_nonzero(result['timeline_actions'][1:]).item() == 0


@pytest.mark.parametrize('kind,column', ((0, -1), (3, -1), (3, 0)))
def test_nonoperation_alias_cannot_invent_an_identity(kind, column):
    result = _capture(_packed(metadata=False), [(kind, -1, 0)], [column])
    assert result['identity_ids'].numel() == 0
    assert result['identity_origins'].shape == (0, 3)
    assert torch.count_nonzero(result['identity_actions']).item() == 0
    assert torch.count_nonzero(result['timeline_actions']).item() == 0


def test_local_identity_without_an_introducing_operation_fails_at_capture():
    packed = _packed()
    with pytest.raises(RuntimeError, match=r'local identity .* has no introducing operation'):
        _capture(packed, [(0, -1, 0)], [-1])


@pytest.mark.parametrize('kind,column', ((1, -1), (2, -1), (2, 1)))
def test_operation_requires_an_in_range_journal_column(kind, column):
    with pytest.raises(ValueError, match='identity journal operations have invalid columns'):
        _capture(_packed(), [(kind, 0, -1)], [column])


def test_shifted_identity_needs_no_introducing_operation():
    packed = _packed()
    packed[3][0, 0, SOURCE] = -1
    result = _capture(packed, [(0, -1, 0)], [-1])
    assert result['identity_origins'].tolist() == [[4, 0, 1]]
    assert result['identity_leaves'].tolist() == [1]
    assert torch.count_nonzero(result['identity_actions']).item() == 0


@pytest.mark.parametrize('origin', ('durable', 'pool'))
def test_existing_identity_needs_no_introducing_operation(monkeypatch, origin):
    from Meaning import ConceptualMeaning
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    symbol = f.noun('carrier')
    point = f.registry._payload(('sym', symbol))
    if origin == 'durable':
        identity, expected = symbol, [1, symbol, 1]
    else:
        row = f.store.append_meaning(ConceptualMeaning.from_description(point))
        occurrence = f.store.occurrence_of(row)
        token = f.cs.mtm.open(occurrence, kind=None, when=0, value=point)
        identity = f.cs.mtm.entries[token].transient
        expected = [2, occurrence[-1], 0]
    result = _capture(_packed(owner=f.cs, identity=identity), [(0, -1, 0)], [-1])
    assert result['identity_origins'].tolist() == [expected]
    assert torch.count_nonzero(result['identity_actions']).item() == 0


def _native_packed_reading(f, method):
    """Run selected native operations, storing no column for either SHIFT."""
    tree = ('bind', (method, 'carrier')) if method == 'lower_unary' else (
        'bind', (method, 'marker', 'carrier'))
    entry = f.program(tree)
    orders = torch.tensor([2] if method == 'lower_unary' else [0, 2])
    from order_fixtures import force_order
    for identity, order in zip(entry.concept_ids.tolist(), orders.tolist()):
        force_order(f.cs, identity, order)
    walk = Walk(f.binary, f.unary, capacity=f.cs.mtm.capacity,
        first_id=int(f.cs.mtm.transient_ids[0]))
    state = walk.empty(entry.leaves[None, 0], 2)
    columns, identity, introduced_at = [], None, None
    for action, (kind, local, word) in enumerate(entry.actions.tolist()):
        if kind == 0:
            state = walk.shift(state, entry.leaves[word][None],
                word=int(entry.concept_ids[word]), index=word, order=int(orders[word]))
            columns.append(-1)
            continue
        rule = (f.binary if kind == 1 else f.unary)[local]
        stm, live = chunks.refresh(state[0], chunks.stack_indices(state[0]))
        state = (stm, *state[1:])
        bind = rule.method_name == 'bind'
        if bind:
            offered, ids, indices, valid = chunks.bind_menu(stm, live[:, :1])
            matches = ((ids[0, 0] == identity) & valid[0, 0]).nonzero().flatten()
            assert len(matches), 'the selected native introduction must remain readable'
            selected_index = int(matches[0])
            operands = torch.stack((live[:, 0], offered[:, 0, selected_index]), 1)
            result = f.operation(rule).compose(operands[:, 0], operands[:, 1],
                dummy=torch.tensor([False]))
            selected = choice(result, kind=kind, local=local, identity=identity,
                chunk=int(indices[0, 0, selected_index]), bind=True, order=1)
            column = 1
        else:
            arity = 2 if kind == 1 else 1
            operands = live[:, [1, 0]] if arity == 2 else torch.stack(
                (live[:, 0], torch.zeros_like(live[:, 0])), 1)
            result = f.operation(rule).compose(*(operands[:, i] for i in range(arity)))
            selected = choice(result, kind=kind, local=local, order=1,
                head=getattr(rule, 'head_role', 0))
            column, introduced_at = 0, action
        state = walk.apply(state, selected, operands, column)
        columns.append(column)
        if identity is None:
            identity = int(state[1][0, 0, IDENTITY])
    stm, scope, records, values = state
    column_tensor = torch.tensor(columns)
    assert columns[0] == -1 and introduced_at > 0
    assert int(records[0, 0, 2]) == identity and int(records[0, 0, 3]) < 0
    captured = capture(f.cs, stm, scope, records, values, batch=0,
        positions=torch.arange(len(entry.leaves)), columns=column_tensor, actions=entry.actions)
    handle = int((captured['identity_ids'] == identity).nonzero().item()) + 1
    end = chunks.stack_view(stm)[0][0, 0]
    reading = replace(entry, close_references=True, leaf_orders=orders,
        operation_values=values[0, column_tensor.clamp_min(0)].reshape(len(entry.actions), 3, -1),
        operation_refs=records[0, column_tensor.clamp_min(0), :2],
        end_state=torch.stack((end, end * 0, end * 0)), **captured)
    return reading, introduced_at, handle


@pytest.mark.parametrize('method', ('lower', 'lower_unary'))
def test_shift_alias_records_real_operation_and_closing_writes_it(monkeypatch, method):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    reading, action, handle = _native_packed_reading(f, method)
    assert reading.identity_origins[handle - 1, :2].tolist() == [3, action]
    assert reading.actions[0, 0].item() == 0
    assert reading.actions[action, 0].item() in (1, 2)
    close = f.cs.components.reference_close
    clause = finish_clause(f.language, reading, registry=f.registry)
    plan = close.preview(clause, context(close, ()))
    introduction = next(item for item in plan['items']
        if item['node'].identity['handle'] == handle and item['node'].identity['introduction'])
    use = next(item for item in plan['items']
        if item['node'].reference_request.get('compose_bind') == handle)
    assert introduction['node'].identity['action'] == action
    assert introduction['selected'] == use['selected'] == ('new', 0)
    prepared, written = close.prepare(clause, plan), {}
    row = f.store.write_clause(prepared, written_rows=written,
        )
    close.commit(plan, row, written, document='packed operation', position=0, training=False)
    first, second = [written[('occurrence', id(item['written_node']))]
                     for item in (introduction, use)]
    assert first != second
    assert f.cs.mtm.token_of(first) == f.cs.mtm.token_of(second) == first
