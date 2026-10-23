"""Fixed-operation identity certificates; no claim about learned parsing."""
from types import SimpleNamespace

import pytest
import torch

from MediumTermMemory import MediumTermMemory
from MTMOverlay import MTMOverlay
from Spaces import LanguageOperationChoice


def address(number):
    return 'ltm', 'address-v1', number


def operations():
    return MTMOverlay((SimpleNamespace(method_name='lift', clause_form='S'),),
                      (SimpleNamespace(method_name='bind', head_role=1),))


def choice(value, *, kind=2, identity=0):
    batch = len(value)
    ids = torch.zeros(batch, dtype=torch.long)
    active = torch.ones(batch, dtype=torch.bool)
    return LanguageOperationChoice(value, ids+kind, ids, ids, active,
        value.new_ones(batch), ids, active, order=ids+1, identity=ids+identity)


def push(overlay, state, value, *, word, index, order=1, resolved=True):
    return overlay.push(state, torch.ones(len(value), dtype=torch.bool), value,
        word=torch.full((len(value),), word), order=torch.full((len(value),), order),
        index=torch.tensor(index), words=8, resolved=torch.full((len(value),), resolved))


def start(value):
    overlay = operations()
    state = overlay.empty(value[:, None].expand(-1, 8, -1), journal=16, words=8)
    state = push(overlay, state, value, word=101, index=0)
    state = overlay.apply(state, choice(value), torch.stack((value, value*0), 1),
                          column=torch.tensor(0), words=8)
    return overlay, state


def event(overlay, state, value, *, index, column):
    state = push(overlay, state, torch.ones_like(value)*.2, word=202, index=index, order=0)
    before = overlay.read(state[0], state[2][:, 1, 0:1])[:, 0]
    return overlay.apply(state, choice(value, kind=1),
        torch.stack((before, torch.ones_like(value)*.2), 1),
        column=torch.tensor(column), words=8)


@pytest.mark.parametrize('compiled', (False, True))
def test_second_verb_receives_first_result_through_bind_with_derivative(compiled):
    first = torch.tensor([[.2, .7, .3]], requires_grad=True)
    scale = torch.tensor(.6, requires_grad=True)
    overlay, state = start(first)
    if compiled:
        overlay.apply = torch.compile(overlay.apply, backend='eager', fullgraph=True)
    changed = first * scale
    state = event(overlay, state, changed, index=1, column=1)
    mention = torch.zeros_like(first)
    state = push(overlay, state, mention, word=303, index=2)
    points, records, handles, _, available = overlay.menu(state, mention[:, None], torch.tensor([[0]]))
    assert available[0, 0, 1] and handles[0, 0, 1] == 1
    torch.testing.assert_close(records[:, 0, 1], changed)
    bound = points[:, 0, 1]
    state = overlay.apply(state, choice(bound, identity=1),
        torch.stack((mention, records[:, 0, 1]), 1), column=torch.tensor(2), words=8)
    second_scale = torch.tensor(.4, requires_grad=True)
    received = overlay.refresh(state, mention[:, None], torch.tensor([[0]]))[:, 0]
    second = received * second_scale
    state = event(overlay, state, second, index=3, column=3)
    torch.testing.assert_close(state[0][:, 0], first * scale * second_scale)
    state[0][:, 0].sum().backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, .24))
    torch.testing.assert_close(scale.grad, (first.detach()*second_scale.detach()).sum())
    assert second_scale.grad > 0


def test_latest_verb_replaces_state_and_two_aliases_share_description_join():
    overlay, state = start(torch.tensor([[.1, .1, .1]]))
    state = event(overlay, state, torch.tensor([[.8, .1, .1]]), index=1, column=1)
    state = push(overlay, state, torch.zeros(1, 3), word=303, index=2)
    old = overlay.read(state[0], torch.tensor([[1]]))[:, 0]
    state = overlay.apply(state, choice(old, identity=1), torch.stack((old*0, old), 1),
                          column=torch.tensor(2), words=8)
    state = event(overlay, state, torch.tensor([[.1, .8, .1]]), index=3, column=3)
    torch.testing.assert_close(state[0][0, 0], torch.tensor([.1, .8, .1]))
    description = torch.tensor([[.1, .1, .9]])
    state = push(overlay, state, description, word=404, index=4)
    points, records, _, _, _ = overlay.menu(state, description[:, None], torch.tensor([[0]]))
    snapshot = records[:, 0, 1].clone()
    state = overlay.apply(state, choice(points[:, 0, 1], identity=1),
        torch.stack((description, records[:, 0, 1]), 1), column=torch.tensor(4), words=8)
    torch.testing.assert_close(snapshot, torch.tensor([[.1, .8, .1]]))
    torch.testing.assert_close(overlay.refresh(state, torch.zeros(1, 2, 3), torch.tensor([[0, 1]])),
        torch.tensor([[[.1, .8, .9], [.1, .8, .9]]]))


@pytest.mark.parametrize('keep_second', (False, True))
def test_trial_values_are_isolated_and_only_selected_values_commit(keep_second):
    from SentenceCompose import select_rows
    overlay, original = start(torch.tensor([[.2, .3]]))
    first = event(overlay, original, torch.tensor([[.8, .1]]), index=1, column=1)
    second = event(overlay, original, torch.tensor([[.1, .9]]), index=1, column=1)
    kept = select_rows(first, second, torch.tensor([keep_second]))
    torch.testing.assert_close(original[0][0, 0], torch.tensor([.2, .3]))
    pool = MediumTermMemory()
    pool.open(address(1), kind=None, when=1, value=original[0][0, 0])
    pool.update(address(1), kept[0][0, 0], when=2)
    torch.testing.assert_close(pool.value_of(address(1)), second[0][0, 0] if keep_second else first[0][0, 0])
    assert not pool.value_of(address(1)).requires_grad


def test_names_events_checkpoint_and_reclamation_have_separate_ownership():
    pool = MediumTermMemory(2)
    x, alias, event_row = address(1), address(2), address(3)
    pool.open(x, kind=None, when=1, reading=101, value=torch.tensor([.5, .1]))
    pool.bind(x, alias, kind=None, when=2, reading=102, value=torch.tensor([.1, .7]))
    pool.event(x, event_row, verb=201, role=0, when=3)
    pool.event(x, event_row, verb=201, role=0, when=4)
    pool.update(x, torch.tensor([.2, .9]), when=4)
    assert pool.entries[x].count == 2 and pool.entries[x].occurrences == (x, alias)
    assert len(pool.entries[x].events) == 1
    assert pool.reading_of(x) == 101 and pool.reading_of(alias) == 102
    restored = MediumTermMemory(2)
    restored.restore(pool.state())
    assert restored.names == pool.names and restored.entries[x].events == pool.entries[x].events
    torch.testing.assert_close(restored.value_of(alias), torch.tensor([.2, .9]))
    restored.forget(event_row)
    assert restored.token_of(alias) == x and not restored.entries[x].events
    restored.forget(x)
    assert not restored.entries and not restored.names


def test_frame_access_does_not_leak_a_native_individuals_other_stream_value():
    from dataclasses import replace
    from IdentityRows import mint
    from ReferenceContext import SituationFrame
    from test_medium_term_memory import keep, mention
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm
    native = mint(resolver.components.owner, atoms[0])
    rows = [keep(resolver, mention(atoms[0], reference=native), position=index)[0] for index in (0, 1)]
    first, second = map(store.occurrence_of, rows)
    assert first != second and pool.token_of(first) != pool.token_of(second)
    pool.update(first, atoms[1], when=20)
    frames = [replace(SituationFrame.from_row(store.row(row)), symbol_ids=(native,),
                      activations=((store.occurrence_of(row), native, 0),)) for row in rows]
    staged = MTMOverlay.stage(resolver, [(frames[0],), (frames[1],)],
        torch.zeros(2, 8, 12), journal=8, words=8)
    for batch, expected in enumerate((atoms[1], atoms[0])):
        current = staged[0][batch, staged[1][batch]]
        assert len(current) > 0
        torch.testing.assert_close(current, expected.expand_as(current), rtol=0, atol=0)
    assert not resolver.individuals(resolver.accessible(()))


@pytest.mark.parametrize('compiled', (False, True))
def test_compose_bind_locations_share_the_operation_chooser_and_zero_dummy(compiled):
    from Language import BindLayer, OperationSelectionLayer
    from ReferenceContext import prepare_operands
    rule = SimpleNamespace(method_name='bind', head_role=1)
    layer = OperationSelectionLayer(d_model=2, ops=[], unary_ops=[BindLayer()])
    with torch.no_grad():
        layer.apply_anchor.fill_(1)
        layer.stop_anchor.fill_(-100)
    value = torch.tensor([[[.1, .2]]], requires_grad=True)
    antecedent = torch.tensor([[[.8, .7], [.3, .4]]], requires_grad=True)
    state = (antecedent, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 1, 6, dtype=torch.long),
             torch.tensor([[[2, 101, 1], [2, 102, 1]]]), torch.zeros(1, 1, 6, dtype=torch.long))
    bank = SimpleNamespace()
    def run(value, antecedent):
        current = (antecedent, *state[1:])
        ids = torch.zeros(1, 1, dtype=torch.long)
        data = prepare_operands(value, ids-1, ids, ids, orders=ids+1,
            rules=(), unary_rules=(rule,), bank=bank, live=None,
            active=torch.ones_like(ids, dtype=torch.bool), identity_state=current, source=ids)
        _, path, route = layer(value, reference_data=data, expectation_prime=torch.ones(1, 2))
        return path, route['identity'], route['logits'], route['probabilities'], route['operands']
    if compiled:
        run = torch.compile(run, backend='aot_eager', fullgraph=True)
    path, identity, logits, probabilities, operands = run(value, antecedent)
    assert identity.tolist() == [1] and logits[0, 0] == 0
    torch.testing.assert_close(path, antecedent[:, :1])
    torch.testing.assert_close(operands[:, 0], value[:, 0])
    torch.testing.assert_close(operands[:, 1], antecedent[:, 0])
    torch.testing.assert_close(probabilities, logits.softmax(-1))
    path.sum().backward(retain_graph=True)
    torch.testing.assert_close(antecedent.grad, torch.tensor([[[1., 1.], [0., 0.]]]))
    assert layer.apply_anchor.grad is None or torch.count_nonzero(layer.apply_anchor.grad) == 0
    (-probabilities[0, 1].log()).backward()
    assert layer.apply_anchor.grad is not None and layer.apply_anchor.grad.abs().sum() > 0
    with torch.no_grad():
        layer.apply_anchor.fill_(-10)
    _, identity, logits, _, _ = run(value, antecedent)
    assert identity.tolist() == [0] and logits[0, 0] == 0 and (logits[0, 1:] < 0).all()


def test_compose_menu_uses_the_resolvers_observed_exclusions():
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    for index in (0, 1):
        row = store.write_clause(Clause(ConceptualMeaning.from_description(atoms[index]),
            point=atoms[index], order=1), document_key='supports', sentence_index=index)
        store.witness_count[row] = 8
    exclusions = torch.tensor(resolver.exclusion_pairs()).reshape(-1, 2)
    state = (atoms[0].reshape(1, 1, -1), torch.ones(1, 1, dtype=torch.bool),
             torch.zeros(1, 1, 6, dtype=torch.long), torch.tensor([[[2, 101, 1]]]),
             torch.zeros(1, 1, 6, dtype=torch.long))
    for own in (atoms[1], atoms[0], atoms[0]+atoms[1], atoms[1]*0):
        *_, available = MTMOverlay.menu(state, own.reshape(1, 1, -1), torch.tensor([[0]]),
                                       exclusions=exclusions)
        assert available[0, 0, 0]
        assert bool(available[0, 0, 1]) == (not resolver.exclusion_query()(own, atoms[0]))


def recorded_identity_reading(f, tree):
    """Execute fixed grammar operations and capture the production tensor journal."""
    from dataclasses import replace
    from ReferenceContext import result_order
    entry = f.program(tree)
    orders = torch.tensor([1 if int(word) in (f.words['subject'][0], f.words['alias'][0]) else 0
                           for word in entry.word_ids])
    entry = replace(entry, leaf_orders=orders, close_references=True)
    overlay = MTMOverlay(f.binary, f.unary)
    words, columns = len(entry.leaves), len(entry.actions)
    state = overlay.empty(entry.leaves.new_zeros(1, 8, 8), words=words, journal=columns)
    stack, stamps, frames, refs, inputs = [], [], [], [], []
    first = 0
    for action, (kind, local, word) in enumerate(entry.actions.tolist()):
        if kind == 0:
            value = entry.leaves[word][None]
            state = overlay.push(state, torch.tensor([True]), value,
                word=entry.concept_ids[word:word+1], order=orders[word:word+1],
                index=torch.tensor(word), words=words, resolved=torch.tensor([True]))
            stack.insert(0, value); stamps.insert(0, int(orders[word]))
            frames.append(value.new_zeros(3, 8)); refs.append((-1, -1))
            continue
        source = torch.arange(len(stack))[None]
        live = overlay.refresh(state, torch.stack(stack, 1), source)
        rule = (f.binary if kind == 1 else f.unary)[local]
        operands = live[:, [1, 0]] if kind == 1 else torch.stack((live[:, 0], live[:, 0]*0), 1)
        handle = 0
        if rule.method_name == 'bind':
            points, records, _, _, available = overlay.menu(state, live[:, :1], torch.tensor([[0]]))
            handle = first
            assert available[0, 0, handle]
            result = f.operation(rule).compose(points[:, :, handle])[:, 0]
            operands = torch.stack((live[:, 0], records[:, 0, handle]), 1)
            if not first:
                first = int(state[2][0, 0, 0])
        else:
            values = (operands[:, 0:1], operands[:, 1:2]) if kind == 1 else (operands[:, 0:1],)
            result = f.operation(rule).compose(*values).reshape(1, -1)
        arity = 2 if kind == 1 else 1
        stamp = result_order(rule, tuple(reversed(stamps[:arity])))
        selected = choice(result, kind=kind, identity=handle)._replace(
            local_op=torch.tensor([local]), order=torch.tensor([stamp]))
        inputs.append((rule.method_name, operands.clone(), result.clone()))
        state = overlay.apply(state, selected, operands, column=torch.tensor(action), words=words)
        if not first and getattr(rule, 'determiner_mode', None) in ('mint', 'bind'):
            first = int(state[2][0, 0, 0])
        del stack[:arity]; del stamps[:arity]
        stack.insert(0, result); stamps.insert(0, stamp)
        frames.append(torch.cat((operands, result[:, None]), 1)[0]); refs.append((-1, -1))
    return replace(entry, operation_values=torch.stack(frames), operation_refs=torch.tensor(refs),
        identity_origins=state[3][0], identity_actions=state[4][0], identity_values=state[0][0],
        identity_leaves=torch.arange(1, words+1),
        end_state=torch.stack((stack[0][0], stack[0][0]*0, stack[0][0]*0))), inputs, first


def test_kept_journal_commits_two_verbs_one_identity_and_immutable_occurrences(monkeypatch):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    from test_medium_term_memory import snapshot, unchanged
    f = SentenceFixture(monkeypatch)
    tree = ('sum',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'place'))),
        ('lift', ('bind', 'alias'), ('verb', 'affect', ('lower', 'd', 'target'))))
    entry, inputs, handle = recorded_identity_reading(f, tree)
    lifts = [record for record in inputs if record[0] == 'lift']
    # The alias is a description mention and joins before the next VP reads it.
    alias = entry.leaves[entry.word_ids.tolist().index(f.words['alias'][0])]
    torch.testing.assert_close(lifts[1][1][:, 0], torch.maximum(lifts[0][2], alias))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, ())
    prepared = resolver.prepare(clause, plan)
    assert not f.cs.mtm.entries and not plan['decisions']
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='identity', sentence_index=0)
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='identity', position=0, training=False)
    unchanged(resolver, retained)
    selected = [item for item in plan['items'] if item['written_node'].identity['handle'] == handle]
    names = [written[('occurrence', id(item['written_node']))] for item in selected]
    assert len(names) == 2
    token = f.cs.mtm.token_of(names[0])
    assert f.cs.mtm.token_of(names[1]) == token
    assert f.cs.mtm.entries[token].count == 2
    assert len(f.cs.mtm.entries[token].events) == 2
    torch.testing.assert_close(f.cs.mtm.value_of(token), lifts[1][2][0])
    f.cs.mtm.update(token, torch.zeros(8), when=20)
    unchanged(resolver, retained)


@pytest.mark.parametrize('corruption', ('membership', 'value', 'event', 'reading'))
def test_invalid_identity_checkpoint_does_not_replace_the_pool(corruption):
    pool = MediumTermMemory()
    pool.open(address(1), kind=None, when=1, value=torch.ones(2))
    before = pool.state()
    bad = pool.state()
    if corruption == 'membership':
        bad['entries'][0]['occurrences'] = (address(2),)
    elif corruption == 'value':
        bad['entries'][0]['value'][0] = float('nan')
    elif corruption == 'event':
        bad['entries'][0]['events'] = ((address(3), 5, 1, 3),)
    else:
        bad['names'] = ((address(1), address(1), 'bad'),)
    with pytest.raises(ValueError):
        pool.restore(bad)
    assert pool.state()['names'] == before['names']
    torch.testing.assert_close(pool.value_of(address(1)), before['entries'][0]['value'])


def test_binding_an_unresolved_object_does_not_complete_its_verb():
    overlay, state = start(torch.tensor([[.6, .2]]))
    # A named object is present, but its value has not been resolved.
    state = push(overlay, state, torch.tensor([[.1, .1]]), word=303, index=2, resolved=False)
    state = overlay.apply(state, choice(torch.tensor([[.1, .1]])),
        torch.tensor([[[.1, .1], [0., 0.]]]), column=torch.tensor(2), words=8)
    state = push(overlay, state, torch.zeros(1, 2), word=404, index=3, resolved=False)
    points, records, _, _, _ = overlay.menu(state, torch.zeros(1, 1, 2), torch.tensor([[0]]))
    state = overlay.apply(state, choice(points[:, 0, 3], identity=3),
        torch.stack((torch.zeros(1, 2), records[:, 0, 3]), 1), column=torch.tensor(3), words=8)
    assert state[2][0, 0, 3] == 0
