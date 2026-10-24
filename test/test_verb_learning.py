"""Kept observations credit a retained verb event once, across sentences."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from Layers import Error
from MeaningKernels import MeaningAction
from MediumTermMemory import MediumTermMemory
from VerbLearning import observe


def test_teacher_graph_is_built_after_the_paired_optimizer_steps():
    from VerbLearning import register
    mtm = MediumTermMemory()
    token, event = ('ltm', 'unit', 1), ('ltm', 'unit', 2)
    before, after, verb = (torch.full((8,), value) for value in (.2, .35, .3))
    action = MeaningAction(8, capacity=1)
    mtm.open(token, kind=11, value=before, when=1)
    mtm.event(token, event, verb=12, role=0, when=2, before=before, basis=-1)
    previous = dict(mtm.entries)
    mtm.update(token, after, when=3)
    records, errors = [], Error()
    observe(action, mtm, previous, {token}, lambda _: verb, errors=errors, deferred=records)
    assert errors.total(objective='reconstruction') is None
    assert len(records) == 1 and all(not value.requires_grad for value in records[0][1:4])
    with torch.no_grad():
        action.readout.add_(.1)  # a later sentence's ordinary paired update
    register(records, errors)
    expected = action.teacher(before, after, verb, group=torch.tensor(-1)) * .1
    torch.testing.assert_close(errors.total(objective='reconstruction'), expected)
    errors.total(objective='reconstruction').backward()
    assert action.readout.grad.abs().sum() > 0 and not records


def record(f, tree, frames=()):
    """Fixed operations on the production frame-staged identity transaction."""
    from MTMOverlay import MTMOverlay
    from ReferenceContext import result_order
    from test_identity_threading import choice
    entry = f.program(tree)
    orders = torch.tensor([1 if word == f.words['subject'][0] else 0 for word in entry.word_ids])
    overlay = MTMOverlay(f.binary, f.unary)
    words, columns = len(entry.leaves), len(entry.actions)
    state = overlay.stage(f.cs.components.reference_close, [frames],
        entry.leaves.new_zeros(1, 8, 8), words=words, journal=columns)
    base = state[0].shape[1] - words - columns
    stack, stamps, recorded, refs = [], [], [], []
    for column, (kind, local, word) in enumerate(entry.actions.tolist()):
        if kind == 0:
            value = entry.leaves[word][None]
            state = overlay.push(state, torch.tensor([True]), value,
                word=entry.concept_ids[word:word+1], order=orders[word:word+1],
                index=torch.tensor(word), words=words, resolved=torch.tensor([True]))
            stack.insert(0, value); stamps.insert(0, int(orders[word]))
            recorded.append(value.new_zeros(3, 8)); refs.append((-1, -1))
            continue
        live = overlay.refresh(state, torch.stack(stack, 1), torch.arange(len(stack))[None])
        rule = (f.binary if kind == 1 else f.unary)[local]
        operands = live[:, [1, 0]] if kind == 1 else torch.stack((live[:, 0], live[:, 0]*0), 1)
        handle = 0
        if rule.method_name == 'bind':
            points, records, _, _, available = overlay.menu(state, live[:, :1], torch.tensor([[0]]))
            handle = 1
            assert base and available[0, 0, handle]
            result = points[:, 0, handle]
            operands = torch.stack((live[:, 0], records[:, 0, handle]), 1)
        else:
            values = (operands[:, 0], operands[:, 1]) if kind == 1 else (operands[:, 0],)
            result = f.operation(rule).compose(*values).reshape(1, -1)
        arity = 2 if kind == 1 else 1
        order = result_order(rule, tuple(reversed(stamps[:arity])))
        selected = choice(result, kind=kind, identity=handle)._replace(
            local_op=torch.tensor([local]), order=torch.tensor([order]))
        state = overlay.apply(state, selected, operands, column=torch.tensor(column), words=words)
        del stack[:arity]; del stamps[:arity]
        stack.insert(0, result); stamps.insert(0, order)
        recorded.append(torch.cat((operands, result[:, None]), 1)[0]); refs.append((-1, -1))
    return replace(entry, leaf_orders=orders, close_references=True,
        operation_values=torch.stack(recorded), operation_refs=torch.tensor(refs),
        identity_origins=state[3][0], identity_actions=state[4][0], identity_values=state[0][0],
        identity_leaves=torch.arange(base+1, base+words+1),
        end_state=torch.stack((stack[0][0], stack[0][0]*0, stack[0][0]*0)))


def test_two_kept_sentences_join_the_copular_observation_and_teach_once(monkeypatch):
    from ClauseJournal import finish_clause
    from ReferenceContext import SituationAddress
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    action = MeaningAction(8, capacity=2)
    for word, value in (('subject', .2), ('move', .3), ('property', .4)):
        symbol = f.noun(word)
        with torch.no_grad():
            f.cs.similarity_codebook.W[f.cs._csw_row_of(symbol)].fill_(value)
    f.ops['lift'] = SimpleNamespace(compose=lambda a, b: action(a, b))
    resolver, errors = f.cs.components.reference_close, Error()
    def close(tree, frames=(), position=0):
        clause = finish_clause(f.language, record(f, tree, frames), registry=f.registry)
        plan = resolver.preview(clause, frames)
        kept = resolver.prepare(clause, plan)
        written = {}
        row = f.store.write_clause(kept, written_rows=written, document_key='teacher', sentence_index=position)
        resolver.commit(plan, row, written, document='teacher', position=position, training=True,
                        verb_action=action, errors=errors)
        return SituationAddress(f.store.occurrence_of(row),
            symbols=written.get(('symbols', id(kept)), ()),
            activations=written.get(('activations', id(kept)), ())).read(f.store)
    frame = close(('lift', ('lower_unary', 'subject'), 'move'))
    token = next(key for key, entry in f.cs.mtm.entries.items() if entry.events)
    assert not resolver.last_verb_teacher
    close(('part', ('bind', 'subject'), 'property'), (frame,), position=1)
    assert len(resolver.last_verb_teacher) == 1
    assert resolver.last_verb_teacher[0]['token'] == token
    torch.testing.assert_close(f.cs.mtm.value_of(token), torch.full((8,), .4))
    assert sum(event.taught for event in f.cs.mtm.entries[token].events) == 1
    errors.total(objective='reconstruction').backward()
    assert action.readout.grad.abs().sum() > 0


def test_observed_state_trains_readout_once_without_concept_gradients():
    mtm = MediumTermMemory()
    token, event = ('ltm', 'unit', 1), ('ltm', 'unit', 2)
    before = torch.tensor([.2, .3, 0., .4, .2, .3, 0., .4], requires_grad=True)
    verb = torch.full((8,), .2, requires_grad=True)
    action = MeaningAction(8, capacity=2)
    mtm.open(token, kind=11, value=before, when=1)
    mtm.event(token, event, verb=12, role=0, when=2, before=before, basis=-1)
    previous = dict(mtm.entries)
    observed = torch.tensor([.3, .4, .8, .45, .3, .4, .8, .45], requires_grad=True)
    mtm.update(token, observed, when=3)
    errors = Error()
    report = observe(action, mtm, previous, {token}, lambda _: verb, errors=errors)
    assert len(report) == 1 and report[0]['defined'] == 6
    term = errors.total(objective='reconstruction')
    term.backward()
    assert action.readout.grad.abs().sum() > 0
    assert before.grad is None and observed.grad is None and verb.grad is None
    # The optimizer receives only the defined-coordinate chart discrepancy.
    assert torch.equal(action.readout.grad[[2]], torch.zeros_like(action.readout.grad[[2]]))
    assert observe(action, mtm, dict(mtm.entries), {token}, lambda _: verb, errors=errors) == []
    copy = MediumTermMemory()
    copy.restore(mtm.state())
    assert copy.entries[token].events[0].taught
    torch.testing.assert_close(copy.entries[token].events[0].before, before.detach())


def test_copular_join_is_kept_and_fires_the_previous_verb_teacher(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from test_identity_threading import recorded_identity_reading
    from ClauseJournal import finish_clause
    from test_medium_term_memory import keep, mention
    from ReferenceContext import SituationAddress
    from ClauseRow import Clause, ClausePredicate
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    action = MeaningAction(8, capacity=2)
    for word, value in (('subject', .2), ('move', .3), ('property', .4)):
        symbol = f.noun(word)
        with torch.no_grad():
            f.cs.similarity_codebook.W[f.cs._csw_row_of(symbol)].fill_(value)
    f.ops['lift'] = SimpleNamespace(compose=lambda a, b: action(a, b))
    entry, _, _ = recorded_identity_reading(f, ('lift', ('lower_unary', 'subject'), 'move'))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, ())
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='unit', sentence_index=0)
    resolver.commit(plan, row, written, document='unit', position=0, training=True, verb_action=action)
    token = next(key for key, value in f.cs.mtm.entries.items() if any(event.before is not None for event in value.events))
    prior = f.cs.mtm.entries[token].value.clone()
    # The second sentence is a fixed part operation on the retained subject.
    from MTMOverlay import MTMOverlay
    from test_identity_threading import choice
    overlay = MTMOverlay((SimpleNamespace(method_name='part', relation_kind='part'),), ())
    values, valid, stack, origins, journal = overlay.empty(prior[None, None].expand(1, 2, 8), words=2, journal=1)
    values[0, 0], valid[0, 0] = prior, True
    stack[0, 1, :4] = torch.tensor([1, 12, 0, 1])
    stack[0, 0, :4] = torch.tensor([0, 13, 0, 1])
    stack[..., 6] = 1
    property_value = torch.full_like(prior, .4)
    previous = dict(f.cs.mtm.entries)
    state = overlay.apply((values, valid, stack, origins, journal), choice(property_value[None], kind=1),
        torch.stack((prior, property_value))[None], column=torch.tensor(0), words=2)
    torch.testing.assert_close(state[0][0, 0], torch.maximum(prior, property_value))
    f.cs.mtm.update(token, state[0][0, 0], when=3)
    errors = Error()
    from IdentityRows import point
    reports = observe(action, f.cs.mtm, previous, {token}, lambda symbol: point(f.cs, symbol), errors=errors)
    assert len(reports) == 1
    loss = errors.total(objective='reconstruction')
    loss.backward()
    assert action.readout.grad.abs().sum() > 0
    with torch.no_grad():
        action.readout.sub_(10 * action.readout.grad)
    before_error = reports[0]['error']
    remembered = previous[token].events[-1]
    after_error = action.teacher(remembered.before, f.cs.mtm.value_of(token), point(f.cs, remembered.verb))
    assert after_error < before_error
