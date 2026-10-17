"""Fixed-operation, native-row closing certificates; no learned-parse claim."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from ClauseRow import Clause
from IndependentComponents import EventChart, ColumnDictionary
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from ReferenceClose import ReferenceClose
from ReferenceContext import SituationFrame


def fixture():
    width, capacity = 12, 64
    atoms = torch.eye(width)
    chart = EventChart(torch.arange(width), atoms, capacity=capacity, content_width=width)
    store = TernaryTruthStore(width, capacity=512)
    nouns = ColumnDictionary(2*capacity)
    owner = SimpleNamespace(nWhat=width, _closed_clause_store=lambda: store)
    component = SimpleNamespace(owner=owner, capacity=capacity, nouns=nouns, _ready=lambda: chart)
    def allocate(value):
        return 100 + len(nouns.ids)
    component._allocate = allocate
    resolver = ReferenceClose(component)
    store.configure_clause_index(concept_point=lambda cid: chart.render(
        nouns.directions[str(cid)][chart.features]) if cid in nouns.ids else None)
    return resolver, store, atoms


def close(resolver, point, predicate, mode, frames, *, document, sentence, selected=None):
    request = dict(mode=mode, leaves=(1,), cue=torch.zeros_like(point), predicate=predicate, role=0)
    np = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                reference_request=request)
    clause = Clause(ConceptualMeaning.from_description(point+predicate), point=point+predicate,
                    children=(np,), refs=(('clause', 0), -1, -1), order=2)
    plan = resolver.preview(clause, frames)
    if selected is not None:
        assert mode != 'mint'
        plan['items'][0]['selected'] = plan['items'][0]['candidates'][selected]
    kept = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(kept, document_key=document, sentence_index=sentence,
                                     written_rows=written)
    resolver.commit(plan, row, written, document=document, position=sentence, training=False)
    frame = SituationFrame.from_row(resolver.store.row(row))
    address = resolver.last_audit[0]['occurrence']
    return frame, address, plan


def test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them():
    resolver, store, atoms = fixture()
    noun, property, runs, sleeps = atoms[0], atoms[4], atoms[6], atoms[7]
    frames, addresses = [], []
    contents = (noun, noun, noun+property, noun+property)
    for sentence, (point, predicate, mode) in enumerate(zip(contents,
            (runs, sleeps, runs, runs), ('mint', 'mint', 'bind', 'bind'))):
        # Freeze an earlier alternative binding in the last reading: the
        # closing revision, not the initial choice, must reconcile it.
        frame, address, _ = close(resolver, point, predicate, mode, frames,
            document='merge', sentence=sentence, selected=1 if sentence == 3 else None)
        frames.append(frame); addresses.append(address)
        if sentence == 1:
            assert len({int(store.refs[store.index_of_row(a), 0]) for a in addresses}) == 2
    assert len({int(store.refs[store.index_of_row(a), 0]) for a in addresses}) == 1
    for address, point in zip(addresses, contents):
        torch.testing.assert_close(store.point_of_row(address), point, rtol=0, atol=0)


@pytest.mark.parametrize('kind', range(4))
def test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding(kind):
    resolver, store, atoms = fixture()
    noun, black, white, runs, sleeps = atoms[kind], *atoms[4:8]
    # Evidence teaches exclusivity; no color/word identities enter the owner.
    for _ in range(8):
        resolver._learn(resolver.coordinates(noun+black), resolver.coordinates(runs))
        resolver._learn(resolver.coordinates(noun+white), resolver.coordinates(sleeps))
    for document, second, distinct in (('mint-pair', 'mint', True), ('one', 'bind', False),
                                      ('revisable', 'bind', True)):
        frames, occurrences = [], []
        contents = (noun, noun, noun+black, noun+(white if distinct else black))
        for sentence, (content, predicate, mode) in enumerate(zip(contents,
                (runs, sleeps, runs, sleeps), ('mint', second, 'bind', 'bind'))):
            frame, address, plan = close(resolver, content, predicate, mode, frames,
                document=(document, kind), sentence=sentence)
            frames.append(frame); occurrences.append(address)
            if mode == 'mint':
                assert plan['items'][0]['selected'][0] == 'new'
            else:
                assert all(isinstance(ref, int) for ref in plan['items'][0]['candidates'])
        references = [int(store.refs[store.index_of_row(address), 0]) for address in occurrences]
        assert len(set(references)) == (2 if distinct else 1)
        assert references[0] == references[2] and references[1] == references[3]
        for address, content in zip(occurrences, contents):
            torch.testing.assert_close(store.point_of_row(address), content, rtol=0, atol=0)


def test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised():
    resolver, store, atoms = fixture()
    frame, address, _ = close(resolver, atoms[0], atoms[6], 'mint', (), document='a', sentence=0)
    before = store.refs.clone()
    _, missing, plan = close(resolver, torch.zeros(12), atoms[6], 'pronoun', (), document='b', sentence=0)
    assert plan['items'][0]['reference'] is None and len(resolver.components.nouns.ids) == 1
    assert int(store.refs[store.index_of_row(missing), 0]) == -1
    torch.testing.assert_close(store.refs[:store.index_of_row(missing)], before[:store.index_of_row(missing)])
    assert resolver.last_audit[0]['missing_reason'] == 'no_accessible_individual'


def test_empty_close_has_no_policy_gradient_or_store_side_effect():
    resolver, store, atoms = fixture()
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=0)
    plan = resolver.preview(clause, ())
    assert float(plan['cost']) == 0 and not plan['cost'].requires_grad
    kept = resolver.prepare(clause, plan)
    assert kept is clause
    written = {}
    row = store.write_clause(kept, document_key='field', sentence_index=0, written_rows=written)
    metadata = dict(store._semantic_rows)
    required, refs = store.metadata_required.clone(), store.refs.clone()
    resolver.commit(plan, row, written, document='field', position=0, training=True)
    assert store._semantic_rows == metadata and resolver.last_audit == ()
    torch.testing.assert_close(store.metadata_required, required, rtol=0, atol=0)
    torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)
    assert resolver.total == 0 and not resolver.components.nouns.ids


def test_reference_evidence_credits_the_departure_without_changing_exact_reconstruction():
    from Layers import Error
    from SentenceCredit import components, comparison, score_function
    registry, like = Error(), torch.zeros(1)
    registry.add('reconstruction.exact', like, category='reconstruction')
    parts = torch.stack([components(registry, like+100, like, reference=like+1),
                         components(registry, like+200, like, reference=like)], 1)
    audit = comparison(parts, torch.ones(1, dtype=torch.bool))
    assert audit['keep_costs'].tolist() == [[0., 0.]] and not bool(audit['wins'].any())
    assert audit['advantage'].tolist() == [-1.] and audit['deciding'] == ['answer']
    probability = torch.tensor([[.5]], requires_grad=True)
    loss, _ = score_function(probability, torch.ones_like(probability),
        torch.ones_like(probability, dtype=torch.bool), audit['costs'])
    loss.sum().backward()
    assert probability.grad.tolist() == [[-1.]]


def test_revision_merges_three_accessible_groups_transitively_across_cued_documents():
    resolver, store, atoms = fixture()
    frames, addresses = [], []
    for document in range(3):
        first, a, _ = close(resolver, atoms[0], atoms[6], 'mint', (),
                            document=document, sentence=0)
        second, b, _ = close(resolver, atoms[0]+atoms[4], atoms[6], 'bind', (first,),
                             document=document, sentence=1)
        frames.extend((first, second)); addresses.extend((a, b))
    _, inaccessible, _ = close(resolver, atoms[0]+atoms[4], atoms[6], 'mint', (),
                               document='outside', sentence=0)
    outside = int(store.refs[store.index_of_row(inaccessible), 0])
    before = {address: store.point_of_row(address).clone() for address in addresses}
    resolver.revise(resolver.accessible(frames))
    assert len({int(store.refs[store.index_of_row(a), 0]) for a in addresses}) == 1
    assert int(store.refs[store.index_of_row(inaccessible), 0]) == outside
    assert sum(bool(w) for w in resolver.components.nouns._witnesses.values()) == 2
    for address, point in before.items():
        torch.testing.assert_close(store.point_of_row(address), point, rtol=0, atol=0)


def test_closing_credit_keeps_both_trial_graphs_and_learns_salience_and_form_cues():
    resolver, store, atoms = fixture()
    frames = []
    for sentence in range(2):
        frame, _, _ = close(resolver, atoms[sentence], atoms[6+sentence], 'mint', frames,
                            document='credit', sentence=sentence)
        frames.append(frame)
    request = dict(mode='bind', leaves=(0,), cue=atoms[4], predicate=atoms[6], role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=1,
                    reference_request=request)
    plans = [resolver.preview(clause, frames) for _ in range(2)]
    optimizer = torch.optim.SGD([*resolver.parameters(), *resolver.components.nouns.parameters()], lr=.01)
    for plan in plans:
        optimizer.zero_grad()
        plan['cost'].backward()
        assert resolver.policy.grad[2:].abs().sum() > 0
        assert resolver.cue_policy.grad.abs().sum() > 0
        optimizer.step()


def test_only_order_one_defers_reference_and_higher_orders_keep_their_contract():
    from ReferenceContext import ReferenceBank, prepare_operands
    values = torch.ones(2, 1, 4)
    ids, stamps = torch.tensor([[1], [2]]), torch.tensor([[1], [3]])
    bank = ReferenceBank(torch.tensor([[100], [200]]), values,
        torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool),
        values[:, 0], torch.zeros(2, dtype=torch.bool), orders=stamps, close_references=True)
    rule = SimpleNamespace(reference_roles=('argin1',), reference_modes=(('argin1', 'pronoun'),))
    positions = torch.zeros_like(ids)
    result = prepare_operands(values, ids, positions, positions, orders=stamps,
        rules=(), unary_rules=(rule,), bank=bank,
        live=(values, ids, stamps, torch.zeros_like(ids, dtype=torch.bool), positions),
        active=torch.ones_like(ids, dtype=torch.bool))
    refs = [result['unary_refs'][b, 0, result['unary_valid'][b, 0], 0].tolist() for b in range(2)]
    assert refs == [[0], [200]]
    rule.reference_modes = ()
    ordinary = prepare_operands(values, ids, positions, positions, orders=stamps,
        rules=(), unary_rules=(rule,), bank=bank,
        live=(values, ids, stamps, torch.zeros_like(ids, dtype=torch.bool), positions),
        active=torch.ones_like(ids, dtype=torch.bool))
    selected = ordinary['unary_valid'][0, 0]
    assert ordinary['unary_refs'][0, 0, selected, 0].tolist() == [1]
    assert ordinary['unary_choices'][0, 0, selected, 0].tolist() == [-1]


def test_deferred_reference_request_survives_kept_trial_detachment(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    f = SentenceFixture(monkeypatch)
    program = record_reading(f.language, replace(
        f.program(('lift', ('lower', 'cue', 'object'), 'predicate')),
        leaf_orders=torch.tensor([0, 0, 0]), close_references=True))
    scored = finish_clause(f.language, program, registry=f.registry)
    retained = program.detached()
    committed = finish_clause(f.language, retained, registry=f.registry)
    assert retained.close_references
    before, after = constituents(scored), constituents(committed)
    assert len(before) == len(after) == 1
    assert before[0].reference_request['mode'] == after[0].reference_request['mode'] == 'mint'
    assert before[0].reference_request['leaves'] == after[0].reference_request['leaves']
    torch.testing.assert_close(before[0].point, after[0].point, rtol=0, atol=0)


@pytest.mark.parametrize('description_order', (0, 2))
def test_kept_journal_supplies_np_content_predicate_and_positional_provenance(monkeypatch, description_order):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    from ReferenceContext import binding_spec, result_order
    f = SentenceFixture(monkeypatch)
    for rule in f.binary:
        mode = rule.determiner_mode
        if mode not in ('mint', 'bind', 'kind'):
            continue
        expected = 2 if mode == 'kind' else 1
        assert binding_spec(rule, 'argin2', description_order) == (expected, mode)
        stamps = torch.tensor([description_order, 17])
        assert result_order(rule, (torch.zeros_like(stamps), stamps)).tolist() == [expected, expected]
    program = f.program(('lift', ('lower', 'cue', 'object'), 'predicate'))
    program = record_reading(f.language, replace(program,
        leaf_orders=torch.tensor([0, description_order, 1]), close_references=True))
    clause = finish_clause(f.language, program, registry=f.registry)
    mentions = constituents(clause)
    assert len(mentions) == 1 and mentions[0].order == 1
    request = mentions[0].reference_request
    assert request['mode'] == 'mint' and request['leaves'] == (1,) and request['role'] == 0
    torch.testing.assert_close(mentions[0].point, program.leaves[1], rtol=0, atol=0)
    torch.testing.assert_close(request['predicate'], program.leaves[2], rtol=0, atol=0)
    torch.testing.assert_close(request['cue'], program.leaves[0], rtol=0, atol=0)
    torch.testing.assert_close(clause.point, program.end_state[0], rtol=0, atol=0)
    # The real conceptual allocator and clause taxonomy accept a first
    # mention immediately, including with statistical learning frozen.
    component = f.cs.components
    component._chart = EventChart(program.word_rows, program.leaves,
        capacity=component.capacity, content_width=8)
    resolver = component.reference_close
    f.cs._online_learning_frozen = True
    plan = resolver.preview(clause, ())
    before = component.nouns.ids
    kept = resolver.prepare(clause, plan)
    assert component.nouns.ids == before
    written = {}
    row = f.store.write_clause(kept, document_key='native', sentence_index=0, written_rows=written)
    resolver.commit(plan, row, written, document='native', position=0, training=False)
    identity = resolver.last_audit[0]['reference']
    assert identity in component.nouns.ids and f.cs._concept_source_order(identity) == 1
    assert component.nouns.witness_count(identity) == 1 and not hasattr(component.nouns, 'observe')
    occurrence = resolver.last_audit[0]['occurrence']
    assert f.store.row(f.store.index_of_row(occurrence))['order'] == 1
    restored = TernaryTruthStore(8, capacity=64)
    restored.load_state_dict(f.store.state_dict())
    restored.load_semantic_extras(f.store.semantic_extras())
    torch.testing.assert_close(restored.refs, f.store.refs, rtol=0, atol=0)
    torch.testing.assert_close(restored.slots, f.store.slots, rtol=0, atol=0)
