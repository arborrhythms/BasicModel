"""Fixed-operation, native-row closing certificates; no learned-parse claim."""
from chunk_fixtures import context as chunk_context
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
from order_fixtures import force_order

from ClauseRow import Clause, Composition, attach_clause_index
from Layers import LTM0
from Meaning import ConceptualMeaning
from ReferenceClose import ReferenceClose
from ReferenceContext import SituationFrame
from part_fixtures import part_references, assert_part_references


def fixture(*, percept=False):
    from test_cs_sparse_weights import _cs
    from IdentityRows import point
    owner = _cs(nS=512, order=3, dim=12)
    store = LTM0(12, capacity=512, percept_width=8 if percept else 0)
    import weakref
    object.__setattr__(owner, "_clause_store_ref", weakref.ref(store))
    resolver = owner.components.reference_close
    attach_clause_index(SimpleNamespace(), store, owner)
    return resolver, store, torch.eye(12)


@torch.no_grad()
def form_exposure(resolver, cue, content):
    """The fixture supplies a previously known native cue region."""
    from IdentityRows import mint
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([101], (), form='fixture-cue')
    row = owner._csw_row_of(word)
    owner.similarity_codebook.W[row].copy_(cue)
    mint(owner, content, form=word)


def close(resolver, point, predicate, mode, frames, *, document, sentence, selected=None,
          dummy=False, dummy_if_missing=False):
    request = dict(mode=mode, leaves=(1,), cue=torch.zeros_like(point), predicate=predicate, role=0)
    if dummy:
        request['compose_bind'] = 0
    term = Composition(point, order=1)
    np = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                reference_request=request, composition=term)
    value = torch.maximum(point, predicate)
    owner = resolver.components.owner
    from Spaces import _concept_alloc_of
    allocator = _concept_alloc_of(owner)
    predicates = resolver.__dict__.setdefault('_fixture_predicates', {})
    key = tuple(predicate.tolist())
    if key not in predicates:
        symbol = allocator.new_concept()
        force_order(owner, symbol, 0)
        row = owner._csw_concept_row(0, symbol)
        with torch.no_grad():
            owner.similarity_codebook.W[row].copy_(predicate)
        predicates[key] = symbol
    composition = Composition(value, (term, Composition(predicate, reference=predicates[key])),
                              operation='lift', order=2)
    clause = Clause(ConceptualMeaning.from_description(value), point=value,
                    children=(np,), refs=(('clause', 0), -1, -1), order=2,
                    composition=composition)
    plan = resolver.preview(clause, chunk_context(resolver, frames))
    if dummy_if_missing and plan['items'][0]['selected'] is None:
        # These complete mentions retain the compose menu's dummy outcome;
        # an abstract thought's unresolved bind has no such recorded choice.
        request['compose_bind'] = 0
        plan = resolver.preview(clause, chunk_context(resolver, frames))
    if selected is not None:
        assert mode != 'mint'
        plan['items'][0]['selected'] = plan['items'][0]['candidates'][selected]
    kept = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(kept, document_key=document, sentence_index=sentence,
                                     written_rows=written)
    resolver.commit(plan, row, written, document=document, position=sentence, training=False)
    frame = replace(SituationFrame.from_row(resolver.store.row(row)),
        symbol_ids=written.get(('symbols', id(kept)), ()),
        activations=written.get(('activations', id(kept)), ()))
    address = resolver.last_audit[0]['occurrence']
    return frame, address, plan


def current_identities(resolver, frames):
    """Observe the working partition without editing a row or a bank code."""
    store, bank = resolver.store, resolver.components.owner.similarity_codebook
    before = store.parts.refs.clone(), store.slots.clone(), bank.W.detach().clone()
    result = {address: identity
              for identity, item in resolver.individuals(resolver.accessible(chunk_context(resolver, frames))).items()
              for address in item['occurrences']}
    for actual, expected in zip((store.parts.refs, store.slots, bank.W), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return result


def test_mint_uses_the_selected_member_order_instead_of_the_source_word_order():
    resolver, store, atoms = fixture()
    request = dict(mode='mint', leaves=(0,), cue=atoms[0], content=atoms[0], role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        order=1, reference_request=request, composition=Composition(atoms[0], order=0))
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    assert kept.composition.order == 1
    before = len(resolver.components.individual_ids)
    written = {}
    row = store.write_clause(kept, written_rows=written)
    resolver.commit(plan, row, written, document='member', position=0, training=True)
    assert int(store.order[row]) == 1
    assert resolver.last_audit[0]['reference'] == store.occurrence_of(row)
    assert len(resolver.components.individual_ids) == before
    assert written[('activations', id(kept))] == ((store.occurrence_of(row), None, 0),)


@pytest.mark.parametrize('order', (0, 3))
def test_recent_token_returns_at_its_kinds_native_conceptual_order(order):
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    kind = owner.new_concept()
    force_order(owner, kind, order)
    seat = owner._csw_concept_row(order, kind)
    with torch.no_grad():
        owner.similarity_codebook.W[seat].copy_(atoms[0])
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        order=1, minted=True, composition=Composition(atoms[0], reference=kind, order=order))
    row = store.write_clause(clause)
    assert store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
    identity = store.occurrence_of(row)
    frame = SituationFrame.from_row(store.row(row))
    from chunk_fixtures import recall
    records = resolver.accessible(recall(resolver, (frame,)))
    candidates = resolver.individuals(records)
    assert identity in candidates and kind not in candidates
    candidate = candidates[identity]
    torch.testing.assert_close(candidate['value'], atoms[0], rtol=0, atol=0)
    assert int(store.order[row]) == 1 and owner._concept_source_order(kind) == order


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
            identities = current_identities(resolver, frames)
            assert len({identities[a] for a in addresses}) == 2
    identities = current_identities(resolver, frames)
    assert len({identities[a] for a in addresses}) == 1
    for address, point in zip(addresses, contents):
        torch.testing.assert_close(store.point_of_row(address), point, rtol=0, atol=0)


def same_kind_fourth_close(kind):
    resolver, store, atoms = fixture()
    noun, black, white, runs, sleeps = atoms[kind], *atoms[4:8]
    # Evidence teaches exclusivity; no color/word identities enter the owner.
    for i in range(8):
        for point, pred, label in ((noun+black, runs, 'a'), (noun+white, sleeps, 'b')):
            close(resolver, point, pred, 'bind', (), document=('exclusion', label), sentence=i, dummy=True)
    for document, second, distinct in (('mint-pair', 'mint', True), ('one', 'bind', False),
                                      ('revisable', 'bind', True)):
        frames, occurrences = [], []
        contents = (noun, noun, noun+black, noun+(white if distinct else black))
        for sentence, (content, predicate, mode) in enumerate(zip(contents,
                (runs, sleeps, runs, sleeps), ('mint', second, 'bind', 'bind'))):
            frame, address, plan = close(resolver, content, predicate, mode, frames,
                document=(document, kind), sentence=sentence,dummy_if_missing=True)
            frames.append(frame); occurrences.append(address)
            if mode == 'mint':
                assert plan['items'][0]['selected'][0] == 'new'
            else:
                assert all(isinstance(ref, tuple) and ref[:2] == ('ltm', store.address_domain)
                           and isinstance(store.location(ref), int)
                           for ref in plan['items'][0]['candidates'])
        identities = current_identities(resolver, frames)
        references = [identities[address] for address in occurrences]
        assert len(set(references)) == (2 if distinct else 1)
        for address, content in zip(occurrences, contents):
            torch.testing.assert_close(store.point_of_row(address), content, rtol=0, atol=0)
        yield references


@pytest.mark.parametrize('kind', range(4))
def test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding(kind):
    list(same_kind_fourth_close(kind))


@pytest.mark.phase('mtm_reidentification')
@pytest.mark.parametrize('kind', range(4))
def test_reidentification_revises_the_earlier_ambiguous_binding(kind):
    for references in same_kind_fourth_close(kind):
        assert references[0] == references[2] and references[1] == references[3]


def test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised():
    resolver, store, atoms = fixture()
    frame, address, _ = close(resolver, atoms[0], atoms[6], 'mint', (), document='a', sentence=0)
    before = part_references(store)
    _, missing, plan = close(resolver, torch.zeros(12), atoms[6], 'pronoun', (), document='b', sentence=0)
    assert plan['items'][0]['reference'] is None and len(resolver.components.individual_ids) == 0
    assert tuple(resolver.components.owner.mtm.entries) == (store.occurrence_of(store.index_of_row(address)),)
    assert not hasattr(store, 'refs')
    assert_part_references(store, before)
    assert resolver.last_audit[0]['missing_reason'] == 'no_accessible_individual'


def test_empty_close_has_no_policy_gradient_or_store_side_effect():
    resolver, store, atoms = fixture()
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=0)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    assert float(plan['cost']) == 0 and not plan['cost'].requires_grad
    kept = resolver.prepare(clause, plan)
    assert kept is clause
    written = {}
    row = store.write_clause(kept, document_key='field', sentence_index=0, written_rows=written)
    metadata = dict(store._semantic_rows)
    required, refs = store.metadata_required.clone(), store.parts.refs.clone()
    resolver.commit(plan, row, written, document='field', position=0, training=True)
    assert store._semantic_rows == metadata and resolver.last_audit == ()
    torch.testing.assert_close(store.metadata_required, required, rtol=0, atol=0)
    torch.testing.assert_close(store.parts.refs, refs, rtol=0, atol=0)
    assert not resolver.components.individual_ids and not resolver._participants()


def test_answer_credit_credits_the_departure_without_changing_exact_reconstruction():
    from Layers import Error
    from SentenceCredit import components, comparison, score_function
    registry, like = Error(), torch.zeros(1)
    registry.add('reconstruction.exact', like, category='reconstruction')
    # Only a supplied referent's answer error enters A; local bind prices do not.
    parts = torch.stack([components(registry, like+100, like, answer=like+1),
                         components(registry, like+200, like, answer=like)], 1)
    audit = comparison(parts, torch.ones(1, dtype=torch.bool))
    assert audit['keep_costs'].tolist() == [[1., 0.]] and bool(audit['wins'].all())
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
    outside = resolver.last_audit[0]['reference']
    before = {address: store.point_of_row(address).clone() for address in addresses}
    identities = current_identities(resolver, frames)
    assert len({identities[a] for a in addresses}) == 1
    assert outside not in set(identities.values())
    introductions = (*addresses[::2], inaccessible)
    assert len(set(introductions)) == 4
    assert {store.occurrence_of(store.index_of_row(a)) for a in introductions} == set(
        resolver.components.owner.mtm.entries)
    assert not resolver.components.individual_ids
    for address, point in before.items():
        torch.testing.assert_close(store.point_of_row(address), point, rtol=0, atol=0)


def test_closing_credit_keeps_both_trial_graphs_and_learns_salience_and_form_cues():
    resolver, store, atoms = fixture()
    frames = []
    for sentence in range(2):
        frame, _, _ = close(resolver, atoms[sentence], atoms[6+sentence], 'mint', frames,
                            document='credit', sentence=sentence)
        frames.append(frame)
    form_exposure(resolver, atoms[4], atoms[0])
    request = dict(mode='bind', leaves=(0,), cue=atoms[4], predicate=atoms[6], role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=1,
                    reference_request=request)
    plans = [resolver.preview(clause, chunk_context(resolver, frames)) for _ in range(2)]
    optimizer = torch.optim.SGD(resolver.parameters(), lr=.01)
    for plan in plans:
        optimizer.zero_grad()
        plan['cost'].backward()
        assert resolver.policy.grad[2:].abs().sum() > 0
        assert resolver.policy.grad[4].abs() > 0
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


@pytest.mark.parametrize('candidate_order', (0, 3))
@pytest.mark.parametrize('reference_role', (False, True))
def test_implicit_reference_offers_retained_definitions_at_their_native_orders(candidate_order, reference_role):
    from Interpret import InterpretLayer
    from ReferenceContext import ReferenceBank, prepare_operands
    values = torch.ones(1, 1, 4)
    ids, stamps = torch.tensor([[7]]), torch.tensor([[1]])
    flags = torch.zeros_like(ids)
    bank = ReferenceBank(torch.tensor([[100]]), values * .5,
        torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool),
        values[:, 0], torch.zeros(1, dtype=torch.bool),
        orders=torch.tensor([[candidate_order]]), close_references=True,
        associations=torch.tensor([[7, 100]]), interpreter=InterpretLayer())
    rule = SimpleNamespace(reference_roles=('argin1',) if reference_role else ())
    result = prepare_operands(values, ids, flags, flags, orders=stamps,
        rules=(), unary_rules=(rule,), bank=bank,
        live=(values, ids, stamps, flags.bool(), flags),
        active=torch.ones_like(ids, dtype=torch.bool))
    valid = result['unary_valid'][0, 0]
    refs = result['unary_refs'][0, 0, valid, 0]
    assert 100 in refs and 7 not in refs
    torch.testing.assert_close(result['unary'][0, 0, valid][refs == 100][0], values[0, 0] * .5)


def test_deferred_reference_request_survives_kept_trial_detachment(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    f = SentenceFixture(monkeypatch, native_cube=True)
    program = record_reading(f.language, replace(
        f.program(('lift', ('lower', 'cue', 'object'), 'predicate')),
        leaf_orders=torch.tensor([0, 1, 0]), close_references=True))
    scored = finish_clause(f.language, program, registry=f.registry)
    retained = program.detached()
    committed = finish_clause(f.language, retained, registry=f.registry)
    assert retained.close_references
    before, after = constituents(scored), constituents(committed)
    assert len(before) == len(after) == 1
    assert before[0].reference_request['mode'] == after[0].reference_request['mode'] == 'mint'
    assert before[0].reference_request['leaves'] == after[0].reference_request['leaves']
    torch.testing.assert_close(before[0].point, after[0].point, rtol=0, atol=0)


def test_objectless_marker_stays_literal_under_an_implicit_reference_read():
    from Interpret import InterpretLayer
    from ReferenceContext import ReferenceBank, prepare_operands
    value = torch.ones(1, 1, 4)
    ids = torch.tensor([[7]])
    zero = torch.zeros_like(ids)
    bank = ReferenceBank(torch.tensor([[100]]), value * .5,
        torch.ones_like(ids, dtype=torch.bool), zero.bool(), value[:, 0], zero[:, 0].bool(),
        orders=zero, interpreter=InterpretLayer(), literal_ids=torch.tensor([7]))
    rule = SimpleNamespace(reference_roles=('argin1',))
    result = prepare_operands(value, ids, zero, zero, orders=zero, rules=(), unary_rules=(rule,),
        bank=bank, live=(value, ids, zero, zero.bool(), zero), active=torch.ones_like(ids, dtype=torch.bool))
    valid = result['unary_valid'][0, 0]
    assert result['unary_refs'][0, 0, valid, 0].tolist() == [7]
    torch.testing.assert_close(result['unary'][0, 0, valid][0], value[0, 0])


@pytest.mark.parametrize('description_order', (1, 2))
def test_kept_journal_supplies_np_content_predicate_and_positional_provenance(monkeypatch, description_order):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    from ReferenceContext import binding_spec, denoted_order
    f = SentenceFixture(monkeypatch, native_cube=True)
    for rule in f.binary:
        mode = rule.determiner_mode
        if mode not in ('mint', 'bind', 'kind'):
            continue
        expected = max(1, description_order - 1)
        assert binding_spec(rule, 'argin2', description_order) == (expected, mode)
        stamps = torch.tensor([description_order, 17])
        assert denoted_order(rule, (torch.zeros_like(stamps), stamps), (), None).tolist() == [expected, 16]
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
    resolver = component.reference_close
    f.cs._online_learning_frozen = True
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    before = component.individual_ids
    kept = resolver.prepare(clause, plan)
    assert component.individual_ids == before
    written = {}
    row = f.store.write_clause(kept, document_key='native', sentence_index=0, written_rows=written)
    resolver.commit(plan, row, written, document='native', position=0, training=False)
    identity = resolver.last_audit[0]['reference']
    assert isinstance(identity, tuple) and identity[:2] == ('ltm', f.store.address_domain)
    assert component.individual_ids == before
    assert f.store.raw_meaning_of(f.store.location(identity)).role_refs[0] == ('sym', program.concept_ids[1].item())
    assert int(f.store.witness_count[f.store.location(identity)]) == 1 and not hasattr(component, 'nouns')
    occurrence = resolver.last_audit[0]['occurrence']
    assert f.store.row(f.store.index_of_row(occurrence))['order'] == 1
    restored = LTM0(8, capacity=64, percept_width=0)
    restored.load_state_dict(f.store.state_dict())
    restored.load_semantic_extras(f.store.semantic_extras())
    assert not hasattr(restored, 'refs') and 'refs' not in restored.state_dict()
    torch.testing.assert_close(restored.slots, f.store.slots, rtol=0, atol=0)
