"""Reference collections, state errors and the bounded shared sentence stores."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from test_meaning_kernels import paired


def receipt(name, data):
    root = os.environ.get('ITEM20_RECEIPT')
    if root:
        path = Path(root); path.mkdir(parents=True, exist_ok=True)
        (path / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


def test_all_link_callers_window_likeness_and_reference_values():
    from test_reference_close import fixture
    from MemoryLink import Occurrence
    from Layers import Error
    owner_resolver, store, atoms = fixture()
    owner = owner_resolver.components.owner
    errors = Error()
    object.__setattr__(owner, '_model', NS(_sentence_close_errors=errors))
    owner.mtm.horizon = 2
    def observed(value, number, kind='state', order=0):
        row = store.append_idea(value, address=number, when=False)
        address = store.occurrence_of(row)
        key = owner.link(Occurrence(address, kind=kind, order=order), {},
            value=value, where=torch.zeros(4), when=number, stream=0)
        owner.mtm.close(store, stream=0)
        return key, address
    first, a = observed(atoms[0], 101)
    second, b = observed(atoms[0], 102)
    assert first == second and owner.mtm.entries[first].occurrences == (a, b)
    recurrence, c = observed(atoms[0], 103)
    assert recurrence == first
    unlike, _ = observed(atoms[1], 104)
    assert unlike != first
    higher, _ = observed(atoms[0], 105, order=1)
    assert higher != first
    owner.mtm.closes[0] += 3
    distant, _ = observed(atoms[0], 106)
    assert distant != first
    reference, _ = observed(atoms[0], 107, kind='reference')
    event, _ = observed(atoms[0], 108, kind='event')
    assert reference != event != distant
    entry = owner.mtm.entries[event]
    assert not hasattr(entry, 'value') and not hasattr(entry.timelines[0], 'value')
    assert 'memory.link_coverage' in errors._terms
    term = errors._terms['memory.link_coverage']
    assert term['objective'] == 'reconstruction' and term['side'] == 'R'
    receipt('link-callers', dict(callers=['reference', 'event', 'state'], consecutive_same=True,
        recurrence=True, distant_dummy=True, cross_order=False, stored_values=0,
        version=owner.mtm.state()['version'], terms=list(errors._terms)))


def test_merge_path_promotion_cycle_and_epoch_withdrawal():
    from test_reference_close import fixture
    from MemoryLink import Occurrence
    from FamiliarTier import promote_identity
    from Memory import OCCURRENCE, SYMBOL
    from ChainRead import path
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    parts = owner._concept_allocator.parts
    parts.begin_epoch()
    keys, addresses = [], []
    for clock in range(4):
        row = store.append_idea(atoms[0], address=201+clock, when=False)
        address = store.occurrence_of(row); addresses.append(address)
        pool.closes[0] = clock
        keys.append(owner.link(Occurrence(address, kind='state', order=0,
            selected=keys[clock % 2] if clock >= 2 else None), {},
            value=atoms[0], where=None, when=clock, stream=0))
    assert keys[0] == keys[2] and keys[1] == keys[3]
    merged = pool.merge(keys[1], keys[0], when=4)
    assert pool.entries[merged].occurrences == tuple(addresses)
    transient = pool.entries[merged].transient
    concept = promote_identity(owner, merged)
    assert pool.entries[merged].transient is None
    members = tuple((OCCURRENCE, address[-1]) for address in addresses)
    assert path(store, concept) == members
    assert all(parts.find(member, (SYMBOL, concept)) is not None for member in members)
    assert all(parts.find(left, right) is not None for left, right in zip(members, members[1:]))
    parts.add(members[-1], members[0], truth_plus=1.)
    assert set(path(store, concept)) == set(members) and len(path(store, concept)) == 4
    removed = parts.begin_epoch()
    assert removed and path(store, concept) == ()
    assert concept in owner._concept_allocator.symbol_types
    receipt('promotion-path', dict(members=4, membership_rows=4, successor_rows=3,
        cycle_visits=4, released=transient, order=owner._concept_source_order(concept),
        withdrawn=len(removed), retained_concept=concept))


def test_promoted_reference_reenters_as_one_higher_order_member():
    from test_reference_close import fixture
    from MemoryLink import Occurrence
    from FamiliarTier import promote_identity
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    key, original = None, []
    for clock in range(3):
        row = store.append_idea(atoms[0], address=301+clock, when=False)
        address = store.occurrence_of(row)
        original.append(address)
        key = owner.link(Occurrence(address, order=0, selected=key), {},
            value=atoms[0], where=None, when=clock, stream=0)
        pool.closes[0] = clock+1
    concept = promote_identity(owner, key)
    assert owner._concept_source_order(concept) == 1
    from chunk_fixtures import context
    candidates = resolver.individuals(resolver.accessible(context(resolver, ()), stream=0))
    choices, _, _ = resolver.score(atoms[0], atoms[0]*0, atoms[0], 0,
        candidates, stream=0, member_order=1)
    assert concept in choices and key not in choices
    higher = []
    for clock in range(3, 5):
        row = store.append_idea(atoms[0], order=1, address=301+clock, when=False)
        address = store.occurrence_of(row)
        higher.append(owner.link(Occurrence(address, concept=concept, order=1, selected=key), {},
            value=atoms[0], where=None, when=clock, stream=0))
        pool.closes[0] = clock+1
    assert higher[0] == higher[1] != key
    assert pool.entries[key].occurrences == tuple(original)
    assert pool.entries[key].refinement['member_order'] == 0
    assert pool.entries[higher[0]].native == concept
    assert pool.entries[higher[0]].refinement['member_order'] == 1
    assert len(pool.entries[higher[0]].occurrences) == 2
    from ChainRead import path
    path_before = path(store, concept)
    assert promote_identity(owner, higher[0]) == concept
    assert path(store, concept) == path_before and len(path_before) == 3
    with pytest.raises(ValueError, match='cannot mix collection kinds or orders'):
        owner.link(Occurrence(original[0], order=2, selected=key), {},
            value=atoms[0], where=None, when=5, stream=0)
    receipt('promoted-reference-reentry', dict(original_members=3, original_order=0,
        promoted_order=1, reentries=2, higher_collections=1, original_path_unchanged=True,
        unrelated_order_rejected=True, concept_offered_separately=True,
        reentry_self_memberships=0))


def test_event_relation_path_promotes_with_native_tags_and_bands():
    from test_reference_close import fixture
    from MemoryLink import Occurrence
    from FamiliarTier import promote_identity
    from Memory import OCCURRENCE, RELATION, SYMBOL
    from ChainRead import path
    resolver, store, atoms = fixture()
    owner, parts = resolver.components.owner, store.parts
    key, members = None, []
    for clock in range(3):
        left = store.append_idea(atoms[0], address=331+2*clock, when=False)
        right = store.append_idea(atoms[1], address=332+2*clock, when=False)
        relation = parts.add((OCCURRENCE, store.occurrence_of(left)[-1]),
            (OCCURRENCE, store.occurrence_of(right)[-1]), truth_plus=1.,
            when=torch.full((4,), .1*(clock+1)), where=torch.full((4,), .2*(clock+1)))
        members.append((RELATION, relation))
        key = owner.link(Occurrence(('sym', relation), kind='event', order=0, selected=key), {},
            value=atoms[0]+atoms[1], where=None, when=clock, stream=0)
        owner.mtm.closes[0] = clock+1
    count = len(store)
    concept = promote_identity(owner, key)
    assert len(store) == count and path(store, concept) == tuple(members)
    for member in members:
        row = parts.find(member, (SYMBOL, concept))
        torch.testing.assert_close(parts.percept[row], parts.percept[member[1]])
    for left, right in zip(members, members[1:]):
        row = parts.find(left, right)
        torch.testing.assert_close(parts.percept[row], parts.percept[right[1]])
    parts.add(members[-1], members[0], truth_plus=1.)
    assert len(path(store, concept)) == 3 and set(path(store, concept)) == set(members)
    receipt('event-relation-promotion', dict(event_members=3, membership_rows=3,
        successor_rows=2, member_tag='RELATION', new_idea_rows=0,
        bands_preserved=True, cycle_visits=3))


def test_replaced_root_links_its_written_event_idea_instead_of_its_relation():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    subject = known_concept(owner, atoms[0], order=1)
    verb = known_concept(owner, atoms[1], order=0)
    first = store.append_idea(atoms[0], document_key='event-subject', when=False)
    subject_address = store.occurrence_of(first)
    subject_key = owner.link(Occurrence(subject_address, referent=subject,
        native=subject, selected=None), {}, value=atoms[0], where=None, when=0, stream=0)
    earlier = store.append_idea(atoms[2], document_key='event-before', when=False)
    earlier_address = store.occurrence_of(earlier)
    event_key = owner.link(Occurrence(earlier_address, kind='event', order=0, selected=None),
        {}, value=atoms[2], where=None, when=0, stream=0)
    terms = (Composition(atoms[0], reference=subject), Composition(atoms[1], reference=verb))
    record = dict(action=0, event=(0, 1, 0, verb, 0, False), result=atoms[2],
        operation='part', origins=torch.tensor([[2, subject_address[-1]]]))
    root = Clause(ConceptualMeaning.from_description(atoms[2]), point=atoms[2],
        order=0, composition=Composition(atoms[2], terms, operation='part'),
        identity=record, evidence=(1., 0.))
    plan = dict(prepared_root=root, items=[], stream=0,
                link_choices={('event', 0): event_key})
    written = {}
    row = store.write_clause(replace(root, when=torch.ones(4)), written_rows=written)
    assert isinstance(row, tuple)
    event_address = written['root_event_occurrence']
    assert event_address[:1] == ('ltm',) and event_address != written['root_occurrence']
    resolver.link_events(plan, written, when=1)
    assert owner.mtm.entries[subject_key].events[-1].occurrence == event_address
    assert owner.mtm.entries[event_key].occurrences == (earlier_address, event_address)
    torch.testing.assert_close(owner.mtm.event_value(owner.mtm.entries[subject_key].events[-1]), atoms[2])
    receipt('root-event-handoff', dict(replaced_roots=1, event_idea_addresses=1,
        relation_substitutions_before=1, relation_substitutions_after=0,
        event_path_members=2, same_recorded_result=True))


def test_word_member_order_is_its_read_concept_not_the_clause_order():
    from test_reference_close import fixture
    from MemoryLink import Occurrence, member_order
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([101], (), form='member')
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(atoms[0])
    key = None
    for clock in range(2):
        row = store.append_idea(atoms[0], order=clock, address=351+clock, when=False)
        site = Occurrence(store.occurrence_of(row), concept=word, selected=key)
        assert member_order(owner, site) == 0
        linked = owner.link(site, {}, value=atoms[0], where=None, when=clock, stream=0)
        assert key is None or key == linked
        key = linked
        pool.closes[0] = clock+1
    assert pool.entries[key].refinement['member_order'] == 0
    assert len(pool.entries[key].occurrences) == 2
    receipt('reference-member-order', dict(clause_orders=[0, 1], read_concept_order=0,
        collections=1, members=2))


def test_reference_departure_filters_collection_order_before_selection():
    from test_reference_close import fixture, known_concept
    from order_fixtures import force_order
    from MemoryLink import Occurrence
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    from ReferenceContext import SituationFrame
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([102], (), form='reference-menu')
    force_order(owner, word, 1)
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(atoms[0])
    assert owner._concept_source_order(owner.interpret.referent(word)) == 1
    keys = []
    for order in (0, 1):
        row = store.append_idea(atoms[0], order=order, address=371+order, when=False)
        keys.append(owner.link(Occurrence(store.occurrence_of(row),
            concept=word if order == 1 else None, order=order, selected=None), {},
            value=atoms[0], where=None, when=0, stream=0))
    request = dict(mode='bind', forward_identity=word, leaves=(0,), cue=atoms[0],
        content=atoms[0], predicate=torch.zeros_like(atoms[0]), role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=1,
        subject_word_id=word, reference_request=request,
        composition=Composition(atoms[0], reference=word, word_id=word, order=1))
    native = known_concept(owner, atoms[0], order=0)
    frame = SituationFrame(1, torch.stack((atoms[0], atoms[0]*0, atoms[0]*0)),
        torch.tensor([True, False, False]), -1, atoms[0], symbol_ids=(native,))
    plan = resolver.preview(clause, context(resolver, (frame,)), stream=0)
    choices = plan['items'][0]['candidates']
    assert keys[0] not in choices and keys[1] in choices and native not in choices
    selected = plan['items'][0]['selected']
    assert selected != keys[0]
    from FamiliarTier import promote_identity
    promoted = promote_identity(owner, keys[1])
    assert owner._concept_source_order(promoted) == 2
    receipt('reference-departure-order', dict(equal_content_candidates_before=2,
        eligible_collection_candidates_after=1, query_order=1, excluded_order=0,
        incompatible_native_candidates_offered=0, filtered_before_selection=True, promoted_order=2))


def test_compose_bind_filters_the_same_collection_order_before_choice():
    from Memory import STMChunks as chunks
    from ReferenceContext import prepare_operands
    state = chunks.empty(1, 8, 12)
    mention = torch.ones(1, 1, 12)
    for order in (0, 1):
        state, _ = chunks.deposit(state, mention[:, 0], torch.tensor([True]),
            order=order, identity=101+order)
    ages = state[chunks.READ].clone()
    for order in (-1, 0, 1):
        menu = prepare_operands(mention.expand(1, 2, 12), torch.full((1, 2), -1),
            torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long),
            orders=torch.full((1, 2), order), rules=(NS(method_name='bind'),), unary_rules=(),
            active=torch.tensor([[True, False]]), chunk_state=state, bank=None, live=None)
        valid = menu['binary_valid']
        selected = menu['binary_refs'][..., 1][valid].tolist()
        assert set(selected) == ({101, 102} if order < 0 else {0, 101+order})
    torch.testing.assert_close(state[chunks.READ], ages, rtol=0, atol=0)
    receipt('compose-bind-order', dict(query_orders=[0, 1], equal_content_candidates=2,
        candidates_before=3, candidates_after=2, includes_dummy=True, menu_is_read_only=True,
        unknown_query_keeps_both_referents=True, unknown_query_has_no_dummy=True))


def test_staged_pending_names_keep_their_member_order_on_arrival_and_refresh():
    from test_reference_arrivals import world, described
    from test_medium_term_memory import keep
    from order_fixtures import force_order
    from PrependedSymbols import SymbolEntries, stage
    from Memory import STMChunks as chunks
    for order in (0, 1):
        resolver, store, atoms, words, _ = world(pending_atom=0)
        owner = resolver.components.owner
        force_order(owner, words['x'], order)
        row, _ = keep(resolver, described(atoms[0], words['x']))
        key = owner.mtm.token_of(store.occurrence_of(row))
        owner.mtm.record_link(key, store.occurrence_of(row), 0, kind='reference', order=order)
        entry = owner.mtm.entries[key]
        assert entry.pending and entry.refinement['member_order'] == order
        entries = SymbolEntries(1, 1, 8, store.nDim)
        model = NS(_concept_owner=lambda: owner, inputSpace=NS(symbol_entries=entries),
            symbolSpace=NS(ltm_store=store), _open_sentence_slot=0)
        state = stage(model, chunks.empty(1, 8, store.nDim), torch.tensor([True]))
        index = (state[chunks.IDENTITY][0] == entry.transient).nonzero().item()
        assert state[chunks.ORDERS][0, index] == order
        assert entries.concept_orders[0, 0] == order
        # A resident seat may still hold its pre-promotion/pre-link stamp.
        # Refresh must update metadata together with its working value.
        state[chunks.ORDERS][0, index] = 1-order
        state[chunks.ROWS][0, index] = 999
        refreshed = stage(model, state, torch.tensor([True]))
        assert refreshed[chunks.ORDERS][0, index] == order
        assert refreshed[chunks.ROWS][0, index] == -1
        assert int((refreshed[chunks.IDENTITY] == entry.transient).sum()) == 1
    receipt('staged-reference-order', dict(member_orders=[0, 1], arrival_orders=[0, 1],
        refreshed_orders=[0, 1], duplicate_seats=0, stale_rows_after=0))


def test_pending_name_arrival_filters_the_written_candidates_order():
    from test_reference_arrivals import world, described, equality
    from test_medium_term_memory import keep
    from ReferenceContext import SituationFrame
    from chunk_fixtures import context
    from order_fixtures import force_order
    resolver, store, atoms, words, _ = world(pending_atom=0)
    owner = resolver.components.owner
    force_order(owner, words['x'], 1)
    row, _ = keep(resolver, described(atoms[0], words['x']))
    key = owner.mtm.token_of(store.occurrence_of(row))
    assert owner.mtm.entries[key].refinement['member_order'] == 1
    clause = equality(atoms[0], words['x'], words['wrong'], subject=atoms[0])
    plan = resolver.preview(clause, context(resolver, (SituationFrame.from_row(store.row(row)),)))
    assert plan['arrival_count'] > 0 and plan['arrivals']
    assert all(not (isinstance(choice, tuple) and choice[:1] == ('arrival',))
        for item in plan['arrivals'] for choice in item['candidates'])
    receipt('pending-arrival-order', dict(pending_member_order=1,
        new_candidate_order=0, proposed_constituents=plan['arrival_count'],
        incompatible_constituents_offered=0, filtered_before_selection=True))


def test_state_teachers_use_later_references_and_keep_stored_values_detached():
    from test_item20_round34_actions import seat
    from memory_fixtures import put
    from VerbLearning import observe
    from ModalLearning import observe as observe_modal
    from Layers import Error
    from Occurrence import ADDRESS_DOMAIN
    owner, store, key, timeline, action, modifier, sources = seat(alternative=False)
    pool = owner.mtm
    previous = dict(pool.entries)
    later = ('ltm', ADDRESS_DOMAIN, 41)
    target = torch.full((8,), .65, requires_grad=True)
    put(store, later, target)
    pool.bind(key, later, kind=11, when=3)
    errors = Error()
    observe(action, pool, previous, {key}, training=True, errors=errors)
    value = errors.total(objective='reconstruction')
    assert torch.isfinite(value).all()
    gradient, = torch.autograd.grad(value.sum(), (action.readout,))
    assert bool(gradient.abs().any())
    assert target.grad is None and all(source.grad is None for source in sources)
    event = previous[key].events[-1]
    from MemoryState import verb
    with torch.no_grad(): action.readout.zero_()
    error, baseline = verb(action, event, target, Error())
    torch.testing.assert_close(error, baseline)
    actual, alternative = ('ltm', ADDRESS_DOMAIN, 42), ('ltm', ADDRESS_DOMAIN, 43)
    put(store, actual, torch.full((8,), .3)); put(store, alternative, torch.full((8,), .7))
    pool.event(key, actual, verb=12, role=0, when=4)
    branch = pool.alternative(key, when=4)
    pool.event(key, alternative, verb=12, role=0, when=4, timeline=branch,
        modifiers=(torch.full((8,), .4),))
    modal_errors = Error()
    observe_modal(modifier, pool, {key}, training=True, errors=modal_errors)
    modal = modal_errors.total(objective='reconstruction')
    gradient_modal, = torch.autograd.grad(modal.sum(), (modifier.readout,))
    assert bool(gradient_modal.abs().any())
    empty = Error()
    observe(action, pool, previous, {key}, training=False, errors=empty)
    observe_modal(modifier, pool, {key}, training=False, errors=empty)
    assert not empty._terms
    receipt('state-teachers', dict(verb=float(value.detach()), modal=float(modal.detach()),
        verb_gradient=float(gradient.norm()), modal_gradient=float(gradient_modal.norm()),
        stored_gradients=False, identity_equals_baseline=True, evaluation_terms=0))


def test_unnamed_alias_reads_a_pending_names_order_without_resolving_it():
    from dataclasses import replace
    from test_reference_arrivals import world, described
    from test_medium_term_memory import keep
    from order_fixtures import force_order
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from MemberRead import read
    from ThoughtReferences import open_slots
    resolver, store, atoms, words, _ = world(pending_atom=0)
    owner = resolver.components.owner
    force_order(owner, words['x'], 1)
    row, _ = keep(resolver, described(atoms[0], words['x']))
    first = store.occurrence_of(row)
    key = owner.mtm.token_of(first)
    assert owner.mtm.entries[key].pending
    own = (atoms[0] * .7).requires_grad_()
    meaning = replace(ConceptualMeaning.from_description(own), bindings={'_identity': first})
    alias = store.append_meaning(meaning, when=False, document_key='unnamed-alias')
    address = store.occurrence_of(alias)
    assert store.order_of_row(row) == store.order_of_row(alias) == 1
    linked = owner.link(Occurrence(address, selected=key), {}, value=own,
                        where=None, when=1, stream=0)
    assert linked == key and owner.mtm.entries[key].pending
    assert ('referent', 0) in open_slots(store.raw_meaning_of(row))
    torch.testing.assert_close(read(owner, address), own.detach())
    assert owner.mtm.reading_of(address) is None
    assert owner.mtm.entries[key].refinement['member_order'] == 1
    receipt('pending-alias-order', dict(named_order=1, alias_order_before=0, alias_order_after=1,
        members=2, pending_after=True, alias_reads_own_idea=True, alias_word_readings=0))


def test_alias_order_follows_the_words_denotation_for_symbol_and_name_references():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from MemoryLink import Occurrence
    from Meaning import ConceptualMeaning
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([103], (), form='alias-order')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept)
    assert owner._concept_source_order(word) == 0
    assert owner._concept_source_order(owner.interpret.referent(word)) == 1
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': ('sym', owner.interpret.referent(word))}), address=391, when=False)
    key = owner.link(Occurrence(store.occurrence_of(first), concept=word, selected=None), {},
        value=atoms[0], where=None, when=0, stream=0)
    for tag in ('sym', 'name'):
        # A new reading selects the denotation before writing its native
        # reference. Reading a stored occurrence does not reinterpret a name.
        reference = (('sym', owner.interpret.referent(word)) if tag == 'sym'
                     else store.occurrence_of(first))
        meaning = replace(ConceptualMeaning.from_description(atoms[0]),
            bindings={'_identity': reference})
        row = store.append_meaning(meaning, when=False, document_key=tag)
        assert store.order_of_row(row) == 1
        assert owner.link(Occurrence(store.occurrence_of(row), selected=key), {},
            value=atoms[0], where=None, when=1, stream=0) == key
    assert len(owner.mtm.entries[key].occurrences) == 3
    receipt('word-alias-order', dict(word_order=0, denotation_order=1,
        alias_orders_before=[0, 0], alias_orders_after=[1, 1], reference_kinds=['sym', 'name'],
        collection_members=3, collection_member_order=1))


def test_shared_definition_witness_resolves_a_null_token_only_once(monkeypatch):
    import FamiliarTier
    from test_reference_close import fixture
    from Meaning import ConceptualMeaning
    from Memory import OCCURRENCE, SYMBOL
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    label = owner.interpret.lookup_word([104], (), form='shared-sign')
    row = store.append_meaning(ConceptualMeaning.from_description(atoms[0]*0),
        order=0, when=False)
    address = store.occurrence_of(row)
    key = owner.mtm.open(address, kind=owner.null_concept_id, when=0, value=atoms[0]*0)
    left, right = (OCCURRENCE, address[-1]), (SYMBOL, label)
    store.parts.add(left, right, truth_plus=1.)
    witness = (left, right, None, label, True)
    owner._kept_definition_witnesses = [witness, witness]
    calls = []
    original = FamiliarTier.define_pending
    def observe(*args, **kwargs):
        calls.append((args[1:], kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(FamiliarTier, 'define_pending', observe)
    FamiliarTier.finish_definitions(owner)
    assert len(calls) == 1
    entry = owner.mtm.entries[key]
    assert entry.symbol is not None and entry.kind == entry.symbol
    assert not entry.pending
    assert not hasattr(owner, '_kept_definition_witnesses')
    receipt('shared-definition-witness', dict(queued_views=2, distinct_kept_relations=1,
        definition_calls_before=2, definition_calls_after=1, promoted_tokens=1))


def test_kept_word_read_reenters_a_definition_learned_after_its_pending_path():
    from test_reference_close import fixture, known_concept
    from FamiliarTier import commit_read_words
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([105], (), form='later-definition')
    key = pool.named(word, value=atoms[0], when=0)
    reading = NS(word_ids=torch.tensor([word]), leaves=atoms[0][None], symbol_where=None)
    def close(position):
        root = store.append_idea(atoms[0], address=401+position, when=False)
        return commit_read_words(owner, reading, batch=0, document='later-definition',
            position=position, written={'root_occurrence': store.occurrence_of(root)})[0]
    first = close(0)
    assert pool.entries[key].refinement['member_order'] == 0
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=.1)
    assert not owner.definitions.objects(word) and owner.interpret.referent(word) == concept
    second = close(1)
    assert pool.entries[key].symbol == concept
    assert pool.entries[key].occurrences == (first,)
    new = pool.token_of(second)
    assert new != key and pool.entries[new].native == concept
    assert pool.entries[new].refinement['member_order'] == 1
    assert store.order_of_row(store.location(second)) == 1
    receipt('kept-word-order-transition', dict(pending_member_order=0, learned_referent_order=1,
        old_path_members_before=1, old_path_members_after=1, new_native_members=1,
        new_collection_order=1, mixed_order_links=0))


def test_unnamed_reference_reenters_a_pending_words_later_definition():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from MemoryLink import Occurrence
    from Meaning import ConceptualMeaning
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([114], (), form='later-reference')
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(atoms[0])
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_forward_references': ((0, word),)}), when=False, document_key='first')
    address = store.occurrence_of(first)
    key = owner.link(Occurrence(address, concept=word, pending=True, selected=None), {},
        value=atoms[0], where=None, when=0, stream=0)
    assert pool.entries[key].refinement['member_order'] == 0
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=.1)
    own = atoms[0] + atoms[1]
    alias = store.append_meaning(replace(ConceptualMeaning.from_description(own),
        bindings={'_identity': ('sym', owner.interpret.referent(word))}),
        when=False, document_key='unnamed')
    new = owner.link(Occurrence(store.occurrence_of(alias), selected=key), {},
        value=own, where=None, when=1, stream=0)
    assert pool.entries[key].symbol == concept and pool.entries[key].occurrences == (address,)
    assert new != key and pool.entries[new].native == concept
    assert pool.entries[new].refinement['member_order'] == 1
    assert pool.reading_of(store.occurrence_of(alias)) is None
    torch.testing.assert_close(pool.read_member(store.occurrence_of(alias)), own)
    receipt('unnamed-known-reentry', dict(old_member_order=0, new_member_order=1,
        original_path_members=1, new_native_members=1, extra_word_readings=0,
        alias_reads_own_idea=True))


def test_shared_sentence_strata_operator_references_gradients_and_expiry():
    from SentenceStrata import SentenceStrata
    from Memory import STM0, STM1, LTM0, LTM1
    from OperationSeats import Seat
    from Understanding import Reading
    words = torch.rand(2, 4, requires_grad=True)
    rules = (NS(method_name='first', canonical='first', surface_name=None),
             NS(method_name='second', canonical='second', surface_name=None))
    seats = {}
    for index, name in enumerate(('first', 'second'), 1):
        seats[name] = seats[index] = Seat(index, name, False, lambda a, b: a+b)
    language = NS(parts_of_speech=NS(seats=seats), _compose_binary_rules=rules, _compose_unary_rules=())
    strata = SentenceStrata(1, 3, 2, 4)
    def reading(operation):
        frames = torch.stack((words.new_zeros(3, 4), words.new_zeros(3, 4),
                              torch.stack((words[0], words[1], words.mean(0)))))
        return Reading(rows=torch.tensor([0, 1]), word_rows=torch.tensor([0, 1]),
            activations=torch.ones(2), leaves=words, actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, operation, -1]]),
            targets=words, end_state=words.mean(0), concept_ids=torch.tensor([10, 11]), operation_values=frames)
    a = strata.capture(reading(0), language, 0)
    b = strata.capture(reading(1), language, 0)
    assert isinstance(a.tree.store, STM0) and STM0.point_of_row is LTM0.point_of_row
    assert isinstance(a.tree.store.parts, STM1) and STM1.references is LTM1.references
    assert a.tree.forest()[0][1] == 0 and b.tree.forest()[0][1] == 1
    assert len(a.tree.store) == 3 and len(a.tree.store.parts) == 2
    assert bool(a.tree.store.parts.operation_seat.eq(-1).all())
    # Conditioning's composition view and MTM resolve the same live entry
    # that reconstruction's forest names; neither owns another value/tree.
    from MemberRead import read
    composition = a.tree.composition(action=2)
    owner = NS(stm=NS(strata=strata, _thought_stores=()))
    address = ('ltm', a.tree.store.address_domain, composition.address)
    torch.testing.assert_close(read(owner, address), composition.point)
    torch.testing.assert_close(torch.stack([child.point for child in composition.children]), words)
    gradient, = torch.autograd.grad(a.leaves.sum(), words)
    torch.testing.assert_close(gradient, torch.ones_like(words))
    reference = strata.keep_close(0, words.mean(0), False)
    assert len(strata.recent_stores[0]) == 1
    assert all(len(bank) == len(bank.parts) == 0 for bank in strata.banks)
    torch.testing.assert_close(strata.read(reference), words.mean(0).detach())
    with pytest.raises(ValueError, match='expired'): a.tree.forest()
    strata.keep_close(0, words.mean(0), False)
    strata.keep_close(0, words.mean(0), False)
    with pytest.raises(ValueError, match='expired'): strata.read(reference)
    receipt('shared-strata', dict(word_bound=3, operator_bound=strata.operators, branch_bound=2*strata.operators,
        words=2, operators=1, branches=2, shared_reads=True,
        readers=['reconstruction forest', 'conditioning composition', 'MTM member'], gradient=gradient.tolist(),
        after_close_operators=0, after_close_branches=0, recent_capacity=2, expired_reference_rejected=True))


def test_reentry_joins_native_identity_and_records_the_current_link_clock():
    from test_reference_close import fixture, known_concept
    from Meaning import ConceptualMeaning
    from dataclasses import replace
    from QueryWork import QueryWorkBudget
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    concept = known_concept(owner, atoms[0])
    references = []
    for at in range(2):
        meaning = replace(ConceptualMeaning.from_description(atoms[0]), bindings={'_identity': ('sym', concept)})
        row = store.append_meaning(meaning, document_key='recall', sentence_index=at, order=1)
        references.append(store.occurrence_of(row))
        pool.closes[0] = at+4
        assert not store.recall_identities((store.row(row),), work=QueryWorkBudget(16), stream=0)
    key = pool.token_of(references[0])
    assert pool.token_of(references[1]) == key and len(pool.entries) == 1
    assert pool.entries[key].refinement['mentions'][references[1]] == (0, 6)
    assert pool.in_window(key, 0)
    receipt('memory-reentry', dict(tokens=1, references=2, last_link=[0, 6], candidate=True))


def test_epoch_withdrawal_expires_shared_short_term_references():
    from test_presentation_memory import fixture, model_for, begin
    from SentenceStrata import SentenceStrata
    from ThoughtStrata import write
    from Meaning import ConceptualMeaning
    from Memory import SYMBOL
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    owner.stm.strata = SentenceStrata(1, 3, 2, atoms.shape[-1],
                                    relation_capacity=store.parts.capacity)
    begin(model)
    a, b = owner.new_concept(), owner.new_concept()
    relation = store.parts.add((SYMBOL, a), (SYMBOL, b), truth_plus=1.)
    strata = owner.stm.strata
    address = strata.keep_close(0, atoms[0], False, relations=(relation,), source=store.parts)
    thoughts = owner.stm._thought_stores[0][0]
    write(thoughts, ConceptualMeaning.from_description(atoms[0]), key=(2, 2, 2, 2))
    assert len(strata.recent_stores[0]) == len(strata.recent_stores[0].parts) == 1
    assert len(thoughts) == 2  # The role and its written thought root.
    begin(model)
    assert not store.parts.contains(relation)
    assert len(strata.recent_stores[0]) == len(strata.recent_stores[0].parts) == len(thoughts) == 0
    with pytest.raises(ValueError, match='expired'):
        strata.read(address)
    current = strata.keep_close(0, atoms[1], False, source=store.parts)
    torch.testing.assert_close(strata.read(current), atoms[1])
    receipt('epoch-shared-strata', dict(recent_before=1, relations_before=1, thought_rows_before=2,
        recent_after=0, relations_after=0, thought_rows_after=0, next_close_readable=True))


def test_thought_trial_rollback_preserves_expired_external_references_without_resurrection():
    from dataclasses import replace
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from ThoughtStrata import construct, write, snapshot, restore
    external = LTM0(4, capacity=4)
    row = external.append_idea(torch.ones(4), when=False)
    address = external.occurrence_of(row)
    stores = construct(1, 2, 4)
    store = stores[0][0]
    store._external_memory = lambda: external
    meaning = replace(ConceptualMeaning.from_description(torch.ones(4)), role_refs=(address, None, None))
    view = write(store, meaning, key=(2, 2, 2, 2))
    saved = snapshot(stores)
    portable = store.semantic_extras()
    external.reset()
    write(store, ConceptualMeaning.from_description(torch.zeros(4)), key=(3, 2, 2, 2))
    restore(stores, saved)
    torch.testing.assert_close(view.read().roles, meaning.roles)
    assert view.read().role_refs[0] == address
    assert external.location(address) is None and len(external) == 0
    with pytest.raises(ValueError, match='semantic context occurrence is unavailable'):
        store.load_semantic_extras(portable)
    receipt('thought-trial-expired-reference', dict(restored_thoughts=1,
        external_rows_before=1, external_rows_after=0, reference_preserved=True,
        checkpoint_validation_preserved=True))


def test_unnamed_leaf_closing_reads_its_idea_without_a_sentinel_concept(monkeypatch):
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from test_bind_thought import native
    from test_medium_term_memory import keep
    f = native(monkeypatch)
    program = record_reading(f.language, f.program(('definite_unary', 'member')))
    program = replace(program, concept_ids=torch.full_like(program.concept_ids, -1),
        word_ids=torch.full_like(program.concept_ids, -1),
        reference_ids=torch.full_like(program.concept_ids, -1),
        operation_refs=torch.full_like(program.operation_refs, -1),
        leaf_orders=torch.ones_like(program.concept_ids), close_references=True)
    clause = finish_clause(f.language, program, registry=f.registry)
    assert clause.reference_request.get('forward_identity') is None
    clause = replace(clause, reference_request=dict(clause.reference_request, compose_bind=0))
    row, plan = keep(f.cs.components.reference_close, clause)
    pool, store = f.cs.mtm, f.cs._closed_clause_store()
    address = store.occurrence_of(row)
    assert pool.reading_of(address) is None
    torch.testing.assert_close(pool.value_of(address), store.point_of_row(address[-1]))
    assert all(name.reading != -1 for name in pool.names.values())
    receipt('unnamed-leaf-reference', dict(forward_names=0, sentinel_concepts=0,
        reference_reads_own_idea=True))


def test_mtm_v12_and_policy_migrations_are_reported_and_invalid_members_are_atomic():
    import copy
    from test_reference_close import fixture
    from Memory import MediumTermMemory
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm
    row = store.append_idea(atoms[0], when=False)
    address = store.occurrence_of(row)
    key = pool.open(address, kind=None, when=0, value=atoms[0])
    pool.record_link(key, address, 0, kind='reference', order=0)
    pool.event(key, address, verb=1, role=0, when=1, before=atoms[1])
    pool.close(store, stream=0)
    old = copy.deepcopy(pool.state()); old['version'] = 11
    for entry in old['entries']:
        entry['timelines'] = tuple((tuple((*event[:7], atoms[2], *event[7:]) for event in history), atoms[3])
                                   for history in entry['timelines'])
    pool.restore(old)
    assert pool.migration_report == dict(source=11, target=12, dropped_values=1, dropped_event_after=1)
    torch.testing.assert_close(pool.value_of(key), atoms[0])
    bad = copy.deepcopy(pool.state()); bad['entries'][0]['refinement']['members'] = ((('ltm', 'bad', 3), None),)
    before = pool.entries
    with pytest.raises(ValueError, match='member reference'): pool.restore(bad)
    assert pool.entries is before
    state = resolver.state_dict(); state['policy'] = state['policy'][:8].clone()
    resolver.load_state_dict(state)
    assert resolver.policy.shape == (12,) and not resolver.policy[8:].any()
    receipt('memory-migrations', dict(mtm=pool.migration_report, policy=resolver.policy_migration,
        invalid_restore_atomic=True))


def test_consecutive_state_links_witness_a_noun_without_a_separate_counter():
    from test_reference_close import fixture
    from MemoryLink import Occurrence
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    owner.concept_mint_count = 3
    introduction = store.append_idea(atoms[0], address=501, when=False)
    address = store.occurrence_of(introduction)
    noun = owner.link(Occurrence(address, kind='reference', order=0), {},
        value=atoms[0], where=None, when=0, stream=0)
    for close in range(3):
        row = store.append_idea(atoms[0], address=502+close, when=False)
        member = store.occurrence_of(row)
        pool.event(noun, member, verb=1, role=0, when=close, before=atoms[0])
        owner.link(Occurrence(member, kind='state', order=0), {}, value=atoms[0], where=None, when=close, stream=0)
        pool.close(store, stream=0)
        pool.promote_candidates(owner)
    entry = pool.entries[noun]
    assert len(entry.refinement['chain_mentions']) == 3
    assert entry.symbol is not None and owner._concept_source_order(entry.symbol) == 1
    assert entry.count == 1  # The ordinary mention clock, not a second recurrence counter.
    receipt('nouns-from-chains', dict(state_links=3, mention_count=entry.count,
        chain_witnesses=3, promoted_order=1))


def test_operator_authority_and_legacy_columns_use_declared_seats(monkeypatch):
    from test_trust_gates import gate_fixture, gate_data, words
    from ConceptTrust import positive_length
    from definition_fixtures import agreeing_views, learned_operation
    from Memory import SYMBOL
    gate_data(monkeypatch, interpret='open', clause='open')
    f = gate_fixture(monkeypatch)
    owner, parts, occupations = f.cs, f.store.parts, f.language.parts_of_speech
    left, right, external = words(f, 3)
    for word in (left, right, external): agreeing_views(owner, word)
    learned_operation(occupations, 'part', left, right, count=3)
    seat = occupations.seats['part']
    row = parts.add((SYMBOL, left), (SYMBOL, right), truth_plus=.8,
                    operator=left, operation_seat=seat.concept)
    parts.refresh_effective()
    expected = .8*min(positive_length(owner, left), positive_length(owner, right),
        positive_length(owner, seat.concept), parts.gates.cap(named=False, operation='part'))
    assert float(parts.effective_plus[row]) == pytest.approx(expected)
    assert int(parts.operator_reference[row]) == -1
    assert seat.native and 'trust' not in occupations.operation('part')
    external_row = parts.add((SYMBOL, right), (SYMBOL, left), truth_plus=.7,
                            operator=external, operation_seat=seat.concept)
    parts.refresh_effective()
    external_expected = .7*min(positive_length(owner, left), positive_length(owner, right),
        positive_length(owner, external), positive_length(owner, seat.concept),
        parts.gates.cap(named=False, operation='part'))
    assert int(parts.operator_reference[external_row]) == external
    assert float(parts.effective_plus[external_row]) == pytest.approx(external_expected)
    state = {key: value.clone() for key, value in parts.state_dict().items()}
    old = state.pop('operation_seat')
    catalog = torch.full_like(old, -1)
    for index, name in enumerate(parts.operation_catalog): catalog[old == occupations.seats[name].concept] = index
    state['grammar_operation'] = catalog
    state['operator'] = state.pop('operator_reference'); state['operator'][row] = left
    parts.load_state_dict(state)
    assert int(parts.operation_seat[row]) == seat.concept and int(parts.operator_reference[row]) == -1
    assert parts.operation_migration['redundant_endpoints'] == 1
    assert int(parts.operator_reference[external_row]) == external
    receipt('operator-authority', dict(effective=float(parts.effective_plus[row]), expected=expected,
        seat=seat.concept, interfaces=len(seat.native), migration=parts.operation_migration,
        nonendpoint_case='operator word distinct from both symbolic endpoints',
        nonendpoint_reference=external, nonendpoint_effective=float(parts.effective_plus[external_row]),
        nonendpoint_expected=external_expected))


def test_one_ltm_idea_generates_a_meaning_compatible_surface_without_input_stm(paired, monkeypatch):
    from test_item20_round2_generation import proposal
    from Models import BasicModel, _ensure_grad_anchors
    from Memory import LTM0
    import ChunkJournal, util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    def forbidden(*args, **kwargs): raise AssertionError('generation read the input tree or STM')
    monkeypatch.setattr(ChunkJournal, 'action_context', forbidden)
    monkeypatch.setattr(ChunkJournal, 'read_identity', forbidden)
    language, words, phrase, parent, bank, groups, structure = proposal('group')
    store = LTM0(12, capacity=4).double()
    row = store.append_idea(parent[0].detach(), when=False)
    idea = store.point_of_row(int(store.address_keys[row]))[None]
    class Owner:
        languageSpace = language
        stm = property(forbidden)
        desiredOutput = property(forbidden)
    def policy(value, features=None):
        root = (value[:, :4] == idea[:, :4]).all(-1)
        return torch.nn.functional.one_hot(torch.where(root, 0, 2), 3).to(value)*10
    language.generate_policy_logits = policy
    event = torch.cat((idea[:, None], torch.zeros_like(words)), 1)
    _ensure_grad_anchors(event.device, (event.dtype,))
    result = BasicModel._output_generate_walk(Owner(), event, 6, basis=words,
        basis_valid=torch.ones(1, 3, dtype=torch.bool), verb_groups=groups,
        candidate_limit=3, return_trace=True)
    assert result[1].item() == 2 and not result[2].any() and result[5].all()
    emitted = result[0][0, :2]
    ids = (emitted[:, None, :4]-words[0, None, :, :4]).square().sum(-1).argmin(-1)
    surface = b' '.join((b'actor', b'acts', b'place')[index] for index in ids.tolist())
    assert surface == b'actor acts'
    receipt('retained-idea-generation', dict(ltm_ideas=len(store), input_tree=False,
        input_stm=False, surface=surface.decode(), exact_recomposition=True))


def test_pool_and_unpooled_stm_share_the_menu_and_states_never_prepend():
    from test_reference_close import fixture
    from MemoryLink import Occurrence, menu
    from Memory import STMChunks
    from PrependedSymbols import SymbolEntries, stage
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    first = store.append_idea(atoms[0], address=601, when=False)
    pooled = owner.link(Occurrence(store.occurrence_of(first), kind='state', order=0), {},
        value=atoms[0], where=None, when=1, stream=0)
    pool.close(store, stream=0)
    second = store.append_idea(atoms[0], address=602, when=False)
    unpooled = store.occurrence_of(second)
    owner.stm.ensure_batch(1)
    refs = torch.tensor([[[STMChunks.OCCURRENCE, unpooled[-1]], [0, 0]]])
    state, _ = STMChunks.deposit(owner.stm.state, atoms[0][None], torch.tensor([True]),
        order=0, refs=refs)
    owner.stm.set_state(state)
    query = store.append_idea(atoms[0], address=603, when=False)
    choices, features, _ = menu(owner, Occurrence(store.occurrence_of(query), kind='state', order=0), {},
        value=atoms[0], where=None, when=2, stream=0)
    assert pooled in choices and unpooled in choices and features.shape[-1] == 12
    assert features[choices.index(unpooled), -1] == 1
    assert pool.token_of(unpooled) is None  # A menu read cannot create the collection.
    pooled_refs = refs.clone(); pooled_refs[0, 0, 1] = 601
    carried, _ = STMChunks.deposit(state, atoms[0][None], torch.tensor([True]),
        order=0, refs=pooled_refs, identity=pool.entries[pooled].transient)
    carried = resolver.committed_chunks(carried, {'items': []}, {}, stream=0)
    assert carried[STMChunks.IDENTITY].eq(-1).all()
    assert (carried[STMChunks.REFS][..., 0, 1] == 601).any()
    entries = SymbolEntries(1, 4, 4, 12)
    model = NS(_concept_owner=lambda: owner, inputSpace=NS(symbol_entries=entries),
        symbolSpace=NS(ltm_store=store), _open_sentence_slot=0)
    stage(model, carried, torch.tensor([True]))
    assert entries.concept_ids[:, :entries.prepend].eq(-1).all()
    receipt('pool-stm-menu', dict(pool_candidate=True, unpooled_stm_candidate=True,
        menu_width=12, stm_feature=1, states_prepended=0, states_published_as_identities=0,
        sentence_occurrence_preserved=True, menu_is_read_only=True))


def test_link_coverage_reaches_live_member_readers_and_modal_is_not_a_state():
    from test_reference_close import fixture
    from MemoryLink import Occurrence, state_eligible, preview
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from Layers import Error
    from dataclasses import replace
    from order_fixtures import force_order
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    concept = owner.new_concept(); force_order(owner, concept, 0)
    row = owner._csw_concept_row(0, concept)
    with torch.no_grad(): owner.similarity_codebook.W[row].copy_(atoms[0]*.5)
    first = store.append_idea(atoms[0], address=611, when=False)
    key = owner.link(Occurrence(store.occurrence_of(first), concept=concept, kind='reference', order=0), {},
        value=atoms[0]*.5, where=None, when=0, stream=0)
    owner.mtm.close(store, stream=0)
    errors = Error(); owner._model._sentence_close_errors = errors
    second = store.append_idea(atoms[0], address=612, when=False)
    own = atoms[0].clone().requires_grad_()
    owner.link(Occurrence(store.occurrence_of(second), kind='reference', order=0, selected=key), {},
        value=own, where=None, when=1, stream=0)
    value = errors.total(objective='reconstruction')
    own_grad, reader_grad = torch.autograd.grad(value.sum(), (own, owner.similarity_codebook.W))
    assert bool(own_grad.abs().any()) and bool(reader_grad[row].abs().any())
    ordinary = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=0)
    modal = replace(ordinary, meaning=replace(ordinary.meaning, bindings={'_modal': True}))
    assert state_eligible(ordinary) and not state_eligible(modal)
    plan = {'items': []}
    before = dict(owner.mtm.entries)
    preview(resolver, plan, modal, None, stream=0, when=2)
    assert not plan['link_choices'] and owner.mtm.entries == before
    receipt('link-coverage-gradient', dict(own_norm=float(own_grad.norm()),
        member_norm=float(reader_grad[row].norm()), owner='reconstruction', side='R',
        modal_links=0, modal_preview_read_only=True))


def test_native_kept_close_uses_all_three_link_callers(monkeypatch):
    """Forced native derivations; the adapter supplies Models' state handoff."""
    from test_modal_timelines import fixture, recording, close
    from MemoryLink import Occurrence, state_eligible
    from Layers import Error
    f,verb,modifier=fixture(monkeypatch)
    owner,resolver=f.cs,f.cs.components.reference_close
    owner.mtm.horizon=2
    errors=Error();owner._model._sentence_close_errors=errors
    calls=[];native_link=owner.link;native_commit=resolver.commit
    def link(site,candidates,**options):
        result=native_link(site,candidates,**options)
        calls.append(dict(kind=site.kind, member=site.address, key=result,
            close=owner.mtm.closes.get(options['stream'],0)+1, order=site.order))
        return result
    def commit(plan,row,written,**options):
        clause=plan['prepared_root']
        if isinstance(row,int) and state_eligible(clause) and f.store.kind_of(row) not in ('estimate','question','cover'):
            plan['state_link']=(Occurrence(f.store.occurrence_of(row),kind='state',order=0),clause.point,clause.where)
        return native_commit(plan,row,written,**options)
    monkeypatch.setattr(owner,'link',link);monkeypatch.setattr(resolver,'commit',commit)
    frames=[];states=[]
    for position in range(3):
        subject=('lower_unary','subject') if position==0 else ('bind','subject')
        program=recording(f,('lift',subject,'move'),frames)
        before=len(calls)
        clause,frame,row,written=close(f,program,frames,position=position,actions=(verb,modifier))
        frames.append(frame)
        current=calls[before:]
        assert {item['kind'] for item in current}=={'reference','event','state'}, current
        assert clause.order==0
        states.append(next(item for item in current if item['kind']=='state'))
    assert states[0]['key']==states[1]['key']==states[2]['key']
    assert [item['close'] for item in states]==[1,2,3]
    before=len(calls)
    program=recording(f,('lift',('bind','subject'),('adverb','move','shift')),frames)
    clause,frame,_,_=close(f,program,frames,position=3,actions=(verb,modifier))
    modal=calls[before:]
    assert not state_eligible(clause) and clause.order>0
    assert all(item['kind']=='reference' for item in modal),modal
    # The same model's close clock advances beyond the window before another
    # real kept derivation. The native state menu must take the dummy.
    owner.mtm.closes[0]+=3
    before=len(calls)
    program=recording(f,('lift',('lower_unary','subject'),'move'),())
    _,_,_,_=close(f,program,(),position=7,actions=(verb,modifier))
    distant=next(item for item in calls[before:] if item['kind']=='state')
    assert distant['key']!=states[-1]['key']
    assert 'memory.link_coverage' in errors._terms
    term=errors._terms['memory.link_coverage']
    assert term['objective']=='reconstruction' and term['side']=='R'
    receipt('native-link-callers', dict(consecutive_closes=[1, 2], recurrence_close=3,
        native_callers=['ReferenceClose.commit reference', 'ReferenceClose.link_events',
                        'ReferenceClose.commit state_link'],
        fixture_adapter='The direct-clause fixture supplies Models state_eligible/address/point handoff.',
        modal_state_links=0, modal_event_links=0, distant_dummy=True,
        registered_owner=term['objective'], registered_side=term['side'], calls=calls))


def test_shared_composition_admission_validates_arrivals_before_binding():
    from test_reference_close import fixture
    from SentenceStrata import SentenceStrata
    from Understanding import Reading
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    resolver,store,atoms=fixture();owner=resolver.components.owner
    word=owner.interpret.lookup_word([97],(),form='arrival-word')
    point=atoms[0]
    language=NS(_compose_binary_rules=(),_compose_unary_rules=(),parts_of_speech=NS(seats={}))
    entry=Reading(rows=torch.tensor([0]),word_rows=torch.tensor([0]),activations=torch.ones(1),
     leaves=point[None],actions=torch.tensor([[0,-1,0]]),targets=torch.tensor([0]),end_state=point[None],concept_ids=torch.tensor([word]))
    strata=SentenceStrata(1,1,8,len(point));strata.capture(entry,language,0)
    view=entry.tree.install_composition(Composition(point,reference=word,word_id=word,leaves=(0,),order=0),word=0)
    request=dict(mode='mint',leaves=(0,),cue=point,predicate=point,content=point,role=0)
    clause=Clause(ConceptualMeaning.from_description(point),point=point,order=1,reference_request=request,composition=view)
    plan=resolver.preview(clause,context(resolver,()),stream=0)
    assert plan['arrival_count']==0 and len(plan['items'])==1 and view.reference==word
    kept=resolver.prepare(clause,plan)
    assert len(resolver._arrival_constituents(clause)[1])==1 and view.reference==-1
    assert plan['prepared_root'] is kept and plan['arrival_count']==0
    receipt('shared-arrival-admission', dict(scored_arrivals=0, post_binding_candidates=1,
        checked_before_binding=True, prepared=True))


def test_two_reference_sites_of_one_shared_occurrence_do_not_self_bind():
    from test_reference_close import fixture
    from SentenceStrata import SentenceStrata
    from Understanding import Reading
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner, point = resolver.components.owner, atoms[0]
    word = owner.interpret.lookup_word([97], (), form='shared-word')
    language = NS(_compose_binary_rules=(), _compose_unary_rules=(), parts_of_speech=NS(seats={}))
    entry = Reading(rows=torch.tensor([0]), word_rows=torch.tensor([0]), activations=torch.ones(1),
        leaves=point[None], actions=torch.tensor([[0, -1, 0]]), targets=torch.tensor([0]),
        end_state=point[None], concept_ids=torch.tensor([word]))
    strata = SentenceStrata(1, 1, 8, len(point))
    strata.capture(entry, language, 0)
    view = entry.tree.install_composition(Composition(point, reference=word, word_id=word,
        leaves=(0,), order=0), word=0)
    request = dict(mode='mint', leaves=(0,), cue=point, predicate=point, content=point, role=0)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                    reference_request=request, composition=view)
    plan = resolver.preview(clause, context(resolver, ()), stream=0)
    assert len(plan['items']) == 1
    item = plan['items'][0]
    item['selected'] = ('new', 0)
    plan['items'].append(dict(item))
    kept = resolver.prepare(clause, plan)
    assert kept.composition is view and view.binding is None
    assert len(view.tree.store) == 1 and len(view.tree.store.parts) == 0
    receipt('shared-occurrence-alias', dict(reference_sites=2, shared_occurrences=1,
        self_bindings=0, extra_branches=0, prepared=True))


def test_written_alias_uses_the_kept_binding_instead_of_earlier_identity_metadata():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    chosen = known_concept(owner, atoms[0], order=1)
    earlier = known_concept(owner, atoms[1], order=0)
    target = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': ('sym', chosen)}), when=False, document_key='chosen-member')
    key = owner.link(Occurrence(store.occurrence_of(target), selected=None), {},
        value=atoms[0], where=None, when=0, stream=0)
    antecedent = Composition(atoms[0], reference=chosen, order=1)
    own = atoms[0] + atoms[1]
    term = Composition(own, reference=earlier, order=1, binding=antecedent)
    clause = Clause(replace(ConceptualMeaning.from_description(own),
        bindings={'_identity': ('sym', earlier)}), point=own, order=1, composition=term)
    row = store.write_clause(clause)
    address = store.occurrence_of(row)
    assert dict(store.raw_meaning_of(row).bindings)['_identity'] == ('sym', chosen)
    assert store.order_of_row(row) == 1
    assert owner.link(Occurrence(address, selected=key), {}, value=own,
                      where=None, when=1, stream=0) == key
    torch.testing.assert_close(owner.mtm.read_member(address), own)
    receipt('written-alias-binding', dict(pre_bind_order=0, selected_order=1,
        written_order=1, members=2, alias_reads_own_idea=True))


def test_written_alias_retains_the_selected_pending_name_and_its_order():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    selected_word = owner.interpret.lookup_word([112], (), form='pending-choice')
    earlier_word = owner.interpret.lookup_word([101], (), form='earlier-choice')
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(selected_word)].copy_(atoms[0])
    owner.interpret.define(earlier_word, known_concept(owner, atoms[0], order=1))
    selected = owner.mtm.named(selected_word, value=atoms[0])
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': selected}), when=False, document_key='pending-choice')
    owner.link(Occurrence(store.occurrence_of(first), concept=selected_word, selected=selected),
        {}, value=atoms[0], where=None, when=0, stream=0)
    own = atoms[0] + atoms[1]
    meaning = replace(ConceptualMeaning.from_description(own),
        bindings={'_forward_references': ((0, earlier_word),)})
    node = Clause(meaning, point=own, order=1, composition=Composition(own, order=-1),
        reference_request=dict(mode='bind', compose_bind=1, leaves=(0,),
            cue=own, content=own, predicate=own, role=0),
        identity=dict(handle=1, origins=torch.tensor([[6, selected_word]])))
    plan = resolver.preview(node, context(resolver, ()), stream=0)
    assert plan['items'][0]['selected'] == selected and resolver.address(selected) is None
    kept = resolver.prepare(node, plan)
    written = {}
    row = store.write_clause(kept, document_key='pending-alias', written_rows=written)
    resolver.commit(plan, row, written, document='pending-alias', position=1, training=False)
    retained = dict(store.raw_meaning_of(row).bindings)['_identity']
    assert retained == store.occurrence_of(first)
    assert owner.mtm.token_of(retained) == selected
    assert store.order_of_row(row) == owner.mtm.entries[selected].refinement['member_order'] == 0
    assert owner.mtm.token_of(store.occurrence_of(row)) == selected
    torch.testing.assert_close(owner.mtm.read_member(store.occurrence_of(row)), own)
    assert owner.mtm.entries[selected].pending
    receipt('written-pending-name', dict(earlier_forward_order=1, selected_order=0,
        written_order=0, members=len(owner.mtm.entries[selected].occurrences),
        pending_after=True, alias_reads_own_idea=True))


def test_written_ltm_selection_replaces_earlier_identity_without_a_binding_child():
    from dataclasses import replace
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    selected = known_concept(owner, atoms[0], order=1)
    earlier = known_concept(owner, atoms[1], order=0)
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': ('sym', selected)}), when=False, document_key='selected-ltm')
    address = store.occurrence_of(first)
    key = owner.link(Occurrence(address, selected=None), {}, value=atoms[0],
        where=None, when=0, stream=0)
    own = atoms[0] + atoms[1]
    node = Clause(replace(ConceptualMeaning.from_description(own),
        bindings={'_identity': ('sym', earlier)}), point=own, order=1,
        composition=Composition(own, reference=earlier, order=-1),
        reference_request=dict(mode='bind', compose_bind=1, leaves=(0,),
            cue=own, content=own, predicate=own, role=0),
        identity=dict(handle=1, origins=torch.tensor([[2, address[-1]]])))
    plan = resolver.preview(node, context(resolver, ()), stream=0)
    kept = resolver.prepare(node, plan)
    assert kept.composition.binding is None
    written = {}
    row = store.write_clause(kept, document_key='selected-alias', written_rows=written)
    resolver.commit(plan, row, written, document='selected-alias', position=1, training=False)
    assert dict(store.raw_meaning_of(row).bindings)['_identity'] == address
    assert store.order_of_row(row) == 1
    assert owner.mtm.token_of(store.occurrence_of(row)) == key
    torch.testing.assert_close(owner.mtm.read_member(store.occurrence_of(row)), own)
    receipt('written-ltm-selection', dict(earlier_order=0, selected_order=1,
        written_order=1, binding_children=0, members=2, alias_reads_own_idea=True))


def test_same_close_binding_keeps_the_introductions_own_mention():
    from test_reference_close import fixture
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    words = [owner.interpret.lookup_word([letter], (), form=f'pending-{letter}')
             for letter in (97, 98)]
    point = atoms[0] + atoms[1]
    operands = tuple(Composition(atoms[i], reference=word, word_id=word,
        order=1, leaves=(i,)) for i, word in enumerate(words))
    term = Composition(point, operands, operation='part', order=1, leaves=(0, 1))
    introduction = Clause(ConceptualMeaning.from_description(point), point=point,
        order=1, composition=term,
        reference_request=dict(mode='bind', compose_bind=0, addressed_introduction=True,
            leaves=(0, 1), cue=point, content=point, role=0),
        identity=dict(handle=1, action=0, introduction=True, origins=torch.tensor([[3, 0]])))
    root = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
        composition=Composition(point, (term,), operation='bind', order=1, leaves=(0, 1)),
        children=(introduction,), refs=(('clause', 0), -1, -1),
        reference_request=dict(mode='bind', compose_bind=1, leaves=(0, 1),
            cue=point, content=point, role=0),
        identity=dict(handle=1, action=1, introduction=False, origins=torch.tensor([[3, 0]])))
    plan = resolver.preview(root, context(resolver, ()), stream=0)
    assert [item['selected'] for item in plan['items']] == [('new', 0), ('new', 0)]
    kept = resolver.prepare(root, plan)
    assert kept.composition.binding is kept.children[0].composition
    written = {}
    row = store.write_clause(kept, document_key='local-introduction', written_rows=written)
    first, alias = (written[('mention', id(item['written_node']))] for item in plan['items'])
    assert row[0] == 'sym'  # The unresolved semantic chain remains a relation.
    assert dict(store.raw_meaning_of(store.location(alias)).bindings)['_identity'] == first
    resolver.commit(plan, row, written, document='local-introduction', position=0, training=False)
    assert owner.mtm.token_of(first) == owner.mtm.token_of(alias)
    assert store.order_of_row(store.location(first)) == store.order_of_row(store.location(alias)) == 0
    torch.testing.assert_close(owner.mtm.read_member(alias), point)
    receipt('same-close-mention-binding', dict(introductions=1, members=2,
        member_order=0, relation_substitutions_before=1, relation_substitutions_after=0,
        semantic_relation_retained=True, alias_reads_own_idea=True))
