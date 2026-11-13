"""Contracts corrected by the item-2.0 round-6 review."""
import pytest
import torch


@pytest.mark.parametrize('repetitions', [1, 2])
@torch.no_grad()
def test_kept_event_link_precedes_its_words_new_trust_views(repetitions):
    from types import SimpleNamespace as NS
    from test_review13_subspaces import fixture
    from FamiliarTier import commit_read_words, kept_close
    from ConceptTrust import observe
    from MemoryLink import Occurrence, menu

    owner, bank, _, _ = fixture(meaning_pairs=4)
    owner._model.wholeSpace = NS(subspace=NS(what=None))
    word = owner.interpret.lookup_word([4], [], form='word')
    row = owner._csw_row_of(word)
    bank.mereology.place_rows([row], torch.zeros(len(bank.W)), b'event-word')
    observe(owner, word, bank.lookup_rows(row))
    member = bank.lookup_rows(row).clone()
    key = owner.link(Occurrence(('sym', word), kind='event', order=0, selected=None), {},
        value=member, where=None, when=1, stream=0)
    event = member.clone()
    start, width = bank.trust.start, bank.trust.width
    event[start+width:start+2*width] = event[start:start+width]
    choices, _, _ = menu(owner, Occurrence((), kind='event', order=0), {},
        value=event, where=None, when=2, stream=0)
    assert key in choices
    store = owner._closed_clause_store()
    event_row = store.append_idea(event, document_key='event', sentence_index=0)
    address = store.occurrence_of(event_row)
    reading = NS(word_ids=torch.full((repetitions,), word),
        leaves=member[None].expand(repetitions, -1), symbol_where=None)
    model = NS(_concept_owner=lambda: owner)
    radius = bank.trust.lengths[row, 0].clone()

    @kept_close
    def close(model, state, sid, active):
        commit_read_words(owner, reading, batch=0, document='event', position=0,
            written={'root_occurrence': address})
        torch.testing.assert_close(bank.trust.lengths[row, 0], radius, rtol=0, atol=0)
        owner.link(Occurrence(address, kind='event', order=0, selected=key), {},
            value=event, where=None, when=2, stream=0)

    close(model, None, 0, torch.tensor([True]))
    assert address in owner.mtm.entries[key].occurrences
    assert bank.trust.views[row] == 1 + repetitions
    assert bank.trust.lengths[row, 0] > radius
    assert bank.trust._deferred_word_views is None


def test_unequal_attribution_shares_preserve_the_signed_demand():
    from basic_fixtures import feature
    from test_cs_sparse_weights import _cs
    owner = _cs()
    feature(owner, 0, 'ps', 65, .25)
    feature(owner, 0, 'ws', 3, -.75, parts=(65,))
    query = torch.zeros(sum(owner._order_caps()), 1, 1, 2)
    query[0, 0, 0] = torch.tensor([.8, .6])
    columns, attributed, spans = owner.cs_percept_attribution(query)
    assert columns.tolist() == [0, 1, 2, 3]
    values = attributed[:, 0, 0, 0]
    torch.testing.assert_close(values, torch.tensor([.2, .15, .45, .6]))
    torch.testing.assert_close(values[[0, 1]] / .25, values[[3, 2]] / .75)
    torch.testing.assert_close(values[[0, 1]] + values[[3, 2]], query[0, 0, 0])
    assert spans is None


@pytest.mark.parametrize('gate', [1., 1e-6])
def test_word_form_uses_positive_definitions_then_its_own_percept(gate):
    from test_review13_subspaces import fixture, word
    owner, book, native, allocator = fixture()
    row = word(owner, [(4, .25 * gate)])
    other = owner.basic_percepts.reserve((7,), ())
    owner.add_concept_feature(row, 'bp', other, .75 * gate)
    own = owner.basic_percepts.words[owner.concept_id_at_row(row)]
    torch.testing.assert_close(book.lookup_rows(row)[:3],
                               .25 * native.W[4] + .75 * native.W[7])
    with torch.no_grad():
        allocator.layer().features.values.zero_()
    torch.testing.assert_close(book.lookup_rows(row)[:3], owner.basic_percepts.code(own))


def test_overlap_underflow_is_a_valid_zero_without_a_seed():
    from Layers import isa_part_op
    child = torch.zeros(2, 3, 4)
    parent = torch.full_like(child, 10.)
    assert isa_part_op(child, parent).item() == 0.
    assert isa_part_op(child, child).item() == 1.


@pytest.mark.parametrize('recovered_rows,observed_rows', [(1, 2), (2, 1)])
def test_native_reverse_loss_charges_missing_and_excess_events(recovered_rows, observed_rows):
    from types import SimpleNamespace
    from Layers import Error, ModelLoss
    from Models import BasicModel
    model = SimpleNamespace(loss=ModelLoss(what_scale=1., where_scale=0., when_scale=0.),
        inputSpace=SimpleNamespace(subspace=SimpleNamespace(nWhere=0, nWhen=0)))
    recovered = torch.ones(1, recovered_rows, 3, requires_grad=True)
    observed = torch.ones(1, observed_rows, 3)
    errors = Error()
    cost = BasicModel._reverse_event_loss(model, recovered, observed,
        registry=errors, name='reconstruction_order_zero')
    assert bool(cost > 0), 'matching a prefix must not erase unreturned or extra native events'
    assert bool(errors.total() > 0)


def test_occurrence_writer_preserves_the_emitted_terminal_despite_priming():
    from types import SimpleNamespace as NS
    from OutputOccurrences import generated
    from SentenceUnderstanding import PrimedSymbols
    codes = torch.tensor([[[1., 0., 1., 0., 0., 0.], [.8, .6, 0., 1., 0., 0.]]])
    bank = PrimedSymbols(torch.tensor([[0, 1]]), codes, torch.tensor([[1., 2.]]),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[[97], [98]]]),
        torch.ones(1, 2, 1, dtype=torch.bool), meaning_start=2)
    empty = NS(_index={}, values=torch.empty(0))
    store = NS(_tensor_row_keys={0:10, 1:11}, _index={}, values=torch.empty(0), conjunctive=empty)
    owner = NS(similarity_codebook=NS(lookup_rows=lambda rows: codes[0, rows],
        mereology=NS(percept_event_width=2)),
        _row_order=lambda row: 0, concept_id_at_row=lambda row: row + 10,
        word_surface_for_row=lambda row: (b'a', b'b')[row],
        cs_reverse_presence=lambda evidence, **kwargs: evidence,
        _concept_allocator=NS(layer=lambda order: store))
    result = generated(NS(_concept_owner=lambda: owner), codes[:, :1],
        torch.tensor([1]), torch.tensor([False]), bank, (), when=torch.tensor([[[1, 2]]]))
    assert result.field.concept_ids.tolist() == [10, 11]
    torch.testing.assert_close(result.field.evidence.flatten(), torch.tensor([1., 0., 0., 0.]))


@pytest.mark.parametrize('operation', ['conjunction', 'disjunction'])
@pytest.mark.parametrize('pair', [(1., 0.), (0., 1.), (1., 1.), (0., 0.)])
def test_disjoint_inverse_keeps_compatible_candidate_poles(operation, pair):
    from MeaningCodes import compose, recover_lanes
    from Interpret import activate_code
    codes = torch.tensor([[[1., 0., 1., 0., 0., 0.],
                           [0., 1., 0., 1., 0., 0.]]])
    candidates = tuple(activate_code(code, torch.ones(1), 2,
        evidence=torch.tensor([pair])) for code in codes.unbind(1))
    parent = torch.cat((codes[:, :, :2].mean(1),
        compose(operation, *(value[:, 2:] for value in candidates))), -1)
    recovered = recover_lanes(operation, parent, *candidates, 2)
    for actual, expected in zip(recovered, candidates):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(compose(operation, *(value[:, 2:] for value in recovered)),
                               parent[:, 2:], rtol=0, atol=0)


@pytest.mark.parametrize('orders', [(0, 0), (0, 1)])
def test_reconciled_reference_candidates_keep_one_collection_order(orders):
    from MemoryLink import Occurrence
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    records = {}
    for index, order in enumerate(orders):
        row = store.append_idea(atoms[index], order=order, time=False)
        address = store.occurrence_of(row)
        token = owner.link(Occurrence(address, order=order, selected=None), {},
            value=atoms[index], where=None, when=index + 1, stream=0)
        # Later common distinguishing evidence permits same-order alias
        # reconciliation; it cannot join two different collection orders.
        records[address[-1]] = (row, dict(identity=token, token=token,
            value=atoms[2], role=0, position=(index + 1, 0)))
    candidates = resolver.individuals(records)
    assert len(candidates) == (1 if orders[0] == orders[1] else 2)
    for identity, candidate in candidates.items():
        selected = owner.mtm.entries[owner.mtm.token_of(identity)]
        read = owner.mtm.entries[owner.mtm.token_of(candidate['token'])]
        assert selected.refinement['member_order'] == read.refinement['member_order']


@pytest.mark.parametrize('origin', [True, False])
def test_collected_state_members_are_not_offered_as_unpooled(origin):
    from Memory import STMChunks
    from MemoryLink import Occurrence, menu
    from test_reference_close import fixture
    resolver,store,atoms=fixture();owner=resolver.components.owner
    first=store.append_idea(atoms[0],address=801,time=False);start=store.occurrence_of(first)
    key=owner.link(Occurrence(start,kind='state',order=0,selected=None),{},value=atoms[0],where=None,when=1,stream=0)
    second=store.append_idea(atoms[0]+atoms[1],address=802,time=False)
    owner.link(Occurrence(store.occurrence_of(second),kind='state',order=0,selected=key),{},value=atoms[0]+atoms[1],where=None,when=2,stream=0)
    third=store.append_idea(atoms[0],address=803,time=False);member=store.occurrence_of(third)
    owner.link(Occurrence(member,kind='state',order=0,selected=key),{},value=atoms[0],where=None,when=3,stream=0)
    fresh=store.append_idea(atoms[0]+atoms[2],address=804,time=False);unpooled=store.occurrence_of(fresh)
    chosen=start if origin else member
    owner.stm.ensure_batch(1);state=owner.stm.state
    for address,value in ((chosen,atoms[0]),(unpooled,atoms[0]+atoms[2])):
     refs=torch.tensor([[[STMChunks.OCCURRENCE,address[-1]],[0,0]]]);state,_=STMChunks.deposit(state,value[None],torch.tensor([True]),order=0,refs=refs)
    owner.stm.set_state(state)
    choices,_,_=menu(owner,Occurrence((),kind='state',order=0),{},value=atoms[0]+atoms[2],where=None,when=4,stream=0)
    assert chosen not in choices
    assert key not in choices
    assert unpooled in choices
    containing = atoms[0] + atoms[1] + atoms[2]
    choices, _, _ = menu(owner, Occurrence((), kind='state', order=0), {},
        value=containing, where=None, when=4, stream=0)
    assert key in choices and unpooled in choices and chosen not in choices


def test_kept_thought_finishes_definitions_before_relation_withdrawal(monkeypatch):
    from types import SimpleNamespace as NS
    from part_fixtures import native_store
    from test_thought_binding import equality
    from ThoughtStream import write
    from FamiliarTier import finish_definitions
    from Memory import SYMBOL
    import Occurrence
    store, (a, b, predicate) = native_store(width=3, points=torch.eye(3))
    owner = store._concept_owner()
    model = NS(symbolSpace=NS(ltm_store=store), grammatical_thoughts=NS(
        signature_for=lambda *args, **kwargs: NS(operation=NS(semantic_id='isEqual'))))
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('definition-boundary', 0))
    store.parts.begin_epoch()
    index = write(model, equality(('sym', a), ('sym', b)), row=0)
    assert store.parts.contains(index[1])
    assert store.parts.references(index[1]) == ((SYMBOL, a), (SYMBOL, b))
    store.parts.begin_epoch()
    # The next input close must not try to finish the preceding epoch's
    # concluded thought. The relation's ordinary withdrawal is unchanged.
    finish_definitions(owner)
    assert not getattr(owner, '_kept_definition_witnesses', ())
    owner._kept_definition_witnesses = [((SYMBOL, a), (SYMBOL, b), None, None, False)]
    with pytest.raises(ValueError, match='definition requires its kept relation'):
        finish_definitions(owner)


@pytest.mark.parametrize('unwitnessed_position', (0, 2))
def test_recurrence_waits_for_each_members_order_before_allocating(unwitnessed_position):
    from dataclasses import replace
    from test_reference_close import fixture
    from DocumentCover import open_cover
    from Meaning import ConceptualMeaning
    from MemoryConsolidation import consolidate
    from order_fixtures import force_order
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    unknown, known = owner.new_concept(), owner.new_concept()
    force_order(owner, known, 0)
    rows = []
    for n in range(3):
        document = ('order-witness', n)
        open_cover(owner, store, stream=0, document=document)
        reference = unknown if n == unwitnessed_position else known
        meaning = replace(ConceptualMeaning.from_description(atoms[0] + atoms[1]),
                          role_refs=(('sym', reference), None, None))
        rows.append(store.append_meaning(meaning, document_key=document,
            sentence_index=n, stream=0, evidence=(1., 0.)))
    allocator = owner._concept_allocator
    before = allocator.next_id
    assert consolidate(owner, store) == ()
    assert allocator.next_id == before
    reading = int(store.recurrence_readings[rows[0]])
    assert all(int(store.recurrence_readings[row]) == reading for row in rows)
    assert int(store.recurrence_documents[reading]) == 3
    assert int(store.recurrence_symbols[reading]) == -1
    force_order(owner, unknown, 0)
    assert consolidate(owner, store) == (('reading', reading),)
    symbol = int(store.recurrence_symbols[reading])
    assert symbol > 0 and allocator.order_of(symbol) == 0


@pytest.mark.parametrize('order', (0, 1))
def test_arrival_eligibility_reads_the_identity_after_shared_site_preparation(order):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from test_reference_close import fixture, known_concept
    from test_reference_arrivals import equality
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    subject = owner.interpret.lookup_word([120], [], form='pending-form')
    word = owner.interpret.lookup_word([121], [], form='arriving-form')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept)
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(atoms[0])
    native = known_concept(owner, atoms[0], order=order)
    meaning = replace(ConceptualMeaning.from_description(atoms[0]),
        role_refs=(('sym', native), None, None),
        bindings={'_forward_references': ((0, subject),)})
    row = store.append_meaning(meaning, document_key='arrival-order', stream=0)
    token = store.occurrence_of(row)
    owner.link(Occurrence(token, concept=native, selected=None), {},
               value=atoms[0], where=None, when=1, stream=0)
    clause = equality(atoms[0], subject, concept, subject=atoms[0])
    child = clause.children[0]
    child = replace(child, composition=replace(child.composition, word_id=word), subject_word_id=word)
    clause = replace(clause, children=(child,), composition=replace(clause.composition,
        children=(clause.composition.children[0], child.composition)))
    site = replace(child, reference_request=dict(mode='bind', forward_identity=word,
        leaves=(1,), content=atoms[0], cue=atoms[0]*0, predicate=atoms[0]*0, role=0))
    item = dict(node=site, source_node=child, selected=('sym', word), mode='bind',
        role=0, leaves=(1,), asserted=(), member_order=1, own_object=False, anonymous=False)
    _, candidates = resolver._arrival_constituents(clause)
    chosen = ('arrival', next(i for i, node in enumerate(candidates) if node is child))
    resolver.select = lambda choices, logits, **kwargs: choices.index(chosen) if chosen in choices else None
    plan = dict(items=[item], records={token: (row, {})}, decisions=[], departure=-1,
                cost=atoms[0].new_zeros(()), stream=0)
    resolver._arrival_plan(clause, plan, {}, stream=0)
    decision = next(item for item in plan['arrivals'] if item['token'] == token)
    assert child.composition.reference == concept, 'a trial read must not publish the preparation'
    assert (chosen in decision['candidates']) == (order == 0)
    prepared = resolver.prepare(clause, plan)
    written = store.write_clause(prepared.children[0], document_key='arrival-written', stream=0)
    assert store.order_of_row(written) == owner._concept_source_order(word) == 0
    options = dict(value=atoms[0], where=None, when=2, stream=0)
    occurrence = Occurrence(store.occurrence_of(written), selected=token)
    if order == 0:
        owner.link(occurrence, {}, **options)
    else:
        with pytest.raises(ValueError, match='cannot mix collection kinds or orders'):
            owner.link(occurrence, {}, **options)


def test_event_staging_keeps_the_accessible_collections_selected_token(monkeypatch):
    from test_modal_timelines import fixture, recording, close, same
    from CaseSelection import CaseBank, stage_event_cases
    f, action, modifier = fixture(monkeypatch)
    _, frame, _, _ = close(f, recording(f, ('lift', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(.8, .2))
    token, entry = next(iter(f.cs.mtm.entries.items()))
    verb = f.words['move'][1]
    code = f.cs.similarity_codebook.W[f.cs._csw_row_of(verb)].clone()
    bank = CaseBank(torch.tensor([verb]), code[None, None], code.new_zeros(1, 1, 2),
        code.new_zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    resolver = f.cs.components.reference_close
    native = resolver.individuals
    def named_view(records):
        candidates = native(records)
        key = next(key for key, value in candidates.items() if value.get('token', key) == token)
        record = candidates.pop(key)
        candidates[verb] = dict(record, token=token)
        return candidates
    monkeypatch.setattr(resolver, 'individuals', named_view)
    assert f.cs.mtm.value_of(verb) is None, 'the view key alone is not the selected collection'
    before = f.cs.mtm.state()
    staged = stage_event_cases(bank, resolver, [(frame,)], limit=4)
    assert staged.codes.shape == (1, 4, 8)
    torch.testing.assert_close(staged.view(torch.tensor([verb])).reference[0], f.cs.mtm.value_of(token))
    for actual, expected in zip(staged.codes[0, 1:],
            (entry.events[0].before, code, f.cs.mtm.event_value(entry.events[0]))):
        torch.testing.assert_close(actual, expected)
    same(f.cs.mtm.state(), before)


@pytest.mark.parametrize('order', (0, 1))
def test_arrival_menu_keeps_the_planned_member_instead_of_a_duplicate_occurrence(order):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from ReferenceContext import SituationFrame
    from chunk_fixtures import context
    from test_reference_close import fixture, known_concept
    from test_reference_arrivals import equality
    resolver,store,a=fixture();owner=resolver.components.owner
    subject=owner.interpret.lookup_word([120],[],form='pending-form')
    word=owner.interpret.lookup_word([121],[],form='arriving-form')
    concept=known_concept(owner,a[0],order=1)
    owner.interpret.define(word,concept)
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(a[0])
        owner.similarity_codebook.W[owner._csw_row_of(subject)].copy_(a[0])
    native=known_concept(owner,a[0],order=order)
    meaning=replace(ConceptualMeaning.from_description(a[0]),role_refs=(('sym',native),None,None),
        bindings={'_forward_references':((0,subject),)})
    row=store.append_meaning(meaning,document_key='arrival-order',stream=0)
    token=store.occurrence_of(row)
    owner.link(Occurrence(token,concept=native,selected=None),{},value=a[0],where=None,when=1,stream=0)
    clause=equality(a[0],subject,concept,subject=a[0])
    child=clause.children[0]
    child=replace(child,composition=replace(child.composition,word_id=word,reference=concept,order=1),subject_word_id=word,order=1)
    clause=replace(clause,children=(child,),composition=replace(clause.composition,
        children=(clause.composition.children[0],child.composition)))
    site=replace(child,reference_request=dict(mode='bind',leaves=(1,),
        content=a[0],cue=a[0]*0,predicate=a[0]*0,role=0))
    clause=replace(clause,companions=(site,))
    _,candidates=resolver._arrival_constituents(clause)
    duplicate=('arrival',next(i for i,node in enumerate(candidates) if node is child))
    def choose(choices,logits,**kw):
        if kw.get('forward_identity')!=subject:return None
        if duplicate in choices:return choices.index(duplicate)
        return next((i for i,key in enumerate(choices) if isinstance(key,tuple) and key[:1]==('new',)),None)
    resolver.select=choose
    plan=resolver.preview(clause,context(resolver,(SituationFrame.from_row(store.row(row)),)),stream=0)
    planned = next(item for item in plan['items'] if item['node'].composition is child.composition)
    arrival = next(item for item in plan['arrivals'] if item['site_occurrence'] == token)
    assert planned['member_order'] == 1
    assert duplicate not in arrival['candidates']
    assert (planned['selected'] in arrival['candidates']) == (order == 1)
    assert all(duplicate not in item['candidates'] for item in plan['arrivals'])
    prepared=resolver.prepare(clause,plan)
    written={};row=store.write_clause(prepared,document_key='arrival-written',stream=0,written_rows=written)
    resolver.commit(plan,row,written,document='arrival-written',position=1,training=False)
    retained = owner.mtm.entries[owner.mtm.token_of(token)]
    assert retained.refinement['member_order'] == order
