"""§50: held references, explicit event states, and exact content association."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def world(*, meaning_pairs=2):
    from ClauseRow import attach_clause_index
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from test_review13_subspaces import fixture, word
    owner, bank, _, _ = fixture(meaning_pairs=meaning_pairs)
    store = LTM0(bank.W.shape[-1], capacity=16)
    owner.symbolSpace = SimpleNamespace(ltm_store=store)
    attach_clause_index(SimpleNamespace(), store, owner)
    rows = [word(owner, [(source, 1.)]) for source in (4, 7)]
    concepts = [owner.concept_id_at_row(row) for row in rows]
    owner._concept_allocator.symbol_types.setdefault(concepts[0], set()).add('word')
    from definition_fixtures import agreeing_views
    agreeing_views(owner, *concepts)
    values = bank.lookup_rows(rows).detach()
    occurrences = [store.occurrence_of(store.append_meaning(
        ConceptualMeaning.from_description(value), document_key='referents',
        sentence_index=index, order=0)) for index, value in enumerate(values)]
    return owner, bank, store, rows, concepts, occurrences


def place(bank, row, pair):
    with torch.no_grad():
        bank.mereology.placement[row] = torch.tensor(pair)
        bank.mereology.placement_witnesses[row] = 1
        bank.mereology.placement_valid[row] = True
        bank.mereology.placement_dirty[row] = True


def test_defined_word_reads_current_denotation_not_its_word_row():
    owner, bank, _, rows, concepts, _ = world()
    name, obj = concepts
    owner.interpret.define(name, obj)
    key = owner.mtm.named(name, value=bank.lookup_rows([rows[0]])[0])
    start = bank.mereology.percept_event_width
    for pair in ((.2, .8), (.9, .1)):
        place(bank, rows[1], pair)
        expected = bank.lookup_rows([rows[1]])[0]
        assert not torch.equal(expected[start:], bank.lookup_rows([rows[0]])[0, start:])
        torch.testing.assert_close(owner.mtm.value_of(key)[start:], expected[start:], rtol=0, atol=0)


@torch.no_grad()
def test_word_reads_its_single_remembered_object_after_epoch_boundary():
    from test_presentation_memory import begin, fixture, model_for
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    begin(model)
    name = owner.interpret.lookup_word([109], [], form='remembered')
    obj = owner.new_concept()
    from order_fixtures import force_order
    force_order(owner, obj, 1)
    row = owner._csw_concept_row(1, obj)
    owner.similarity_codebook.W[row].copy_(atoms[0])
    owner.interpret.define(name, obj)
    begin(model)
    assert not owner.definitions.objects(name)
    assert owner.interpret.remembered_objects(name) == (obj,)
    key = owner.mtm.named(name, value=atoms[1])
    owner.similarity_codebook.W[row].copy_(atoms[2])
    torch.testing.assert_close(owner.mtm.value_of(key), atoms[2], rtol=0, atol=0)


def test_referent_is_frozen_and_ambiguity_is_shared_with_reading():
    owner, _, _, _, concepts, _ = world()
    name, obj = concepts
    owner._concept_allocator.named_identities[name] = {obj}
    before = owner.mtm.state()
    for mandatory in (False, True):
        owner.interpret.mandatory = mandatory
        for frozen in (False, True):
            owner._online_learning_frozen = frozen
            assert owner.interpret.referent(name) == obj
            assert owner.mtm.state()['names'] == before['names']
            if not mandatory:
                # Choosing a lexical/sign read still omits interpretation;
                # a held word reference resolves independently of that choice.
                assert owner.interpret.reading(name)[0] == name
                assert owner.mtm.state()['names'] == before['names']
    other = owner.new_concept()
    from order_fixtures import force_order
    force_order(owner, other, 0)
    owner._csw_concept_row(0, other)
    owner._concept_allocator.named_identities[name].add(other)
    for read in (owner.interpret.referent, owner.interpret.reading):
        with pytest.raises(ValueError, match='ambiguous identity reading'):
            read(name)


def test_description_reads_its_own_distinguishing_coordinate():
    owner, bank, store, rows, concepts, occurrences = world()
    start = bank.mereology.percept_event_width
    point = bank.lookup_rows([rows[1]])[0].detach().clone()
    point[start:] = torch.tensor([.1, .8, .2, .9])
    store.slots[store.location(occurrences[1])] = point
    key = owner.mtm.open(occurrences[1], kind=concepts[1], reading=concepts[1], when=0, value=point)
    assert not torch.equal(bank.lookup_rows([rows[1]])[0, start:], point[start:])
    torch.testing.assert_close(owner.mtm.value_of(key)[start:], point[start:], rtol=0, atol=0)


@pytest.mark.parametrize('held', ['native', 'symbol'])
def test_identity_joins_its_explicit_concept_reference(held):
    owner, bank, _, rows, concepts, occurrences = world()
    place(bank, rows[0], (.2, .8))
    place(bank, rows[1], (.7, .3))
    values = bank.lookup_rows(rows)
    key = owner.mtm.open(occurrences[0], kind=None, reading=concepts[0], when=0,
                         value=values[0], **({'native': concepts[1]} if held == 'native' else {}))
    if held == 'symbol':
        owner.mtm.promote(key, concepts[1])
    start = bank.mereology.percept_event_width
    torch.testing.assert_close(owner.mtm.value_of(key)[start:], values[:, start:].amax(0), rtol=0, atol=0)


def test_chart_without_meaning_band_keeps_its_stored_form_and_location():
    owner, bank, _, rows, concepts, occurrences = world(meaning_pairs=0)
    held = torch.full_like(bank.lookup_rows([rows[0]])[0], .4)
    key = owner.mtm.open(occurrences[0], kind=None, reading=concepts[0], when=0, value=held)
    assert bank.mereology.percept_event_width == held.numel()
    assert not torch.equal(bank.lookup_rows([rows[0]])[0], held)
    torch.testing.assert_close(owner.mtm.value_of(key), held, rtol=0, atol=0)


def test_alternative_start_survives_the_first_event():
    owner, bank, _, rows, concepts, occurrences = world()
    key = owner.mtm.open(occurrences[0], kind=None, reading=concepts[0], when=0,
                         value=bank.lookup_rows([rows[0]])[0])
    alternative = torch.full_like(owner.mtm.entries[key].value, .6)
    branch = owner.mtm.alternative(key, when=1, before=alternative)
    place(bank, rows[0], (.1, .9))
    actual = owner.mtm.value_of(key).clone()
    owner.mtm.event(key, occurrences[1], verb=concepts[1], role=0, when=2, timeline=branch)
    torch.testing.assert_close(owner.mtm.value_of(key), actual, rtol=0, atol=0)
    torch.testing.assert_close(owner.mtm.value_of(key, timeline=branch), alternative, rtol=0, atol=0)


def test_kept_thought_adds_its_written_occurrence_reference():
    from Meaning import ConceptualMeaning
    from ThoughtStream import write
    from ThoughtReferences import with_slots
    owner, bank, store, rows, _, occurrences = world()
    point = bank.lookup_rows([rows[1]])[0].detach().clone()
    start = bank.mereology.percept_event_width
    store.slots[store.location(occurrences[0]), start:] = torch.tensor([.8, .1, .6, .2])
    point[start:] = torch.tensor([.2, .9, .1, .7])
    key = owner.mtm.open(occurrences[0], kind=None, value=store.point_of_row(occurrences[0][-1]), when=0)
    before = owner.mtm.value_of(key).clone()
    meaning = ConceptualMeaning(torch.stack((before, torch.zeros_like(point), point)),
        torch.ones(3, dtype=torch.bool), role_refs=(key, ('sym', 42), occurrences[1]),
        bindings={'_equality': True, '_producing_operation': 'query', '_bound_roles': (('referent', 2),)},
        sentence_kind='relation')
    result = write(SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store)),
                   with_slots(meaning, (), pair=(1., 0.)), row=0)
    written = store.occurrence_of(result)
    assert owner.mtm.token_of(written) == key and owner.mtm.reading_of(written) is None
    torch.testing.assert_close(owner.mtm.value_of(key)[start:],
                               torch.maximum(before, point)[start:], rtol=0, atol=0)
    assert not owner.mtm.entries[key].events


def test_update_requires_an_event_on_the_identity():
    owner, bank, _, rows, concepts, occurrences = world()
    key = owner.mtm.open(occurrences[0], kind=None, reading=concepts[0], when=0,
                         value=bank.lookup_rows([rows[0]])[0])
    with pytest.raises(ValueError, match='without an event'):
        owner.mtm.update(key, owner.mtm.entries[key].value, when=1)


@pytest.mark.parametrize('kind', ['reading', 'occurrence'])
@pytest.mark.parametrize('eventful', [False, True])
def test_ownerless_semantic_reference_fails_loudly(kind, eventful):
    from Memory import MediumTermMemory
    pool = MediumTermMemory()
    key = ('ltm', 'address-v1', 11) if kind == 'occurrence' else ('numeric', 11)
    pool.open(key, kind=None, when=0, value=torch.ones(3), reading=7 if kind == 'reading' else None)
    if eventful:
        from Memory import Event, Timeline
        pool.entries[key] = replace(pool.entries[key], timelines=(
            Timeline((Event(('numeric', 2), 1, 0, 1),), torch.ones(3)),))
    with pytest.raises(ValueError, match='without an owner'):
        pool.value_of(key)


def test_ownerless_numerical_pool_keeps_explicit_value_api():
    from Memory import MediumTermMemory
    pool = MediumTermMemory()
    pool.open(('numeric', 1), kind=None, when=0, value=torch.tensor([.2, .7]))
    torch.testing.assert_close(pool.value_of(('numeric', 1)), torch.tensor([.2, .7]), rtol=0, atol=0)
    pool.event(('numeric', 1), ('numeric', 2), verb=1, role=0, when=1)
    pool.update(('numeric', 1), torch.tensor([.3, .8]), when=2)
    torch.testing.assert_close(pool.value_of(('numeric', 1)), torch.tensor([.3, .8]), rtol=0, atol=0)


def test_tiny_positive_containment_is_exact_and_empty_content_is_not_evidence():
    from IdentityRows import meet_ratios
    query = torch.tensor([1e-30, 0.])
    covered, covers = meet_ratios(query, torch.tensor([1e-30, 1e-30]))
    assert covered == .5 and covers == 1
    assert all(ratio == 0 for ratio in meet_ratios(torch.zeros(2), query))


def test_reclamation_protects_a_held_reading():
    from ConceptForgetting import reclaim_concepts
    from test_concept_forgetting import _cs, concept
    owner = _cs(8)
    weak, row = concept(owner, magnitude=.1)
    strong, _ = concept(owner, magnitude=.9)
    owner.mtm.named(weak, value=owner.similarity_codebook.W[row])
    result = reclaim_concepts(owner)
    assert weak not in result['retired'] and owner._csw_row_of(weak) is not None
    assert strong in result['retired']


def test_ingestion_pool_has_its_owner_and_reads_current_meanings():
    from contextlib import nullcontext
    from Models import BasicModel
    from Meaning import ConceptualMeaning
    owner, bank, store, rows, concepts, _ = world()
    caller = owner.mtm
    captured = []
    def batch(**kwargs):
        pool = owner.mtm
        assert pool is not caller
        assert pool._owner() is owner
        assert pool.field_width == caller.field_width
        assert int(pool.transient_ids[0]) == owner.nVectors
        key = pool.named(concepts[0], value=bank.lookup_rows([rows[0]])[0])
        place(bank, rows[0], (.6, .4))
        start = bank.mereology.percept_event_width
        current = bank.lookup_rows([rows[0]])[0]
        torch.testing.assert_close(pool.value_of(key)[start:], current[start:], rtol=0, atol=0)
        captured.append(pool)
        store.append_meaning(ConceptualMeaning.from_description(current),
            document_key='ingestion', sentence_index=0)
    model = SimpleNamespace(eval=lambda: None, set_sigma=lambda _: None,
        _what_memory=lambda: None, _concept_owner=lambda: owner,
        _runtime_batch=lambda *args, **kwargs: nullcontext(),
        inputSpace=SimpleNamespace(prepInput=lambda _: torch.ones(1, 1)),
        data=SimpleNamespace(train_input=('supplied',)), runBatch=batch,
        _clear_recall_history=lambda: None)
    # Exercise the native ingestion constructor and its try/finally owner swap.
    with pytest.raises(ValueError, match='did not produce a selected grammatical closing'):
        BasicModel._ltm_ingest_truth_texts(model, store, ['supplied'])
    assert len(captured) == 1 and owner.mtm is caller


def test_accessible_reads_each_identity_once(monkeypatch):
    from test_reference_close import fixture
    from Meaning import ConceptualMeaning
    from Memory import STMChunks
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm
    uses = [store.occurrence_of(store.append_meaning(ConceptualMeaning.from_description(atoms[0]),
            document_key='uses', sentence_index=i)) for i in range(3)]
    key = pool.open(uses[0], kind=None, value=atoms[0], when=0)
    for occurrence in uses[1:]:
        pool.bind(key, occurrence, kind=None, value=atoms[0], when=1)
    original, calls = pool.value_of, []
    def counted(reference, **kwargs):
        calls.append(reference)
        return original(reference, **kwargs)
    monkeypatch.setattr(pool, 'value_of', counted)
    records = resolver.accessible(STMChunks.empty(1, 8, len(atoms)), include_pool=True)
    assert len(records) == 3 and calls == [key]


@pytest.mark.parametrize('case', ['exclusive', 'part', 'kind', 'empty', 'tiny'])
def test_content_association_requires_exact_containment_in_both_menus(case):
    from test_reference_close import fixture
    from Memory import STMChunks
    resolver, _, atoms = fixture()
    if case == 'exclusive':
        mention, candidate, allowed = atoms[0]+atoms[1], atoms[0]+atoms[2], False
    elif case == 'part':
        mention, candidate, allowed = atoms[0], atoms[0]+atoms[1]+atoms[2], True
    elif case == 'kind':
        mention, candidate, allowed = atoms[0]+atoms[1], atoms[0], True
    elif case == 'empty':
        mention, candidate, allowed = atoms[0]*0, atoms[0], False
    else:
        mention, candidate, allowed = atoms[0]*1e-30, (atoms[0]+atoms[1])*1e-30, True
    choices, _, _ = resolver.score(mention, atoms[5], atoms[6]*0, 0,
        {17:dict(content=candidate,role=0,position=(0,0))})
    assert (17 in choices) == allowed
    state = STMChunks.empty(1, 8, len(atoms))
    state, at = STMChunks.deposit(state, candidate[None], torch.tensor([True]),
                                  identity=17, order=1, grammar_order=0)
    valid = STMChunks.bind_menu(state, mention[None,None])[-1]
    assert bool(valid[0,0,int(at)+1]) == allowed


def test_content_association_ignores_form_and_location():
    from Memory import STMChunks
    state = STMChunks.empty(1, 8, 6)
    candidate = torch.tensor([[1., 0., 1., 0., .2, .8]])
    mention = torch.tensor([[[0., 1., 0., 1., .2, 0.]]])
    state, at = STMChunks.deposit(state, candidate, torch.tensor([True]), identity=17)
    assert bool(STMChunks.bind_menu(state, mention, meaning_start=4)[-1][0,0,int(at)+1])
