"""Input-only episodes and content-free anchors on the sentence clock."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import attach_clause_index, ClauseCapacityError
from Meaning import ConceptualMeaning
from Memory import LTM0
from RowTime import encoder
from Occurrence import operand_name


def test_pending_selected_phrase_keeps_its_own_occurrence_when_the_head_is_resolved():
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from MemoryLink import Occurrence
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([122], (), form='pending-witness')
    earlier = known_concept(owner, atoms[1], order=1)
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_forward_references': ((0, word),), '_open_references': (('referent', 0),)}),
        )
    address = store.occurrence_of(first)
    key = owner.link(Occurrence(address, pending=True), {}, value=atoms[0],
        where=None, when=0, stream=0)
    own = atoms[0] + atoms[1]
    node = Clause(ConceptualMeaning.from_description(own), point=own, order=1,
        composition=Composition(own, (Composition(atoms[1], reference=earlier),),
            operation='bind', reference=address[-1], head_role=1, order=1),
        reference_request=dict(mode='bind', compose_bind=1, forward_identity=word,
            leaves=(0, 1), cue=own, content=own, predicate=own, role=0),
        identity=dict(handle=1, origins=torch.tensor([[2, address[-1]]])))
    plan = resolver.preview(node, context(resolver, ()), stream=0)
    kept = resolver.prepare(node, plan)
    assert kept.refs[0] == -1 and kept.composition.reference == operand_name(address[-1])
    written = {}
    row = store.write_clause(kept,  written_rows=written)
    assert store.referent_of(row) == address
    assert store.order_of_row(row) == plan['items'][0]['member_order'] == 0
    resolver.commit(plan, row, written, document='pending-alias', position=1, training=False)
    assert owner.mtm.token_of(store.occurrence_of(row)) == key


@pytest.mark.parametrize('operation', ('lift', 'verb'))
def test_composed_alias_keeps_the_selected_occurrence_instead_of_its_original_head(operation):
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from MemoryLink import Occurrence
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    selected = known_concept(owner, atoms[0], order=0)
    earlier = known_concept(owner, atoms[1], order=1)
    first = store.append_meaning(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': ('sym', selected)}))
    address = store.occurrence_of(first)
    key = owner.link(Occurrence(address, selected=None), {}, value=atoms[0],
        where=None, when=0, stream=0)
    own = atoms[0] + atoms[1]
    operands = (Composition(atoms[1], reference=earlier), Composition(atoms[1], reference=earlier))
    node = Clause(ConceptualMeaning.from_description(own), point=own, order=1,
        composition=Composition(own, operands, operation=operation, order=1),
        reference_request=dict(mode='bind', compose_bind=1, leaves=(0, 1),
            cue=own, content=own, predicate=own, role=0),
        identity=dict(handle=1, origins=torch.tensor([[2, address[-1]]])))
    plan = resolver.preview(node, context(resolver, ()), stream=0)
    kept = resolver.prepare(node, plan)
    assert kept.composition.operation == operation and kept.refs[0] == operand_name(address[-1])
    written = {}
    row = store.write_clause(kept,  written_rows=written)
    assert store.referent_of(row) == address
    assert store.order_of_row(row) == plan['items'][0]['member_order'] == 0
    resolver.commit(plan, row, written, document='composed-alias', position=1, training=False)
    assert owner.mtm.token_of(store.occurrence_of(row)) == key
    assert owner.mtm.entries[key].refinement['member_order'] == 0


@pytest.mark.parametrize('operation', ('lift', 'verb'))
def test_nested_introduction_prepares_its_child_before_claiming_the_written_order(operation):
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention, keep
    from ClauseRow import Clause, Composition
    from FamiliarTier import promote_identity
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([122], (), form='lexical-witness')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=1.)
    child = replace(mention(atoms[0], reference=concept),
        meaning=replace(ConceptualMeaning.from_description(atoms[0]),
                        bindings={'_identity': ('sym', word)}))
    other = Composition(atoms[1])
    operands = (child.composition, other) if operation == 'lift' else (other, child.composition)
    root = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=1,
        children=(child,), composition=Composition(atoms[0], operands, operation=operation, order=1),
        reference_request=dict(mode='mint', leaves=(0, 1), cue=atoms[0]*0,
            content=atoms[0], predicate=atoms[0]*0, role=0))
    assert resolver.sites(root)[0]['node'] is child
    row, plan = keep(resolver, root)
    key = owner.mtm.token_of(store.occurrence_of(row))
    assert [item['member_order'] for item in plan['items']] == [0, 0]
    assert store.order_of_row(row) == owner.mtm.entries[key].refinement['member_order'] == 0
    assert owner._concept_source_order(promote_identity(owner, key)) == 1


@pytest.mark.parametrize('cleared_row', (None, 0))
def test_hard_clear_releases_thought_views_before_epoch_time_repeats(cleared_row):
    from Memory import ShortTermMemory
    from ThoughtChunks import held, put
    from ThoughtReferences import bindings
    from clock_fixtures import for_stm
    stm = ShortTermMemory(batch=2, capacity=4, concept_dim=8)
    model = for_stm(stm)
    old = ConceptualMeaning.from_description(torch.ones(8))
    put(stm, old, row=0)
    second = put(stm, old, row=0)
    repeated_when = bindings(second)['_chunk_when']
    other = put(stm, old, row=1)
    old_view = stm._thought_contents[0][1][1]
    stores = tuple(tuple(id(store) for store in row) for row in stm._thought_stores)
    stm.clear(cleared_row)
    # A later presentation can reuse this stamp in a different chunk seat.
    model.when_time.fill_(1)
    fresh = ConceptualMeaning.from_description(torch.full((8,), .25))
    current = put(stm, fresh, row=0)
    assert bindings(current)['_chunk_when'] == repeated_when
    resident = held(stm, 0)
    assert len(resident) == 1 and resident[0][0] == repeated_when
    torch.testing.assert_close(resident[0][1].roles, fresh.roles)
    with pytest.raises(ValueError, match='expired'):
        old_view.read()
    assert tuple(tuple(id(store) for store in row) for row in stm._thought_stores) == stores
    if cleared_row is None:
        assert not held(stm, 1)
    else:
        assert held(stm, 1)[0][0] == bindings(other)['_chunk_when']


def episode_fixture(monkeypatch, capacity=2):
    from test_cs_sparse_weights import _cs
    from test_clause_acceptance import SentenceFixture
    from clock_fixtures import install
    f = SentenceFixture(monkeypatch, cs=_cs(nS=512, order=5, dim=12), percept_width=12)
    f.store = LTM0(12, capacity=64, percept_width=12, episodic_concepts=capacity)
    f.model = install(f.model, tick=17)
    f.model.symbolSpace = SimpleNamespace(ltm_store=f.store)
    f.model._open_sentence_slot = 0
    f.model._sentence_ticks[0] = 17
    f.cs._model = f.model
    attach_clause_index(f.model, f.store, f.cs)
    entry = (torch.tensor([2, 7][:capacity]), torch.tensor([-.6, .8][:capacity]))
    f.model._sentence_episodes = {0: tuple(value[None] for value in entry)}
    return f, entry








def test_thought_write_during_input_close_does_not_take_the_input_episode(monkeypatch):
    from Models import _append_observed_meaning
    from ThoughtStream import write
    f, entry = episode_fixture(monkeypatch)
    clause = f.clause('alpha')
    original = f.store.write_clause
    thoughts = []
    def close(*args, **kwargs):
        if kwargs.get('episode') is not None:
            # The input entry remains staged while the kept thought is written.
            thoughts.append(write(f.model, replace(clause.meaning,
                bindings=dict(clause.meaning.bindings, _producing_operation='query')), row=0))
        return original(*args, **kwargs)
    monkeypatch.setattr(f.store, 'write_clause', close)
    own = _append_observed_meaning(f.store, clause, stream=0, episode=entry)
    assert thoughts and all(row != own for row in thoughts)
    assert torch.equal(f.store.episodic_concepts[own], entry[0])
    for row in thoughts:
        assert (f.store.episodic_concepts[row] == -1).all()
    assert f.model._episode_counts['written'] == 1


def test_composition_reads_time_differences_and_preserves_the_stored_stamp():
    from Spaces import ConceptualSpace
    clock = encoder()
    reference = None
    for tick in (0, 25, 50, 100, 200, 400, 777, 1500, 3000):
        clock.t = tick + 7
        value = torch.cat((torch.ones(1, 2, 10), clock.encode(torch.tensor([[tick, tick+7]]))), -1)
        before = value.clone()
        sub = SimpleNamespace(is_empty=lambda: False, materialize=lambda: value,
            nWhat=6, nWhere=4, nWhen=4, whenEncoding=clock)
        read, _, stored = ConceptualSpace._bind_read(sub, 14)
        assert torch.equal(stored, before) and torch.equal(value, before)
        assert not read[0, 1, -4:].any()
        assert bool(read[0, 0, -4:].any())
        if reference is None:
            reference = read
        assert torch.equal(read, reference)


@pytest.mark.parametrize('member_order', (0, 1))
def test_state_trial_uses_the_retained_member_order_before_identity_promotion(monkeypatch, member_order):
    from test_reference_close import fixture, known_concept
    from chunk_fixtures import context
    from ClauseRow import Clause, Composition
    from FamiliarTier import promote_identity
    import MemoryLink
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    concept = known_concept(owner, atoms[0], order=member_order)
    clause = Clause(replace(ConceptualMeaning.from_description(atoms[0]),
        bindings={'_identity': ('sym', concept)}), point=atoms[0], order=0,
        composition=Composition(atoms[0], reference=concept, order=0))
    orders = []
    native_menu = MemoryLink.menu
    def menu(owner, site, candidates, **kwargs):
        if site.kind == 'state':
            orders.append(site.order)
        return native_menu(owner, site, candidates, **kwargs)
    monkeypatch.setattr(MemoryLink, 'menu', menu)
    keys = []
    for position in range(3):
        plan = resolver.preview(clause, context(resolver, ()), stream=0)
        prepared = resolver.prepare(clause, plan)
        row = store.write_clause(prepared, )
        assert orders[-1] == store.order_of_row(row) == member_order
        key = owner.link(MemoryLink.Occurrence(store.occurrence_of(row), kind='state',
            order=store.order_of_row(row), selected=plan['link_choices'][('state', 0)]), {},
            value=clause.point, where=None, when=position+1, stream=0)
        owner.mtm.close(store, stream=0)
        keys.append(key)
    assert keys[0] == keys[1] == keys[2]
    raised = promote_identity(owner, keys[0])
    assert owner._concept_source_order(raised) == member_order + 1
    assert owner.mtm.entries[keys[0]].refinement['member_order'] == member_order


def test_lexical_introduction_and_checked_thought_keep_the_written_member_order():
    from types import SimpleNamespace
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention, keep
    from ThoughtReferences import with_slots
    from ThoughtStream import write
    from FamiliarTier import promote_identity
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([122], (), form='lexical-witness')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=1.)
    node = mention(atoms[0], reference=word)
    node = replace(node, subject_word_id=word,
        meaning=replace(node.meaning, bindings={'_identity': ('sym', word)}),
        reference_request=dict(node.reference_request, forward_identity=word))
    row, plan = keep(resolver, node)
    occurrence = store.occurrence_of(row)
    key = owner.mtm.token_of(occurrence)
    assert owner._concept_source_order(word) == 0
    assert owner._concept_source_order(owner.interpret.referent(word)) == 1
    assert plan['items'][0]['member_order'] == store.order_of_row(row) == 0
    assert owner.mtm.entries[key].refinement['member_order'] == 0
    meaning = ConceptualMeaning(torch.stack((atoms[0], atoms[1], atoms[0]*0)),
        torch.ones(3, dtype=torch.bool), role_refs=(occurrence, ('sym', concept), None),
        bindings={'_bound_roles': (('referent', 0),), '_producing_operation': 'ask',
                  '_chain_view': True}, sentence_kind='relation')
    proof = write(SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store)),
        with_slots(meaning, (), pair=(1., 0.)), row=0)
    assert store.order_of_row(proof) == 0
    assert owner.mtm.token_of(store.occurrence_of(proof)) == key
    assert len(owner.mtm.entries[key].occurrences) == 2
    promoted = promote_identity(owner, key)
    assert owner._concept_source_order(promoted) == 1


def force_relation(model, monkeypatch):
    chooser = model.languageSpace._tree_layer(2).chooser
    part = next(i for i, rule in enumerate(model.languageSpace._compose_binary_rules)
                if rule.method_name == 'part')
    binary, unary = chooser.score_binary, chooser.score_unary
    def score_binary(*args, **kwargs):
        stop, scores = binary(*args, **kwargs)
        indices = kwargs.get('op_indices', torch.arange(scores.shape[-1], device=scores.device))
        return stop, scores.masked_fill(indices != part, -torch.inf) + 1e6
    def score_unary(*args, **kwargs):
        stop, scores = unary(*args, **kwargs)
        if kwargs.get('op_offset') == model._stm_reducer().r_reduce + model._stm_reducer().r_apply:
            return stop, scores
        indices = kwargs.get('op_indices')
        forced = torch.full_like(scores, -torch.inf)
        if indices is not None:
            forced = torch.where(indices >= model._stm_reducer().r_reduce + model._stm_reducer().r_apply,
                                 scores, forced)
        return stop, forced
    monkeypatch.setattr(chooser, 'score_binary', score_binary)
    monkeypatch.setattr(chooser, 'score_unary', score_unary)
