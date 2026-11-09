"""Completion certificates for recorded orders, counted raises and epoch reads."""
from types import SimpleNamespace
import math

import pytest
import torch

from Memory import SYMBOL, OCCURRENCE


@pytest.mark.parametrize('operation', ['equal', 'part'])
def test_same_order_identity_row_survives_the_epoch_boundary(operation):
    from ClauseRow import Clause, Composition, predicate_identity
    from FamiliarTier import promote_identity
    from Meaning import ConceptualMeaning
    from test_presentation_memory import begin, fixture, model_for, write
    resolver, store, points = fixture()
    owner = resolver.components.owner
    alloc = owner._concept_allocator
    begin(model_for(owner, store))
    symbols = []
    for i, point in enumerate(points[:2]):
        occurrence = write(store, point, ('admission', i), order=0)
        token = owner.mtm.open(occurrence, kind=None, value=point, when=i)
        symbols.append(promote_identity(owner, token))
    left, right = symbols
    operands = tuple(Composition(point, reference=cid, order=1) for cid, point in zip(symbols, points))
    term = Composition(operands[0].point, operands, operation=operation, order=1)
    clause = Clause(ConceptualMeaning.from_description(term.point), relation=operation,
        refs=(left, predicate_identity(operation), right), composition=term, order=1)
    trust = min(float(store._concept_point(cid).norm()) for cid in symbols)
    written = store.write_clause(clause, evidence=(1., .25))
    row = alloc.parts.find(left, right)
    assert written == ('sym', row) and store.row(written)['evidence'] == pytest.approx((trust, .25*trust))
    assert ('sym', left) not in alloc.refs(right, 'part')
    assert ('sym', right) not in alloc.refs(left, 'whole')
    begin(model_for(owner, store))
    assert [alloc.order_of(cid) for cid in symbols] == [1, 1]
    assert alloc.parts.find(left, right) is None


def test_order_boundary_membership_follows_one_rule():
    from test_orders_learned import equality_fixture
    owner, _, alloc = equality_fixture()
    from ConceptIndex import taxonomy_children
    concrete, whole, kind = (alloc.new_concept() for _ in range(3))
    alloc.reference_orders.update({concrete: 0, whole: 0, kind: 1})
    alloc.parts.add(concrete, whole)
    assert ('sym', concrete) not in alloc.refs(whole, 'part')
    alloc.parts.add(concrete, whole, truth_plus=1.)
    alloc.parts.add(whole, whole, truth_plus=1.)
    assert alloc.parts.find(concrete, whole) is not None
    assert ('sym', concrete) not in alloc.refs(whole, 'part')
    assert concrete not in taxonomy_children(owner, whole)
    assert alloc.parts.find(whole, whole) is not None
    assert not alloc.class_member(concrete, whole)
    alloc.parts.add(whole, kind, truth_plus=1.)
    alloc.parts.add(kind, whole, truth_plus=1.)
    assert ('sym', whole) in alloc.refs(kind, 'part')
    assert ('sym', kind) not in alloc.refs(whole, 'part')
    assert [alloc.settle(cid) for cid in (concrete, whole, kind)] == [0, 0, 1]


def test_fold_and_cover_read_only_recorded_class_members():
    from DefinitionFold import examples
    from FamiliarTier import class_members
    from test_definition_fold import fixture
    owner, _, alloc, whole, _, source, witness, field = fixture()
    member = owner.concept_id_at_row(source)
    alloc.reference_orders.update({member: 1, whole: 1})
    alloc.parts.add(member, whole, truth_plus=1.)
    assert not examples(owner, field) and not class_members(owner, whole)
    assert not class_members(owner, member)
    alloc.reference_orders[whole] = 2
    assert [e.witness for e in examples(owner, field)] == [witness]
    assert class_members(owner, whole)[source] > 0
    del alloc.reference_orders[whole]
    alloc.parts.truth_plus[witness] = 0
    example, = examples(owner, field)
    assert example.member == (SYMBOL, member) and example.evidence[0] == 0 and example.evidence[1] > 0


@pytest.mark.parametrize('shared_first', (False, True))
def test_three_distinct_members_at_either_end_name_one_class(shared_first):
    from test_corpus_orders import named_individual, symbol_equality
    from test_orders_learned import equality_fixture
    owner, store, alloc = equality_fixture()
    members = [named_individual(owner, store, alloc, str(i), 3*i+1) for i in range(4)]
    word = owner.interpret.lookup_word([20], [21], form='group')
    raised = set(alloc.raised)
    for i, member in enumerate(members):
        ends = (word, member) if shared_first else (member, word)
        symbol_equality(owner, store, *ends)
        assert alloc.parts.find(*ends) is not None
        assert alloc.order_of(member) == 1 and alloc.order_of(word) == 0
        if i < 2:
            assert set(alloc.raised) == raised and not owner.definitions.objects(word)
        else:
            parent, = set(alloc.raised) - raised
            assert owner.definitions.objects(word) == (parent,) and alloc.order_of(parent) == 2
            assert set(alloc.refs(parent, 'part')) == {('sym', x) for x in members[:i+1]}
    alloc.definitions.rebuild()
    assert owner.definitions.objects(word) == (parent,)


def test_a_previously_named_shared_concept_denotes_its_raised_class():
    from test_corpus_orders import named_individual, symbol_equality
    from test_orders_learned import equality_fixture
    owner, store, alloc = equality_fixture()
    shared, *members = [named_individual(owner, store, alloc, str(i), 3*i+1) for i in range(4)]
    name, = owner.definitions.words(shared)
    for member in members[:2]:
        symbol_equality(owner, store, shared, member)
        assert owner.interpret.forward(name) == shared
    symbol_equality(owner, store, shared, members[2])
    parent = owner.interpret.forward(name)
    assert alloc.order_of(parent) == 2 and alloc.order_of(shared) == 1
    assert set(alloc.refs(parent, 'part')) == {('sym', member) for member in members}
    assert 'word' not in alloc.symbol_types[shared]


def test_raise_count_rejects_one_and_obeys_a_larger_count(monkeypatch):
    from Layers import ConceptAllocator
    from util import TheXMLConfig
    from test_orders_learned import equality_fixture, equal_close, event
    owner, store, alloc = equality_fixture()
    read = TheXMLConfig.space
    count = 1
    monkeypatch.setattr(TheXMLConfig, 'space', lambda space, key, *args, **kwargs:
        count if key == 'raiseCount' else read(space, key, *args, **kwargs))
    with pytest.raises(ValueError, match='raiseCount'):
        ConceptAllocator()
    count = 4
    owner, store, alloc = equality_fixture()
    word = owner.interpret.lookup_word([1], [2], form='group')
    for i in range(count):
        equal_close(owner, store, event(owner, store, i+1), word)
        assert bool(alloc.raised) == (i+1 == count)
    assert len(alloc.raised) == 1


@pytest.mark.parametrize('difference,ready', [(8e-5, False), (2e-4, True)])
def test_bounded_binary_progress_uses_absolute_form_error(difference, ready):
    from test_decomposition_chooser import search
    parent = torch.tensor([[0., 0.], [1e-30, 1e-30]])
    bank = (parent + difference)[:, None, :]
    _, _, valid, _ = search(bank, parent, require_progress=True)
    assert valid.tolist() == [ready, ready]


def test_epoch_gradient_combines_replaced_definition_values_by_name_and_edge():
    from Layers import SparseLayer
    from SentenceTraining import measure, gradient_receipt
    matrix = SparseLayer(4, 2)
    matrix.add_edge(0, 0)
    matrix.add_edge(0, 1)
    model = torch.nn.Module()
    model._measure_sentence_gradients = True
    model.separate_chooser = False
    layer = SimpleNamespace(definition_matrices=lambda: (matrix,))
    model.conceptualSpaces = [SimpleNamespace(_concept_allocator=SimpleNamespace(layer=lambda _: layer))]
    model.objective_parameter_groups = lambda opt: dict(reconstruction=(matrix.values,),
        output=(), expectation=(), attention=())
    def sample(r, a):
        p = matrix.values
        measure(model, dict(reconstruction=p @ p.new_tensor(r), output=p @ p.new_tensor(a)),
                torch.optim.SGD([p], lr=.1))
        assert p.grad is None
    sample([3., 4.], [5., 6.])
    matrix.remove_edges([(0, 0)])
    matrix.add_edge(0, 0)  # Same coordinates, a fresh allocation.
    sample([-4., 12.], [-6., 12.])
    reading = gradient_receipt(model, clear=True)['understanding']
    assert reading['reconstruction_norm'] == pytest.approx(math.hypot(3., 12.))
    assert reading['output_norm'] == pytest.approx(13.)
    assert not model._understanding_epoch


def test_lesson_sides_read_the_kept_occurrence_and_leave_an_unresolved_slot():
    from corpus_lessons import observed_sides
    from FamiliarTier import promote_identity
    from test_presentation_memory import fixture, write
    resolver, store, points = fixture()
    owner = resolver.components.owner
    address = write(store, points[0], 'reading', order=0)
    token = owner.mtm.open(address, kind=None, value=points[0], when=0)
    symbol = promote_identity(owner, token)
    assert observed_sides(owner, ((OCCURRENCE, address[-1]),)) == [
        dict(reference=(OCCURRENCE, address[-1]), symbol=symbol, order=1),
        dict(reference=None, symbol=None, order=None)]


def test_ambiguous_pointer_identity_across_time_is_empty(tmp_path):
    from ClauseRow import Clause, Composition, predicate_identity
    from FamiliarTier import promote_identity
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    from test_one_backward import receipt
    from test_reference_close import fixture

    resolver, store, points = fixture(percept=True)
    owner = resolver.components.owner
    alloc, pool = owner._concept_allocator, owner.mtm
    assert alloc.raise_count == 3
    relata = []
    for i, form in enumerate(('a', 'b', 'c')):
        label = owner.interpret.lookup_word([i+1], (), form=form)
        row = store.append_meaning(ConceptualMeaning.from_description(points[i]), order=0,
                                   document_key=('given', form), when=False)
        occurrence = store.occurrence_of(row)
        token = pool.open(occurrence, kind=None, value=points[i], when=i)
        # These relata are supplied as known operands of the pointer test.
        # Three agreeing occurrences meet the new definition threshold.
        for view in range(2):
            extra = store.append_meaning(ConceptualMeaning.from_description(points[i]), order=0,
                document_key=('given-view', form, view), when=False)
            pool.bind(token, store.occurrence_of(extra), kind=None, value=points[i], when=i)
        symbol = promote_identity(owner, token, word=label)
        assert alloc.order_of(symbol) == 1
        relata.append((label, symbol))
    vectors = torch.stack([store._concept_point(cid) for _, cid in relata])
    support = vectors.ne(0)
    assert support.any(1).all() and (support.sum(0) <= 1).all()

    word = owner.interpret.lookup_word([4], (), form='this')
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(points[3])
    assert not owner.definitions.objects(word)
    facts = []
    identity = None
    for i, ((label, member), sentence) in enumerate(zip(relata, ('This is a.', 'This is b.', 'This is c.'))):
        reference, row, order = owner.interpret.reading(word)
        point = owner.similarity_codebook.lookup_rows(row).detach()
        subject = Composition(point, reference=reference, word_id=word, order=order, leaves=(0,))
        other = Composition(store._concept_point(member), reference=member, word_id=label, order=1, leaves=(2,))
        meaning = ConceptualMeaning.from_description(point)
        children, role = (), reference
        if i == 0:
            # The native reading returned a pending name handle. Carry its
            # actual retained origin through the kept journal's ordinary bind.
            key = pool.token_of(reference)
            assert key == ('name', word) and pool.entries[key].transient == reference
            mention = Clause(meaning, point=point, order=order, composition=subject, subject_word_id=word,
                reference_request=dict(mode='bind', role=0, leaves=(0,), cue=point, content=point,
                                       forward_identity=word, reading_identity=1),
                identity=dict(handle=1, origins=torch.tensor([[6, word, 0, 0]])))
            children, role = (mention,), ('clause', 0)
        else:
            assert reference == identity
        clause = Clause(meaning, relation='equal', refs=(role, predicate_identity('equal'), member),
            order=order, evidence=(1., 0.), children=children,
            composition=Composition(point, (subject, other), operation='equal', order=order))
        plan = resolver.preview(clause, context(resolver, ()))
        kept = resolver.prepare(clause, plan)
        written = {}
        address = store.write_clause(kept, document_key='pointer-over-time', sentence_index=i,
                                     evidence=(1., 0.), text=sentence, written_rows=written)
        resolver.commit(plan, address, written, document='pointer-over-time', position=i, training=False)
        if i == 0:
            identity, = owner.definitions.objects(word)
            first_mention = plan['mention_occurrences'][id(plan['items'][0]['written_node'])]
            assert pool.entries[pool.token_of(first_mention)].symbol == identity
            assert alloc.order_of(identity) == 1
            from ConceptTrust import observe
            for view in range(2):
                extra = store.append_meaning(ConceptualMeaning.from_description(points[3]), order=0,
                    document_key=('pointer-view', view), when=False)
                observe(owner, identity, points[3])
        pair = ((SYMBOL, identity), (SYMBOL, member))
        direct, reverse = alloc.parts.find(*pair), alloc.parts.find(*reversed(pair))
        stored = alloc.parts.references(address[1])
        facts.append(dict(sentence=sentence, word=word, identity=identity, relatum=member,
            identity_order=alloc.order_of(identity), word_objects=owner.definitions.objects(word),
            first_mention=first_mention, row=address[1], stored_references=stored,
            stored_pair_lookup=alloc.parts.find(*stored), identity_to_relatum=direct,
            relatum_to_identity=reverse, evidence=store.row(address)['evidence']))
        receipt(tmp_path, 'pointer-across-time', dict(relata=[cid for _,cid in relata],
            member_rows=vectors.tolist(), steps=facts))
        if i == 0:
            occurrence = (OCCURRENCE, first_mention[-1])
            assert ('occ', first_mention[-1]) in alloc.refs(identity, 'part')
            assert alloc.parts.find(occurrence, (SYMBOL, member)) == address[1]
        else:
            assert direct == address[1]
        parent = alloc.relate_idx.get(('part-kind', (SYMBOL, identity), 2))
        if i < 2:
            assert parent is None and owner.interpret.denotations(word) == (identity,)
        else:
            assert parent is not None and alloc.order_of(parent) == 2
            assert owner.interpret.denotations(word) == (parent,)
            members = {cid for tag, cid in alloc.refs(parent, 'part') if tag == 'sym'}
            assert members == {cid for _, cid in relata}
            assert all(alloc.order_of(cid) + 1 == alloc.order_of(parent) for cid in members)
            invariant = torch.stack([store._concept_point(cid) for cid in sorted(members)]).amin(0)
            assert torch.count_nonzero(invariant) == 0
            receipt(tmp_path, 'pointer-across-time', dict(steps=facts, parent=parent,
                members=sorted(members), member_rows=vectors.tolist(), invariant=invariant.tolist()))


def test_pointer_identity_across_time_keeps_bound_predicate(monkeypatch, tmp_path):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    from chunk_fixtures import recall
    from test_clause_acceptance import SentenceFixture
    from test_one_backward import receipt

    f = SentenceFixture(monkeypatch)
    cat, sleeps, black = (f.noun(form) for form in ('cat', 'sleeps', 'black'))
    cat_point, asleep, predicate = (f.registry._payload(('sym', cid)) for cid in (cat, sleeps, black))
    point = torch.maximum(cat_point, asleep)
    antecedent = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
        composition=Composition(point, (Composition(cat_point, reference=cat, order=1),
                                        Composition(asleep, reference=sleeps)),
                                operation='lift', formal_name='lift', head_role=2, order=1))
    first = f.store.write_clause(antecedent, document_key='bound-over-time', sentence_index=0,
                                 text='A cat sleeps.')
    address = f.store.occurrence_of(first)
    assert dict(f.store.raw_meaning_of(first).bindings)['_identity'] == ('sym', cat)
    request = dict(mode='bind', leaves=(0,), cue=torch.zeros_like(point), predicate=predicate, role=0)
    term = Composition(point, order=1)
    mention = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                     composition=term, reference_request=request)
    value = torch.maximum(point, predicate)
    clause = Clause(ConceptualMeaning.from_description(value), point=value, children=(mention,),
        composition=Composition(value, (term, Composition(predicate, reference=black)),
                                operation='lift', formal_name='lift', head_role=2))
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, recall(resolver, (SituationFrame.from_row(f.store.row(first)),)))
    item, = plan['items']
    assert address in item['candidates'] and item['selected'] == address
    kept = resolver.prepare(clause, plan)
    written = {}
    second = f.store.write_clause(kept, document_key='bound-over-time', sentence_index=1,
                                  text='It is black.', written_rows=written)
    resolver.commit(plan, second, written, document='bound-over-time', position=1, training=False)
    second_mention = plan['mention_occurrences'][id(item['written_node'])]
    token = f.cs.mtm.token_of(address)
    assert address != second_mention and token is not None and f.cs.mtm.token_of(second_mention) == token
    assert dict(f.store.raw_meaning_of(second).bindings)['_identity'] == address
    torch.testing.assert_close(f.store.slots[second], value, rtol=0, atol=0)
    receipt(tmp_path, 'bound-predicate-across-time', dict(first_mention=address, second_mention=second_mention,
        token=token, binding=dict(f.store.raw_meaning_of(second).bindings)['_identity'],
        point=point.tolist(), predicate=predicate.tolist(), kept_value=f.store.slots[second].tolist()))


def _sign_equation(f, form, label, position):
    from Understanding import Reading
    from reading_fixtures import finish_reading
    from chunk_fixtures import context
    owner, store = f.cs, f.store
    ids, words, rows, orders, values = [], [], [], [], []
    for spelling in (form, label):
        word = owner.interpret.word_for_form(spelling)
        if word is None:
            word = owner.interpret.lookup_word([len(owner._concept_allocator.word_forms)+1], (), form=spelling)
            with torch.no_grad():
                owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(torch.eye(store.nDim)[len(words)])
        identity, row, order = owner.interpret.reading(word)
        ids.append(identity); words.append(word); rows.append(row); orders.append(order)
        values.append(owner.similarity_codebook.lookup_rows(row).detach())
    sign = next(i for i,r in enumerate(f.unary) if r.method_name == 'sign')
    equal = next(i for i,r in enumerate(f.binary) if r.method_name == 'equal')
    encoded = f.operation(f.unary[sign]).compose(values[0])
    torch.testing.assert_close(f.cs.sign.reverse(encoded), values[0])
    end = f.operation(f.binary[equal]).compose(encoded, values[1])
    reading = Reading(rows=torch.tensor(rows), word_rows=torch.tensor(rows), word_ids=torch.tensor(words),
        activations=torch.ones(2), leaves=torch.stack(values), concept_ids=torch.tensor(ids),
        leaf_orders=torch.tensor(orders), reference_orders=torch.tensor(orders),
        actions=torch.tensor([[0,-1,0],[2,sign,-1],[0,-1,1],[1,equal,-1]]),
        targets=torch.tensor([equal,len(f.binary)+len(f.unary),len(f.binary)+sign,len(f.binary)+len(f.unary)]),
        end_state=torch.stack((end,end*0,end*0)), lexical_forms=(form,label))
    clause=finish_reading(f.language,reading,registry=f.registry)
    resolver=owner.components.reference_close
    plan=resolver.preview(clause,context(resolver,()))
    site, = [i for i in plan['items'] if i['node'].reference_request.get('sign_leaf')]
    assert site['candidates'] == ()
    kept=resolver.prepare(clause,plan)
    written={}
    ref=store.write_clause(kept,document_key='sign-equations',sentence_index=position,
        written_rows=written,evidence=(1.,0.),text=f'{form} is {label}.')
    resolver.commit(plan,ref,written,document='sign-equations',position=position,training=False)
    return words, plan['mention_occurrences'][id(site['written_node'])]


def test_two_kept_signs_survive_a_full_pool_until_their_definition_is_read(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from Memory import MediumTermMemory
    from chunk_fixtures import context
    from test_clause_acceptance import SentenceFixture

    fixture = SentenceFixture(monkeypatch)
    owner, store = fixture.cs, fixture.store
    pool = owner.mtm = MediumTermMemory(capacity=2, n_vectors=owner.nVectors)
    value = torch.zeros(store.nDim)
    previous = [('ltm', 'old', i) for i in (1, 2)]
    for address in previous:
        pool.open(address, kind=None, when=0, value=value)
    pool.close(None, stream=0, references=previous)
    sides = []
    for slot in (0, 1):
        leaf = Composition(value, leaves=(slot,))
        sign = Composition(value, (leaf,), operation='sign', leaves=(slot,))
        sides.append(Clause(ConceptualMeaning.from_description(value), point=value, order=0,
            composition=sign, reference_request=dict(mode='mint', sign_leaf=True, leaves=(slot,),
                                                    content=value, cue=None, role=slot*2)))
    clause = Clause(ConceptualMeaning.from_description(value), children=tuple(sides),
        relation='equal', refs=(('clause',0), -1, ('clause',1)), composition=Composition(
            value, tuple(side.composition for side in sides), operation='equal'))
    resolver = owner.components.reference_close
    plan = resolver.preview(clause, context(resolver, ()), stream=0)
    kept = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(kept, written_rows=written, evidence=(1.,0.))
    resolver.commit(plan, row, written, document='retained', position=0, training=False)
    references = list(plan['mention_occurrences'].values())
    assert len(references) == 2 and set(references) == set(pool.entries)
    assert all(pool.entries[r].kind == owner.null_concept_id for r in references)
    assert store.parts.find((OCCURRENCE,references[0][-1]), (OCCURRENCE,references[1][-1])) is not None


def test_sign_names_absorbs_and_preserves_the_null_concept(monkeypatch, tmp_path):
    from test_clause_acceptance import SentenceFixture
    from test_one_backward import receipt
    f=SentenceFixture(monkeypatch)
    owner=f.cs
    alloc=owner._concept_allocator
    null=owner.null_concept_id
    before=[v.clone() for v in (owner.null_concept_row,owner.null_concept_lanes,owner.null_concept_certainty)]
    opened=[]
    open_token=owner.mtm.open
    def observe(occurrence, **kw):
        key=open_token(occurrence,**kw)
        if kw.get('kind')==null:
            opened.append((key,kw['value'].detach().clone()))
        return key
    monkeypatch.setattr(owner.mtm,'open',observe)
    (first,label),one=_sign_equation(f,'This','Felix',0)
    target,=owner.definitions.objects(label)
    (second,_),two=_sign_equation(f,'That','Felix',1)
    assert owner.definitions.objects(label)==(target,) and alloc.order_of(target)==1
    assert set(alloc.refs(target,'part'))=={('occ',one[-1]),('occ',two[-1])}
    assert owner.interpret.no_object(first) and owner.interpret.no_object(second)
    assert len(opened)==2 and opened[0][0]!=opened[1][0] and all(not value.any() for _,value in opened)
    assert all(owner.mtm.entries[owner.mtm.token_of(ref)].symbol==target for ref in (one,two))
    assert not any((SYMBOL, null) in alloc.parts.references(i) or (SYMBOL, first) in alloc.parts.references(i)
                   for i in alloc.parts.indices().tolist())
    assert null not in alloc.reference_orders and alloc.settle(null) is None
    with pytest.raises(ValueError,match='no order'):
        alloc.order_of(null)
    for name,obj in ((first,null),(null,target)):
        with pytest.raises(ValueError,match='null-concept'):
            owner.interpret.define(name,obj)
    after=(owner.null_concept_row,owner.null_concept_lanes,owner.null_concept_certainty)
    assert all(not x.any() and torch.equal(x,y) and not y.requires_grad for x,y in zip(before,after))
    receipt(tmp_path,'frozen-null-concept',dict(identity=null, form=after[0].tolist(), lanes=after[1].tolist(),
        certainty=after[2].item(), unchanged=True, occurrences=[one,two], named_identity=target))


def test_sign_question_reads_the_latest_named_token_and_no_concept_for_an_unnamed_token(monkeypatch):
    from BindingAnswers import matches, references
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    from ThoughtReferences import with_slots
    from chunk_fixtures import context
    from test_clause_acceptance import SentenceFixture
    f=SentenceFixture(monkeypatch)
    (_, label), occurrence = _sign_equation(f,'This','Felix',0)
    resolver=f.cs.components.reference_close
    # §50/§52: a sign without observed meaning cannot be associated by
    # empty containment. Supply the same observed state to both tokens;
    # the assertions below still distinguish only their naming status.
    verb = f.noun('observed-change')
    observed = torch.eye(f.store.nDim)[2]
    def observe(reference, when):
        pool = f.cs.mtm
        pool.event(reference, reference, verb=verb, role=0, when=when)
        pool.update(reference, observed, when=when)
        assert resolver.coordinates(pool.value_of(reference)).sum() > 0
    observe(occurrence, 1)
    target, = f.cs.interpret.remembered_objects(label)
    with torch.no_grad():
        f.cs.similarity_codebook.W[f.cs._csw_row_of(target)].copy_(observed)
    assert f.cs.definitions.objects(label) == (target,)
    model=SimpleNamespace(_concept_owner=lambda: f.cs, symbolSpace=SimpleNamespace(ltm_store=f.store))
    # §50 item 101 / §52 item 111: the retained nonempty occurrence is a
    # part of this query; an empty query supplies no association evidence.
    query=with_slots(ConceptualMeaning.from_description(torch.ones(f.store.nDim)),(('referent',0),))
    # Select the latest admissible bind through the ordinary chooser seam.
    # Both outcomes below use this same policy and the same question.
    monkeypatch.setattr(resolver, 'select', lambda choices, logits, *, individuals, **_:
        max(range(len(choices)), key=lambda i: individuals[choices[i]]['position']) if choices else None)
    def ask(latest):
        frames=tuple(SituationFrame.from_row(f.store.row(f.store.location(ref)))
                     for ref in dict.fromkeys((occurrence,latest)))
        decisions=[]
        result=resolver.bind_site(query,0,frames,decisions=decisions)
        assert latest in decisions[0]['candidates']
        return result
    named=ask(occurrence)
    assert named.get('reference')==occurrence and matches(model,named['meaning'],'Felix')
    assert f.cs.interpret.lexical_form(f.cs.mtm.entries[f.cs.mtm.token_of(occurrence)].symbol) == 'Felix'
    # Keep a second sign leaf with no definition on its other side.
    word=f.cs.interpret.word_for_form('This')
    ref,row,order=f.cs.interpret.reading(word)
    value=f.cs.similarity_codebook.lookup_rows(row).detach()
    leaf=Composition(value,reference=ref,word_id=word,order=order,leaves=(0,))
    site=Clause(ConceptualMeaning.from_description(value),point=value,order=0,
        composition=Composition(value,(leaf,),operation='sign',leaves=(0,)),
        reference_request=dict(mode='mint', sign_leaf=True, role=0, leaves=(0,),content=value*0,cue=None))
    plan=resolver.preview(site, context(resolver,()))
    kept=resolver.prepare(site,plan)
    written={}
    address=f.store.write_clause(kept,document_key='independent-sign',written_rows=written)
    resolver.commit(plan,address,written,document='independent-sign',position=0,training=False)
    unnamed=f.store.occurrence_of(address)
    assert f.cs.mtm.entries[f.cs.mtm.token_of(unnamed)].kind==f.cs.null_concept_id
    observe(unnamed, 2)
    from Intuition import extract, read
    from QueryWork import QueryWorkBudget
    from dataclasses import replace
    unbound = replace(ConceptualMeaning.from_description(torch.ones(f.store.nDim)),
                      role_refs=(unnamed,None,None))
    assert not read(f.cs,extract(f.cs,unbound,work=QueryWorkBudget(16))).any()
    empty=ask(unnamed)
    assert not references(empty['meaning']) and not matches(model,empty['meaning'],'Felix')


def test_ltm0_amended_row_reads_referents_and_rejects_old_checkpoints(tmp_path):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from test_one_backward import receipt
    from test_orders_learned import equality_fixture

    _, store, alloc = equality_fixture()
    identity = alloc.new_concept()
    alloc.reference_orders[identity] = 2
    idea = replace(ConceptualMeaning.from_description(store.slots.new_ones(store.nDim)),
                   role_refs=(('sym', identity), None, None))
    row = store.append_meaning(idea, evidence=(.6, .3), order=0, document_key='row-layout')
    assert store.order_of_row(row) == 2 and store.row(row)['evidence'] == pytest.approx((.6, .3))
    assert store.addressing.address_keys[row] == store.row(row)['row_id']
    assert store.when.shape == (store.capacity, 4)
    assert store.when.untyped_storage().data_ptr() == store.slots.untyped_storage().data_ptr()

    columns = dict(slots=store.slots, when=store.when, truth_plus=store.truth_plus,
                   truth_minus=store.truth_minus, access_count=store.access_count, surprise=store.surprise,
                   document_keys=store.addressing.source_documents,
                   document_authorities=store.addressing.source_authorities)
    allocations = {name: dict(shape=list(value.shape), pointer=value.data_ptr(),
                              bytes=value.numel() * value.element_size()) for name, value in columns.items()}
    store.set_origin(row, store.ORIGIN_USER, asserted=(float(store.truth_plus[row]), float(store.truth_minus[row])))
    for column in ('order', 'origin', 'polarity'):
        legacy = store.state_dict()
        legacy[column] = torch.zeros(store.capacity)
        with pytest.raises(ValueError, match='column: ' + column):
            store.load_state_dict(legacy, strict=False)
    restored = type(store)(store.nDim, capacity=store.capacity, percept_width=store.percept_width,
        stream_capacity=store.stream.shape[1], recurrence_capacity=store.recurrence_capacity,
        relation_capacity=store.parts.capacity)
    restored.load_state_dict(store.state_dict())
    assert restored.source_document(row) == store.ORIGIN_USER
    restored.load_semantic_extras(store.semantic_extras())
    assert restored.index_of_row(store.row(row)['row_id']) == row
    assert all(value.data_ptr() == allocations[name]['pointer'] for name, value in columns.items())
    receipt(tmp_path, 'ltm0-amended-allocation', dict(columns=allocations, schema=int(store.truth_schema),
        source_documents=len(restored._document_sources), recovered_order=store.order_of_row(row)))


def test_serial_concept_expectation_does_not_select_a_concept():
    from ConceptSpread import open_prediction, rule_order
    from test_orders_learned import equality_fixture
    owner,_,_=equality_fixture()
    owner._expected_knowing.fill_(1.)
    model=SimpleNamespace(_concept_owner=lambda:owner,word_brackets=True,serial_object_meta=True)
    open_prediction(model,torch.tensor([True]))
    assert model._concept_prediction is None and rule_order(owner,0)==()
    torch.testing.assert_close(owner.relevance_weights(batch=1), (1+owner._desire[:1]).clamp_min(0))


def test_leaf_decisions_are_two_level_and_sign_is_available_at_known_words():
    from ReferenceContext import ReferenceBank, prepare_operands
    from Interpret import SignLayer, InterpretLayer, leaf_readings
    from ClauseScope import ClauseScope
    x = torch.tensor([[[.25, .75], [.5, .5]]])
    ids = torch.tensor([[7, 8]])
    active = torch.ones_like(ids, dtype=torch.bool)
    interpreter = InterpretLayer()
    rules = tuple(SimpleNamespace(method_name=name, arity=1, reference_roles=(), reference_modes=())
                  for name in ('interpret', 'sign'))
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=rules)
    bank = ReferenceBank(torch.tensor([[12, 13]]), torch.tensor([[[1., 0.], [0., 1.]]]), active,
        ~active, x[:, 0], torch.tensor([False]), interpreter=interpreter,
        associations=torch.tensor([[7, 13]]), lexical_capacity=1, close_references=True)

    def menu(signable, interpretable):
        return prepare_operands(x, ids, torch.zeros_like(ids), torch.zeros_like(ids), orders=torch.zeros_like(ids),
            rules=(), unary_rules=rules, bank=bank, live=None, active=active,
            signable=signable, interpretable=interpretable)

    forced = menu(active, ~active)
    assert forced['unary_valid'][0].tolist() == [[False, True], [False, True]]
    assert leaf_readings(language, torch.tensor([[0, -1, 0], [0, -1, 1]]), 2, forced=True) == ('symbol', 'symbol')
    assert leaf_readings(language, torch.tensor([[0, -1, 0], [2, 1, -1], [0, -1, 1]]),
                         2, forced=True) == ('sign', 'symbol')
    interpreter.mandatory = False
    optional = menu(~active, active)
    assert optional['unary_ops'] == (0, 0, 1)
    assert optional['unary_valid'][0].tolist() == [[True, False, False], [False, True, False]]
    assert optional['unary_refs'][0, 0, 0, 0] == 13
    assert leaf_readings(language, torch.tensor([[0, -1, 0], [0, -1, 1]]), 2, forced=False) == ('no_op', 'no_op')
    interpreted = menu(active, ~active)
    assert interpreted['unary_valid'][0, :, -1].all()
    assert leaf_readings(language, torch.tensor([[0, -1, 0], [2, 0, -1], [2, 1, -1]]),
                         1, forced=False) == ('sign',)
    with pytest.raises(ValueError, match='interpreted leaf'):
        leaf_readings(language, torch.tensor([[0, -1, 0], [2, 1, -1]]), 1, forced=False)
    with pytest.raises(ValueError, match='uninterpreted leaf'):
        leaf_readings(language, torch.tensor([[0, -1, 0], [2, 0, -1]]), 1, forced=True)
    scope = ClauseScope((), rules)
    assert scope.unary_interpret.tolist() == [True, False]
    assert InterpretLayer.mandatory and not SignLayer.mandatory and SignLayer.invertible


def test_compose_bind_keeps_its_addressed_shifted_introduction(monkeypatch):
    """An outer bind consumes the handle made at shift, even if the lexical order reads back as zero."""
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from chunk_fixtures import context
    from reading_fixtures import record_reading
    from test_clause_acceptance import SentenceFixture

    f = SentenceFixture(monkeypatch)
    reading = record_reading(f.language, f.program(('bind', 'alias')))
    reading = replace(reading, close_references=True, leaf_orders=torch.tensor([1]),
        reference_orders=torch.tensor([0]), identity_ids=torch.tensor([4120]),
        identity_leaves=torch.tensor([1]), identity_values=reading.leaves.clone(),
        identity_origins=torch.tensor([[4, 5, 1]]),
        identity_actions=torch.tensor([[0, 0, 0, -1, -1, 0], [1, 0, 0, -1, 1, 0]]))
    close = f.cs.components.reference_close
    clause = finish_clause(f.language, reading, registry=f.registry)
    plan = close.preview(clause, context(close, ()))
    assert [site['selected'] for site in plan['items']] == [('new', 0), ('new', 0)]
    assert [site['node'].identity['introduction'] for site in plan['items']] == [True, False]
    prepared, written = close.prepare(clause, plan), {}
    address = f.store.write_clause(prepared, document_key='shift and bind', written_rows=written)
    close.commit(plan, address, written, document='shift and bind', position=0, training=False)
    first, second = (f.store.occurrence_of(written[id(site['written_node'])]) for site in plan['items'])
    assert first != second and f.cs.mtm.token_of(first) == f.cs.mtm.token_of(second) == first

    # Without the shift's journal address there is no introduction to consume.
    lost = finish_clause(f.language, replace(reading, identity_leaves=torch.tensor([0])), registry=f.registry)
    with pytest.raises(RuntimeError, match='a compose bind names an unwritten introduction'):
        close.preview(lost, context(close, ()))


def test_kept_event_retains_its_determination_and_shift(monkeypatch):
    """A wrapped determination supplies the event subject and a shifted leaf supplies its object."""
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from ClauseRow import predicate_identity
    from chunk_fixtures import context
    from reading_fixtures import record_reading
    from test_clause_acceptance import SentenceFixture

    f = SentenceFixture(monkeypatch)
    tree = ('equal', ('non', ('definite_unary', 'left')), 'right')
    reading = record_reading(f.language, f.program(tree))
    actions = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, 0, -1, -1, 0],
                            [1, 0, 0, -1, -1, 0], [0, 0, 0, 0, 0, 0],
                            [1, 1, 2, predicate_identity('equal'), -1, 1]])
    reading = replace(reading, close_references=True, identity_actions=actions,
        identity_origins=torch.tensor([[3, 1, 1], [4, 4, 1]]),
        identity_ids=torch.tensor([4120, 4121]), identity_leaves=torch.tensor([0, 2]),
        identity_values=reading.leaves.clone(), leaf_orders=torch.tensor([1, 1]),
        leaf_evidence=torch.tensor([[1., 0.], [1., 0.]]), reference_orders=torch.tensor([0, 0]))
    close = f.cs.components.reference_close
    clause = finish_clause(f.language, reading, registry=f.registry)
    plan = close.preview(clause, context(close, ()))
    introductions = {s['node'].identity['handle']: s for s in plan['items'] if s['node'].identity['introduction']}
    assert introductions[1]['node'].identity['action'] == 1
    assert introductions[2]['node'].identity['action'] is None
    prepared, written = close.prepare(clause, plan), {}
    address = f.store.write_clause(prepared, document_key='wrapped determination', written_rows=written)
    close.commit(plan, address, written, document='wrapped determination', position=0, training=False)
    roles = {}
    for handle, site in introductions.items():
        occurrence = f.store.occurrence_of(written[id(site['written_node'])])
        token = f.cs.mtm.entries[f.cs.mtm.token_of(occurrence)]
        roles[handle] = [event.role for event in token.events if event.occurrence == address]
    assert roles == {1:[0], 2:[2]}


def _optional_reading(f, tree):
    """Force the first decision through its ordinary reference menu."""
    from Understanding import Reading
    from ReferenceContext import ReferenceBank, prepare_operands
    owner = f.cs
    words, ids, rows, word_rows, values, orders, actions, frames, references = [], [], [], [], [], [], [], [], []
    def visit(node):
        if isinstance(node, str):
            word = owner.interpret.word_for_form(node)
            if word is None:
                word = owner.interpret.lookup_word([len(owner._concept_allocator.word_forms)+1], (), form=node)
                with torch.no_grad():
                    owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(torch.eye(f.store.nDim)[len(words)])
            identity, row, order = owner.interpret.reading(word)
            value = owner.similarity_codebook.lookup_rows(row).detach()
            actions.append((0, -1, len(words)))
            frames.append(torch.zeros(3, f.store.nDim))
            references.append((-1, -1))
            words.append(word); ids.append(identity); rows.append(row); values.append(value); orders.append(order)
            word_rows.append(owner._csw_row_of(word))
            return value, identity, word
        name, *children = node
        operands = [visit(child) for child in children]
        rules = f.binary if len(children) == 2 else f.unary
        local = next(i for i, rule in enumerate(rules) if rule.method_name == name)
        rule = rules[local]
        identity, word = -1, -1
        if name == 'interpret':
            value, source, word = operands[0]
            objects = owner.interpret.denotations(word)
            points = [owner.similarity_codebook.lookup_rows(owner._csw_row_of(obj)).detach() for obj in objects]
            bank = ReferenceBank(torch.tensor([objects], dtype=torch.long),
                torch.stack(points)[None] if points else value.new_zeros(1, 0, len(value)),
                torch.ones(1, len(objects), dtype=torch.bool), torch.zeros(1, len(objects), dtype=torch.bool),
                value[None], torch.tensor([False]), interpreter=owner.interpret, close_references=True,
                associations=torch.tensor([(word, obj) for obj in objects], dtype=torch.long).reshape(-1, 2),
                lexical_capacity=len(objects))
            options = prepare_operands(value[None, None], torch.tensor([[source]]), torch.tensor([[0]]),
                torch.tensor([[0]]), orders=torch.tensor([[0]]), rules=(), unary_rules=(rule,), bank=bank,
                live=None, active=torch.tensor([[True]]), signable=torch.tensor([[False]]),
                interpretable=torch.tensor([[True]]))
            selected, = options['unary_valid'][0, 0].nonzero().flatten().tolist()
            value = options['unary'][0, 0, selected]
            identity = int(options['unary_refs'][0, 0, selected, 0])
            operands = [(value, identity, word)]
        point = f.operation(rule).compose(*(value.reshape(1, 1, -1) for value, _, _ in operands)).reshape(-1)
        if name != 'interpret' and rule.head_role:
            identity, word = operands[rule.head_role-1][1:]
        actions.append((1 if len(children) == 2 else 2, local, -1))
        frames.append(torch.stack((operands[0][0], operands[1][0] if len(children) == 2 else point*0, point)))
        references.append((operands[0][1], operands[1][1] if len(children) == 2 else -1))
        return point, identity, word
    point, _, _ = visit(tree)
    stop = len(f.binary) + len(f.unary)
    targets = [stop if kind == 0 else local if kind == 1 else len(f.binary)+local
               for kind, local, _ in reversed(actions)]
    return Reading(rows=torch.tensor(rows), word_rows=torch.tensor(word_rows), word_ids=torch.tensor(words),
        concept_ids=torch.tensor(ids), activations=torch.ones(len(words)), leaves=torch.stack(values),
        actions=torch.tensor(actions), targets=torch.tensor(targets), end_state=torch.stack((point, point*0, point*0)),
        leaf_orders=torch.tensor(orders), reference_orders=torch.tensor(orders),
        operation_values=torch.stack(frames), operation_refs=torch.tensor(references),
        symbol_where=torch.arange(4, dtype=point.dtype).expand(len(words), 4).clone(),
        symbol_when=point.new_tensor([1., 0., 0., 1.]).expand(len(words), 4).clone(),
        interpret_forced=owner.interpret.mandatory)


def test_no_op_keeps_the_form_then_binds_and_acquires_a_kind(monkeypatch, tmp_path):
    from test_clause_acceptance import SentenceFixture
    from test_one_backward import receipt
    from ClauseJournal import finish_clause
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    from chunk_fixtures import context, recall
    from Interpret import leaf_readings
    from order_fixtures import force_order
    f = SentenceFixture(monkeypatch)
    owner, store = f.cs, f.store
    for form in ('The', 'dog', 'went'):
        f.noun(form)
    before = owner.null_concept_read()
    owner.interpret.mandatory = False
    reading = _optional_reading(f, ('lift', ('lower', ('interpret', 'The'), ('interpret', 'dog')),
                                   ('verb', ('interpret', 'went'), 'woof')))
    decisions = leaf_readings(f.language, reading.actions, len(reading.leaves), forced=False)
    assert decisions == ('symbol', 'symbol', 'symbol', 'no_op')
    word = owner.interpret.word_for_form('woof')
    form = owner.similarity_codebook.lookup_rows(owner._csw_row_of(word)).detach().clone()
    assert form.any() and torch.equal(reading.leaves[-1], form)
    assert owner.mtm.token_of(('name', word)) is None
    clause = finish_clause(f.language, reading, registry=f.registry)
    resolver = owner.components.reference_close
    plan = resolver.preview(clause, context(resolver, ()))
    site, = [item for item in plan['items'] if item['node'].reference_request.get('no_op_leaf')]
    assert site['candidates'] == ()
    opened = []
    original = owner.mtm.open
    def observe(occurrence, **kwargs):
        token = original(occurrence, **kwargs)
        if kwargs.get('kind') == owner.null_concept_id:
            assert torch.equal(kwargs['value'], form) and kwargs['value'].any()
            opened.append(token)
        return token
    monkeypatch.setattr(owner.mtm, 'open', observe)
    kept = resolver.prepare(clause, plan)
    written = {}
    address = store.write_clause(kept, document_key='no-op-reading', sentence_index=0,
        text='The dog went woof.', evidence=(1., 0.), written_rows=written)
    resolver.commit(plan, address, written, document='no-op-reading', position=0, training=False,
                    reading=reading)
    occurrence = plan['mention_occurrences'][id(site['written_node'])]
    token, = opened
    assert owner.mtm.token_of(occurrence) == token
    assert owner.mtm.entries[token].kind == owner.null_concept_id
    assert torch.equal(owner.mtm.entries[token].value, form)
    assert not owner.mtm.promote_candidates(owner)
    # The no-op has no inverse operation. Read-back is its retained form.
    assert decisions[-1] == 'no_op'
    assert torch.equal(site['node'].composition.point, reading.leaves[-1])
    bark_word = owner.interpret.lookup_word([92], (), form='bark')
    bark = owner.new_concept()
    force_order(owner, bark, 2)
    row = owner._csw_concept_row(2, bark)
    with torch.no_grad():
        owner.similarity_codebook.W[row].copy_(torch.eye(store.nDim)[4])
    owner.interpret.define(bark_word, bark)
    predicate = store._concept_point(bark)
    that = owner.interpret.lookup_word([91], (), form='That')
    term = Composition(form, reference=that, word_id=that, order=1, leaves=(0,))
    mention = Clause(ConceptualMeaning.from_description(form), point=form, order=1,
        subject_word_id=that, composition=term,
        reference_request=dict(mode='bind', leaves=(0,), cue=form, predicate=predicate, role=0))
    complement = Composition(predicate, reference=bark, order=2)
    clause = Clause(ConceptualMeaning.from_description(predicate), relation='equal',
        refs=(('clause', 0), -1, bark), children=(mention,), evidence=(1., 0.),
        composition=Composition(predicate, (term, complement), operation='equal'))
    monkeypatch.setattr(resolver, 'select', lambda choices, logits, **kwargs: choices.index(occurrence))
    plan = resolver.preview(clause, recall(resolver, (SituationFrame.from_row(store.row(store.location(occurrence))),)))
    item, = plan['items']
    assert occurrence in item['candidates'] and item['selected'] == occurrence
    kept = resolver.prepare(clause, plan)
    written = {}
    address = store.write_clause(kept, document_key='no-op-reading', sentence_index=1,
        text='That was a bark.', evidence=(1., 0.), written_rows=written)
    resolver.commit(plan, address, written, document='no-op-reading', position=1, training=False)
    later = plan['mention_occurrences'][id(item['written_node'])]
    assert owner.mtm.token_of(later) == token and owner.mtm.entries[token].kind == bark
    assert owner.interpret.no_object(word) and owner.definitions.objects(word) == ()
    alloc = owner._concept_allocator
    assert not any((SYMBOL, word) in alloc.parts.references(row) for row in alloc.parts.indices().tolist())
    assert before == owner.null_concept_read()
    receipt(tmp_path, 'no-op-form-and-kind', dict(forced=False, decisions=decisions,
        form=form.tolist(), token=token, occurrence=occurrence, bound=later, kind=bark,
        word=word, word_objects=owner.definitions.objects(word), null_unchanged=True))


def test_forced_symbol_reads_the_existing_object(monkeypatch, tmp_path):
    from test_clause_acceptance import SentenceFixture
    from test_one_backward import receipt
    from Interpret import leaf_readings
    from ClauseJournal import finish_clause
    from chunk_fixtures import context
    f = SentenceFixture(monkeypatch)
    obj = f.noun('woof')
    word = f.words['woof'][0]
    assert f.cs.interpret.mandatory
    reading = _optional_reading(f, 'woof')
    assert leaf_readings(f.language, reading.actions, 1, forced=True) == ('symbol',)
    assert int(reading.concept_ids[0]) == obj
    assert torch.equal(reading.leaves[0], f.store._concept_point(obj))
    clause = finish_clause(f.language, reading, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, context(resolver, ()))
    assert not any(item['node'].reference_request.get('no_op_leaf') for item in plan['items'])
    receipt(tmp_path, 'forced-symbol-control', dict(forced=True, word=word, object=obj,
        decisions=['symbol'], row=int(reading.rows[0])))


def test_compose_leaf_decisions_match_eager_and_compiled(tmp_path):
    from Memory import STMChunks as chunks
    from ChunkJournal import ChunkJournal, empty_scope, SOURCE, IDENTITY, WORD, LEAVES
    from ClauseScope import ClauseScope
    from Interpret import InterpretLayer, SignLayer
    from Language import OperationSelectionLayer, LanguageSpace, _UnaryGrammarOpAdapter
    from ReferenceContext import ReferenceBank
    from operation_fixtures import selected_action
    from test_one_backward import receipt
    interpreter, sign = InterpretLayer(), SignLayer()
    rules = tuple(SimpleNamespace(method_name=name, arity=1, reference_roles=(), reference_modes=())
                  for name in ('interpret', 'sign'))
    layer = OperationSelectionLayer(d_model=2, ops=(),
        unary_ops=(_UnaryGrammarOpAdapter(interpreter), _UnaryGrammarOpAdapter(sign)))
    form, obj = torch.tensor([[.25, .75]]), torch.tensor([[.8, .2]])
    bank = ReferenceBank(torch.tensor([[13]]), obj[:, None], torch.tensor([[True]]),
        torch.tensor([[False]]), form, torch.tensor([False]), interpreter=interpreter,
        associations=torch.tensor([[7, 13]]), orders=torch.tensor([[1]]), lexical_capacity=1,
        close_references=True)
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
        _tree_layer=lambda arity: layer, _compose_binary_rules=(), _compose_unary_rules=rules,
        _structural_context=lambda **kwargs: None, _reference_bank=bank,
        chunk_journal=ChunkJournal((), rules))
    scope_owner = ClauseScope((), rules)
    marker = scope_owner.unary_interpret
    allocation = dict(shape=list(marker.shape), pointer=marker.data_ptr(),
                      bytes=marker.numel()*marker.element_size(), scope_width=14)

    def leaf(value, identity, order):
        state = chunks.empty(1, 4, 2)
        state, _ = chunks.deposit(state, value, torch.tensor([True]), identity=identity, order=order, stack=True)
        scope = empty_scope(state[0])
        scope[0, 0, 1] = identity
        scope[0, 0, SOURCE] = -1
        scope[0, 0, IDENTITY] = scope[0, 0, WORD] = identity
        scope[0, 0, LEAVES] = 1
        return state, scope

    def choose(state, scope):
        return LanguageSpace.choose_operation(owner, state, torch.tensor([True]), slots=1, reference_scope=scope)

    records = []
    for forced, action, kind in ((True, 1, 2), (True, 4, 0), (False, 6, 0), (False, 0, 2)):
        interpreter.mandatory = forced
        state, scope = leaf(obj if forced else form, 13 if forced else 7, 1 if forced else 0)
        with selected_action(layer, torch.tensor([action])):
            eager = choose(state, scope)
            compiled = torch.compile(choose, backend='eager', fullgraph=True)(state, scope)
        for result in (eager, compiled):
            choice, refs, relations, operands = result
            assert choice.kind.item() == kind and choice.valid.item()
            if not forced and kind == 2:
                assert refs[0, 0] == 13 and choice.order.item() == 1
                assert torch.equal(choice.candidate, obj)
                updated, _ = scope_owner.apply(scope, choice, 0)
                assert updated[0, 0, 0].bitwise_and(ClauseScope.INTERPRETED_LEAF)
            if forced and kind == 2:
                assert choice.order.item() == 0 and choice.local_op.item() == 1
        torch.testing.assert_close(eager[0].candidate, compiled[0].candidate, rtol=0, atol=0)
        records.append(dict(forced=forced, selected='sign' if forced and kind else
                            'symbol' if forced else 'interpret' if kind else 'no_op',
                            eager_compiled_equal=True))
    assert marker.data_ptr() == allocation['pointer'] and list(marker.shape) == allocation['shape']
    assert scope.shape[-1] == allocation['scope_width']
    receipt(tmp_path, 'two-level-compiled-choices', dict(choices=records, allocation=allocation))


def test_optional_interpretation_keeps_the_pending_definition_path(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from ClauseJournal import finish_clause
    from chunk_fixtures import context
    from order_fixtures import force_order
    f = SentenceFixture(monkeypatch)
    force_order(f.cs, f.noun('kind'), 1)
    f.cs.interpret.mandatory = False
    reading = _optional_reading(f, ('equal', ('interpret', 'Name'), ('interpret', 'kind')))
    clause = finish_clause(f.language, reading, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    written = {}
    address = f.store.write_clause(kept, document_key='optional-symbol', evidence=(1., 0.), written_rows=written)
    resolver.commit(plan, address, written, document='optional-symbol', position=0, training=False, reading=reading)
    word = f.cs.interpret.word_for_form('Name')
    identity, = f.cs.interpret.denotations(word)
    assert f.cs._concept_allocator.order_of(identity) == 1
    assert not f.cs.interpret.no_object(word)
