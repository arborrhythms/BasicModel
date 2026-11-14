"""Free variables, stored columns, exhaustion and formation provenance (§14)."""
from Memory import STM0
from order_fixtures import force_order
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, fill, has_open_slots, open_slots, with_slots
from meaning_fixtures import unsupported as question


def test_ignorance_is_not_a_free_variable_or_an_answer_cost_gate():
    from BindingAnswers import cost, matches
    atom = replace(ConceptualMeaning.from_description(torch.ones(4)),
                   role_refs=(('sym', 5), None, None))
    region = question(atom)
    assert not open_slots(region) and not has_open_slots(region)
    assert bindings(region)['_open_references'] == ()
    data = bindings(region)
    data['_bound_roles'] = (('referent', 0),)
    region = replace(region, bindings=data)
    model = SimpleNamespace(_concept_owner=lambda:SimpleNamespace(word_concepts=lambda _: (5,)),
                            symbolSpace=SimpleNamespace(ltm_store=None))
    assert evidence_pair(region) == (0., 0.)
    assert cost(model, region, 'five') == 0 and matches(model, region, 'five')
    assert cost(model, replace(region, role_refs=(None,None,None)), 'five') == 1


def test_formation_provenance_is_not_a_chooser_input():
    from ThoughtFeatures import semantic_metadata
    base = ConceptualMeaning.from_description(torch.ones(4))
    records = [replace(base, bindings={'_formation_records':
        ({'role':0,'choice':choice,'probability':probability,'reference':reference},),
        '_formation_reason':'search_exhausted'})
        for choice,probability,reference in (('open',.01,0),('mint',.9,-1),('bind',.25,12345))]
    for expected, actual in zip(semantic_metadata((base,)*3),semantic_metadata(records)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_copula_fills_the_addressed_open_column_without_a_word_alias():
    from Memory import LTM0
    from Occurrence import slot_key
    from ThoughtStream import write
    from part_fixtures import native_store
    store, _ = native_store(4, capacity=32)
    points = {5:torch.tensor([.1,.2,.3,.4]), 7:torch.tensor([.4,.3,.2,.1])}
    store.configure_clause_index(concept_point=points.get)
    root = ConceptualMeaning(torch.ones(3,4),torch.ones(3,dtype=torch.bool),
        role_refs=(None,('sym',7),('sym',7)), sentence_kind='relation')
    root = question(root, (('referent',0),))
    pending = replace(ConceptualMeaning.from_description(root.roles[0]), bindings=bindings(root))
    row = store.append_meaning(pending,  evidence=(0.,0.),
                               )
    address, position = store.occurrence_of(row), store.row_name(row)
    column = slot_key(store.row_name(row), 0)
    assert store.slot_of_reference(column) == (row,0)
    assert store.semantic_reference(column) == ('slot',address,0)
    original = store.slots[row].clone(), store.raw_meaning_of(row)
    # The copular reader's resolved column is a working conclusion. Persist
    # its close at the addressed question boundary; never upsert the source.
    answered = fill(replace(root, bindings=dict(bindings(root), _query_occurrence=address)),
        dict(reference=('sym',5), binding_value=points[5], support_true=1.,
             support_false=0.), slots=(('referent',0),))
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
    conclusion = write(model, answered, row=0)
    assert conclusion != row
    resolved = store.meaning_of(conclusion)
    assert not open_slots(resolved)
    assert resolved.role_refs[0] == ('sym',5)
    assert bindings(resolved)['_bound_roles'] == (('referent',0),)
    assert not bindings(resolved).get('_pending', False) and not has_open_slots(resolved)
    assert store.occurrence_of(row) == address and store.row_name(row) == position
    torch.testing.assert_close(store.slots[row], original[0], rtol=0, atol=0)
    assert open_slots(store.meaning_of(row)) == open_slots(original[1])
    assert open_slots(store.raw_meaning_of(row)) == open_slots(original[1])
    # Rebuilding the derived posting (as on checkpoint load) retains its ID.
    store.rebuild_caches()
    assert store.slot_of_reference(column) == (row,0)


@pytest.mark.parametrize('supported', [True, False])
def test_one_empty_region_query_permits_conclude_and_only_source_mints(monkeypatch, supported):
    from test_item6_2_thinking import world, run
    model, registry, store, (a,b,c) = world()
    root = registry.form('isPart', a, open_roles=('argin2',))
    root = with_slots(replace(root,
        bindings={'_formation_records': ({'role':2, 'choice':'open', 'probability':.25},),
                  '_source_evidence':(1.,0.) if supported else (0.,0.)}),
        (('referent',2),), pair=(1.,0.) if supported else (0.,0.))
    chosen=[]
    def choose(_root,_active,actions,**_kwargs):
        if None in actions:
            chosen.append('conclude')
            return None
        action = next(action for action in actions if action.semantic_id == 'query')
        chosen.append('query')
        return action
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    model.thought_search_exhaustion = 1
    result = run(model, root, work_budget=32)
    assert chosen == ['query', 'conclude']
    assert bool(open_slots(result.meaning)) is not supported
    assert len(store.parts) == 1
    retained = store.part_row(int(store.parts.indices()[-1]))
    assert bool(open_slots(retained['meaning'])) is not supported
    assert retained['evidence'] == ((1., 0.) if supported else (0., 0.))
    if supported:
        child = result.meaning.constituents[-1]
        assert bindings(child)['_formation_reason'] == 'search_exhausted'
        assert dict(bindings(child)['_formation_records'][0]) == dict(role=2,choice='open',probability=.25)
        assert bindings(result.meaning)['_bound_roles'] == (('referent',2),)
        from BindingAnswers import cost
        owner = SimpleNamespace(word_concepts=lambda _: (9,), _csw_row_of=lambda _:0,
            similarity_codebook=SimpleNamespace(lookup_rows=lambda _,**kw:torch.ones(1,root.roles.shape[-1])))
        scored = SimpleNamespace(_concept_owner=lambda:owner,symbolSpace=SimpleNamespace(ltm_store=store))
        # The minted empty region has no content meeting the target.
        assert cost(scored,result.meaning,'unrelated') == 1.


def test_binding_answer_cost_scores_filled_referent_while_relation_is_open():
    from BindingAnswers import cost, matches
    value = question(ConceptualMeaning(torch.ones(3,4),torch.ones(3,dtype=torch.bool),
        role_refs=(None,None,('sym',7)), sentence_kind='relation'))
    value = fill(value, dict(reference=('sym',5), value=torch.ones(4)), slots=(('referent',0),))
    model = SimpleNamespace(_concept_owner=lambda:SimpleNamespace(word_concepts=lambda _: (5,)),
                            symbolSpace=SimpleNamespace(ltm_store=None))
    assert open_slots(value) == (('relation',1),)
    assert cost(model,value,'five') == 0
    assert not matches(model,value,'five')  # the frozen committed-row verifier is stricter


def test_episode_work_is_reported_without_a_local_thought_judge(monkeypatch):
    from test_item6_2_thinking import credit_chain_world, run
    model, registry, store, goal, menu = credit_chain_world(monkeypatch)
    result = run(model, goal, work_budget=64)
    assert 0 < result.work.spent <= 64
    assert result.records[-1].kind == 'finish'
    assert model._last_thought_comparison is None


def test_region_query_ignores_free_variable_code_and_preserves_a_pending_match():
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from part_fixtures import native_store
    from ClauseRow import Clause, predicate_identity
    store, ids = native_store(width=8, points=torch.eye(8)[:3])
    source = ConceptualMeaning(torch.eye(8)[:3], torch.ones(3,dtype=torch.bool),
        role_refs=(None, ('sym', predicate_identity('equal')), ('sym', ids[2])), sentence_kind='relation')
    pending = with_slots(ConceptualMeaning.from_description(source.roles[0]), (('referent',0),), pair=(1.,0.))
    child = Clause(pending, point=source.roles[0])
    row = store.write_clause(Clause(source, relation='equal', children=(child,),
        refs=(('clause', 0), predicate_identity('equal'), ids[2])), evidence=(1.,0.))
    source = store.raw_meaning_of(row)
    query = question(replace(source,roles=source.roles.clone()))
    query.roles[0].fill_(-100.)
    reader = ThoughtLTMCapability(store=lambda:store,equal=lambda *_:None,tau_id=.99)
    result = reader.best_match(query,max_records=16,work=QueryWorkBudget(128))
    assert result['frames'][0]['occurrence'] == store.occurrence_of(row)
    assert open_slots(result['frames'][0]['meaning']) == (('referent',0),)
    own = bindings(query)
    own['_query_occurrence'] = store.occurrence_of(row)
    empty = reader.best_match(replace(query,bindings=own),max_records=16,work=QueryWorkBudget(128))
    assert not empty['frames'] and not empty['incomplete']
    # A fully free region is a query mask, not an invalid zero-role meaning.
    free = question(replace(source,role_refs=(None,None,None)),
        (('referent', 0), ('relation', 1), ('referent', 2)))
    empty = reader.best_match(free,max_records=16,work=QueryWorkBudget(128))
    assert not empty['frames'] and not empty['incomplete']
    assert free.role_mask.tolist() == [True,True,True]


def test_ordinary_initial_binding_distribution_includes_every_retained_candidate(tmp_path, eager_reading, monkeypatch):
    """Certificate b: own content neutrality; form and expectation are exempt."""
    import json
    from pathlib import Path
    from util import XMLConfig
    from binding_distribution_probe import binding_distributions, retained_bank_distributions
    from math_chain_ordinary import ordinary_model, train_documents, stage, observer_class
    from math_chain_corpus import ChainDocument
    from MathChainTraining import present
    parity = json.loads((Path(__file__).parent/'fixtures/item4-parity.json').read_text())
    parse = XMLConfig._parse_xml
    def configured(path):
        result = parse(path)
        result.setdefault('architecture', {}).setdefault('training', {}).update(parity['training'])
        return result
    monkeypatch.setattr(XMLConfig, '_parse_xml', staticmethod(configured))
    model = ordinary_model(tmp_path)
    try:
        assert not model.expectation_prime_enabled
        # Existing candidates require an existing situation. Read real
        # documents without an optimizer, then prove the chooser is still
        # at initialization before the training certificate starts.
        chooser = model._selected_thought_chooser(None)
        initial = {name: value.detach().clone() for name, value in chooser.named_parameters()}
        compose = model.languageSpace.language_layer.operation_layer.chooser
        compose_initial = {name: value.detach().clone() for name, value in compose.named_parameters()}
        retained_records = retained_bank_distributions(model)
        provision = tmp_path/'provision'
        provision.mkdir()
        # Cover small and larger retained menus with different real
        # documents, not repeated initializations or a selected RNG state.
        # Each history length has eight document streams, and every
        # named variable is queried at each history length.
        names, values = ('w', 'x', 'y', 'z'), ('one', 'two', 'three', 'four')
        provision_documents = []
        for count in range(1, 5):
            for stream in range(8):
                offset = stream % len(names)
                premises = tuple(f'{names[(offset+i) % 4]} is {values[i]}.' for i in range(count))
                provision_documents.append(ChainDocument(f'prior:{count}:{stream}',
                    (*premises, f'what is {names[offset]} ?'), question=count))
        stage(model, provision_documents)
        with binding_distributions() as prior_records, observer_class()(provision) as observer:
            observer.context = dict(epoch=0, phase='initialization_read')
            present(model, split='train', after_batch=observer.after_batch)
        assert all(torch.equal(value, initial[name]) for name, value in chooser.named_parameters())
        assert all(torch.equal(value, compose_initial[name]) for name, value in compose.named_parameters())
        documents = [ChainDocument(str(i), ('what is y ?',), question=0) for i in range(8)]
        with binding_distributions() as records:
            rows = train_documents(model, documents, tmp_path)
        # Both contexts use the identical untrained chooser; the first has
        # backward candidates within its real multi-sentence documents.
        records = retained_records + prior_records + records
        assert records
        ratios = [p*len(item['alternatives']) for item in records for p in item['probabilities']]
        retained = sum(sum(value not in (0,-2) for value in item['alternatives']) for item in records)
        report = dict(forced=False, configuration='test/fixtures/item4-parity.json',
            measured_term='chooser_content', held_out_terms=['form', 'expectation'],
            conditional_band=[.9,1.1], alternatives_relative_to_uniform=[min(ratios),max(ratios)],
            initialization_documents=len(provision_documents), retained_premise_counts=[1,2,3,4],
            controlled_retained_menus=len(retained_records),
            initialization_read_menus=len(prior_records),
            training_menus=len(records)-len(prior_records)-len(retained_records),
            retained_candidates=retained, menus=len(records), questions=len(rows),
            observed_committed_open=sum(bool(row['open']) for row in rows),
            observed_episodes=sum(row['episode'] for row in rows),
            departures=[row['departure'] for row in rows])
        (tmp_path/'distribution.json').write_text(json.dumps(report, indent=2)+'\n')
        (tmp_path/'menus.json').write_text(json.dumps(records)+'\n')
        assert retained, 'the full-candidate band must exercise retained candidates'
        assert min(ratios) >= .9 and max(ratios) <= 1.1, report
    finally:
        model.End()


@pytest.mark.phase("copular_identity")
def test_forced_ordinary_answer_fills_committed_question_without_its_own_episode(tmp_path, eager_reading, restore_phase):
    """Certificate c, FORCED operation/binding; actual paired suffix and costs."""
    import json
    from BindingAnswers import matches
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    model = ordinary_model(tmp_path)
    questions, report = [], []
    documents = [ChainDocument(str(i), ('what is y ?', 'the answer is five.'),
                              question=0, answer='five') for i in range(8)]
    def after(model, split, rows, result, observed, observer):
        store = model.symbolSpace.ltm_store
        if not questions:
            for field in model._sentence_fields[0]:
                index = store.index_of_row(field.row_id)
                assert ('referent',0) in open_slots(store.meaning_of(index))
                questions.append((index,store.occurrence_of(index)))
        else:
            assert not model._last_closing_thoughts
            for index, address in questions:
                value = store.meaning_of(index)
                report.append(dict(occurrence=address, open=open_slots(value),
                                   correct=matches(model,value,'five')))
                assert store.occurrence_of(index) == address
                assert not open_slots(value) and matches(model,value,'five')
    try:
        restore_phase(lambda path: model.load_weights(str(path), require_match=True))
        with ForcedGrammar(model, open_names=('what',)) as forced:
            rows = train_documents(model, documents, tmp_path, after=after)
        assert len(questions) == 8 and len(report) == 8
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True, rows=report,
            departures=[row['departure'] for row in rows], choices=forced.records),indent=2)+'\n')
        assert len({row['departure'] for row in rows if row['departure'] >= 0}) > 1
    finally:
        model.End()


def test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage(tmp_path, monkeypatch, eager_reading):
    """g is a structural query/conclude exercise on the forced c/e grammars."""
    import json
    import ThoughtStream
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    model = ordinary_model(tmp_path)
    # This forced mechanism fixture supplies its vocabulary, as the pending-
    # premise certificate does. Reading now takes DEF objects; putting the
    # fixture's orders on forms would leave every site a new pending token.
    from test_definition_rows import admit
    owner = model._concept_owner()
    for word in ('x', 'what', 'y', 'is', 'plus', 'two'):
        _, obj = admit(model, word)
        force_order(owner, obj, 1 if word in ('x', 'what') else 0)
    from dataclasses import replace
    from Queries import ThoughtLTMCapability
    content_query = ThoughtLTMCapability.query
    empty_scopes = []
    def empty_query_scope(capability, pattern, **options):
        # This exercise supplies an empty eligible scope, including the
        # vocabulary's real definition rows. The ordinary retrieval still
        # scans and charges its work; only the request's scope is supplied.
        store = model.symbolSpace.ltm_store
        excluded = tuple(store.occurrence_of(index) for index in range(len(store)))
        excluded += tuple(('sym', index) for index in store.relations().tolist())
        data = dict(bindings(pattern), _query_exclude=excluded)
        data.pop('_query_event', None)
        result = content_query(capability, replace(pattern, bindings=data), **options)
        assert not result['witnesses']
        empty_scopes.append(dict(excluded=len(excluded), records_scanned=result['records_scanned']))
        return result
    monkeypatch.setattr(ThoughtLTMCapability, 'query', empty_query_scope)
    chooser = model._selected_thought_chooser(None)
    logits = chooser.thought_logits
    offered = []
    def query_then_conclude(active, requests):
        value = logits(active, requests)
        def operation(request):
            if request is None:
                return 'conclude'
            signature = model.grammatical_thoughts.signature_for(request, verify_reference=False)
            return None if signature is None else signature.operation.semantic_id
        offered.append(tuple(operation(candidate) for candidate in requests))
        # The fixture supplies an operation, not a logit preference that a
        # sampled departure can replace. Conclude becomes available only
        # after the real search rule permits it; all closing assertions stay.
        chosen = 'conclude' if None in requests else 'query'
        allowed = torch.tensor([operation(candidate) == chosen for candidate in requests],
                               device=value.device)
        assert allowed.any()
        return value.masked_fill(~allowed, -torch.inf)
    monkeypatch.setattr(chooser, 'thought_logits', query_then_conclude)
    commits, trials, examples = [], {}, []
    commit = ThoughtStream.commit
    def committed_episode(*args, **kwargs):
        index = commit(*args, **kwargs)
        commits.append(index)
        result = args[1]
        trial = trials.get(id(result.records))
        if trial is not None:
            trial.update(committed=index, occurrence=bindings(result.meaning).get('_query_occurrence'))
        return index
    monkeypatch.setattr(ThoughtStream, 'commit', committed_episode)
    native_close = model._commit_sentence
    def kept_close(*args, **kwargs):
        value = native_close(*args, **kwargs)
        for row, result in model._last_closing_thoughts:
            examples.append(trials[id(result.records)])
        return value
    monkeypatch.setattr(model, '_commit_sentence', kept_close)
    def episode(original, model, meaning, **kwargs):
        before_commits = len(commits)
        before_offered = len(offered)
        result = original(model, meaning, **kwargs)
        if not result.records:
            # The native compose comparison can keep an already closed
            # field. Conclude at that opening creates no episode or commit.
            assert not open_slots(meaning) and result.meaning is meaning
            assert len(commits) == before_commits and result.work.spent == 0
            trials[id(result.records)] = dict(already_closed=True, before=(), after=(),
                source_pair=bindings(meaning).get('_source_evidence'),
                operations=(), history=(), work=result.work.spent)
            return result
        assert len(commits) == before_commits  # Every trial ends before the kept write.
        operations = [record.operation for record in result.records if record.kind == 'thought']
        trials[id(result.records)] = dict(already_closed=False, source_pair=bindings(meaning).get('_source_evidence'),
            occurrence=None, committed=None,
            operations=operations, offered_operations=sorted({op for menu in offered[before_offered:] for op in menu if op is not None}), work=result.work.spent, allowance=model.attention_budget,
            history=[record.kind for record in result.records],
            before=open_slots(meaning), after=open_slots(result.meaning),
            minted=any(bindings(child).get('_formation_reason') == 'search_exhausted'
                       for child in result.meaning.constituents))
        return result
    try:
        documents = [ChainDocument('declarative', ('y is x plus two.',)),
                     ChainDocument('question', ('what is y ?',), question=0)]
        with ForcedGrammar(model, open_names=('what','x')) as forced:
            rows = train_documents(model, documents, tmp_path, episode=episode)
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True,
            forcing='c/e operation and binding choices; structural query then conclude with empty eligible query scope',
            empty_query_scopes=empty_scopes,
            examples=examples, rows=rows, choices=forced.records),indent=2)+'\n')
        assert len(examples) == 2
        store = model.symbolSpace.ltm_store
        for item in examples:
            if item['already_closed']:
                assert not item['before'] and not item['after'] and not item['history']
                continue
            # This structural exercise selects query before conclude; a
            # sampled preference alone cannot guarantee an empty search.
            assert 'query' in item['offered_operations']
            assert 'query' in item['operations']
            assert item['history'][-1] == 'finish'
            assert item['work'] <= item['allowance']
            index = store.location(item['occurrence'])
            # The close persists the working conclusion at its own address.
            # The original occurrence retains its open reference.
            committed = store.meaning_of(item['committed'])
            assert open_slots(store.raw_meaning_of(index))
            if 'cutoff' in item['history']:
                # A live departure may spend the whole allowance before a
                # complete empty search. Budget exhaustion cannot authorize
                # minting, even when the source supplied evidence.
                assert item['work'] == item['allowance']
                assert item['after'] == item['before'] and open_slots(committed)
                assert not item['minted'] and item['committed'] == item['occurrence']
            elif tuple(item['source_pair']) == (0.,0.):
                assert item['after'] and open_slots(committed)
                assert open_slots(store.raw_meaning_of(index))
                assert not item['minted']
            else:
                assert item['minted'] and not item['after'] and not open_slots(committed)
                assert item['committed'] != index
    finally:
        model.End()


class UnsupportedReversePremiseBinding(AssertionError):
    """The reverse-order bind consumed a kind with no supplied antecedent."""


@pytest.mark.parametrize('reverse', [False, True])
def test_forced_ordinary_pending_premise_in_both_orders(tmp_path, reverse, eager_reading, request):
    """Certificate e: live bind choices may close correctly or remain pending."""
    import json
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    from BindingAnswers import matches
    model = ordinary_model(tmp_path, budget=0)
    owner = model._concept_owner()
    # The forced premise fixture needs an order-1 referent. Reading now
    # interprets every word, so provision its DEF object instead of putting
    # order 1 on the lexical form and implicitly raising it again.
    from test_definition_rows import admit
    _, obj = admit(model, 'x')
    force_order(owner, obj, 1)
    # Other words were order-0 forms in this forced grammar fixture. Give
    # them their order-0 objects so D6 preserves that supplied vocabulary.
    for word in ('y', 'is', 'plus', 'two', 'three'):
        admit(model, word)
    sentences = ('y is x plus two.', 'x is three.')
    if reverse:
        sentences = sentences[::-1]
    documents = [ChainDocument(str(i), sentences) for i in range(8)]
    pending, report = [], []
    batches = 0

    def check(store, index, names):
        # Classify the input to the working read, before asking it to resolve
        # the slot. The ordinary live chooser does not visit this branch on
        # every run. A correct read on this input is a strict XPASS.
        raw = store.raw_meaning_of(index)
        site = store.occurrence_of(index)
        entry = owner.mtm.entries.get(site) if reverse else None
        unsupported_arrival = False
        from Memory import NULL_CONCEPT
        raw_ref = raw.role_refs[0]
        open_word = raw_ref is not None and raw_ref[0] == 'sym' and owner.interpret.no_object(raw_ref[1])
        if (entry is not None and not entry.pending and entry.native is None and entry.symbol is None
                and entry.kind is not None and entry.kind != NULL_CONCEPT and entry.kind not in names
                and not open_word and any(event.role == 0 for event in entry.events)
                and any(name in names for _, name in bindings(raw).get('_forward_references', ()))):
            kind = replace(ConceptualMeaning.from_description(raw.roles[0]),
                           role_refs=(('sym', entry.kind), None, None),
                           bindings={'_bound_roles': (('referent', 0),)})
            unsupported_arrival = not matches(model, kind, 'three')
            if unsupported_arrival:
                request.node.add_marker(pytest.mark.xfail(strict=True, raises=UnsupportedReversePremiseBinding,
                    reason='The reverse premise can select a fresh MTM kind that does not supply three and close x to that kind instead of preserving its open address.'))
                (tmp_path/'unsupported-arrival.json').write_text(json.dumps(
                    dict(site=site, kind=entry.kind, events=[e.occurrence for e in entry.events],
                         target=owner.word_concepts('three'))) + '\n')
        value = store.meaning_of(index)
        correct = matches(model, value, 'three')
        slots = open_slots(value)
        forwards = bindings(value).get('_forward_references', ())
        report.append(dict(occurrence=store.occurrence_of(index), open=slots,
            refs=value.role_refs, bound_correct=correct,
            pending_x=any(identity in names for _, identity in forwards)))
        if correct:
            assert not slots and not bindings(value).get('_pending')
        else:
            if not any(identity in names and ('referent', role) in slots for role, identity in forwards):
                message = 'x must bind to three or retain its named open referent'
                if unsupported_arrival:
                    raise UnsupportedReversePremiseBinding(message)
                raise AssertionError(message)
            assert not any(kind == 'referent' and role in {r for r, name in
                bindings(store.raw_meaning_of(index)).get('_forward_references', ()) if name in names}
                for kind, role in bindings(value).get('_bound_roles', ())), 'x bound to a wrong antecedent'

    def after(model, split, rows, result, observed, observer):
        nonlocal batches
        batches += 1
        store = model.symbolSpace.ltm_store
        names = set(model._concept_owner().word_concepts('x'))
        if batches == 1 and not reverse:
            for index in range(len(store)):
                value = store.meaning_of(index)
                forwards = () if value is None else bindings(value).get('_forward_references', ())
                if forwards and all(identity in names for _, identity in forwards):
                    pending.append((index, store.occurrence_of(index)))
            assert len(pending) >= 2, 'the first premise must persist pending x constituents in the batch'
        if batches == 2:
            for index, address in pending:
                assert store.occurrence_of(index) == address
                check(store, index, names)
            if reverse:
                for index in range(len(store)):
                    raw = bindings(store.raw_meaning_of(index)).get('_forward_references', ())
                    read = bindings(store.meaning_of(index)).get('_forward_references', ())
                    if any(identity in names for _, identity in (*raw, *read)):
                        check(store, index, names)
    try:
        with ForcedGrammar(model, open_names=('x',), live_references=('x',)) as forced:
            rows = train_documents(model, documents, tmp_path, after=after)
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True, reverse=reverse,
            closure_forced=False, attention_budget=0,
            ending='correct binding or named open referent',
            bound=sum(item['bound_correct'] for item in report),
            open=sum(not item['bound_correct'] for item in report),
            resolved=report, rows=rows, choices=forced.records), indent=2)+'\n')
        assert len({row['departure'] for row in rows if row['departure'] >= 0}) > 1
    finally:
        model.End()


@pytest.mark.phase('pending_premise_learning')
@pytest.mark.parametrize('reverse', [False, True])
def test_live_pending_premise_closes_in_both_orders(tmp_path, reverse, eager_reading, restore_phase):
    """Item 0 learning evidence: live closure with the strict arrival assertions."""
    import json
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    from BindingAnswers import matches
    # An exhausted search now legitimately mints a declarative's forward
    # referent. Exercise the other specified ending (budget exhaustion),
    # leaving the pending column for the next sentence's address upsert.
    model = ordinary_model(tmp_path, budget=0)
    restore_phase(lambda path: model.load_weights(str(path), require_match=True))
    sentences = ('y is x plus two.', 'x is three.')
    if reverse:
        sentences = sentences[::-1]
    documents = [ChainDocument(str(i), sentences) for i in range(8)]
    pending, report = [], []
    batches = 0
    def after(model, split, rows, result, observed, observer):
        nonlocal batches
        batches += 1
        store = model.symbolSpace.ltm_store
        if batches == 1 and not reverse:
            names = set(model._concept_owner().word_concepts('x'))
            for index in range(len(store)):
                value = store.meaning_of(index)
                forwards = () if value is None else bindings(value).get('_forward_references', ())
                # A live departure may also open a different name. This
                # certificate follows the actual pending x columns; arrival
                # of x must not fill an unrelated y or numeral variable.
                if forwards and all(identity in names for _, identity in forwards):
                    pending.append((index,store.occurrence_of(index)))
            assert len(pending) >= 2, 'the first premise must persist pending x constituents in the batch'
        if batches == 2:
            for index, address in pending:
                value = store.meaning_of(index)
                correct = matches(model,value,'three')
                report.append(dict(occurrence=address,open=open_slots(value),refs=value.role_refs,
                                   bound_correct=correct))
                assert store.occurrence_of(index) == address and not open_slots(value)
                assert correct, 'the arriving x is three must supply the bound referent'
                assert not bindings(value).get('_pending')
            if reverse:
                names = set(model._concept_owner().word_concepts('x'))
                for index in range(len(store)):
                    value = store.meaning_of(index)
                    forwards = bindings(value).get('_forward_references', ())
                    # Explore may open an unrelated name (the captured
                    # row 5 names word 14, "is"). Universally require x
                    # to resolve, matching the forward arm's name scope.
                    assert not any(identity in names for _, identity in forwards)
                    original = store.raw_meaning_of(index)
                    pending_names = bindings(original).get('_forward_references', ())
                    if any(identity in names for _, identity in pending_names):
                        assert not any(kind == 'referent' for kind, _ in open_slots(value))
                # The second sentence does not contain "three". Its actual
                # committed graph must carry that earlier referent through
                # the bound x operand, rather than merely lose the variable.
                known = set(model._concept_owner().word_concepts('three'))
                def carries_three(reference):
                    pending_refs, seen = [reference], set()
                    while pending_refs:
                        ref = pending_refs.pop()
                        if ref in known:
                            return True
                        if ref in seen or ref in (-1,0):
                            continue
                        seen.add(ref)
                        if store.parts.contains(ref):
                            pending_refs.extend(store.parts.refs[ref].tolist())
                        child = store.index_of_row(ref)
                        if child is not None:
                            # Identity is interpreted on read, never written
                            # back into the retained occurrence's raw slots.
                            value = store.meaning_of(child)
                            for reference in value.role_refs:
                                if reference is not None:
                                    assert reference[0] in ('sym', 'ltm')
                                    pending_refs.append(reference[1] if reference[0] == 'sym'
                                                        else reference[2])
                    return False
                for field in model._sentence_fields[0]:
                    index = store.location(field.row_id)
                    report.append(dict(occurrence=store.occurrence_of(index),
                        bound_correct=carries_three(int(store.row(index)['row_id']))))
                assert sum(item['bound_correct'] for item in report) >= 2
    try:
        with ForcedGrammar(model, open_names=('x',), live_references=('x',)) as forced:
            rows = train_documents(model, documents, tmp_path, after=after)
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True, reverse=reverse,
            attention_budget=0, ending='budget exhaustion preserves pending reference',
            resolved=report, rows=rows, choices=forced.records),indent=2)+'\n')
        assert len({row['departure'] for row in rows if row['departure'] >= 0}) > 1
    finally:
        model.End()
