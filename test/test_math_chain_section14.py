"""Free variables, stored columns, exhaustion and formation provenance (§14)."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from Meaning import ConceptualMeaning
from ThoughtReferences import (bindings, evidence_pair, fill, needs_episode,
                               open_slots, question, with_slots)


def test_ignorance_is_not_a_free_variable_or_an_answer_cost_gate():
    from BindingAnswers import cost, matches
    atom = replace(ConceptualMeaning.from_description(torch.ones(4)),
                   role_refs=(('sym', 5), None, None))
    region = question(atom)
    assert not open_slots(region) and needs_episode(region)
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
    from Layers import TernaryTruthStore
    from Occurrence import slot_key
    from ThoughtStream import write
    store = TernaryTruthStore(4, capacity=32)
    points = {5:torch.tensor([.1,.2,.3,.4]), 7:torch.tensor([.4,.3,.2,.1])}
    store.configure_clause_index(concept_point=points.get)
    root = ConceptualMeaning(torch.ones(3,4),torch.ones(3,dtype=torch.bool),
        role_refs=(None,('sym',7),('sym',7)), sentence_kind='relation')
    root = question(root, (('referent',0),))
    row = store.append_meaning(root, kind='question', evidence=(0.,0.),
                               document_key='original question', sentence_index=1)
    address, position = store.occurrence_of(row), float(store.sentence_index[row])
    column = slot_key(int(store.address_keys[row]), 0)
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
    assert not bindings(resolved).get('_pending', False) and not needs_episode(resolved)
    assert store.occurrence_of(row) == address and float(store.sentence_index[row]) == position
    torch.testing.assert_close(store.slots[row], original[0], rtol=0, atol=0)
    assert open_slots(store.meaning_of(row)) == open_slots(original[1])
    assert open_slots(store.raw_meaning_of(row)) == open_slots(original[1])
    # Rebuilding the derived posting (as on checkpoint load) retains its ID.
    store._semantic_rows = dict(store._semantic_rows)
    assert store.slot_of_reference(column) == (row,0)


@pytest.mark.parametrize('supported', [True, False])
def test_one_empty_region_query_permits_conclude_and_only_source_mints(monkeypatch, supported):
    from test_item6_2_thinking import world, run
    model, registry, store, (a,b,c) = world()
    root = registry.form('isPart', a, open_roles=('argin2',))
    root = with_slots(replace(root, mode='assertive' if supported else 'interrogative',
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
    assert store.row(len(store)-1)['kind'] == ('inference' if supported else 'question')
    if supported:
        child = result.meaning.constituents[-1]
        assert bindings(child)['_formation_reason'] == 'search_exhausted'
        assert dict(bindings(child)['_formation_records'][0]) == dict(role=2,choice='open',probability=.25)
        assert bindings(result.meaning)['_bound_roles'] == (('referent',2),)
        from BindingAnswers import cost
        owner = SimpleNamespace(word_concepts=lambda _: (9,), _csw_row_of=lambda _:0,
            similarity_codebook=SimpleNamespace(lookup_rows=lambda _,**kw:torch.ones(1,root.roles.shape[-1])))
        scored = SimpleNamespace(_concept_owner=lambda:owner,symbolSpace=SimpleNamespace(ltm_store=store))
        assert .5 <= cost(scored,result.meaning,'unrelated') < 1.


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


def test_episode_work_is_reported_without_changing_the_reconstruction_and_answer_judge(monkeypatch):
    from test_item6_2_thinking import credit_chain_world, run
    model, registry, store, goal, menu = credit_chain_world(monkeypatch)
    run(model, goal, work_budget=64)
    audit = model._last_thought_comparison
    assert audit['components'] == ((0.,0.),(0.,0.))
    assert audit['work_spent'][0] > audit['work_spent'][1]
    assert audit['costs'] == (0.,0.)
    assert audit['keep_costs'] == (0.,0.) and not audit['explore_kept']
    assert model._last_thought_score_function['surrogate'] is None


def test_region_query_ignores_free_variable_code_and_preserves_a_pending_match():
    from Layers import TernaryTruthStore
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from index_fixtures import one_hot_unfold
    store = TernaryTruthStore(8,capacity=16)
    store.configure_leaf_index(code_row=lambda ref:ref[-1], read=one_hot_unfold)
    source = ConceptualMeaning(torch.eye(8)[:3],torch.ones(3,dtype=torch.bool),
        role_refs=(None,('sym',1),('sym',2)),sentence_kind='relation')
    pending = with_slots(source,(('referent',0),),pair=(1.,0.))
    row = store.append_meaning(pending,kind='question',evidence=(1.,0.))
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
    free = question(replace(source,role_refs=(None,None,None)))
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
    """g on the authorized forced c/e grammars; both explore suffixes stay live."""
    import json
    import ThoughtStream
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    model = ordinary_model(tmp_path)
    chooser = model._selected_thought_chooser(None)
    logits = chooser.thought_logits
    def query_then_conclude(active, requests):
        value = logits(active, requests)
        def operation(request):
            if request is None:
                return 'conclude'
            signature = model.grammatical_thoughts.signature_for(request, verify_reference=False)
            return None if signature is None else signature.operation.semantic_id
        bonus = value.new_tensor([10. if candidate is None else
            5. if operation(candidate) == 'query' else 0. for candidate in requests])
        return value + bonus
    monkeypatch.setattr(chooser, 'thought_logits', query_then_conclude)
    commits = []
    commit = ThoughtStream.commit
    def committed_episode(*args, **kwargs):
        index = commit(*args, **kwargs)
        commits.append(index)
        return index
    monkeypatch.setattr(ThoughtStream, 'commit', committed_episode)
    examples = []
    def episode(original, model, meaning, **kwargs):
        before_commits = len(commits)
        result = original(model, meaning, **kwargs)
        assert len(commits) == before_commits + 1
        operations = [record.operation for record in result.records if record.kind == 'thought']
        examples.append(dict(source_pair=bindings(meaning).get('_source_evidence'),
            occurrence=bindings(meaning).get('_query_occurrence'), committed=commits[-1],
            operations=operations,
            before=open_slots(meaning), after=open_slots(result.meaning),
            minted=any(bindings(child).get('_formation_reason') == 'search_exhausted'
                       for child in result.meaning.constituents)))
        return result
    try:
        documents = [ChainDocument('declarative', ('y is x plus two.',)),
                     ChainDocument('question', ('what is y ?',), question=0)]
        with ForcedGrammar(model, open_names=('what','x')) as forced:
            rows = train_documents(model, documents, tmp_path, episode=episode)
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True,
            forcing='c/e operation and binding choices, including query/conclude logit preference; live masked departures',
            examples=examples, rows=rows, choices=forced.records),indent=2)+'\n')
        assert len(examples) == 2
        store = model.symbolSpace.ltm_store
        for item in examples:
            assert item['operations'] == ['query','conclude']
            index = store._index_occurrences[item['occurrence']]
            # The close persists the working conclusion at its own address.
            # The original occurrence retains its open reference.
            committed = store.meaning_of(item['committed'])
            assert open_slots(store.raw_meaning_of(index))
            if tuple(item['source_pair']) == (0.,0.):
                assert item['after'] and open_slots(committed)
                assert store.KINDS[int(store.record_kind[index])] == 'question'
                assert not item['minted']
            else:
                assert item['minted'] and not item['after'] and not open_slots(committed)
                assert item['committed'] != index
    finally:
        model.End()


@pytest.mark.parametrize('reverse', [False, True])
def test_forced_ordinary_pending_premise_in_both_orders(tmp_path, reverse, eager_reading):
    """Certificate e, FORCED grammar, zero budget preserves pending arrivals."""
    import json
    from forced_math_grammar import ForcedGrammar
    from math_chain_ordinary import ordinary_model, train_documents
    from math_chain_corpus import ChainDocument
    from BindingAnswers import matches
    # An exhausted search now legitimately mints a declarative's forward
    # referent. Exercise the other specified ending (budget exhaustion),
    # leaving the pending column for the next sentence's address upsert.
    model = ordinary_model(tmp_path, budget=0)
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
                if store.KINDS[int(store.record_kind[index])] == 'estimate':
                    continue
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
                for index in range(len(store)):
                    if store.KINDS[int(store.record_kind[index])] == 'estimate':
                        continue
                    value = store.meaning_of(index)
                    assert not bindings(value).get('_forward_references')
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
                            pending_refs.extend(store.refs[child].tolist())
                    return False
                for field in model._sentence_fields[0]:
                    index = store.location(field.row_id)
                    report.append(dict(occurrence=store.occurrence_of(index),
                        bound_correct=carries_three(int(store.row(index)['row_id']))))
                assert sum(item['bound_correct'] for item in report) >= 2
    try:
        with ForcedGrammar(model, open_names=('x',), named_bindings={'x':'three'}) as forced:
            rows = train_documents(model, documents, tmp_path, after=after)
        (tmp_path/'certificate.json').write_text(json.dumps(dict(forced=True, reverse=reverse,
            attention_budget=0, ending='budget exhaustion preserves pending reference',
            resolved=report, rows=rows, choices=forced.records),indent=2)+'\n')
        assert len({row['departure'] for row in rows if row['departure'] >= 0}) > 1
    finally:
        model.End()
