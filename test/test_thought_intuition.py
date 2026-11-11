"""Round B: one reference layer, live answer reads and a detached predictor."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from Expectation import ActivationPredictor
from Meaning import ConceptualMeaning
from Intuition import Idea, Term, read, evaluate, selected, encode
from ConceptSpread import Prediction, loss
from ThoughtReferences import bindings
from test_thought_infer import setup, invoke, idea


def test_three_rule_union_not_first_and_one_candidate_per_firing():
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    for a, b, pair in ((refs[0], refs[1], (.8, .2)), (refs[0], refs[2], (.6, .3)),
                       (refs[1], refs[2], (.7, .4))):
        store.append_relation(a[1], b[1], truth_plus=pair[0], truth_minus=pair[1])
    owner._expected_knowing[0, [owner._csw_row_of(ref[1]) for ref in refs], 0] = 1
    result = invoke(model, registry, registry.form('intuit', idea(registry, refs[0])))
    meaning = result.evidence['meaning']
    from Intuition import held_terms
    terms = held_terms(meaning)
    assert len(terms) == 3
    fields = [evaluate(owner, term)[0] for term in terms]
    torch.testing.assert_close(meaning.roles.new_tensor(bindings(meaning)['_intuition_field']), torch.stack(fields).amax(0))
    from Intuition import replay
    replayed, pair = replay(result.evidence['intuition_derivation'])
    torch.testing.assert_close(replayed, torch.stack(fields).amax(0))
    assert pair.tolist() == pytest.approx([result.support_true, result.support_false])
    from ThoughtStream import candidates
    offers = candidates(registry, meaning, meaning, meaning, ())
    choices = [item for item in offers if bindings(item.request).get('_intuition_rule') is not None]
    assert len(choices) == 3
    for offer, term in zip(choices, terms):
        assert not offer.open_roles
        assert bindings(offer.request)['_evidence_pair'] == pytest.approx(evaluate(owner, term)[1].tolist())
        answer = invoke(model, registry, offer.request)
        assert answer.evidence['sources'] == (term.premise, term.rule)
        # Restore the reference plan for the next independent selection.
        from ThoughtChunks import put
        put(owner.stm, meaning, bindings(meaning)['_chunk_when'], 0, existing=True)
    plan = Idea(observed=torch.tensor([.2, .8]))
    bare = SimpleNamespace(similarity_codebook=SimpleNamespace(W=torch.zeros(2)),
        _concept_allocator=SimpleNamespace(layer=lambda _:SimpleNamespace(conjunctive=None)), outputShape=(1,2))
    torch.testing.assert_close(read(bare, replace(plan, negative=True)), read(bare, plan).flip(-1))


def test_rule_conflict_keeps_two_lanes_and_truth_has_no_parameter():
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    root = Idea(observed=torch.full((owner.outputShape[-1],), .7))
    goal = Idea(observed=torch.full((owner.outputShape[-1],), .5))
    t = Term(('sym', 1), root, goal, (.8, 0.), idea(registry, refs[1]), (root,), (2.,2.,2.,2.))
    positive, _ = evaluate(owner, t)
    negative, _ = evaluate(owner, replace(t, consequent=replace(goal, negative=True)))
    both = torch.maximum(positive, negative)
    assert bool((both.amin(-1) > 0).all())
    assert all(type(value) is float for value in t.trust)


def test_answer_trains_only_fired_intuition_weights_and_ltm_is_unchanged():
    from Intuition import seed, evidence_pair
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    store.append_relation(refs[0][1], refs[1][1], truth_plus=.9, truth_minus=.2)
    known = Idea(observed=torch.full((owner.outputShape[-1],), .3, requires_grad=True))
    antecedent = Idea(observed=torch.full((owner.outputShape[-1],), .8, requires_grad=True))
    goal = Idea(observed=torch.full((owner.outputShape[-1],), .5, requires_grad=True))
    term = seed(owner, Term(('sym', 123), antecedent, goal, (.9, .2),
        idea(registry, refs[1]), (known,), (2.,)*4), QueryWorkBudget(256))
    other = seed(owner, replace(term, rule=('sym', 124)), QueryWorkBudget(256))
    optimizer = torch.optim.SparseAdam(owner.intuition.parameters(), lr=.1)
    warmed, _ = selected(owner, other)
    (warmed.roles - .8).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    def bytes_of(module):
        return {key: (str(value.dtype), tuple(value.shape),
                      value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                if torch.is_tensor(value) else repr(value)
                for key, value in module.state_dict().items()}
    before = bytes_of(store)
    pair = evidence_pair(owner, term)
    value, _ = selected(owner, term)
    loss_value = (value.roles - .8).square().mean()
    loss_value.backward()
    assert before == bytes_of(store)
    for parameter, used in ((owner.intuition.pi, term.pi), (owner.intuition.sigma, (term.sigma,))):
        gradient = parameter.grad.coalesce()
        assert gradient.values().abs().sum() > 0
        assert set(gradient.indices()[0].tolist()) <= set(used)
    assert all(x.observed.grad is None for x in (known, antecedent, goal))
    assert all(p.grad is None for matrix in owner._concept_allocator.layer(0).definition_matrices()
               for p in matrix.parameters())
    original = {name:p.detach().clone() for name,p in owner.intuition.named_parameters()}
    optimizer.step()
    assert not torch.equal(original['sigma'][term.sigma], owner.intuition.sigma[term.sigma])
    assert torch.equal(original['sigma'][other.sigma], owner.intuition.sigma[other.sigma])
    torch.testing.assert_close(evidence_pair(owner, term), pair, rtol=0, atol=0)
    assert before == bytes_of(store)
    later, _ = selected(owner, term)
    assert not torch.equal(later.roles, value.roles)


def test_predictor_target_and_consumers_detach_and_predictor_trains():
    predictor = ActivationPredictor(2)
    keys = torch.tensor([[[.2,.3],[.8,.9],[.1,.1]]], requires_grad=True)
    knowing = torch.full((1,3,2), .3, requires_grad=True)
    context = torch.full((1,4), .2, requires_grad=True)
    prediction = predictor(keys, context, knowing).sigmoid()
    target = torch.full_like(prediction, .9, requires_grad=True)
    record = Prediction(keys, context, knowing, prediction.detach(), torch.arange(3),
        torch.tensor([True]), target, torch.ones(1,3,dtype=torch.bool))
    model = SimpleNamespace(_concept_prediction=record,
        _concept_owner=lambda:SimpleNamespace(concept_predictor=predictor))
    loss(model).sum().backward()
    assert target.grad is None and knowing.grad is None and keys.grad is None and context.grad is None
    assert sum(p.grad.abs().sum() for p in predictor.parameters()) > 0
    assert not record.value.requires_grad


def test_sparse_capacity_and_storage_are_fixed_and_exhaustion_is_atomic():
    from IntuitionLayer import IntuitionLayer
    layer = IntuitionLayer(2)
    pointers = {name:p.data_ptr() for name,p in layer.named_parameters()}
    assert layer.seed(('sym',1), ((0,), (1,))) == ((0,1),0)
    with pytest.raises(RuntimeError, match='intuitionCapacity'):
        layer.seed(('sym',2), ((0,),))
    assert len(layer._indices['sigma']) == 1
    assert pointers == {name:p.data_ptr() for name,p in layer.named_parameters()}
    assert all(p.shape == (2,1) for p in layer.parameters())


def test_pervasion_minimum_and_close_agreement_disagreement():
    from Intuition import pervade, judge
    from ThoughtReferences import with_slots, evidence_pair
    property_field = torch.tensor([[1.,0.], [1.,0.]])
    parts = torch.tensor([[[.8,.4],[.8,.4]], [[.5,.9],[.5,.9]]])
    torch.testing.assert_close(pervade(parts, property_field), torch.tensor([.5,.4]))
    base = ConceptualMeaning.from_description(torch.ones(4))
    explicit = with_slots(replace(base, bindings={'_producing_operation':'infer'}), (), pair=(.8,0.))
    implicit = with_slots(replace(base, bindings={'_evidence_kind':'pervasion'}), (), pair=(.6,0.))
    assert evidence_pair(judge(explicit, [SimpleNamespace(meaning=implicit)])) == (.8,0.)
    implicit = with_slots(implicit, (), pair=(0.,.7))
    assert evidence_pair(judge(explicit, [SimpleNamespace(meaning=implicit)])) == (.8,.7)


def test_firing_metadata_reaches_chooser_and_choice_loss_reaches_its_parameters():
    from ThoughtReferences import with_slots
    model, registry, store, refs = setup()
    root = idea(registry, refs[0])
    low = with_slots(registry.form('intuit', root), (), pair=(.1, .2))
    high = with_slots(low, (), pair=(.9, .2))
    chooser = model.shared_grammar
    # Supply an active candidate-content path; this is a wiring probe,
    # independent of a random hidden layer's saturation.
    with torch.no_grad():
        first, last = chooser.chooser.mlp[0], chooser.chooser.mlp[-1]
        first.weight.zero_(); first.bias.fill_(1.)
        first.weight[:, chooser.d_model:2*chooser.d_model].copy_(
            torch.eye(first.out_features, chooser.d_model))
        last.weight.fill_(1./last.in_features); last.bias.zero_()
    logits = chooser.thought_logits(root, (low, high))
    assert not torch.equal(logits[:, 0], logits[:, 1])
    loss = -logits.log_softmax(-1)[0, 1]
    loss.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in chooser.parameters())


def test_prepended_candidate_enters_by_working_value():
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    store.append_relation(refs[0][1], refs[1][1], truth_plus=.8, truth_minus=.3)
    owner._expected_knowing[0, owner._csw_row_of(refs[0][1]), 0] = .4
    value = torch.linspace(.1, .7, owner.outputShape[-1])
    entries = SimpleNamespace(prepend=2, concept_ids=torch.tensor([[987, -1]]),
        lookup_atoms=torch.stack((value, torch.zeros_like(value)))[None])
    object.__setattr__(owner, '_model', SimpleNamespace(inputSpace=SimpleNamespace(symbol_entries=entries),
                                                      symbolSpace=model.symbolSpace))
    result = invoke(model, registry, registry.form('intuit', idea(registry, refs[0])))
    from Intuition import held_terms
    terms = held_terms(result.evidence['meaning'])
    assert terms
    assert any(plan.observed is not None and torch.equal(plan.observed, value) for plan in terms[0].knowing)


def test_intuit_reaches_completed_field_without_serial_grammar_options(monkeypatch):
    from Language import Grammar
    from Queries import GrammaticalThoughtRegistry
    from test_item6_2_thinking import run
    model, old, store, refs = setup()
    model._what_memory().finish_thought(idea(old, refs[0]))
    model._end_finished_selected_thought_episodes()
    grammar = Grammar()
    grammar.configure({'Symbolic': {'thought': {'rule': [
        'intuit_argout1 = intuit.thought(intuit_argin1)']}}})
    registry = GrammaticalThoughtRegistry.install(model._concept_owner(), grammar)
    owner = model._concept_owner()
    store.append_relation(refs[0][1], refs[1][1], truth_plus=.8, truth_minus=.2)
    owner._expected_knowing[0, owner._csw_row_of(refs[0][1]), 0] = 1.
    owner._observed_knowing[0, owner._csw_row_of(refs[0][1]), 0] = 1.
    with torch.no_grad():
        for ref, value in zip(refs, (.7,.5,.3)):
            owner.similarity_codebook.W[owner._csw_row_of(ref[1])].fill_(value)
    assert registry.executable_operation_ids == ('intuit',)
    assert not grammar.rules_upward and not grammar.rules_downward
    object.__setattr__(model, 'grammatical_thoughts', registry)
    model.symbolSpace.grammatical_thoughts = registry
    selected = []
    def choose(root, active, actions, **kwargs):
        if selected:
            assert None in actions
            return None
        action = next(action for action in actions if action is not None and action.semantic_id == 'intuit')
        selected.append(action)
        return action
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    question = replace(idea(old, refs[0]), role_refs=(None,None,None))
    from ThoughtReferences import with_slots, open_slots
    question = with_slots(question, (('referent',0),))
    result = run(model, question, registry=registry, work_budget=128)
    assert selected and not open_slots(result.meaning)
    assert bindings(result.meaning)['_intuition_terms']
    assert result.meaning.roles.any()
    assert result.records[-1].kind == 'finish'
    assert all('conceptualMode' not in Path(path).read_text() for path in ('data/model.xml','data/model.xsd'))


def test_native_part_narrowing_and_pervasion_replay():
    from Intuition import pervasion
    from QueryWork import QueryWorkBudget
    from ThoughtReferences import evidence_pair
    from order_fixtures import force_order
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    obj, first, second = refs
    prop = ('sym', owner.new_concept())
    force_order(owner, prop[1], 0)
    owner._csw_concept_row(0, prop[1])
    for ref in (*refs, prop):
        with torch.no_grad():
            owner.similarity_codebook.W[owner._csw_row_of(ref[1])].fill_(1.)
    store.append_relation(first[1], obj[1], truth_plus=.8, truth_minus=.4)
    store.append_relation(second[1], obj[1], truth_plus=.5, truth_minus=.9)
    question = registry.form('isPart', obj, prop)
    result = pervasion(owner, question, work=QueryWorkBudget(256))
    assert result['evidence_kind'] == 'pervasion'
    assert evidence_pair(result['meaning']) == pytest.approx((.5,.4))
    assert set(result['witnesses']) == {first, second}
    from Intuition import pervade
    proof = result['pervasion_derivation']
    torch.testing.assert_close(pervade(proof['parts'], proof['property']), torch.tensor([.5,.4]))


@pytest.mark.parametrize('semantic_id', ['part', 'isPart'])
def test_pervasion_requires_a_property_relation_and_its_result_persists(semantic_id):
    from Intuition import pervasion, execute
    from QueryWork import QueryWorkBudget
    from ThoughtStream import write
    from order_fixtures import force_order
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    subject, part, prop = refs
    for ref in refs:
        force_order(owner, ref[1], 0)
        with torch.no_grad():
            owner.similarity_codebook.W[owner._csw_row_of(ref[1])].fill_(1.)
    store.append_relation(part[1], subject[1], truth_plus=.8, truth_minus=.4)
    question = registry.form(semantic_id, subject, prop)
    result = pervasion(owner, question, work=QueryWorkBudget(256))
    assert result['evidence_kind'] == 'pervasion'
    assert write(model, result['meaning'], row=0) is not None
    # The captured failure had the same three-role shape but another
    # predicate. It must use the ordinary DNF instead of fabricating a
    # property judgement that the semantic writer cannot admit.
    unrelated = replace(question, role_refs=(subject, prop, part))
    work = QueryWorkBudget(256)
    assert pervasion(owner, unrelated, work=work) is None and work.spent == 0
    ordinary = execute(owner, unrelated, (), row=0, work=work, premise=(1.,)*4)
    assert ordinary['evidence_kind'] == 'intuition'
    assert int(ordinary['meaning'].role_mask.sum()) == 1
    assert write(model, ordinary['meaning'], row=0) is not None


def test_initial_weights_equal_literal_not_then_min_and_union():
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    D = owner.outputShape[-1]
    positive = Idea(observed=torch.full((D,), .7))
    negative = replace(positive, observed=torch.full((D,), .4), negative=True)
    antecedent = Idea(children=(positive, negative))
    knowing = Idea(children=(replace(positive, observed=torch.full((D,), .5)), negative))
    consequent = Idea(observed=torch.full((D,), .6))
    term = seed(owner, Term(('sym', 78), antecedent, consequent, (.9,.2),
        idea(registry, refs[1]), (knowing,), (1.,)*4), QueryWorkBudget(256))
    actual, pair = evaluate(owner, term)
    a, k, c = (read(owner, plan) for plan in (antecedent, knowing, consequent))
    coverage = torch.minimum(a,k).sum()/a.sum()
    expected_pair = torch.minimum(coverage, a.new_tensor(term.trust))
    expected = torch.maximum(torch.minimum(c,expected_pair[0]),torch.minimum(c.flip(-1),expected_pair[1]))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(pair,expected_pair,rtol=0,atol=0)


def test_output_ownership_excludes_understanding_and_expectation():
    from ObjectiveOwnership import backward_owned
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    from Models import BasicModel
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    model.conceptualSpaces = torch.nn.ModuleList([owner])
    parameters = list(owner.concept_predictor.parameters()) + list(owner.intuition.parameters())
    optimizer = SimpleNamespace(param_groups=[dict(params=parameters)])
    groups = BasicModel.objective_parameter_groups(model, optimizer)
    assert {id(p) for p in owner.intuition.parameters()} <= {id(p) for p in groups['output']}
    assert {id(p) for p in owner.concept_predictor.parameters()} <= {id(p) for p in groups['expectation']}
    root = Idea(observed=torch.full((owner.outputShape[-1],), .5))
    term = seed(owner, Term(('sym',99),root,root,(.8,.2),idea(registry,refs[1]),(root,),(1.,)*4),QueryWorkBudget(256))
    value = selected(owner,term)[0].roles.sum()
    backward_owned({'reconstruction':value,'expectation':value},groups)
    assert all(p.grad is None for p in owner.intuition.parameters())
    backward_owned({'output':value},groups)
    assert owner.intuition.sigma.grad.coalesce().values().abs().sum() > 0


def test_predictor_spread_desire_and_placement_have_separate_surfaces():
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    row = owner._csw_row_of(refs[0][1])
    owner.prime_seen(torch.tensor([[row]]))
    seen = owner.placement_context().clone()
    owner._expected_knowing[0, row] = torch.tensor([.3,.7])
    before = owner.relevance_weights(batch=1).clone()
    assert before[0,row] == pytest.approx(1.7)
    owner.prime_desire(torch.tensor([[row]]), valence=-.2)
    assert owner.relevance_weights(batch=1)[0,row] == pytest.approx(1.5)
    torch.testing.assert_close(owner.placement_context(),seen,rtol=0,atol=0)
    assert not owner.relevance_weights(batch=1).requires_grad


def test_sparse_intuition_allocation_read(tmp_path):
    import json, os
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from IntuitionLayer import IntuitionLayer
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    shapes = set()
    class AllocationRead(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if torch.is_tensor(value):
                    shapes.add(tuple(value.shape))
            return result
    pointers = {name:p.data_ptr() for name,p in owner.intuition.named_parameters()}
    known = Idea(observed=torch.full((owner.outputShape[-1],), .3))
    with AllocationRead():
        term = seed(owner, Term(('sym',123), known, known, (.8,.2),
            idea(registry, refs[1]), (known,), (2.,)*4), QueryWorkBudget(256))
        selected(owner, term)[0].roles.sum().backward()
    V = owner.nVectors
    assert all(shape != (V,V) for shape in shapes)
    assert pointers == {name:p.data_ptr() for name,p in owner.intuition.named_parameters()}
    folder=Path(os.environ.get('ITEM25_RECEIPT',tmp_path));folder.mkdir(parents=True,exist_ok=True)
    (folder/'allocation-read.json').write_text(json.dumps(dict(torch_version=str(torch.__version__), inventory=V,
        capacity=owner.intuition.capacity, parameters={name:list(p.shape) for name,p in owner.intuition.named_parameters()},
        pointers=pointers,pointers_unchanged=True,shapes=sorted(shapes),inventory_square=False),indent=2)+'\n')


def test_knowing_effect_preserves_field_shape_and_unaddressed_content(monkeypatch):
    from Intuition import write_knowing
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    rows = torch.tensor([owner._csw_row_of(ref[1]) for ref in refs])
    field = torch.zeros(3,1,2,2)
    field[0,0,0] = torch.tensor([.2,.1])
    carrier = owner.subspace
    carrier._concept_activations = field
    carrier._concept_inventory_rows = rows
    carrier._concept_where = torch.tensor([[[0, 1], [0, 0]]])
    from Intuition import pack_term, chart, leaf
    term = Term(('sym', 1), Idea(), Idea(row=int(rows[0])), (.7, .4),
                idea(registry, refs[0]), (), (2.,) * 4)
    meaning = replace(term.meaning, bindings={'_intuition_terms': (pack_term(term),)})
    code = chart(owner, leaf(owner, int(rows[0]))).amax(-1)
    monkeypatch.setattr('Intuition.evaluate', lambda owner, term:
                        (code[:, None] * code.new_tensor([.7, .4]), code.new_tensor([.7, .4])))
    write_knowing(model,{'meaning': meaning, 'knowing':((int(rows[0]),(.7,.4)),)},0)
    after = carrier._concept_activations
    assert after.shape == field.shape
    torch.testing.assert_close(after[0,0,0],torch.tensor([.7,.4]))
    assert torch.equal(after[1:],field[1:])
    assert torch.equal(after[:,:,1],field[:,:,1])


def test_exhausted_search_keeps_a_checked_intuition_conclusions_pair():
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    from ThoughtReferences import with_slots, conclude_exhausted, evidence_pair
    from ThoughtAnswer import accepted
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    observed = Idea(observed=torch.full((owner.outputShape[-1],), .4))
    # The original question had no source evidence. Its checked answer now
    # has an independent proof, which search exhaustion must not discard.
    consequent = with_slots(idea(registry, refs[1]), (), source_evidence=(0., 0.))
    term = seed(owner, Term(('sym',987),observed,observed,(.8,.2),
        consequent,(observed,),(1.,)*4),QueryWorkBudget(256))
    answer, pair = selected(owner,term)
    assert accepted(answer) and pair[0] > 0
    result = conclude_exhausted(answer)
    assert evidence_pair(result) == pytest.approx(pair.tolist())


def test_answer_adapter_reads_live_reference_plan_and_credits_fired_weights_and_chooser(monkeypatch):
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    from ThoughtAnswer import scorer, train_readers
    from SentenceUnderstanding import SentenceUnderstanding, PrimedSymbols
    from Layers import Error
    model, registry, store, refs = setup()
    owner = model._concept_owner();D=owner.outputShape[-1]
    known=Idea(observed=torch.full((D,),.3));antecedent=Idea(observed=torch.full((D,),.8))
    goal=Idea(observed=torch.full((D,),.5))
    term=seed(owner,Term(('sym',123),antecedent,goal,(.9,.2),idea(registry,refs[1]),(known,),(2.,)*4),QueryWorkBudget(256))
    conclusion,_=selected(owner,term)
    # The resident value is detached; the adapter must recover the live plan.
    conclusion=conclusion.detached()
    language=[None]*12
    language[6]=torch.zeros(1,1,D);language[10]=torch.zeros(1,1,3*D);language[11]=torch.ones(1,1,dtype=torch.long)
    bank=PrimedSymbols(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,D),torch.ones(1,1),
        torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,1,dtype=torch.long),torch.ones(1,1,1,dtype=torch.bool))
    record=SentenceUnderstanding(torch.zeros(1,D),torch.zeros(1,3,D),torch.ones(1,dtype=torch.long),
        language[6],language[11],torch.zeros(1,1,D),torch.zeros(1,1,dtype=torch.long),
        torch.ones(1,1,dtype=torch.bool),bank,torch.tensor(0))
    observation=dict(meanings=[conclusion],clauses=[None],record=record)
    object.__setattr__(model,'inputSpace',SimpleNamespace(data=SimpleNamespace(what=lambda _:None)))
    model._sentence_answer_questions=(None,)
    model.teacher=SimpleNamespace(errors=Error())
    model.answer_probe=torch.nn.Linear(D,1,bias=False)
    with torch.no_grad():model.answer_probe.weight.fill_(1/D)
    def read(state,sentence,active,view,registry):
        prediction=model.answer_probe(view['record'].root).flatten()
        cost=(prediction-.8).square()
        registry.add('answer',cost,category='prediction',objective='output')
        return cost
    monkeypatch.setattr(model,'_sentence_reader_error',read)
    score,trials=scorer(model,((),tuple(language),()),observation,0)
    value=score(0,SimpleNamespace(meaning=conclusion))
    train_readers(model,trials)
    request=registry.form('intuit',idea(registry,refs[0]))
    logits=model.shared_grammar.thought_logits(conclusion,(None,request))
    policy=-logits.log_softmax(-1)[0,1]*value.detach()
    (model.errors.total().sum()+policy).backward()
    assert owner.intuition.pi.grad.coalesce().values().abs().sum()>0
    assert owner.intuition.sigma.grad.coalesce().values().abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.shared_grammar.parameters())
    assert not conclusion.roles.requires_grad


def test_unit_weights_equal_hand_computed_three_rule_dnf_with_literal_negation():
    from IntuitionLayer import IntuitionLayer
    from Intuition import seed
    from QueryWork import QueryWorkBudget
    owner=SimpleNamespace(similarity_codebook=SimpleNamespace(W=torch.zeros(2)),
        outputShape=(1,2),intuition=IntuitionLayer(32))
    def value(a,b,negative=False):
        return Idea(observed=torch.tensor([a,b]),negative=negative)
    knowing=(value(.8,.2),value(.4,.9,True))
    # The first antecedent has both a positive and a negative literal.
    # Its meet mass is 1.4 and its antecedent mass is 2.1, hence 2/3.
    antecedents=(Idea(children=(value(.5,.6),value(.7,.3,True))),
                 value(.3,.6,True),value(.9,.1))
    consequents=(value(.9,.4),value(.8,.2,True),value(.3,.6))
    trusts=((.8,.1),(.4,.7),(.5,.3))
    terms=[seed(owner,Term(('sym',i),a,b,t,ConceptualMeaning.from_description(torch.ones(2)),
            knowing,(1.,)*4),QueryWorkBudget(128))
        for i,(a,b,t) in enumerate(zip(antecedents,consequents,trusts))]
    results=[evaluate(owner,term) for term in terms]
    expected_fields=(torch.tensor([[2/3,.1],[.4,.1]]),
                     torch.tensor([[.7,.4],[.2,.2]]),
                     torch.tensor([[.3,.3],[.5,.3]]))
    for (field,pair),expected,trust in zip(results,expected_fields,((2/3,.1),(.4,.7),(.5,.3))):
        torch.testing.assert_close(field,expected)
        torch.testing.assert_close(pair,torch.tensor(trust))
    torch.testing.assert_close(torch.stack([field for field,_ in results]).amax(0),
        torch.tensor([[.7,.4],[.5,.3]]))


def test_intuition_sparse_checkpoint_restores_native_addresses_and_rejects_malformed_banks():
    import copy
    from IntuitionLayer import IntuitionLayer
    first=IntuitionLayer(4)
    first.seed(('sym',7),((0,3,False),(1,4,True)))
    with torch.no_grad():first.sigma[0]=.4
    state=copy.deepcopy(first.state_dict())
    restored=IntuitionLayer(4)
    pointers={name:p.data_ptr() for name,p in restored.named_parameters()}
    restored.load_state_dict(state)
    assert restored._indices==first._indices
    assert pointers=={name:p.data_ptr() for name,p in restored.named_parameters()}
    torch.testing.assert_close(restored.sigma,first.sigma,rtol=0,atol=0)
    for bank in ('pi','sigma'):
        invalid=copy.deepcopy(first.validate_extra_state(state['_extra_state']))
        invalid['indices'][bank][('duplicate',)]=0
        with pytest.raises(ValueError,match='invalid sparse addresses'):
            restored.validate_extra_state(invalid)
    invalid=copy.deepcopy(first.validate_extra_state(state['_extra_state']));invalid['capacity']=8
    with pytest.raises(ValueError,match='constructed capacity'):
        restored.validate_extra_state(invalid)


def test_observed_reference_plan_contains_order_zero_content_without_percept_stamps():
    from Intuition import observation, pack_idea
    owner=SimpleNamespace(similarity_codebook=SimpleNamespace(mereology=SimpleNamespace(
        percept_width=2,percept_event_width=4)))
    first=torch.tensor([.2,.3,.4,.5,.6,.7],requires_grad=True)
    later=first.detach().clone();later[2:4]=torch.tensor([.8,.9])
    a,b=observation(owner,first),observation(owner,later)
    assert pack_idea(a)==pack_idea(b)
    torch.testing.assert_close(a.observed,torch.tensor([.2,.3,0.,0.,.6,.7]),rtol=0,atol=0)
    assert not a.observed.requires_grad


@pytest.mark.parametrize('budget', [0, 1])
def test_exhausted_intuit_preserves_the_goal_without_depositing_its_request(budget):
    from QueryWork import QueryWorkBudget
    from ThoughtStream import materialize, integrate
    from ThoughtChunks import stage
    model, registry, store, refs = setup()
    goal = registry.form('isPart', refs[0], refs[1])
    request = registry.form('intuit', goal)
    stm = model.conceptualSpace.stm
    before = tuple(value.clone() for value in stm.state)
    contents = stm._thought_contents
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(request, row=0,
            work=QueryWorkBudget(budget), continuation=None)
        result = registry.execute(request, context)
        assert result.evidence['incomplete'] == ('work_budget',)
        assert result.evidence.get('meaning') is None and result.evidence.get('value') is None
        assert materialize(result) is result
        assert stage(model._what_memory(), result, 0) is result
        assert integrate(registry, goal, result, ()) is goal
    assert stm._thought_contents is contents
    for original, unchanged in zip(before, stm.state, strict=True):
        torch.testing.assert_close(original, unchanged, rtol=0, atol=0)


def test_retained_mtm_transient_operand_uses_its_working_value_and_rejects_unknown_ids():
    from Intuition import extract, chart
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    working = torch.full((owner.outputShape[-1],), .3, requires_grad=True)
    key = owner.mtm.open(('test-occurrence', 1), kind=None, when=0,
        value=torch.full_like(working, .9))
    transient = owner.mtm.entries[key].transient
    assert transient >= owner.nVectors and owner._csw_row_of(transient) is None
    meaning = replace(ConceptualMeaning.from_description(working),
        role_refs=(('sym', transient), None, None))
    work = QueryWorkBudget(8)
    plan = extract(owner, meaning, work=work)
    torch.testing.assert_close(read(owner, plan), chart(owner, working), rtol=0, atol=0)
    assert not read(owner, plan).requires_grad and work.counts['payload'] == 1
    owner.mtm.forget(key)
    with pytest.raises(ValueError, match='unallocated concept'):
        extract(owner, meaning, work=QueryWorkBudget(8))


@pytest.mark.parametrize('source_pair', [(1., 0.), (.3, .8)])
def test_returned_intuition_proof_does_not_certify_an_unchanged_parent(source_pair, monkeypatch):
    from test_item6_2_thinking import world, run
    from ThoughtReferences import with_slots, evidence_pair, open_slots
    from Thoughts import replay_thoughts
    model, registry, store, (a, b, c) = world()
    goal = registry.form('isPart', a, c)
    goal = with_slots(replace(goal, role_refs=(None, goal.role_refs[1], None)),
        (('referent', 0), ('referent', 2)), pair=source_pair)
    chosen = []
    def choose(root, active, actions, **kwargs):
        if chosen and None in actions:
            return None
        operation = 'ask' if not chosen else 'intuit'
        value = next(action for action in actions if action is not None and action.semantic_id == operation)
        chosen.append(operation)
        return value
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    # The native return consumes the final unit. The empty child intuition
    # leaves the caller's still-open proposition and source evidence intact.
    result = run(model, goal, work_budget=13)
    assert chosen == ['ask', 'intuit'] and result.work.spent == 13
    assert result.meaning.role_refs == goal.role_refs and open_slots(result.meaning)
    assert evidence_pair(result.meaning) == source_pair
    returned = next(record for record in result.records if record.kind == 'return')
    assert returned.result.evidence['intuition_derivation'] == ()
    assert (returned.support_true, returned.support_false) == (0., 0.)
    assert result.result.semantic_id == 'ask'
    assert result.records[-1].result is None
    assert model._what_memory().thought_reference(returned) in result.records[-1].sources
    assert replay_thoughts(result.records).finished
    corrupt = tuple(replace(record, support_true=.5) if record is returned else record
                    for record in result.records)
    with pytest.raises(ValueError, match='intuition pair differs'):
        replay_thoughts(corrupt)
    # A changed pair on the same proposition still carries its proof and is
    # checked by replay. Only the unrelated parent/child attribution is removed.
    from ThoughtStream import completion_result
    child = returned.result.evidence['meaning']
    assert completion_result(with_slots(child, (), pair=(.5, 0.)), returned.result) is returned.result


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_finish_keeps_an_empty_child_proof_off_an_open_zero_referent(dtype):
    from types import MappingProxyType
    from Meaning import ConceptualMeaning
    from Queries import ThoughtResult
    from ThoughtReferences import with_slots
    from ThoughtStream import completion_result
    empty = ConceptualMeaning.from_description(torch.zeros(392, dtype=dtype))
    child = with_slots(empty, (), pair=(0., 0.))
    caller = with_slots(replace(empty, sentence_kind='idea'), (('referent', 0),),
                        pair=(1., 1.), source_evidence=(1., 1.))
    proof = ThoughtResult('ask', 'conceptual-subgoal', 'idea', 'unverified', child,
        MappingProxyType({'meaning': child, 'intuition_derivation': (), 'support_true': 0., 'support_false': 0.}))
    assert torch.equal(caller.roles, child.roles) and torch.equal(caller.role_mask, child.role_mask)
    assert completion_result(caller, proof) is None
    assert completion_result(with_slots(child, (), pair=(1., 1.)), proof) is proof


def test_empty_child_proof_does_not_certify_a_zero_valued_witnessed_caller():
    from types import MappingProxyType
    from Meaning import ConceptualMeaning
    from Queries import ThoughtResult
    from ThoughtReferences import with_slots
    from ThoughtStream import completion_result
    empty = ConceptualMeaning.from_description(torch.zeros(392))
    child = with_slots(replace(empty, bindings={'_thought_witnesses': ()}), (), pair=(0., 0.))
    caller = with_slots(replace(empty, bindings={'_thought_witnesses': (('sym', 7),)}), (), pair=(.3, 0.))
    proof = ThoughtResult('ask', 'conceptual-subgoal', 'idea', 'unverified', child,
        MappingProxyType({'meaning': child, 'intuition_derivation': (), 'support_true': 0., 'support_false': 0.}))
    assert completion_result(caller, proof) is None
    assert completion_result(with_slots(child, (), pair=(.3, 0.)), proof) is proof


def test_exhausted_ask_child_without_result_keeps_its_evidence(monkeypatch):
    from test_item6_2_thinking import world
    from ThoughtReferences import evidence_pair, with_slots
    from Thoughts import replay_thoughts
    model, registry, store, (a, b, c) = world()
    pair = (.75, .25)
    goal = registry.form('isPart', a, c)
    goal = with_slots(replace(goal, role_refs=(None, goal.role_refs[1], None)),
                      (('referent', 0), ('referent', 2)), pair=pair)
    chosen = []
    def choose(root, active, actions, **kwargs):
        chosen.append('ask')
        assert len(chosen) == 1
        return next(action for action in actions if action is not None and action.semantic_id == 'ask')
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    # Six native units pay for ask's reads and its descent. The descent
    # exhausts the shared meter before a child operation can produce a result.
    with model._query_boundary_scope((0,)):
        result = model._run_selected_thought_once(goal, work_budget=6)
    assert chosen == ['ask'] and result.work.spent == 6
    returned = next(record for record in result.records if record.kind == 'return')
    assert evidence_pair(returned.meaning) == pair
    assert (returned.support_true, returned.support_false) == pair
    assert returned.result.evidence['incomplete'] == ('work_budget',)
    assert returned.result.evidence['value'] is None
    assert evidence_pair(returned.result.evidence['meaning']) == pair
    assert replay_thoughts(result.records).finished


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_nested_return_keeps_empty_proof_off_known_referent_fallback(dtype, monkeypatch):
    from test_item6_2_thinking import world
    from ThoughtReferences import evidence_pair, open_slots, with_slots
    from Thoughts import replay_thoughts
    model, registry, store, (a, b, known) = world()
    model._concept_owner().to(dtype=dtype)
    model.shared_grammar.to(dtype=dtype)
    store.to(dtype=dtype)
    source_pair = (1., 0.)
    goal = registry.form('isPart', a, known)
    goal = with_slots(replace(goal, roles=goal.roles.to(dtype),
        role_refs=(None, None, known), sentence_kind='relation',
        bindings=dict(_chain_view=True, _unresolved_operands=((2, known),))),
        (('referent', 0), ('referent', 1)), pair=source_pair, source_evidence=source_pair)
    sequence = ('ask', 'ask', 'bind', 'intuit')
    chosen = []
    def choose(root, active, actions, **kwargs):
        assert len(chosen) < len(sequence), ('unexpected extra thought', chosen, kwargs['work'].spent)
        operation = sequence[len(chosen)]
        selected = next(action for action in actions
                        if action is not None and action.semantic_id == operation)
        chosen.append(operation)
        return selected
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    # The captured lesson spends 32 units; this smaller store needs 23 for
    # the same four choices. Exhaust at the intuition so both asks drain.
    # Exercise the native episode through finish, before the separate durable
    # close: the captured error occurs here, before a sentence row is committed.
    with model._query_boundary_scope((0,)):
        result = model._run_selected_thought_once(goal, work_budget=23)
    assert chosen == list(sequence) and result.work.spent == 23
    assert result.meaning.role_refs == goal.role_refs
    assert open_slots(result.meaning) == open_slots(goal)
    assert evidence_pair(result.meaning) == source_pair
    returns = [record for record in result.records if record.kind == 'return']
    assert len(returns) == 2
    inner, outer = returns
    assert inner.result.evidence['intuition_derivation'] == ()
    assert inner.result.evidence['meaning'].role_refs == (None, None, None)
    assert (inner.support_true, inner.support_false) == (0., 0.)
    assert 'intuition_derivation' not in outer.result.evidence
    assert outer.result.evidence['value'].evidence['intuition_derivation'] == ()
    assert outer.result.evidence['value'].evidence['meaning'].role_refs == (None, None, None)
    assert outer.result.evidence['meaning'].role_refs == goal.role_refs
    assert (outer.support_true, outer.support_false) == source_pair
    assert 'intuition_derivation' not in result.records[-1].result.evidence
    assert model._what_memory().thought_reference(inner) in outer.sources
    assert replay_thoughts(result.records).finished
    # The original same-proposition proof remains checked at its own return.
    # Altering that pair must not become another reason to drop the proof.
    corrupt = tuple(replace(record, support_true=.5) if record is inner else record
                    for record in result.records)
    with pytest.raises(ValueError, match='intuition pair differs'):
        replay_thoughts(corrupt)
