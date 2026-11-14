"""The signed trust's actual ingestion, serving and proof writers (§17)."""
from dataclasses import replace
import math

import pytest
import torch


@pytest.fixture
def truth_model(eager_reading):
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG, identified_input=False)
    model._ltm_provisioned = True
    try:
        yield model
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('model_trust', (0., .25, 1.))
@pytest.mark.parametrize('writer', ('store_truths', 'provision_ltm'))
def test_50_authority_keeps_its_signed_trust_independently_of_model_trust(truth_model, monkeypatch, model_trust, writer):
    from util import TheXMLConfig
    model = truth_model
    model.trust = model_trust
    model._concept_owner()._trust = model_trust
    store = model.symbolSpace.ltm_store
    values = (.3, -.6, 1.)
    if writer == 'store_truths':
        model.store_truths([dict(content='hello', trust=value) for value in values])
    else:
        get = TheXMLConfig.get
        monkeypatch.setattr(TheXMLConfig, 'get', lambda key, *args, **kwargs:
            dict(truth=[dict(text='hello', trust=value) for value in values])
            if key == 'architecture.truthSet' else get(key, *args, **kwargs))
        assert model.provision_ltm() == len(values)
    assert len(store) == len(values)
    assert [store.row(index)['trust'] for index in range(len(store))] == pytest.approx(values)
    assert all(bool(store.episodic_values[index].any()) for index in range(len(store)))


def test_52_resubmission_appends_without_changing_trust_or_summing_existence(truth_model, monkeypatch):
    from ExtendedRecall import matches, existence
    model = truth_model
    store = model.symbolSpace.ltm_store
    meanings = []
    write = store.write_clause
    def observed(clause, **kwargs):
        meanings.append(clause.meaning)
        return write(clause, **kwargs)
    monkeypatch.setattr(store, 'write_clause', observed)
    entries = [dict(content='hello', trust=.3)]
    model.store_truths(entries)
    before = store.trust[:len(store)].clone()
    first = existence(store, matches(store, meanings[-1].extension))[0]
    assert tuple(first) == pytest.approx((.3, 0.))
    model.store_truths(entries)
    assert len(store) == 2 * len(before)
    torch.testing.assert_close(store.trust[:len(before)], before, rtol=0, atol=0)
    torch.testing.assert_close(existence(store, matches(store, meanings[-1].extension))[0], first)
    assert all(not hasattr(store, name) for name in
        ('set_trust', 'set_evidence', 'clear_origin', 'withdraw_origin'))


@pytest.mark.parametrize('a,b', ((.8, .6), (.4, .9)))
def test_53_infer_writes_the_minimum_of_its_addressed_premises(a, b):
    from test_thought_infer import setup, operands, invoke
    from ThoughtReferences import evidence_pair
    from ThoughtStream import write
    model, registry, store, refs = setup()
    premise, rule, rule_row = operands(model, registry, store, refs,
        premise_pair=(a, 0.), rule_pair=(b, 0.))
    result = invoke(model, registry, registry.form('infer', premise, rule))
    assert result.evidence['derivation']['match_support_true'] == 1
    assert evidence_pair(result.evidence['meaning']) == pytest.approx((min(a, b), 0.))
    before = len(store)
    written = write(model, result.evidence['meaning'], row=0)
    assert written == before and len(store) == before + 1
    assert store.row(written)['trust'] == pytest.approx(min(a, b))


def test_53_a_kept_both_lane_thought_writes_two_signed_occurrences():
    from test_thought_infer import setup, idea
    from ThoughtStream import write
    model, registry, store, refs = setup()
    meaning = idea(registry, refs[0], (.7, .3))
    before = len(store)
    write(model, meaning, row=0)
    assert len(store) == before + 2
    assert store.trust[before:len(store)].tolist() == pytest.approx([.7, -.3])
    assert store.row_name(before) < store.row_name(before+1)
    assert store.referent_of(before) == store.referent_of(before+1) == refs[0]


def test_54_truth_view_reads_nonzero_trust_and_reports_without_saved_sources(truth_model):
    model = truth_model
    store, truth = model.symbolSpace.ltm_store, model.symbolSpace.truth_layer
    model.store_truths([dict(content='hello', trust=.8), dict(content='hello', trust=-.6)])
    retained = store.trust[:len(store)].clone()
    conversation = len(store)
    model.infer('hello')
    assert len(store) > conversation and not store.trust[conversation:len(store)].any()
    truth.sync_from_ltm()
    selected = store.trust[:len(store)].ne(0).nonzero().flatten().tolist()[-truth.max_truths:]
    assert truth._sources == [store.occurrence_of(row) for row in selected]
    assert truth._trusts == pytest.approx([.8, -.6])
    score, contradictions = truth.consistency(return_report=True)
    assert contradictions and float(score) < 1
    assert truth.suggest_clarifications()
    assert all('origin' not in store.row(row) and 'text' not in store.row(row) for row in selected)
    torch.testing.assert_close(store.trust[:conversation], retained, rtol=0, atol=0)


def test_55_served_truths_run_the_activity_pass_before_writing_under_pressure(truth_model):
    from Forgetting import ForgettingConfig
    model = truth_model
    store = model.symbolSpace.ltm_store
    config = ForgettingConfig.from_xml()
    count = math.ceil(config.high_water * store.capacity)
    for _ in range(count):
        store.append_idea(torch.zeros(store.nDim), trust=0.)
    before = int(store.forget_passes)
    model.store_truths([dict(content='hello', trust=.3)])
    assert int(store.forget_passes) == before + 1
    assert len(store) == math.floor(config.low_water * store.capacity) + 1
    assert store.row(len(store)-1)['trust'] == pytest.approx(.3)


@pytest.mark.parametrize('conversation', (False, True))
def test_51_training_and_serving_keep_the_same_open_answer_and_different_authority(truth_model, monkeypatch, conversation):
    from Extension import Context, Node
    from Language import AskLayer
    from Queries import ThoughtLTMCapability, QueryWorkBudget
    from ThoughtReferences import with_slots
    from What import What
    model = truth_model
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    owner._trust = 1.
    names = [rule.method_name for rule in model.languageSpace._compose_binary_rules]
    selected = names.index('conjunction')
    # Supply a grammatical choice in the existing full menu. The two repeated
    # words have one object, so both operands have a nonempty intersection.
    def binary(x, candidates, *_args, **kwargs):
        scores = x.new_full(candidates.shape[:-1], -1e6)
        indices = kwargs.get('op_indices')
        if indices is None:
            scores[..., selected] = 0.
        else:
            scores = torch.where(indices == selected, 0., scores)
        return x.new_full((*x.shape[:2], 1), 1e6), scores
    monkeypatch.setattr(model.symbolSpace.languageLayer.operation_layer.chooser, 'score_binary', binary)
    closed = []
    write = store.write_clause
    def observed(clause, **kwargs):
        closed.append(clause)
        return write(clause, **kwargs)
    monkeypatch.setattr(store, 'write_clause', observed)
    if conversation:
        model.infer('hello hello')
    else:
        with model._runtime_batch(['hello hello']):
            inputs = model.inputSpace.prepInput(list(model.data.train_input))
            model.runBatch(train=True, optimizer=model.getOptimizer(lr=.001), batchSize=1,
                split='runtime', batch_override=(inputs, torch.empty(1, 0)), source_rows=[0],
                questions=(What.present(0, split='runtime'),))
    assert len(store) == 1 and len(closed) == 1
    clause = closed[0]
    certainty = max(clause.evidence)
    assert certainty > 0
    assert store.row(0)['trust'] == pytest.approx(0. if conversation else certainty)
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda a,b: 0., tau_id=.5)
    complete = reader.best_match(clause.meaning, max_records=8, work=QueryWorkBudget(128))
    assert (complete['support_true'], complete['support_false']) == pytest.approx(
        (0., 0.) if conversation else (certainty, 0.))
    root, = clause.meaning.extension
    assert root.context.name == 'conjunction' and len(root.children) == 2
    ask, context = AskLayer(), Context(name='ask', open=True)
    hole = Node(ask.compose_extend(root.children[1].field, context=context),
                (root.children[1],), ask, context)
    children = (root.children[0], hole)
    requested = replace(root, children=children, field=root.operator.compose_extend(
        *(child.field for child in children), context=root.context))
    question = with_slots(replace(clause.meaning, extension=(requested,)),
                          (('referent', 2),), pair=(0., 0.))
    generated = []
    native = model.generate
    rules = [rule.method_name for rule in model.languageSpace._generate_binary_rules]
    binary_index = rules.index('conjunction')
    stop = len(rules) + len(model.languageSpace._generate_unary_rules)
    def selected_inverse(*args, **kwargs):
        trace = torch.full((1, model._walk_budget()), -1, dtype=torch.long)
        trace[0, :3] = torch.tensor([binary_index, stop, stop])
        kwargs['walk_options'] = dict(kwargs.get('walk_options') or {}, kept_actions=trace)
        result = native(*args, **kwargs)
        generated.append(result)
        return result
    # This independent inverse is executed and checked by the native walk;
    # it is a supplied legal selection, not a saved tree on the retained row.
    monkeypatch.setattr(model, 'generate', selected_inverse)
    assert not store.meaning_of(0).extension
    opened = reader.best_match(question, max_records=8, work=QueryWorkBudget(128))
    assert generated and generated[-1].counts.tolist() == [2]
    assert opened['answer_operand'] is not None
    # A generated point can activate several concepts. The lexical readback
    # row does not name its answer; the open operand retains the whole set.
    expected = generated[-1].extension[0][0].operand((('conjunction', 1),))
    actual = opened['answer_operand'].extension[0].field
    ids = torch.cat((expected.ids, actual.ids)).unique()
    torch.testing.assert_close(actual.aligned(ids), expected.aligned(ids))
    torch.testing.assert_close(actual.aligned(root.children[1].field.ids),
                               root.children[1].field.values)
    assert int(actual.values.any(-1).sum()) > 1
    assert opened['reference'] is None
    assert opened['witnesses'] == (store.occurrence_of(0),)
    assert (opened['support_true'], opened['support_false']) == pytest.approx(
        (0., 0.) if conversation else (certainty, 0.))
