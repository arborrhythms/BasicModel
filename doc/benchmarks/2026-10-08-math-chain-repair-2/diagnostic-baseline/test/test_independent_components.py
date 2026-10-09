"""Item 6.5 mechanism certificates; no passing seed is selected."""
import itertools
from types import SimpleNamespace

import pytest
import torch


def test_definedness_keeps_direction_input_uncertainty_and_evidence_separate():
    from MeaningCodes import defined_code, definedness, compose
    from Interpret import activate_code
    direction = torch.tensor([1., 2., 0.])
    counts = torch.tensor([0., 1., 4., 16., 10000.], dtype=torch.float64)
    magnitudes = definedness(counts)
    assert (magnitudes.diff() > 0).all()
    torch.testing.assert_close(1 - magnitudes, 4 / (counts + 4))
    for n, pair in itertools.product([0, 1, 4, 16], itertools.product([0., 1.], repeat=2)):
        code = defined_code(direction, n)
        m = n / (n + 4)
        torch.testing.assert_close(code.norm(), code.new_tensor(m))
        if n:
            torch.testing.assert_close(code / code.norm(), direction / direction.norm())
        atoms = torch.cat((torch.tensor([1., 0.]), code, torch.zeros(3)))[None]
        leaf = activate_code(atoms, torch.tensor([.7]), 2, evidence=torch.tensor([pair]))
        torch.testing.assert_close(leaf[0, :2], torch.tensor([.7, 0.]))
        torch.testing.assert_close(leaf[0, 2:5], code * pair[0])
        torch.testing.assert_close(leaf[0, 5:], code * pair[1])
        conjunction = compose('conjunction', leaf[:, 2:], torch.ones(1, 6))
        assert conjunction[:, :3].norm() <= m + 1e-7
        input_value = .3 * direction / direction.norm()
        torch.testing.assert_close(input_value @ code, code.new_tensor(.3 * m))


def test_row_definedness_survives_rewitness_and_checkpoint_without_changing_truth():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    from MeaningCodes import sentence_key
    store = TernaryTruthStore(8, capacity=4)
    meaning = ConceptualMeaning.from_description(torch.ones(8))
    key = sentence_key([b'a', b'cat'])
    args = dict(document_key='fixture', sentence_index=0, content_key=key, evidence=(1., 1.))
    row = store.append_meaning(meaning, **args)
    first = store.identity_code(row, 64, 3)
    assert first.norm().item() == pytest.approx(1 / 5)
    store.append_meaning(meaning, **args)
    second = store.identity_code(row, 64, 3)
    torch.testing.assert_close(second / second.norm(), first / first.norm())
    assert second.norm().item() == pytest.approx(2 / 6)
    assert store.c_plus[row] == store.c_minus[row] == 1
    restored = TernaryTruthStore(8, capacity=4)
    restored.load_state_dict(store.state_dict())
    torch.testing.assert_close(restored.identity_code(row, 64, 3), second)


def test_word_magnitude_counts_containing_rows_and_excludes_definition_rows():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    from MeaningCodes import sentence_key
    from MereologicalCodes import MereologicalCodes
    store = TernaryTruthStore(20, capacity=8)
    owner = SimpleNamespace(_closed_clause_store=lambda: store,
        _csw_row_of=lambda cid: None, _definition_index=lambda: None)
    codes = MereologicalCodes(owner, torch.zeros(2, 20), percept_width=4, percept_event_width=4)
    for position, kind in enumerate((store.REL_NONE, store.REL_NONE, store.REL_DEF)):
        row = store.append_meaning(ConceptualMeaning.from_description(torch.ones(20)),
            rel_type=kind, document_key='fixture', sentence_index=position,
            content_key=sentence_key([str(position).encode()]))
        store._append_leaf_terms(row, ((0,), (), ()), (True, True, True))
    count, value = codes.occurrence_terms()[0]
    assert count == 2 and value.norm().item() == pytest.approx(2 / 6)
    assert not value[8:].any() and not value.requires_grad


def dictionary(width=4, sources=2):
    from IndependentComponents import SparseDictionary
    return SparseDictionary(width, max_sources=sources, prior_scale=.05,
                            mint_threshold=.2, recurrence=4)


def witness(dictionary, value, start=0, count=4):
    for occurrence in range(start, start + count):
        dictionary.observe(value, witness=occurrence)


def test_fixed_column_and_individuation_are_recurrence_effects():
    columns = dictionary()
    first, second = torch.eye(4)[:2]
    witness(columns, first, count=3)
    assert columns.ids == ()
    columns.observe(first, witness=3)
    assert len(columns.ids) == 1
    original = columns.ids[0]
    witness(columns, second, start=4, count=3)
    assert columns.ids == (original,)
    columns.observe(second, witness=7)
    assert len(columns.ids) == 2
    stream = torch.stack((first, second, first, torch.zeros_like(first), second))
    result = columns.encode(stream)
    assert result.codes[:, 0].ne(0).tolist() == [True, False, True, False, False]
    assert result.codes[:, 1].ne(0).tolist() == [False, True, False, False, True]
    assert columns.ids[0] == original
    torch.testing.assert_close(columns.matrix().norm(dim=-1), torch.ones(2))


def test_two_cats_share_a_word_but_have_distinct_recurring_properties():
    columns = dictionary()
    # Coordinate zero is the shared word; the next two are properties.
    first = torch.tensor([1., 1., 0., 0.]) / 2**.5
    second = torch.tensor([1., 0., 1., 0.]) / 2**.5
    witness(columns, first)
    witness(columns, second, start=10, count=3)
    assert len(columns.ids) == 1
    columns.observe(second, witness=13)
    assert len(columns.ids) == 2
    codes = columns.encode(torch.stack((first, second, first, first + second))).codes
    assert codes.ne(0).tolist() == [[True, False], [False, True], [True, False], [True, True]]


def test_mint_threshold_duplicate_witness_and_relevance_do_not_delete_columns():
    columns = dictionary()
    first, second = torch.eye(4)[:2]
    for _ in range(9):
        columns.observe(first, witness='same occurrence')
    assert not columns.ids
    witness(columns, first)
    assert len(columns.ids) == 1
    witness(columns, .01 * second, start=10)
    assert len(columns.ids) == 1
    with torch.no_grad():
        columns.relevance[str(columns.ids[0])].zero_()
    assert columns.pruning_candidates() == columns.ids
    assert len(columns.ids) == 1
    assert not columns.encode(first[None]).codes.any()


def test_population_loss_is_batch_partition_invariant_and_has_one_backward():
    columns = dictionary()
    witness(columns, torch.eye(4)[0])
    witness(columns, torch.eye(4)[1], start=10)
    population = torch.tensor([[1., 0., .2, 0.], [0., .7, 0., .1], [1., .8, 0., 0.]])
    whole = columns.loss(population)
    singles = torch.stack([columns.loss(row[None]) for row in population]).mean()
    chunks = (columns.loss(population[:2]) * 2 + columns.loss(population[2:])) / 3
    torch.testing.assert_close(whole, singles)
    torch.testing.assert_close(whole, chunks)
    calls = []
    handle = columns.directions[str(columns.ids[0])].register_hook(lambda grad: calls.append(grad))
    whole.backward()
    handle.remove()
    assert len(calls) == 1 and calls[0].abs().sum() > 0


def test_role_superposition_does_not_claim_role_binding():
    from IndependentComponents import noun_frame
    cat, verb, dog = torch.eye(4)[:3]
    first = torch.stack((cat, verb, dog))
    reversed_roles = torch.stack((dog, verb, cat))
    assert not torch.equal(first, reversed_roles)
    torch.testing.assert_close(noun_frame(first), noun_frame(reversed_roles))
    columns = dictionary()
    witness(columns, cat)
    witness(columns, dog, start=10)
    torch.testing.assert_close(columns.loss(noun_frame(first)[None]),
                               columns.loss(noun_frame(reversed_roles)[None]))


def test_change_dictionary_is_one_sparse_per_sentence():
    changes = dictionary(width=12, sources=1)
    first, second = torch.eye(12)[:2]
    witness(changes, first)
    witness(changes, second, start=10)
    result = changes.encode(torch.stack((first, second, first, first + second)))
    assert result.codes.ne(0).sum(-1).tolist() == [1, 1, 1, 1]
    assert result.codes[:3].argmax(-1).tolist() == [0, 1, 0]


def test_dictionary_checkpoint_preserves_pending_recurrence_and_learned_parameters():
    import copy
    source = dictionary()
    first, second = torch.eye(4)[:2]
    witness(source, first)
    witness(source, second, start=10, count=2)
    restored = dictionary()
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    assert restored.ids == source.ids
    torch.testing.assert_close(restored.encode(first[None]).codes, source.encode(first[None]).codes)
    witness(restored, second, start=12, count=2)
    assert len(restored.ids) == 2 and len(source.ids) == 1


def test_identity_candidates_share_the_operation_softmax_and_live_credit():
    from ReferenceContext import ReferenceBank, prepare_operands
    from Language import OperationSelectionLayer
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x
    rule = SimpleNamespace(reference_orders=(('I1', 1),),
                           reference_kinds=(('I1', 'particular'),))
    word = torch.tensor([[[0., 0., 1., 0.]]])
    columns = torch.eye(4)[:2][None]
    bank = ReferenceBank(torch.tensor([[11, 12]]), columns,
                         torch.ones(1, 2, dtype=torch.bool),
                         torch.zeros(1, 2, dtype=torch.bool), word[:, 0], torch.tensor([True]))
    live = (word, torch.tensor([[3]]), torch.tensor([[0]]),
            torch.tensor([[False]]), torch.tensor([[0]]))
    args = dict(rules=(), unary_rules=(rule,), bank=bank, live=live,
                active=torch.tensor([[True]]))
    candidates = prepare_operands(word, torch.tensor([[3]]), torch.tensor([[0]]),
                                  torch.tensor([[0]]), **args)
    refs = candidates['unary_refs'][0, 0, :, 0]
    legal = candidates['unary_valid'][0, 0]
    assert set(refs[legal].tolist()) == {-1, 11, 12}
    layer = OperationSelectionLayer(d_model=4, ops=(), unary_ops=(Identity(),), chooser='mlp')
    for identity in (11, 12, -1):
        action = ((refs == identity) & legal).nonzero()[0, 0].reshape(1)
        _, _, route = layer(word, reference_data=candidates, replay_action=action)
        assert route['refs'][0, 0] == identity and route['op'][0] == 0
        torch.testing.assert_close(route['probabilities'], route['logits'].softmax(-1))
        credit = route['logits'].log_softmax(-1).gather(1, action[:, None]).sum()
        grads = torch.autograd.grad(credit, tuple(layer.chooser.parameters()), allow_unused=True)
        assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)
    # A pronoun has precisely the bounded noun columns, with no dictionary
    # decision or mint fallback. Selection remains the same global action.
    rule.reference_kinds = (('I1', 'pronoun'),)
    pronoun = prepare_operands(word, torch.tensor([[3]]), torch.tensor([[0]]),
                               torch.tensor([[0]]), **args)
    assert set(pronoun['unary_refs'][0, 0, pronoun['unary_valid'][0, 0], 0].tolist()) == {11, 12}


def test_event_chart_excludes_address_bands_and_preserves_evidence_poles():
    from IndependentComponents import EventChart
    # The active meaning width may be narrower than the reserved pole stride.
    codes = torch.zeros(2, 16)
    codes[0, 4], codes[1, 5] = 1., 1.
    chart = EventChart(torch.tensor([1, 3]), codes, capacity=6, content_width=14,
                       meaning_start=4, meaning_pairs=2, pole_stride=5)
    value = torch.arange(16, dtype=torch.float32, requires_grad=True)
    torch.testing.assert_close(chart.project(value), value[[4, 5, 9, 10]])
    chart.project(value).square().sum().backward()
    assert value.grad.nonzero().flatten().tolist() == [4, 5, 9, 10]
    restored = chart.render(chart.project(value))
    assert restored.nonzero().flatten().tolist() == [4, 5, 9, 10]


def component_fixture():
    from IndependentComponents import IndependentComponents, EventChart
    owner = SimpleNamespace(nVectors=8, nWhat=4, stm=SimpleNamespace(capacity=8),
                            _csw_row_of=lambda identity: identity + 3)
    model = IndependentComponents(owner, weight=.1, prior_scale=.05, mint_threshold=.2)
    model._chart = EventChart(torch.arange(4), torch.eye(4), capacity=8, content_width=4)
    model._population = torch.zeros(0, 8)
    for index in range(2):
        witness(model.nouns, model._chart.expand(torch.cat((torch.eye(4)[index], torch.zeros(4)))),
                start=10 * index)
    return model


def test_prediction_reencodes_live_columns_but_detaches_targets_and_older_context():
    from Layers import BracketExpectation
    columns = component_fixture()
    predictor = BracketExpectation(n_symbols=4, max_depth=8, n_dim=4, concept_dim=4,
                                   batch=1, expectation_scope='structured')
    sources = []
    def encode(value):
        result = columns.source(value)
        result.retain_grad()
        sources.append(result)
        return result
    object.__setattr__(predictor, '_source_encoder', encode)
    older = torch.nn.Parameter(torch.ones(3, 4))
    current = torch.nn.Parameter(torch.eye(4)[:3])
    target = torch.nn.Parameter(torch.full((3, 4), .25))
    for value in (older, current, target):
        predictor.predict_and_observe_stm_end_state([3], [value], documents=['stream'], layout='infix')
    loss = predictor.consume_inter_loss()
    loss.backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in sources)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in columns.nouns.parameters())
    assert older.grad is current.grad is target.grad is None
    assert all(not roles.requires_grad for _, roles, _ in predictor.get_stm_chain())


def test_native_column_reads_use_witness_magnitude_and_only_situation_candidates():
    from ReferenceContext import SituationFrame
    columns = component_fixture()
    for identity in columns.nouns.ids:
        assert columns.column_point(identity).norm().item() == pytest.approx(.5)
    row = torch.eye(4)[0]
    roles = torch.stack((row, torch.zeros_like(row), torch.zeros_like(row)))
    frame = SituationFrame(1, roles, torch.tensor([True, False, False]), 91, row)
    assert columns.candidates((frame,)) == (columns.nouns.ids[0],)
    assert columns.candidates(()) == ()
    witness(columns.nouns, columns._chart.expand(columns._chart.project(row)), start=40)
    assert columns.column_point(columns.nouns.ids[0]).norm().item() == pytest.approx(8 / 12)


def test_kind_bind_and_mint_closings_obey_the_selected_choice():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    existing = columns.nouns.ids
    counts = [columns.nouns.witness_count(identity) for identity in existing]
    store = TernaryTruthStore(4, capacity=16)
    columns.owner._closed_clause_store = lambda: store
    columns._allocate = lambda direction: 100
    meaning = ConceptualMeaning.from_description(torch.eye(4)[2])
    for position in range(12):
        row = store.append_meaning(meaning, document_key='selected-readings',
            sentence_index=position, order=2 if position < 4 else 1, kind='observation')
        references = () if position < 4 else (existing[0],) if position < 8 else (-1,)
        columns.commit(row, meaning, individual_references=references)
        if position < 11:
            assert columns.nouns.ids == existing
        if position < 8:
            assert not columns.nouns._pending
    assert columns.nouns.ids == (*existing, 100)
    assert columns.nouns.witness_count(100) == 4
    assert [columns.nouns.witness_count(identity) for identity in existing] == counts


@pytest.mark.parametrize('mode,reference,expected', [
    ('mint', -1, (-1,)), ('bind', 77, (77,)), ('kind', -1, ()),
])
def test_selected_grammar_controls_admission_without_word_lookup(monkeypatch, mode, reference, expected):
    from dataclasses import replace
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ReferenceContext import selected_individual_references
    fixture = SentenceFixture(monkeypatch)
    def determined(tree, selected_mode):
        program = fixture.program(tree)
        actions = program.actions.clone()
        actions[-1, 1] = next(i for i, rule in enumerate(fixture.binary)
                              if rule.determiner_mode == selected_mode)
        return record_reading(fixture.language, replace(program, actions=actions))
    program = determined(('lower', 'marker', 'description'), mode)
    refs = program.operation_refs.clone()
    refs[-1, 1] = reference
    program = replace(program, operation_refs=refs)
    assert selected_individual_references(fixture.language, program) == expected
    # Enclosing extension scope must erase an inner mint request.
    nested = determined(('lower', 'marker', ('lower', 'another-marker', 'description')), 'kind')
    assert selected_individual_references(fixture.language, nested) == ()


def test_ltm_population_is_detached_and_independent_of_current_batch_partition(monkeypatch):
    import Spaces
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    store = TernaryTruthStore(4, capacity=8)
    for index, value in enumerate(torch.eye(4)[:3]):
        store.append_meaning(ConceptualMeaning.from_description(value),
                             document_key='stream', sentence_index=index, kind='observation')
    for index, kind in enumerate(('fact', 'estimate'), start=3):
        store.append_meaning(ConceptualMeaning.from_description(torch.ones(4)),
                             document_key='stream', sentence_index=index, kind=kind)
    owner = columns.owner
    owner._closed_clause_store = lambda: store
    owner._order0_inventory_row = lambda row: row < 4
    owner.similarity_codebook = SimpleNamespace(W=torch.eye(4), lookup_rows=lambda rows: torch.eye(4)[rows])
    monkeypatch.setattr(Spaces, '_concept_alloc_of', lambda _: SimpleNamespace(
        layer=lambda: SimpleNamespace(_tensor_row_keys={i: i for i in range(4)})))
    columns.begin_forward()
    assert columns._population.shape == (3, 8) and not columns._population.requires_grad
    meanings = [ConceptualMeaning.from_description(value) for value in torch.eye(4)[:2]]
    together = columns.cost(meanings, None, torch.tensor([True, True])).mean()
    separate = torch.stack([columns.cost([meaning], None, torch.tensor([True]))[0]
                            for meaning in meanings]).mean()
    torch.testing.assert_close(together, separate)
    columns._population = columns._population[:0]
    columns._population_limits = []
    assert not torch.isclose(columns.cost(meanings, None, torch.tensor([True, True])).mean(), together)


def test_unretained_innovation_cannot_change_population_cost(monkeypatch):
    import Spaces
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    store = TernaryTruthStore(4, capacity=8)
    meaning = ConceptualMeaning.from_description(torch.eye(4)[0])
    row = store.append_meaning(meaning, document_key='retained', sentence_index=0, kind='observation')
    owner = columns.owner
    owner._closed_clause_store = lambda: store
    owner._order0_inventory_row = lambda physical: physical < 4
    owner.similarity_codebook = SimpleNamespace(W=torch.eye(4), lookup_rows=lambda rows: torch.eye(4)[rows])
    monkeypatch.setattr(Spaces, '_concept_alloc_of', lambda _: SimpleNamespace(
        layer=lambda: SimpleNamespace(_tensor_row_keys={i: i for i in range(4)})))
    prediction = SimpleNamespace(roles=torch.zeros(3, 4))
    columns.begin_forward()
    expected = columns.cost([meaning], [prediction], torch.tensor([True]))
    retained = int(store.row_ids[row])
    absent = retained + 100
    columns._innovations[absent] = torch.ones(3 * columns.capacity)
    columns.begin_forward()
    actual = columns.cost([meaning], [prediction], torch.tensor([True]))
    torch.testing.assert_close(actual, expected)
    assert absent not in columns._innovations
    columns._innovations[retained] = torch.ones(3 * columns.capacity)
    columns.begin_forward()
    assert retained in columns._innovations
    assert not torch.allclose(columns.cost([meaning], [prediction], torch.tensor([True])), expected)


def test_continuity_is_binding_evidence_and_never_a_column_signature():
    from IndependentComponents import EventChart
    from ReferenceContext import SituationFrame
    columns = component_fixture()
    basis = torch.zeros(4, 12)
    basis[:, 4:8] = torch.eye(4)
    columns._chart = EventChart(torch.arange(4), basis, capacity=8, content_width=12,
                                meaning_start=4, meaning_pairs=4)
    columns.owner.similarity_codebook = SimpleNamespace(mereology=SimpleNamespace(
        percept_width=0, percept_event_width=4))
    identity = columns.nouns.ids[0]
    original = columns.column_point(identity).detach().clone()
    frame = SituationFrame(1, torch.zeros(3, 12), torch.tensor([True, False, False]),
                           91, None, where=torch.tensor([2., 3.]), when=torch.tensor([8., 9.]))
    located = columns.situated_point(identity, frame)
    torch.testing.assert_close(located[:4], torch.tensor([2., 3., 8., 9.]))
    torch.testing.assert_close(columns._chart.project(located), columns._chart.project(original))
    torch.testing.assert_close(columns.column_point(identity), original)


def test_prediction_role_cost_uses_column_coordinates_with_a_frozen_target():
    columns = component_fixture()
    prediction = torch.nn.Parameter(torch.tensor([[.8, .2, 0., 0.],
                                                  [.1, .7, 0., 0.], [.5, .1, 0., 0.]]))
    target = torch.nn.Parameter(torch.eye(4)[:3])
    actual, observed = columns.prediction_coordinates(prediction, target)
    assert actual.shape == observed.shape == (3, 2) and not observed.requires_grad
    (actual - observed).square().mean().backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert target.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in columns.nouns.parameters())


def test_relative_clause_sources_extend_the_ordinary_ceiling_only_to_stm_capacity():
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    columns = component_fixture()
    for index in (2, 3):
        witness(columns.nouns, columns._chart.expand(torch.cat((torch.eye(4)[index], torch.zeros(4)))),
                start=10 * index)
    meaning = ConceptualMeaning.from_description(torch.ones(4))
    leaf = Clause(meaning, point=meaning.roles[0])
    relative = Clause(meaning, point=meaning.roles[0], children=(leaf,))
    assert columns.source_limit(leaf) == 2 and columns.source_limit(relative) == 4
    deep = relative
    for _ in range(10):
        deep = Clause(meaning, point=meaning.roles[0], children=(deep,))
    assert columns.source_limit(deep) == 8
    observation = columns._chart.expand(columns._chart.project(torch.ones(4)))
    assert columns.nouns.encode(observation).codes.ne(0).sum() == 2
    assert columns.nouns.encode(observation, max_sources=columns.source_limit(relative)).codes.ne(0).sum() == 4
    population = torch.stack((observation, observation))
    pooled = columns.nouns.loss(population, source_limits=[2, 4])
    separate = (columns.nouns.loss(population[:1], source_limits=[2])
                + columns.nouns.loss(population[1:], source_limits=[4])) / 2
    torch.testing.assert_close(pooled, separate)


def test_temporal_shuffle_reports_loss_of_predictability_without_changing_identity(record_property):
    columns = dictionary(sources=1)
    first, second = torch.eye(4)[:2]
    witness(columns, first)
    witness(columns, second, start=10)
    sequence = torch.stack((first, first, first, second, second, second, first, first))
    shuffled = sequence[torch.tensor([0, 3, 1, 4, 6, 5, 2, 7])]
    series, control = columns.encode(sequence).codes, columns.encode(shuffled).codes
    record_property('ordered_persistence_error', float(series.detach().diff(dim=0).square().mean()))
    record_property('shuffled_persistence_error', float(control.detach().diff(dim=0).square().mean()))
    # This is a reported temporal diagnostic, not a learned-identity gate.
    torch.testing.assert_close(columns.loss(sequence), columns.loss(shuffled))


def test_native_admission_optimizer_ownership_and_checkpoint(tmp_path, monkeypatch):
    import copy
    from pathlib import Path
    import Language
    import Models
    import util
    from data import TheData
    from Spaces import _concept_alloc_of
    from IndependentComponents import EventChart
    source = Path(__file__).resolve().parents[1] / 'data' / 'XOR_grammar.xml'
    config = tmp_path / 'components.xml'
    config.write_text(source.read_text().replace('<nVectors>6</nVectors>', '<nVectors>32</nVectors>').replace('<architecture>',
        '<architecture><ltmConsolidation>true</ltmConsolidation>', 1))
    util.init_config(path=str(config), defaults_path=str(source.parent / 'model.xml'))
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load('xor', dat=dict(cfg['architecture']['data']))
    model, _ = Models.BaseModel.from_config(str(config), data=TheData)
    try:
        owner = model._concept_owner()
        columns = owner.components
        event = owner.new_concept()
        event_row = owner._csw_concept_row(0, event)
        width = owner.similarity_codebook.W.shape[-1]
        code = torch.zeros(1, width)
        derived = owner.similarity_codebook.mereology
        code[0, derived.percept_event_width] = 1.
        columns._chart = EventChart(torch.tensor([event_row]), code, capacity=owner.nVectors,
            content_width=owner.nWhat, meaning_start=derived.percept_event_width,
            meaning_pairs=derived.meaning_pairs, pole_stride=derived.reserved_pairs)
        value = columns._chart.expand(torch.tensor([1., 0.]))
        for occurrence in range(4):
            columns.nouns.observe(value, witness=occurrence, allocate=columns._allocate)
        identity, = columns.nouns.ids
        assert _concept_alloc_of(owner).order_of(identity) == 1
        assert owner._csw_row_of(identity) != event_row
        read = owner.similarity_codebook.lookup_rows(torch.tensor(owner._csw_row_of(identity)))
        torch.testing.assert_close(read[derived.percept_event_width:],
                                   columns.column_point(identity)[derived.percept_event_width:])
        optimizer = model.getOptimizer(lr=.01)
        owners = model.objective_parameter_groups(optimizer)
        ids = {id(p) for p in columns.parameters()}
        assert ids <= {id(p) for p in owners['expectation']}
        assert all(not ids.intersection(id(p) for p in params)
                   for name, params in owners.items() if name != 'expectation')
        state = copy.deepcopy(columns.state_dict())
        from IndependentComponents import IndependentComponents
        restored = IndependentComponents(owner, weight=.1, prior_scale=.05, mint_threshold=.2)
        restored.load_state_dict(state)
        assert restored.nouns.ids == (identity,)
        torch.testing.assert_close(restored.nouns.encode(value).codes, columns.nouns.encode(value).codes)
        assert restored.nouns.witness_count(identity) == 4
        path = tmp_path / 'admitted-columns.ckpt'
        model.save_weights(path)
        restored_model, _ = Models.BaseModel.from_config(str(config), data=TheData)
        try:
            assert restored_model.load_weights(path, strict=True, require_match=True)
            restored_owner = restored_model._concept_owner()
            assert restored_owner.components.nouns.ids == (identity,)
            assert restored_owner._csw_row_of(identity) == owner._csw_row_of(identity)
            assert restored_owner.components.nouns.witness_count(identity) == 4
            torch.testing.assert_close(restored_owner.components.nouns.encode(value).codes,
                                       columns.nouns.encode(value).codes)
        finally:
            restored_model.End()
            restored_model.symbolSpace.soft_reset()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
