"""Producer certificates for the five findings of the whole-path review."""
from types import SimpleNamespace as NS, MethodType

import torch


def test_packed_output_retains_each_streams_last_sentence():
    from Output import AnswerConstruction
    from WholePaths import keep_output

    model = NS(inputSpace=NS(_packed_sentence_slot_mask=torch.tensor([[True, False], [True, True]])))
    def path(values, texts, actions):
        x = torch.tensor(values, dtype=torch.float32)[:, None, :]
        return NS(construction=AnswerConstruction(x, None, x, x, x, (), texts),
                  output_trace=(torch.tensor(actions),))
    first = path([[1., 2.], [3., 4.]], ('first', 'earlier'), [[1, 2], [3, 4]])
    second = path([[0., 0., 0.], [5., 6., 7.]], ('', 'last'), [[-1, -1, -1], [5, 6, 7]])
    keep_output(model, [first], torch.zeros(2, dtype=torch.long), 0)
    keep_output(model, [second], torch.zeros(2, dtype=torch.long), 1)
    assert model._kept_path_output.texts == ('first', 'last')
    for name in ('actual', 'concepts', 'percepts', 'surface'):
        torch.testing.assert_close(getattr(model._kept_path_output, name)[:, 0],
                                   torch.tensor([[1., 2., 0.], [5., 6., 7.]]))
    assert model._kept_path_output_actions.tolist() == [[1, 2, -1], [5, 6, 7]]


def test_thought_departure_preserves_the_other_streams_reading():
    from Memory import ShortTermMemory
    from Models import BasicModel
    from WalkTrials import ThoughtTrialState
    stm = ShortTermMemory(batch=2, capacity=2, concept_dim=4)
    memory = NS(_what_slots=[{'value': 2}, {'value': 3}], _address_sources={0: 'first', 1: 'second'})
    owner = NS(stm=stm, subspace=NS(), _cs_field_where=torch.tensor([[[2., 3.]], [[3., 4.]]]),
               _cs_field_when=torch.tensor([[[7., 7.]], [[7., 7.]]]))
    model = NS(when_time=torch.tensor(7), when_encoding=NS(t=7),
               conceptualSpace=owner, _what_memory=lambda: memory,
               _assert_query_boundary=lambda row: None)
    model.present = lambda: int(model.when_time)
    model.run_selected_thought = MethodType(BasicModel.run_selected_thought, model)
    fork = dict(state=ThoughtTrialState(model), work=object())
    # Row 1's reading departure is already complete when row 0 rethinks.
    memory._what_slots = [{'value': 20}, {'value': 30}]
    owner._cs_field_where = torch.tensor([[[20., 21.]], [[30., 31.]]])
    model.when_time.fill_(11)
    model._path_thinking = NS(thought_trace=lambda row: ({}, fork if row == 0 else None))
    observed = []
    def run(meaning, *, row, **kwargs):
        observed.append((row, memory._what_slots[row]['value']))
        return NS(records=())
    model._run_selected_thought_once = run
    model.run_selected_thought(None, row=0)
    model.run_selected_thought(None, row=1)
    assert observed == [(0, 2), (1, 30)]
    torch.testing.assert_close(owner._cs_field_where, torch.tensor([[[2., 3.]], [[30., 31.]]]))
    assert model.present() == 11


def test_every_all_path_cost_has_two_total_exposures():
    from Layers import Error
    from WholePaths import objectives
    active = torch.ones(1, dtype=torch.bool)
    parameter = torch.nn.Parameter(torch.tensor(1.))
    paths = []
    for _ in range(3):
        errors = Error(row_mask=active)
        for name, category, owner in (
            ('reconstruction', 'reconstruction', 'reconstruction'),
            ('expectation', 'expectation', 'expectation'),
            ('compose.lesson', 'grammar', 'compose_lesson'),
            ('generate.lesson', 'grammar', 'generate_lesson'),
            ('attention', 'other', 'attention'),
            ('regularizer', 'reg', 'reconstruction')):
            errors.add(name, parameter[None], category=category, objective=owner)
        errors.add('answer', parameter[None], category='prediction', objective='output')
        paths.append(NS(registry=errors))
    parts = torch.tensor([[[1., 1., 1.], [2., 1., 1.], [3., 1., 1.]]])
    draws = [dict(compose_round=torch.tensor([-1])) for _ in range(2)]
    costs, weights = objectives(paths, parts, active, draws)
    for name in ('reconstruction', 'expectation', 'compose_lesson', 'generate_lesson',
                 'attention', 'penalty.reconstruction', 'penalty.expectation', 'penalty.attention'):
        torch.testing.assert_close(costs[name], 2 * parameter)
    torch.testing.assert_close(costs['output'], parameter)
    torch.testing.assert_close(costs['penalty.output'], parameter)
    assert weights.tolist() == [[1., 0., 0.]]


def test_no_departure_does_not_share_later_registry_writes():
    from Layers import Error
    from WholePaths import Path, Thinking, finish
    active = torch.ones(1, dtype=torch.bool)
    registry = Error(row_mask=active)
    registry.add('reconstruction', torch.tensor([2.]), objective='reconstruction')
    base = Path((), (), {}, None, registry, torch.tensor([[2., 0., 0.]]), Thinking(), None, None)
    other = finish(None, (), 0, active, base=base, draw={'walk': torch.tensor([-1])})
    registry.add('attention.reading_lesson', torch.tensor([3.]), objective='attention')
    assert 'attention.reading_lesson' not in other.registry._terms
    assert other.registry.total(objective='reconstruction') == 2


def test_two_explore_paths_credit_their_own_compose_and_attention_state(tmp_path, monkeypatch):
    import WholePaths
    import SentenceCredit
    import util
    from test_compiled_word_chunk import _tiny_canonical_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'explorePaths': 2, 'omissionTarget': 1})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    captured, credited = {}, []
    original_finish, original_credit = WholePaths.finish, model._compose_score_function_loss
    def finish(*args, **kwargs):
        result = original_finish(*args, **kwargs)
        forced = model._compose_forced_slots
        captured[id(result.registry)] = (None if forced is None else forced.clone(), model._attention_words)
        return result
    def departure(narrowing, compose, *, active, used=(), **kwargs):
        rounds = compose[0].nonzero().flatten()
        assert len(rounds) >= 2, 'the fixture must expose two legal compose departures'
        at = rounds[len(used)].reshape(1)
        result = {name + '_round': torch.full_like(at, -1) for name in SentenceCredit.WALKS}
        result.update(walk=torch.ones_like(at), rounds=compose.sum(-1),
            walk_count=torch.ones_like(at), walk_rounds=compose.sum(-1) - len(used),
            narrowing=torch.zeros_like(active), compose_round=at,
            round=at + (0 if narrowing is None else narrowing.alternatives.shape[1]))
        return result
    def credit(state, costs):
        forced, reading = captured[id(model._sentence_cost_registry)]
        torch.testing.assert_close(model._compose_forced_slots, forced)
        assert model._attention_words is reading
        credited.append(forced.clone())
        return original_credit(state, costs)
    monkeypatch.setattr(WholePaths, 'finish', finish)
    monkeypatch.setattr(SentenceCredit, 'departure', departure)
    monkeypatch.setattr(model, '_compose_score_function_loss', credit)
    try:
        source = model.inputSpace.prepInput(['alpha gamma alpha gamma'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        assert len(credited) == 2 and not torch.equal(*credited)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_free_percept_reconstruction_excludes_omitted_places(monkeypatch):
    import SentenceField
    from Layers import Error
    from Understanding import ConceptualField
    active = torch.ones(1, dtype=torch.bool)
    predicted = torch.tensor([[[2.], [0.]]], requires_grad=True)
    observed = torch.tensor([[[1.], [9.]]])
    field = ConceptualField(torch.tensor([0]), torch.ones(1, 1, 2, 2),
        torch.tensor([[[0., 1.], [1., 2.]]]), torch.ones(1, 2, 2), None)
    first = NS(_reconstruction_percepts=predicted, materialize=lambda: predicted.transpose(1, 2))
    second = NS(_reconstruction_percepts=observed)
    generation = NS(field=field, trace=(torch.zeros(1, 1), None, None, torch.zeros(1, 3), None, None, None),
                    words=torch.zeros(1, 2), counts=torch.tensor([2]), truncated=torch.zeros(1, dtype=torch.bool))
    monkeypatch.setattr(SentenceField, 'reconstruct_generation', lambda *a, **k: (generation, second))
    monkeypatch.setattr(SentenceField, 'publish', lambda *a: None)
    model = NS(_staged_concepts_in=observed.transpose(1, 2),
        _omission_targets={'raw': observed.transpose(1, 2), 'bytes': torch.tensor([[False, True]])},
        loss=NS(reconstruction_scale=1.), _input_reconstructions={})
    def loss(value, target, *, registry, name, weight):
        registry.add(name + '.what', (value - target).square().sum((1, 2)), objective='reconstruction')
    model._reverse_event_loss = loss
    record = NS(open_field=field, word_values=observed, word_valid=torch.ones(1, 2, dtype=torch.bool),
                percept_expected=torch.zeros_like(observed), depths=torch.ones(1, 1), sentence=0)
    errors = Error(row_mask=active)
    SentenceField.reconstruct(model, record, errors, active, previous=(first, observed))
    value = errors._value(errors._terms['reconstruction.free_percept']).sum()
    torch.testing.assert_close(value, torch.tensor(1.))
    gradient, = torch.autograd.grad(value, predicted)
    torch.testing.assert_close(gradient, torch.tensor([[[2.], [0.]]]))


def test_partial_omission_admits_only_the_visible_form_parts(tmp_path, monkeypatch):
    from Omission import Draw
    from WordIdentity import base_atoms
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'explorePaths': 0})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    def hide(valid, *, training):
        forms = torch.zeros_like(valid)
        forms[:, 1, 0] = valid[:, 1, 0]
        return Draw(torch.zeros(valid.shape[:2], dtype=torch.bool), forms,
                    torch.zeros(len(valid), dtype=torch.long))
    monkeypatch.setattr(model.omission, 'draw', hide)
    try:
        source = model.inputSpace.prepInput(['alpha gamma'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        identity = model.perceptualSpace.percept_store.identity
        parts = base_atoms(b'gamma')
        assert b'gamma' not in identity.words and b'gamma' not in identity.word_witnesses
        assert parts[0] not in identity.atoms
        assert all(part in identity.atoms for part in parts[1:])
        ids = model.inputSpace._ar_word_part_ids[0, 1]
        assert int(ids[0]) == -1 and bool((ids[1:len(parts)] >= 0).all())
        assert bool(model.inputSpace._ar_embedded_N[0, 1, :identity.width].any())
        torch.testing.assert_close(model._omission_targets['word_values'][0, 1, :identity.width],
                                   identity.form(b'gamma'))
        assert model._omission_targets['word_bytes'][0][1] == b'gamma'
    finally:
        model.End()
        model.symbolSpace.soft_reset()
