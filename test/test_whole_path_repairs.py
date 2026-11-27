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
    __import__('clock_fixtures').install(model, 7)
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


def test_only_the_admitted_reconstruction_trains_at_sentence_exit():
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
    torch.testing.assert_close(costs['reconstruction'], parameter)
    for name in ('expectation', 'compose_lesson', 'generate_lesson',
                 'attention', 'penalty.reconstruction', 'penalty.expectation', 'penalty.attention'):
        torch.testing.assert_close(costs[name], 2 * parameter)
    torch.testing.assert_close(costs['output'], parameter)
    torch.testing.assert_close(costs['penalty.output'], parameter)
    assert weights.tolist() == [[1., 0., 0.]]


def test_reconstruction_gradient_follows_each_streams_selected_path():
    from Layers import Error
    from WholePaths import objectives
    active = torch.tensor([True, True, False])
    values = [torch.ones(3, requires_grad=True) for _ in range(3)]
    paths = []
    for value in values:
        errors = Error(row_mask=active)
        errors.add('reconstruction_order_zero', value, category='reconstruction')
        paths.append(NS(registry=errors))
    parts = torch.zeros(3, 3, 3)
    parts[:, :, 0] = torch.tensor([[3., 2., 1.], [1., 2., 3.], [2., 1., 3.]])
    draws = [dict(compose_round=torch.full((3,), -1)) for _ in range(2)]
    costs, _ = objectives(paths, parts, active, draws)
    costs['reconstruction'].backward()
    torch.testing.assert_close(values[0].grad, torch.tensor([0., .5, 0.]))
    assert values[1].grad is None
    torch.testing.assert_close(values[2].grad, torch.tensor([.5, 0., 0.]))


def test_no_departure_does_not_share_later_registry_writes():
    from Layers import Error
    from WholePaths import Path, Thinking, finish
    active = torch.ones(1, dtype=torch.bool)
    registry = Error(row_mask=active)
    registry.add('reconstruction', torch.tensor([2.]), objective='reconstruction')
    base = Path((), (), {}, registry, torch.tensor([[2., 0., 0.]]), Thinking(), None, None)
    other = finish(None, (), 0, active, base=base, draw={'walk': torch.tensor([-1])}, trust=1.)
    registry.add('attention.output', torch.tensor([3.]), objective='attention')
    assert 'attention.output' not in other.registry._terms
    assert other.registry.total(objective='reconstruction') == 2


def test_attention_outcome_credits_only_the_sampled_departure():
    from Layers import Error
    from WholePaths import credit
    # One cheaper answer, one costlier answer, one tie. Changing the
    # expectation column alone must leave the outcome gradient zero.
    scores = torch.zeros(3, 2, 3, requires_grad=True)
    probabilities = scores.softmax(-1)[..., 2]
    rounds = torch.tensor([0, 1, 1])
    proposal = torch.zeros_like(probabilities)
    proposal[torch.arange(3), rounds] = .5
    reading = NS(actions=torch.full((3, 2), 2), probabilities=probabilities, proposal=proposal)
    draw = dict(attention_round=rounds, thought_round=torch.full_like(rounds, -1),
                walk_count=torch.ones_like(rounds), walk_rounds=torch.full_like(rounds, 2))
    active = torch.ones(3, dtype=torch.bool)
    base = NS(parts=torch.ones(3, 3), output_trace=None)
    other = NS(parts=torch.tensor([[1., 1., 0.], [1., 1., 2.], [1., 0., 1.]]),
               output_trace=None, attention_words=reading, draw=draw, registry=Error(row_mask=active))
    credit(None, base, other, active)
    other.registry.total(objective='attention').sum().backward()
    expected = torch.zeros_like(scores)
    # Two possible rounds, proposal probability 1/2. The gradient of
    # softmax at these equal scores is (-1, -1, 2)/9 for the sampled action.
    expected[1, 1] = scores.new_tensor([-1., -1., 2.]) * (4. / 9.)
    expected[0, 0] = -expected[1, 1]
    torch.testing.assert_close(scores.grad, expected)


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
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
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
            narrowing=torch.zeros_like(active), compose_round=at, interpret_round=torch.full_like(at, -1),
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
    nothing = NS(amounts=torch.zeros(1, 2, 256))
    first = NS(_reconstruction_percepts=predicted, materialize=lambda: predicted.transpose(1, 2),
               _radix_activity=nothing)
    model = NS(_staged_concepts_in=observed.transpose(1, 2),
        _omission_targets={'raw': observed.transpose(1, 2), 'bytes': torch.tensor([[False, True]])},
        loss=NS(reconstruction_scale=1.), _input_reconstructions={},
        perceptualSpace=NS(percept_store=NS(get_id=lambda raw: raw[0] if raw[0] < 16 else None,
                                            inverse_table=[bytes([v]) for v in range(16)],
                                            _basis=NS(lookup_rows=lambda rows: torch.eye(16)[rows]))))
    def loss(value, target, *, registry, name, weight):
        registry.add(name + '.what', (value - target).square().sum((1, 2)), objective='reconstruction')
    model._reverse_event_loss = loss
    record = NS(open_field=field, word_values=observed, word_valid=torch.ones(1, 2, dtype=torch.bool),
                percept_expected=torch.zeros_like(observed), depths=torch.ones(1, 1), sentence=0)
    errors = Error(row_mask=active)
    SentenceField.charge(model, errors, 'reconstruction_order_zero', first, field, active, hidden=False)
    SentenceField.reconstruct(model, record, errors, active, second=first)
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
        assert b'gamma' not in identity.words
        # Every delivered byte is seated before the sentence, a hidden one too.
        assert all(part in identity.atoms for part in parts)
        ids = model.inputSpace._ar_word_part_ids[0, 1]
        assert int(ids[0]) == -1 and bool((ids[1:len(parts)] >= 0).all())
        assert bool(model.inputSpace._ar_embedded_N[0, 1, :identity.width].any())
        # The hidden target is the word's visible reading: its cover by the existing byte parts.
        ps = model.perceptualSpace
        rows = ps.form_part_rows[torch.tensor(list(b'gamma'))][None, None]
        offsets = torch.arange(len(b'gamma'))[None, None]
        visible = ps.synthesize_word_parts(rows, torch.ones_like(rows, dtype=torch.bool), offsets)
        torch.testing.assert_close(model._omission_targets['word_values'][0, 1, :ps.nWhat],
                                   visible[0, 0, :ps.nWhat], rtol=0, atol=0)
        assert model._omission_targets['word_bytes'][0][1] == b'gamma'
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_extension_attachment_keeps_reference_plans_on_the_scored_clause(tmp_path, monkeypatch):
    from test_item18_expect import lesson
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from Understanding import ConceptualField
    from SentenceUnderstanding import SentenceUnderstanding
    import Models
    model = lesson(tmp_path, monkeypatch)
    owner = model._concept_owner()
    point = owner.similarity_codebook.W.new_zeros(owner.nDim)
    point[0] = 1.
    # The ownership mechanism receives a declared completed binding. The
    # learned chooser is not required to create a reference site for this test.
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
        composition=Composition(point, order=1), reference_request=dict(mode='bind',
            leaves=(0, 1, 2, 3), cue=point*0, predicate=point, content=point,
            role=0, compose_bind=0))
    original_boundary = Models._boundary_observation_view
    def boundary(*args, clauses, **kwargs):
        result = original_boundary(*args, clauses=clauses, **kwargs)
        clauses[0] = clause
        return result
    monkeypatch.setattr(Models, '_boundary_observation_view', boundary)
    monkeypatch.setattr(model, '_capture_reading_programs', lambda **kw: (None, {0: (None,)}))
    root = point[None]
    depth = torch.ones(1, dtype=torch.long)
    lang = [None]*31
    lang[6], lang[10], lang[11] = root[:, None], root.repeat(1,3)[:, None], depth[:, None]
    # This fixture supplies its completed clause directly, with no arrivals
    # or chooser events. Supply the native carry's empty cost columns too.
    model.inputSpace._word_active_mask = torch.ones(1, 4, dtype=torch.bool)
    model.inputSpace._packed_sentence_ids = torch.zeros(1, 4, dtype=torch.long)
    lang[15] = torch.full((1, 12), -1, dtype=torch.long)
    lang[28] = torch.full((1, 12), -1, dtype=torch.long)
    lang[29] = torch.full((1, 4, 3), -1, dtype=torch.long)
    lang[30] = torch.zeros(1, 12, dtype=torch.long)
    field = ConceptualField(torch.empty(0,dtype=torch.long), point.new_zeros(0,1,4,2),
        point.new_zeros(1,4,2), point.new_zeros(1,4,2), None)
    record = SentenceUnderstanding(root,root[:,None].expand(-1,3,-1),depth,root[:,None],
        depth[:,None],root[:,None].expand(-1,4,-1),torch.full((1,4),-1),
        torch.ones(1,4,dtype=torch.bool),None,torch.tensor(0),open_field=field)
    model._sentence_trial = 'base'
    model._component_reading = True
    checked = []
    try:
        observation = model._sentence_observation((tuple(v[:1] for v in owner.stm.state),lang,None),0,
            torch.ones(1,dtype=torch.bool),record=record)
        for scored, plan in zip(observation['clauses'], observation['reference_plans']):
            nodes = set()
            def visit(node):
                if node is None or id(node) in nodes:
                    return
                nodes.add(id(node))
                for child in (*node.children, *node.companions):
                    visit(child)
            visit(scored)
            for item in plan['items']:
                if item.get('parent') is None and not item.get('lexical'):
                    assert id(item.get('source_node', item['node'])) in nodes
                    checked.append(item)
        assert checked
        assert observation['clauses'][0] is not clause
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_sentence_inverts_the_live_open_field_once_before_its_looks(tmp_path, monkeypatch, eager_reading):
    import SentenceField
    import WholePaths
    from SentenceLooks import Looks
    from test_item17_echo import echo
    from test_item17_places import model_for, nominate

    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('Felix', 'This', 'is', 'This is Felix.'))
        data = model.inputSpace.data
        monkeypatch.setattr(data, 'has_supervised_outputs', True)
        monkeypatch.setattr(data, 'text_answers', {'train': ['Felix'], 'validation': []})
        reconstruct, look, objectives = SentenceField.reconstruction, Looks.run, WholePaths.objectives
        opened, events, returned = [], [], []

        def inverted(owner, field, **kwargs):
            surface = reconstruct(owner, field, **kwargs)
            opened.append(field)
            returned.append(surface.materialize().detach().clone())
            events.append('inverse')
            return surface

        def observed(path, arrive, **kwargs):
            assert path.model._sentence_open_fields[path.sid] is opened[0]
            events.append('look')
            return look(path, arrive, **kwargs)

        def measured(paths, parts, active, draws, **kwargs):
            costs, weights = objectives(paths, parts, active, draws, **kwargs)
            grad, = torch.autograd.grad(costs['reconstruction'], opened[0].evidence,
                                        retain_graph=True)
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            for path in paths:
                torch.testing.assert_close(path.observation['input_reconstruction'].event, returned[0])
            events.append('cost')
            return costs, weights

        monkeypatch.setattr(SentenceField, 'reconstruction', inverted)
        monkeypatch.setattr(Looks, 'run', observed)
        monkeypatch.setattr(WholePaths, 'objectives', measured)
        echo(model, 0, 'This is Felix.', train=True, optimizer=model.getOptimizer())
        assert events.count('inverse') == 1 and 'look' in events
        assert events[0] == 'inverse' and events[-1] == 'cost'
    finally:
        model.End()
