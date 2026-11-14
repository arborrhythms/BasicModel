"""The whole derivation owns its keep, exposure, and single departure credit."""
import pytest
import torch


def test_thought_content_growth_preserves_structure_and_trial_rollback():
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtStrata import construct, bound_to_sentence, write, snapshot, restore
    stores = construct(1, 2, 4)
    bound_to_sentence(stores, words=3, operators=3)
    store = stores[0][0]
    def nested(count):
        value = ConceptualMeaning.from_description(torch.ones(4))
        for _ in range(count - 1):
            value = replace(value, role_refs=(('constituent', 0), None, None),
                            constituents=(value,))
        return value
    first = write(store, nested(1), key=(2, 2, 2, 2))
    saved = snapshot(stores)
    rng = torch.random.get_rng_state().clone()
    second = write(store, nested(6), key=(3, 2, 2, 2))
    assert store.capacity == store.parts.capacity == 12
    assert len(store) == 12 and len(store.parts) == 11
    assert store._thought_capacity_bound == 4 * (3 + 3)
    assert torch.equal(torch.random.get_rng_state(), rng)
    value, count = second.read(), 1
    while value.constituents:
        value, count = value.constituents[0], count + 1
    assert count == 6
    with pytest.raises(ValueError, match='expired'):
        first.read()
    restore(stores, saved)
    assert store.capacity == 8 and len(store) == 2
    torch.testing.assert_close(first.read().roles, nested(1).roles)
    with pytest.raises(OverflowError, match='declared sentence bound'):
        write(store, nested(13), key=(4, 2, 2, 2))
    torch.testing.assert_close(first.read().roles, nested(1).roles)


def test_native_thought_bound_uses_the_declared_sentence_structure(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
                                   concept_rows=32)
    try:
        stm = model.conceptualSpace.stm
        bound = 4 * (stm.strata.words + stm.strata.operators)
        assert all(store._thought_capacity_bound == bound
                   for row in stm._thought_stores for store in row)
        assert all(store.capacity == 4 * stm.capacity
                   for row in stm._thought_stores for store in row)
        assert bound > 4 * stm.capacity
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_retained_named_transient_captures_its_concept_not_its_mtm_address():
    from types import SimpleNamespace
    from Models import BasicModel
    from SentenceStrata import SentenceStrata
    token = SimpleNamespace(order=1, pending=False, transient=91)
    pool = SimpleNamespace(entries={'retained': token}, reading_of=lambda key: 17)
    strata = SentenceStrata(1, 1, 2, 4)
    model = SimpleNamespace(_reconstruction_stack=lambda: SimpleNamespace(),
        _concept_owner=lambda: SimpleNamespace(mtm=pool),
        inputSpace=SimpleNamespace(_ar_word_concept_ids=torch.tensor([[17]])),
        symbolSpace=SimpleNamespace(), languageSpace=SimpleNamespace(),
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(strata=strata)))
    positions = torch.tensor([[0]])
    values = torch.ones(1, 1, 4)
    program = (positions, torch.tensor([[[0, -1, 0]]]), values, torch.tensor([[-1]]))
    entry, = BasicModel._program_entries(model, program, values, positions, positions,
        values, values, concept_ids=torch.tensor([[91]]))
    assert entry.concept_ids.tolist() == [17]
    assert entry.word_ids.tolist() == [17]
    torch.testing.assert_close(entry.leaves, values[0])
    assert pool.entries['retained'] is token and token.transient == 91


def test_complete_path_cost_and_compose_exposure():
    from SentenceCredit import comparison, reader_weights, score_function
    # The reconstruction departure improves R but damages its eventual answer.
    parts = torch.tensor([[[3., 100., 1.], [1., 0., 5.], [2., 0., 1.]],
                          [[1., 0., 1.], [1., 9., 1.], [3., 0., 0.]]])
    active = torch.ones(2, dtype=torch.bool)
    draws = [dict(compose_round=torch.tensor([-1, 2])), dict(compose_round=torch.tensor([3, -1]))]
    audit = comparison(parts, active)
    assert audit['winner'].tolist() == [2, 0]
    torch.testing.assert_close(reader_weights(parts, active, draws),
                               torch.tensor([[.5, 0., .5], [.5, .5, 0.]]))
    probability = torch.tensor([[.3], [.4]], requires_grad=True)
    credit, _ = score_function(probability, torch.ones_like(probability),
        active[:, None], audit['costs'][:, :2])
    credit.sum().backward()
    torch.testing.assert_close(probability.grad, torch.tensor([[2.], [0.]]))


def test_departures_are_distinct_and_correct_remaining_proposal():
    from SentenceCredit import departure, WALKS
    active = torch.ones(64, dtype=torch.bool)
    empty = active[:, None][:, :0]
    eligible = active[:, None].expand(-1, 3)
    draws = []
    for count in range(6):
        draw = departure(None, empty, active=active, reconstruction=eligible,
                         output=eligible, used=draws)
        for previous in draws:
            assert not bool(((previous['walk'] == draw['walk']) &
                ((previous['reconstruction_round'] == draw['reconstruction_round']) &
                 (previous['output_round'] == draw['output_round']))).any())
        assert draw['rounds'].tolist() == [6-count] * len(active)
        assert bool((draw['walk_rounds'] > 0).all())
        draws.append(draw)
    exhausted = departure(None, empty, active=active, reconstruction=eligible,
                          output=eligible, used=draws)
    assert exhausted['walk'].tolist() == [-1] * len(active)


def test_omission_protects_a_whole_word_and_never_draws_padding():
    from Omission import Omission
    policy = Omission(.9, 1., 1024)
    policy.rates.fill_(1.)
    valid = torch.tensor([[[1, 1, 0], [1, 1, 1], [0, 0, 0]]], dtype=torch.bool).expand(32, -1, -1)
    draw = policy.draw(valid, training=True)
    assert not bool((draw.forms & ~valid).any())
    assert not bool(draw.forms[torch.arange(len(valid)), draw.protected].any())
    assert draw.words.sum(-1).tolist() == [1] * len(valid)
    assert not bool(policy.draw(valid, training=False).forms.any())


def test_omission_rates_checkpoint_and_disabled_rng_identity():
    from Omission import Omission
    visible, omitted = torch.tensor([.8, 1., .95]), torch.tensor([0., 0., .91])
    torch.testing.assert_close(Omission.rate(visible, omitted, .9, .5), torch.tensor([0., .1, .5]))
    policy = Omission(1., .5, 1024)
    policy.rates.fill_(.5)
    policy.accuracies.copy_(torch.tensor([[.95, .2], [.97, .4]]))
    policy.observations.fill_(10)
    before = torch.random.get_rng_state().clone()
    draw = policy.draw(torch.ones(2, 3, 4, dtype=torch.bool), training=True)
    assert not bool(draw.forms.any())
    assert torch.equal(torch.random.get_rng_state(), before)
    restored = Omission(1., .5, 1024)
    restored.load_state_dict(policy.state_dict())
    assert restored.report() == policy.report()


@pytest.mark.parametrize('explores', [1, 2])
def test_native_numeric_paths_share_one_head_and_one_update(explores, monkeypatch, tmp_path):
    from pathlib import Path
    import Models
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    source = Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'
    config = tmp_path / 'paths.xml'
    config.write_text(source.read_text().replace('<architecture>', f'<architecture><explorePaths>{explores}</explorePaths>', 1))
    model, _, data = _fresh_model(str(config))
    calls, steps = [], []
    generate, step = model.generate, model._sentence_train_step
    def generated(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)
    def stepped(loss):
        steps.append(True)
        return step(loss)
    monkeypatch.setattr(model, 'generate', generated)
    monkeypatch.setattr(model, '_sentence_train_step', stepped)
    try:
        batch = next(iter(data.data_loader(split='train', num_streams=4)))
        raw, target = model.inputSpace.prepInput(batch[0]), model.outputSpace.prepOutput(batch[1])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.01), batchSize=4,
                       split='train', batch_override=(raw, target))
        assert not hasattr(model, 'comparison_reader')
        assert not any('comparison_reader' in name for name in model.state_dict())
        assert len(steps) == 1
        assert len(calls) <= 1 + explores
        audit = model._last_sentence_credit
        assert audit['costs'].shape == (4, explores + 1)
        assert len(audit['departures']) == explores
        for draw in audit['departures']:
            assert bool((draw['output_round'] == -1).all())
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_native_omission_keeps_targets_off_the_reading_path(tmp_path, monkeypatch):
    from Omission import Draw
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    def hide_second(valid, *, training):
        omitted = torch.zeros_like(valid)
        if training:
            omitted[:, 1] = valid[:, 1]
        return Draw(omitted.any(-1), omitted, torch.zeros(len(valid), dtype=torch.long))
    monkeypatch.setattr(model.omission, 'draw', hide_second)
    captured = []
    score = model._sentence_path_cost
    def inspect(*args, **kwargs):
        targets = model._omission_targets
        assert targets is not None and bool(targets['draw'].forms.any())
        assert bool((model.inputSpace._ar_word_part_ids[:, 1] == -1).all())
        assert bool((model.inputSpace._ar_word_concept_ids[:, 1] == -1).all())
        hidden = targets['bytes']
        assert not bool(model._staged_concepts_in[:, 0][hidden].any())
        assert bool(targets['raw'][:, 0][hidden].any())
        value = score(*args, **kwargs)
        original = value[2]['reverse'][1]
        width = original.shape[1]
        hidden = torch.nn.functional.pad(targets['bytes'][:, :width],
            (0, max(0, width - targets['bytes'].shape[1])))
        erased = original.masked_fill(hidden[..., None], 0)
        assert model._reverse_event_loss(original, original) == 0
        assert model._reverse_event_loss(erased, original) > 0
        from Omission import visible_events
        assert model._reverse_event_loss(*visible_events(model, erased, original)) == 0
        captured.append(value)
        return value
    monkeypatch.setattr(model, '_sentence_path_cost', inspect)
    try:
        source = model.inputSpace.prepInput(['alpha gamma'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        assert captured
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('explores', [1, 2])
def test_native_verbal_paths_finish_before_one_commit(tmp_path, monkeypatch, explores):
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import WhatAnswer
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'answerSynthesis': True, 'explorePaths': explores, 'omissionTarget': 1})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = True
    monkeypatch.setattr(model.inputSpace.data, 'what', WhatAnswer.silence)
    generate, think, commit = model.generate, model.run_selected_thought, model._commit_sentence
    calls = []
    def generated(*args, **kwargs):
        calls.append('generate')
        return generate(*args, **kwargs)
    def thought(*args, **kwargs):
        assert 'commit' not in calls
        calls.append('think')
        return think(*args, **kwargs)
    def committed(*args, **kwargs):
        calls.append('commit')
        return commit(*args, **kwargs)
    monkeypatch.setattr(model, 'generate', generated)
    monkeypatch.setattr(model, 'run_selected_thought', thought)
    monkeypatch.setattr(model, '_commit_sentence', committed)
    try:
        source = model.inputSpace.prepInput(['alpha gamma'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
            batch_override=(source, torch.empty(1, 0)))
        assert calls.count('commit') == 1
        assert calls.count('generate') <= 2 * (1 + explores)
        assert calls.count('think') <= 1 + explores
        assert model._last_sentence_credit['costs'].shape == (1, 1 + explores)
        assert model._kept_path_output is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_hidden_word_changes_only_loss_targets_through_the_close(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from Omission import Draw
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    initial = torch.random.get_rng_state().clone()
    captures = []
    from dataclasses import fields, is_dataclass
    from collections.abc import Mapping
    def snapshot(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if is_dataclass(value):
            return {field.name: snapshot(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(snapshot(item) for item in value)
        return value
    def identical(first, second, label):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0, msg=label)
        elif isinstance(first, Mapping):
            assert first.keys() == second.keys(), label
            for key in first:
                identical(first[key], second[key], label)
        elif isinstance(first, (tuple, list)):
            assert len(first) == len(second), label
            for a, b in zip(first, second, strict=True):
                identical(a, b, label)
        else:
            assert first == second, label
    for text in ('alpha gamma', 'alpha delta'):
        torch.random.set_rng_state(initial)
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
            concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
            architecture_overrides={'explorePaths': 0})
        model._tensor_peer_while_eager = True
        model.inputSpace.data.has_supervised_outputs = False
        def hide(valid, *, training):
            forms = torch.zeros_like(valid)
            forms[:, 1] = valid[:, 1]
            return Draw(forms.any(-1), forms, torch.zeros(len(valid), dtype=torch.long))
        monkeypatch.setattr(model.omission, 'draw', hide)
        # Compare the forward and its writes before learning from distinct targets.
        monkeypatch.setattr(model, '_sentence_train_step', lambda loss: None)
        record = {}
        import WholePaths
        finish = WholePaths.finish
        def observed(*args, **kwargs):
            path = finish(*args, **kwargs)
            record['root'] = path.observation['record'].root.detach().clone()
            record['reconstructed'] = path.observation['order_zero_reverse'].materialize().detach().clone()
            record['target'] = path.observation['reverse'][1].detach().clone()
            return path
        with monkeypatch.context() as patch:
            patch.setattr(WholePaths, 'finish', observed)
            model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                batch_override=(model.inputSpace.prepInput([text]), torch.empty(1, 0)))
        store = model.symbolSpace.ltm_store
        record['writes'] = store.slots.detach().clone()
        native = model.perceptualSpace.percept_store
        record['percept_store'] = snapshot(dict(addresses=native.hash_map,
            values=native._basis.W, witnesses=native.identity.word_witnesses,
            words=native.identity.words, size=len(native)))
        record['occurrences'] = snapshot((model._sentence_sources, tuple(store.row_name(i) for i in range(len(store)))))
        record['word_values'] = model.inputSpace._ar_embedded_N.detach().clone()
        owner = model._concept_owner()
        record['expectation'] = snapshot(dict(
            knowing=owner._observed_knowing, expected=owner._expected_knowing,
            concept=model._concept_prediction,
            pending=model.symbolSpace.expectation._inter_last_meaning,
            context=model.symbolSpace.expectation._inter_context))
        if captures:
            assert model._reverse_event_loss(captures[0]['target'], record['target']) > 0
        captures.append(record)
        model.End()
        model.symbolSpace.soft_reset()
    for key in ('percept_store', 'occurrences', 'root', 'reconstructed', 'writes', 'word_values', 'expectation'):
        identical(captures[0][key], captures[1][key], key)
    assert not torch.equal(captures[0]['target'], captures[1]['target'])


def test_omission_target_one_matches_removing_omission_from_the_native_path(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import Omission
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    initial = torch.random.get_rng_state().clone()
    states = []
    for removed in (False, True):
        torch.random.set_rng_state(initial)
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
            concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
            architecture_overrides={'omissionTarget': 1})
        model._tensor_peer_while_eager = True
        model.inputSpace.data.has_supervised_outputs = False
        with monkeypatch.context() as patch:
            if removed:
                patch.setattr(Omission, 'stage', lambda owner: setattr(owner, '_omission_targets', None))
            model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                batch_override=(model.inputSpace.prepInput(['alpha gamma']), torch.empty(1, 0)))
        states.append(({name: value.detach().clone() for name, value in model.state_dict().items()
                        if not name.startswith('omission.')}, torch.random.get_rng_state().clone()))
        model.End()
        model.symbolSpace.soft_reset()
    assert torch.equal(states[0][1], states[1][1])
    assert states[0][0].keys() == states[1][0].keys()
    for name, value in states[0][0].items():
        torch.testing.assert_close(value, states[1][0][name], rtol=0, atol=0, msg=name)


def test_output_departure_runs_only_the_last_stage(monkeypatch):
    from types import SimpleNamespace as NS
    from Layers import Error
    from WalkTrials import ThoughtTrialState
    from WholePaths import Path, Thinking, finish
    import WholePaths
    active = torch.tensor([True])
    model = NS(_what_memory=lambda: None, when_time=torch.tensor(7),
        when_encoding=NS(t=7), present=lambda: 7, _sentence_fields={}, _sentence_answer_predictions=None,
        inputSpace=NS(data=NS(has_supervised_outputs=True)))
    before = ThoughtTrialState(model)
    registry = Error(row_mask=active)
    registry.add('reconstruction', torch.tensor([2.]), objective='reconstruction')
    observation = dict(record=NS(root=torch.ones(1,2)), thoughts=(), thought_fields=())
    base = Path((), (), observation, None, registry, torch.tensor([[2.,0.,3.]]),
                Thinking(), before, before, output_trace=('held',))
    def forbidden(*args, **kwargs):
        raise AssertionError('an output departure repeated its prefix')
    model._sentence_path_cost = forbidden
    monkeypatch.setattr('SentenceField.reconstruct', forbidden)
    monkeypatch.setattr('ThoughtClosing.close', forbidden)
    calls = []
    def output(*args, **kwargs):
        assert kwargs['trace'] is base.output_trace
        calls.append(True)
        return torch.tensor([1.]), None, None
    monkeypatch.setattr(WholePaths, 'output', output)
    draw = dict(walk=torch.tensor([6]), reconstruction_round=torch.tensor([-1]),
                thought_round=torch.tensor([-1]), output_round=torch.tensor([0]))
    path = finish(model, (), 0, active, base=base, draw=draw)
    assert len(calls) == 1 and path.observation['record'] is base.observation['record']
    assert path.parts.tolist() == [[2.,0.,1.]]


def test_no_concluded_answer_has_no_occupied_role_or_surface(tmp_path, monkeypatch, eager_reading):
    from test_output_path_supervised import _native_answer_model
    from test_output_walk import _capture_program_probe, _stop
    from What import What
    model = _native_answer_model(tmp_path, False)
    model.eval()
    _stop(model)
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ['1', '2'])
            held = model._resolve_answer(understanding, (What.supervised(0), What.supervised(1)), selected_thoughts=())
            occupied = model._answer_occupancy(held, held.conceptual_answer)
            assert not occupied.any()
            quiet = model.reverseOutput(understanding, held)
            assert quiet.texts == ('', '')
            assert not quiet.actual.any() and not quiet.concepts.any()
            torch.testing.assert_close(model._output_target_cost(quiet, ['', '']), torch.zeros(2))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_old_twin_checkpoint_keys_are_dropped_and_named(tmp_path, monkeypatch):
    from test_mm_xor import _fresh_model
    import Models
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    path = tmp_path / 'reader.ckpt'
    messages = []
    try:
        model.save_weights(str(path))
        saved = torch.load(path, weights_only=False)
        saved['state_dict']['comparison_reader.weights.0'] = torch.ones(1, 1)
        torch.save(saved, path)
        monkeypatch.setattr(Models, 'TheMessage', lambda value, *args, **kwargs: messages.append(str(value)))
        assert model.load_weights(str(path), strict=True, require_match=True)
        assert any('Dropped' in message and 'comparison_reader.weights.0' in message for message in messages)
        assert not hasattr(model, 'comparison_reader')
        assert not any('comparison_reader' in key for key in model.state_dict())
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_kept_verbal_output_reaches_the_live_understanding_and_conditioner(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import MappingProxyType
    from test_compiled_word_chunk import _tiny_canonical_model
    from Models import SelectedThoughtResult
    from Queries import ThoughtResult
    from QueryWork import QueryWorkBudget
    from What import WhatAnswer
    import WholePaths, util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, concept_meaning_width=8,
        training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'answerSynthesis': True, 'explorePaths': 0, 'omissionTarget': 1})
    model._tensor_peer_while_eager = True
    from test_reverse_traversal import _select_completed_binary_path
    # The whole grammar is open (§17.18). Select a parametric composition
    # for this gradient contract; the first catalog rule is logical part.
    _select_completed_binary_path(model, operation='compound')
    model.inputSpace.data.has_supervised_outputs = True
    monkeypatch.setattr(model.inputSpace.data, 'what', WhatAnswer.silence)
    # An explicit continuous thought transformation isolates the graph contract
    # from the untrained controller's choice to conclude nothing.
    thought_inputs = []
    def conclude(meaning, **kwargs):
        thought_inputs.append(meaning.roles)
        from Meaning import ConceptualMeaning
        point = meaning.roles[meaning.role_mask].mean(0) * .7
        value = replace(ConceptualMeaning.from_description(point),
                        sentence_kind='idea',
                        bindings={'_evidence_kind': 'truth', '_producing_operation': 'isTrue'})
        result = ThoughtResult('isTrue', 'concept', 'truth', 'truth', value, MappingProxyType({}))
        return SelectedThoughtResult(value, dict(support_true=1., support_false=0.), QueryWorkBudget(32), (), result)
    monkeypatch.setattr(model, 'run_selected_thought', conclude)
    observed = []
    native = WholePaths.output
    def output(owner, state, observation, *args, **kwargs):
        result = native(owner, state, observation, *args, **kwargs)
        construction = result[1]
        parameter = owner.question_conditioners[str(owner.conceptualSpace.stm.concept_dim)].weight
        root = thought_inputs[-1]
        # The output trains its target loss and the three declared inverse
        # residual lanes. A discrete terminal's byte identity alone need not
        # vary with every coordinate of the composed meaning.
        loss = result[0].sum() + result[2][3].sum()
        gradients = torch.autograd.grad(loss, (root, parameter), retain_graph=True, allow_unused=True)
        parameters = tuple(owner.parameters())
        upstream = torch.autograd.grad(root.square().sum(), parameters, retain_graph=True, allow_unused=True)
        sources = tuple(p for p, g in zip(parameters, upstream) if g is not None and bool(g.abs().any()))
        assert sources, 'the thought reads an understanding produced by model parameters'
        sent = torch.autograd.grad(loss, sources, retain_graph=True, allow_unused=True)
        assert any(g is not None and bool(g.abs().any()) for g in sent)
        observed.append(tuple(g is not None and bool(g.abs().any()) for g in gradients))
        return result
    monkeypatch.setattr(WholePaths, 'output', output)
    try:
        optimizer = model.getOptimizer(lr=.003)
        model.runBatch(train=True, optimizer=optimizer, batchSize=1,
            batch_override=(model.inputSpace.prepInput(['alpha beta']), torch.empty(1, 0)))
        assert observed == [(True, True)]
        conditioner = model.question_conditioners[str(model.conceptualSpace.stm.concept_dim)].weight
        assert any(p is conditioner for group in optimizer.param_groups for p in group['params'])
        assert bool(conditioner.detach().abs().any()), 'the native sentence update trains its live conditioner'
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_unsupervised_report_uses_the_kept_thought_and_stays_silent(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import WhatAnswer
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'answerSynthesis': True, 'explorePaths': 1, 'omissionTarget': 1})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    monkeypatch.setattr(model.inputSpace.data, 'what', WhatAnswer.silence)
    import ThoughtClosing
    # Supply no concluded episode; arbitrary native chooser draws may
    # otherwise conclude an answer, which is a different reporting case.
    monkeypatch.setattr(ThoughtClosing, 'close', lambda *args, **kwargs: ())
    generate, commit = model.generate, model._commit_sentence
    reports, committed = [], []
    def generated(*args, **kwargs):
        if committed:
            reports.append(kwargs['occupied'].detach().clone())
        return generate(*args, **kwargs)
    def close(*args, **kwargs):
        result = commit(*args, **kwargs)
        committed.append(True)
        return result
    monkeypatch.setattr(model, 'generate', generated)
    monkeypatch.setattr(model, '_commit_sentence', close)
    def no_second_resolution(*args, **kwargs):
        raise AssertionError('the completed sentence opened a second answer resolution')
    monkeypatch.setattr(model, 'resolveAnswer', no_second_resolution)
    try:
        source = model.inputSpace.prepInput(['alpha gamma'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
            batch_override=(source, torch.empty(1, 0)))
        assert not model._last_closing_thoughts
        assert len(reports) == 1 and not reports[0].any()
        assert model._last_answer_construction.texts == ('',)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_thought_preview_keeps_scored_reference_arrivals_unchanged():
    from types import SimpleNamespace as NS
    from test_reference_close import fixture
    from SentenceStrata import SentenceStrata
    from Understanding import Reading
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from chunk_fixtures import context
    from RowTime import encoder
    from WholePaths import fields
    resolver, store, atoms = fixture()
    owner, point = resolver.components.owner, atoms[0]
    word = owner.interpret.lookup_word([97], (), form='arrival-word')
    language = NS(_compose_binary_rules=(), _compose_unary_rules=(), parts_of_speech=NS(seats={}))
    entry = Reading(rows=torch.tensor([0]), word_rows=torch.tensor([0]), activations=torch.ones(1),
        leaves=point[None], actions=torch.tensor([[0, -1, 0]]), targets=torch.tensor([0]),
        end_state=point[None], concept_ids=torch.tensor([word]))
    strata = SentenceStrata(1, 1, 8, len(point))
    strata.capture(entry, language, 0)
    view = entry.tree.install_composition(Composition(point, reference=word, word_id=word,
        leaves=(0,), order=0), word=0)
    request = dict(mode='mint', leaves=(0,), cue=point, predicate=point, content=point, role=0)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                    reference_request=request, composition=view)
    plan = resolver.preview(clause, context(resolver, ()), stream=0)
    model = NS(_concept_owner=lambda: owner, symbolSpace=NS(ltm_store=store),
        conceptualSpace=NS(_incoming_trust_multiplier=lambda: 1.), grammatical_thoughts=None,
        when_encoding=encoder(), _sentence_ticks={0: 1})
    observation = dict(clauses=(clause,), reference_plans=(plan,), entries=(entry,))
    assert plan['arrival_count'] == 0
    for _ in range(2):
        assert fields(model, observation, 0)[0] is not None
        assert view.reference == word
        assert len(resolver._arrival_constituents(clause)[1]) == plan['arrival_count']
        assert 'prepared_root' not in plan and 'written_node' not in plan['items'][0]
    committed = resolver.prepare(clause, plan)
    assert plan['prepared_root'] is committed and view.reference == -1
