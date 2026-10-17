"""Stage 4e waits train the shared chooser; gold never installs a reading."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from GrammarLessons import compose_loss, compose_examples, constituent_examples
from Language import OperationSelectionLayer
from constituent_corpus import corpus
from constituent_measurement import grade, summarize


def test_explicit_brackets_label_student_waits_and_wrong_folds_without_replacing_them():
    from GrammarLessons import ComposeLessonStates, score_student_state
    class Sum(torch.nn.Module):
        def forward(self, left, right):
            return left+right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(),),
                                    unary_ops=(torch.nn.Identity(),), chooser='mlp')
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'),),
        _compose_unary_rules=(SimpleNamespace(method_name='identity'),))
    capture = ComposeLessonStates()
    # Actual sequence: wait after leaf 0, prematurely fold leaves 0+1, then
    # see leaf 2. The gold tree needs (1,2), which is now unreachable.
    windows = [torch.stack((torch.eye(4)[0], torch.zeros(4))), torch.eye(4)[:2],
               torch.stack((torch.eye(4)[0]+torch.eye(4)[1], torch.eye(4)[2]))]
    for slot, depth, kind, window in zip((0, 3, 6), (1, 2, 2), (0, 1, 0), windows):
        kwargs = dict(depth=torch.tensor([min(2, depth)]), load_depth=torch.tensor([depth]),
                      slots=2, allowance=7, rounds_left=3)
        _, _, route = layer(window[None], **kwargs)
        route = dict(route, kind=torch.tensor([kind]))
        capture.record(slot, window[None], kwargs, route)
    attempted = torch.zeros(1, 25, dtype=torch.bool)
    attempted[0, [0, 3, 6]] = True
    program = SimpleNamespace(leaves=torch.eye(4)[:3], actions=torch.tensor([0, 1, 0]),
        student_states=capture.sentence(0, torch.arange(3), width=3,
            closing_base=9, closing_width=16, attempted=attempted))
    lesson = dict(tree=['sum', 0, ['sum', 1, 2]], waits=True)
    states = constituent_examples(language, program, lesson['tree'])
    assert [state['target'] for state in states] == [-1, -1, -1]
    assert states[-1]['spans'] == ((0, 1), (2,))
    for state in states:
        torch.testing.assert_close(score_student_state(layer, state), state['observed_probabilities'])
    original = program.actions.clone()
    loss = compose_loss(language, [program], [lesson])
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.any()) for p in layer.chooser.parameters())
    torch.testing.assert_close(program.actions, original, rtol=0, atol=0)
    # When actual operands do match a bracket, its declared operator is taught.
    program.student_states = [dict(states[1], spans=((1,), (2,)))]
    assert constituent_examples(language, program, lesson['tree'])[0]['target'] == 0
    del program.student_states
    with pytest.raises(ValueError, match='captured student states'):
        compose_loss(language, [program], [lesson])


def test_constituent_grader_rejects_missing_duplicate_and_unlowered_occurrences():
    rows = corpus()
    assert all(text['text'] == lesson['text'] == ' '.join(label['words'])
               for _, text, lesson, label in rows)
    train = [row for row in rows if row[0] == 'train']
    held = [row for row in rows if row[0] == 'eval']
    assert len(train) == 72 and len(held) == 40
    assert not {row[1]['text'] for row in train} & {row[1]['text'] for row in held}
    graded = []
    for _, _, _, label in held:
        observed = dict(leaf_positions=list(range(len(label['words']))), brackets=label['brackets'],
            lowerings=[dict(span=np['span'], before=0, after=1) for np in label['nps']],
            occurrences=[dict(description=np['description'], address=100+i, written=True, order=1)
                         for i, np in enumerate(label['nps'])],
            individual_writes=[100+i for i in range(len(label['nps']))])
        graded.append(grade(label, observed))
    assert summarize(graded)['passed']
    assert not summarize([grade(row[3], None) for row in held])['passed']
    label = held[0][3]
    observed = graded[0]['observed']
    observed['occurrences'] *= 2
    observed['individual_writes'] *= 2
    observed['lowerings'][0]['after'] = 0
    failed = grade(label, observed)
    assert failed['duplicate_writes'] and not failed['nps'][0]['reduction']
    assert not failed['nps'][0]['one_occurrence']
    assert not grade(held[-1][3], observed)['unlowered_clean']


def test_declared_forms_keep_distinct_targets_when_the_implementation_is_shared():
    class Head(torch.nn.Module):
        def forward(self, left, right):
            return right
    names = ('lower', 'definite', 'universal')
    layer = OperationSelectionLayer(d_model=4, ops=tuple(Head() for _ in names), chooser='mlp')
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=tuple(SimpleNamespace(method_name='lower', surface_name=name) for name in names),
        _compose_unary_rules=())
    program = SimpleNamespace(leaves=torch.eye(4)[:2],
                              student_states=[dict(spans=((0,), (1,)))])
    for expected, name in enumerate(names):
        tree = [name, 0, 1]
        binary, _ = compose_examples(language, program, tree)
        assert [target for _, target, _, _ in binary if target >= 0] == [expected]
        assert [row['target'] for row in constituent_examples(language, program, tree)
                if row['target'] >= 0] == [expected]


@pytest.mark.slow
def test_native_constituent_lesson_captures_student_states_without_cases(tmp_path, monkeypatch):
    from identity_experiment import make_model
    import GrammarLessons
    from What import What
    model = make_model(tmp_path, monkeypatch)
    seen = []
    native = GrammarLessons.compose_loss
    def check(language, programs, lessons, **kwargs):
        for program, lesson in zip(programs, lessons):
            assert 'compound' not in str(lesson['tree'])
            states = constituent_examples(language, program, lesson['tree'])
            assert states and all(state['window'].shape[-2:] == (2, 232) for state in states)
            for state in states:
                torch.testing.assert_close(GrammarLessons.score_student_state(
                    language._tree_layer(2), state), state['observed_probabilities'], rtol=1e-5, atol=1e-6)
            seen.extend(states)
        return native(language, programs, lessons, **kwargs)
    monkeypatch.setattr(GrammarLessons, 'compose_loss', check)
    try:
        lessons = [next(lesson for split, text, lesson, label in corpus()
                        if split == 'train' and label['family'] == family)
                   for family in ('np', 'intransitive', 'transitive', 'preposition')]
        model.inputSpace.data.grammar_lessons = {'train': lessons}
        inputs = model.inputSpace.prepPackedInput([[lesson['text']] for lesson in lessons])
        from constituent_budget import paired_updates_only
        optimizer = model.getOptimizer(lr=.003)
        with paired_updates_only(model, optimizer) as updates:
            model.runBatch(train=True, optimizer=optimizer, batchSize=4,
                split='train', batch_override=(inputs, torch.empty(4, 0)), source_rows=[[b] for b in range(4)],
                questions=tuple(What.present(b, split='train') for b in range(4)))
            assert updates['paired'] == 2
            pressure = model.languageSpace._tree_layer(2).reduce_pressure
            before = pressure.detach().clone()
            pressure.grad = torch.ones_like(pressure)
            disabled = updates['batch_end_disabled']
            optimizer.step()
            assert updates['batch_end_disabled'] == disabled+1
            torch.testing.assert_close(pressure, before, rtol=0, atol=0)
        assert len(seen) > 8 and model.languageSpace._compose_lesson_states is None
    finally:
        model.End()


def test_wait_lesson_trains_pressure_with_the_order_lesson_optimizer(tmp_path, monkeypatch):
    from identity_experiment import make_model
    from ObjectiveOwnership import backward_owned
    model = make_model(tmp_path, monkeypatch)
    try:
        layer = model.languageSpace._tree_layer(2)
        optimizer = model.getOptimizer(lr=.003)
        groups = model.objective_parameter_groups(optimizer)
        owned = {id(p) for p in groups['compose_lesson']}
        assert owned == {id(p) for p in layer.chooser.parameters()} | {id(layer.reduce_pressure)}
        compose = next(o for o in optimizer.optimizers if getattr(o, 'objective_owner', None) == 'compose_lesson')
        attention = next(o for o in optimizer.optimizers if getattr(o, 'objective_owner', None) == 'attention')
        assert isinstance(compose._inner, torch.optim.Adam) and isinstance(attention._inner, torch.optim.Adam)
        for key in ('lr', 'betas', 'eps', 'weight_decay', 'capturable'):
            assert compose.param_groups[0][key] == attention.param_groups[0][key]
        assert sum(p is layer.reduce_pressure for g in optimizer.param_groups for p in g['params']) == 1
        window = torch.randn(1, 2, 232)
        def wait_cost():
            _, _, route = layer(window, slots=2, allowance=7, rounds_left=3)
            return -route['probabilities'][0, -1].log()
        before = wait_cost()
        backward_owned({'compose_lesson': before}, groups)
        assert layer.reduce_pressure.grad > 0
        assert all(p.grad is None for g in optimizer.param_groups for p in g['params'] if id(p) not in owned)
        optimizer.step()
        assert layer.reduce_pressure < 1.
        assert wait_cost() < before
    finally:
        model.End()


@pytest.mark.learning_certificate
@pytest.mark.artifact_eval
def test_four_constituent_gates_on_trained_artifact():
    import json
    import os
    from pathlib import Path
    from constituent_corpus import load
    path = os.environ.get('BASICMODEL_CONSTITUENT_ARTIFACT')
    if not path:
        pytest.skip('set BASICMODEL_CONSTITUENT_ARTIFACT to a stage 4e run')
    texts, labels = load('eval')
    records = json.loads((Path(path)/'kept-programs.json').read_text())
    readings = [row for row in records if not row['training']]
    assert Counter(row['id'] for row in readings) == Counter(row['id'] for row in texts)
    result = summarize([grade(labels[row['id']], row['observed']) for row in readings])
    assert result['passed'], result['gates']
