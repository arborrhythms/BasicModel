"""Stage 4e waits train the shared chooser; gold never installs a reading."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from GrammarLessons import compose_loss, compose_examples, constituent_examples
from Language import OperationSelectionLayer
from constituent_corpus import corpus
from constituent_measurement import grade, summarize


def test_explicit_brackets_teach_waits_and_multiple_folds_without_changing_student():
    class Sum(torch.nn.Module):
        def forward(self, left, right):
            return left+right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(),), chooser='mlp')
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'),), _compose_unary_rules=())
    program = SimpleNamespace(leaves=torch.eye(4), actions=torch.tensor([[0, -1, 0]]),
        compose_capacity=8, compose_closing_budget=16)
    # The two top leaves must first wait; the third word permits two folds.
    lesson = dict(tree=['sum', ['sum', 0, ['sum', 1, 2]], 3], waits=True)
    original = {name: value.clone() for name, value in vars(program).items() if torch.is_tensor(value)}
    states = constituent_examples(language, program, lesson['tree'])
    assert [(state['target'], state['depth']) for state in states] == [
        (-1, 1), (-1, 2), (0, 3), (0, 2), (-1, 1), (0, 2), (-1, 1), (-1, 1)]
    from Layers import Error
    errors = Error()
    loss = compose_loss(language, [program], [lesson], errors=errors)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in layer.chooser.parameters())
    for name, value in original.items():
        torch.testing.assert_close(getattr(program, name), value, rtol=0, atol=0)
    # Wait-only supervision is a real gradient, not the old skipped target.
    layer.zero_grad()
    singleton = SimpleNamespace(leaves=program.leaves[:1], compose_capacity=8, compose_closing_budget=16)
    # Add a genuine competing unary so STOP has a nontrivial softmax.
    other = OperationSelectionLayer(d_model=4, ops=(Sum(),), unary_ops=(torch.nn.Identity(),), chooser='mlp')
    language._tree_layer = lambda _: other
    language._compose_unary_rules = (SimpleNamespace(method_name='identity'),)
    compose_loss(language, [singleton], [dict(tree=0, waits=True)]).backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in other.chooser.parameters())


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
            lowerings=[dict(span=np['span'], before=2, after=1) for np in label['nps']],
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
    observed['lowerings'][0]['before'] = 0
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
    program = SimpleNamespace(leaves=torch.eye(4)[:2], compose_capacity=8, compose_closing_budget=16)
    for expected, name in enumerate(names):
        tree = [name, 0, 1]
        binary, _ = compose_examples(language, program, tree)
        assert [target for _, target, _, _ in binary if target >= 0] == [expected]
        assert [row['target'] for row in constituent_examples(language, program, tree)
                if row['target'] >= 0] == [expected]


def test_constituent_teacher_runs_all_constructions_with_native_meaning_bands(tmp_path, monkeypatch):
    from identity_experiment import make_model
    model = make_model(tmp_path, monkeypatch)
    try:
        from CaseSelection import CaseBank
        capacity = int(model.conceptualSpace.stm.capacity)
        closing = min(2*capacity, int(model.syntacticOrder)) if model.syntacticOrder else 2*capacity
        seen = set()
        for _, _, lesson, label in corpus():
            key = label['family'], label['words'][0]
            if key in seen:
                continue
            seen.add(key)
            program = SimpleNamespace(leaves=torch.randn(len(label['words']), 232),
                compose_capacity=capacity, compose_closing_budget=closing,
                concept_ids=torch.arange(1, len(label['words'])+1))
            count = len(program.leaves)
            # A mechanism fixture supplies existing memberships. Their
            # absence is reported, never replaced by a fabricated head.
            missing = constituent_examples(model.languageSpace, program, lesson['tree'])
            if label['family'] in ('np', 'intransitive', 'description'):
                assert missing[-1]['unavailable'] == 'no_native_cases'
            program.case_bank = CaseBank(program.concept_ids, program.leaves,
                torch.ones(count, count, 2), torch.ones(count, count, 2))
            states = constituent_examples(model.languageSpace, program, lesson['tree'])
            assert all(state['window'].shape == (2, 232) for state in states)
            assert states[-1]['target'] == -1 and states[-1]['depth'] == 1
            assert torch.isfinite(compose_loss(model.languageSpace, [program], [lesson]))
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
