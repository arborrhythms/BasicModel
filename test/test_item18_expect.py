"""Round B: omission supplies expectation's detached, native-field teacher."""
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)


def test_82_omission_teacher_keeps_the_staged_input_bracket():
    from types import SimpleNamespace
    from Omission import full_percepts
    from Layers import RadixLayer
    store = RadixLayer(4, initial_cap=8)
    rows = [store.insert(raw) for raw in (b'a', b'b', b'c')]
    prepared = dict(raw=torch.tensor([[[97, 32, 98, 32, 99]]]),
        units=((b'a b c', ((0, 1), (2, 3), (4, 5))),),
        plans=([{}, {}],), sentences=torch.zeros(1, 2, dtype=torch.long))
    model = SimpleNamespace(perceptualSpace=SimpleNamespace(percept_store=store))
    full_percepts(model, prepared)
    ids, spans, words = prepared['full_percepts']
    assert ids.tolist() == [rows[:2]]
    assert spans.tolist() == [[[0, 1], [2, 3]]]
    assert words.tolist() == [[[0, 1], [2, 3]]]


def lesson(tmp_path, monkeypatch, mapping='sigmaPi'):
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        conceptual_overrides={'expectationMap': mapping})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def read(model, text, *, training=False, optimizer=None):
    source = model.inputSpace.prepInput([text])
    return model.runBatch(train=training, optimizer=optimizer or model.getOptimizer(lr=.003), batchSize=1,
                         batch_override=(source, torch.empty(1, 0)))


def omit_second(monkeypatch, model):
    from Omission import Draw
    def draw(valid, *, training):
        hidden = torch.zeros_like(valid)
        hidden[:, 1] = valid[:, 1]
        return Draw(hidden.any(-1), hidden, torch.zeros(len(valid), dtype=torch.long))
    monkeypatch.setattr(model.omission, 'draw', draw)


def test_81_expectation_restores_the_omitted_native_word(tmp_path, monkeypatch):
    from ConceptSpread import open_prediction, loss, rule_order
    model = lesson(tmp_path, monkeypatch)
    ordinary = model.omission.draw
    try:
        from PrependedSymbols import word_view
        examples = ('alpha gamma', 'beta delta')
        hidden_rows = []
        for text in examples:
            read(model, text)
            hidden_rows.append(int(word_view(model.inputSpace, 'concept_rows')[0, 1]))
        assert all(row >= 0 for row in hidden_rows)
        owner = model._concept_owner()
        omit_second(monkeypatch, model)
        optimizer = model.getOptimizer(lr=.01)
        for presentation in range(200):
            read(model, examples[presentation % 2], training=True, optimizer=optimizer)
        for text, hidden_row in zip(examples, hidden_rows):
            read(model, text)
            prediction = model._concept_prediction
            assert prediction.target[0, hidden_row, 0] > 0
            assert owner._observed_knowing[0, hidden_row].sum() == 0
            values = owner._expected_knowing[0, :owner._order_caps()[0], 0]
            absent = owner._observed_knowing[0, :len(values)].amax(-1) == 0
            absent[hidden_row] = False
            assert values[hidden_row] > values[absent].max()
            assert hidden_row in rule_order(owner, 0)
            assert not owner._expected_knowing.requires_grad
            print(dict(presentations=200, text=text, hidden_row=hidden_row,
                       expected=float(values[hidden_row]), other_absent=float(values[absent].max())))
    finally:
        model.omission.draw = ordinary
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma', 'beta', 'delta']))
        model.End(); model.symbolSpace.soft_reset()


@pytest.mark.parametrize('mapping', ['sigmaPi', 'predictor'])
def test_82_omission_is_the_only_teacher_and_consumers_detach(tmp_path, monkeypatch, mapping):
    from ConceptSpread import loss, rule_order, open_prediction
    from Expectation import ActivationPredictor
    model = lesson(tmp_path, monkeypatch, mapping)
    ordinary = model.omission.draw
    try:
        read(model, 'alpha gamma')
        assert loss(model) is None
        assert 'expectation.concepts' not in model.errors._terms
        owner = model._concept_owner()
        assert any(isinstance(module, ActivationPredictor) for module in owner.modules()) == (mapping == 'predictor')
        omit_second(monkeypatch, model)
        read(model, 'alpha gamma')
        open_prediction(model, torch.ones(1, dtype=torch.bool), sid=0)
        prediction = model._concept_prediction
        assert prediction.target is not None and not prediction.target.requires_grad
        assert prediction.value.requires_grad
        assert rule_order(owner, 0) == tuple(i for i in owner._expected_knowing[0].amax(-1).argsort(
            descending=True, stable=True).tolist())
        assert not owner._expected_knowing.requires_grad
        loss(model).sum().backward()
        module = owner.concept_predictor if mapping == 'predictor' else owner.expect_stack
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
        assert all(p.grad is None for p in owner.similarity_codebook.parameters())
    finally:
        model.omission.draw = ordinary
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma']))
        model.End(); model.symbolSpace.soft_reset()
