"""Item 1.8 §68.54: one expected field, deposited without a second teacher."""
import pytest
import torch


def test_intuit_deposits_the_expected_basic_block_then_closes(monkeypatch):
    from test_thought_infer import setup
    from Understanding import ConceptualField
    from SentenceField import publish, knowing
    from IntuitionField import open_field, read
    from ConceptSpread import open_prediction
    from QueryWork import QueryWorkBudget
    import FieldProjection
    model, _, _, refs = setup()
    owner = model._concept_owner()
    object.__setattr__(model, 'conceptualSpaces', (owner,))
    ids = torch.tensor([ref[1] for ref in refs])
    field = ConceptualField(ids, torch.zeros(len(ids), 1, 1, 2),
        torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    width = owner._order_caps()[0]
    # A controlled map tests lane/row transport, independently of learning.
    expected = torch.arange(2 * width).reshape(1, width, 2).float() / (2 * width)
    monkeypatch.setattr(owner.expect_stack, 'forward', lambda value: expected.flatten(1))
    publish(model, open_field(model, field))
    open_prediction(model, torch.tensor([True]))
    before_close = []
    close = FieldProjection.close
    def capture(space, rows, evidence, *args, **kwargs):
        before_close.append((rows, evidence.detach().clone()))
        return close(space, rows, evidence, *args, **kwargs)
    monkeypatch.setattr(FieldProjection, 'close', capture)
    read(owner, row=0, work=QueryWorkBudget(64))
    assert len(before_close) == 1
    rows, deposited = before_close[0]
    for index, address in enumerate(rows):
        if address < width:
            torch.testing.assert_close(deposited[index, 0, 0], owner._expected_knowing[0, address])
    assert not knowing(model).evidence.requires_grad
    assert not hasattr(model, '_intuition_teacher')


def test_expectation_cost_and_gradients_keep_the_omission_teacher(tmp_path, monkeypatch):
    from test_item18_expect import lesson, read, omit_second
    from ConceptSpread import open_prediction, complete
    from Layers import Error
    model = lesson(tmp_path, monkeypatch)
    try:
        read(model, 'alpha gamma')
        omit_second(monkeypatch, model)
        read(model, 'alpha gamma')
        owner = model._concept_owner()
        open_prediction(model, torch.tensor([True]), sid=0)
        prediction = model._concept_prediction
        expected = (prediction.value - prediction.target.detach()).square().mean((1, 2))
        expected = torch.where(prediction.active, expected, 0.)
        registry = Error()
        complete(model, torch.tensor([True]), registry)
        assert set(registry._terms) == {'expectation.concepts'}
        actual = registry.total()
        torch.testing.assert_close(actual, expected.mean(), rtol=0, atol=1e-6)
        parameters = tuple(owner.expect_stack.parameters())
        wanted = torch.autograd.grad(expected.sum(), parameters, retain_graph=True, allow_unused=True)
        got = torch.autograd.grad(actual.sum(), parameters, allow_unused=True)
        for left, right in zip(got, wanted):
            if left is None or right is None:
                assert left is right
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_only_the_concept_owner_builds_the_expected_map_and_optimizer(tmp_path, monkeypatch):
    from test_math_chain import build_model
    model = build_model()
    try:
        owner = model._concept_owner()
        assert len(model.conceptualSpaces) == 3
        for space in model.conceptualSpaces:
            assert (space.expect_stack is not None) == (space is owner)
            assert not hasattr(space, 'intuit_stack')
        optimizer = model.getOptimizer(lr=.003)
        held = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert {id(p) for p in owner.expect_stack.parameters()} <= held
        assert not any('intuit_stack' in name for name, _ in model.named_parameters())
    finally:
        model.End(); model.symbolSpace.soft_reset()
