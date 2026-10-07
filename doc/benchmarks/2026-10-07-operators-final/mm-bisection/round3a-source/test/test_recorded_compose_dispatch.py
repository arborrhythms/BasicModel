"""A recorded compose step executes its selected operator, without rescoring."""
import pytest
import torch

from test_reverse_traversal import _traversal_model


def test_recorded_binary_executes_only_selected_operators(tmp_path, monkeypatch):
    model = _traversal_model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        names = list(binary.op_names)
        chosen = names.index("sum")
        calls = []
        for name, op in zip(names, binary.ops):
            original = op.forward_with_context

            def observe(*args, _original=original, _name=name, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(op, "forward_with_context", observe)
        width = int(model.conceptualSpace.stm.concept_dim)
        left, right = torch.randn(2, width), torch.randn(2, width)
        language.forward_binary_step(
            left, right, torch.full((2,), chosen), torch.tensor([True, False]))
        assert calls == ["sum"]
        calls.clear()
        language.forward_binary_step(
            left, right, torch.full((2,), chosen), torch.zeros(2, dtype=torch.bool))
        assert calls == []
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_recorded_dispatch_preserves_values_and_gradients_for_mixed_rows(tmp_path, compiled):
    model = _traversal_model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        names = list(binary.op_names)
        selected = torch.tensor([names.index("sum"), names.index("lift"), 0])
        valid = torch.tensor([True, True, False])
        # The actual replay supplies two views of the same stack storage.
        torch.manual_seed(461)
        width = int(model.conceptualSpace.stm.concept_dim)
        operands = (torch.randn(3, 2, width) * .1).requires_grad_()
        left, right = operands[:, 1], operands[:, 0]
        window = torch.stack((left, right), dim=1)
        candidates = binary._stacked_reduced(window)[:, 0]
        indices = selected[:, None, None].expand(-1, 1, width)
        expected = torch.where(valid[:, None], candidates.gather(1, indices)[:, 0], right)
        parameters = tuple(binary.parameters())
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (operands, *parameters), allow_unused=True)
        call = language.forward_binary_step
        if compiled:
            call = torch.compile(call, backend="eager", fullgraph=True)
        actual = call(left, right, selected, valid)
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (operands, *parameters), allow_unused=True)
        torch.testing.assert_close(actual, expected)
        for parameter, a, b in zip((operands, *parameters), actual_grads, expected_grads):
            a = torch.zeros_like(parameter) if a is None else a
            b = torch.zeros_like(parameter) if b is None else b
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
