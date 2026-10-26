"""Incoming placement-review repairs, before the shared-chunk migration."""
import pytest
import torch


@pytest.mark.parametrize('has_cases', (False, True))
def test_unary_case_keeps_a_single_position_per_stream(monkeypatch, has_cases):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, CompoundUnaryLayer
    from CaseSelection import CaseView
    base = CompoundLayer().double()
    unary = CompoundUnaryLayer(base).double()
    values = torch.linspace(.1, .8, 24, dtype=torch.float64).reshape(2, 1, 12).requires_grad_()
    codes = torch.linspace(.15, .75, 72, dtype=torch.float64).reshape(2, 3, 12).requires_grad_()
    weights = torch.zeros(2, 3, 2, dtype=torch.float64)
    if has_cases:
        weights[0, 0] = 1.
        weights[1, 2] = 1.
    view = CaseView(codes, weights, torch.ones(2, 12, dtype=torch.float64),
                    torch.zeros(2, dtype=torch.bool), codes[:, 1])
    actual = unary.compose(values, case_view=view)
    expected = torch.stack([unary.compose(values[b], case_view=CaseView(
        codes[b:b+1], weights[b:b+1], view.test[b:b+1],
        view.selected[b:b+1], view.reference[b:b+1])) for b in range(2)])
    assert actual.shape == (2, 1, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (values, codes, base.meaning_kernel.bias)
    wanted = torch.autograd.grad(expected.sum(), inputs, retain_graph=True, allow_unused=True)
    observed = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    for first, second in zip(observed, wanted):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_recurrence_capacity_is_independent_fixed_and_versioned():
    from Layers import LTM0
    from MemoryConsolidation import recurrence_groups
    store = LTM0(4, capacity=9, recurrence_capacity=2)
    assert store.recurrence_codes.shape == (2, 4)
    for name in ('documents', 'weights', 'used', 'symbols'):
        assert getattr(store, 'recurrence_'+name).shape == (2,)
    assert len(store._recurrence_identities) == 2
    assert store.recurrence_readings.shape == (9,)
    pointers = {name:value.data_ptr() for name,value in store.named_buffers()}
    for point in torch.eye(4)[:2]:
        store.append_idea(point, trust=1.)
    recurrence_groups(store)
    assert store.recurrence_used.all()
    saved = store.state_dict()
    restored = LTM0(4, capacity=9, recurrence_capacity=2)
    restored.load_state_dict(saved)
    restored.load_semantic_extras(store.semantic_extras())
    assert restored.recurrence_readings.tolist() == store.recurrence_readings.tolist()
    with pytest.raises(ValueError, match='construction'):
        LTM0(4, capacity=9, recurrence_capacity=3).load_state_dict(saved)
    older = dict(saved, recurrence_version=torch.tensor(1))
    with pytest.raises(ValueError, match='recurrenceCapacity checkpoint version'):
        restored.load_state_dict(older)
    with pytest.raises(ValueError, match='semantic checkpoint version'):
        restored.load_semantic_extras(dict(store.semantic_extras(), version=8))
    store.append_idea(torch.eye(4)[2], trust=1.)
    with pytest.raises(RuntimeError, match='recurrenceCapacity'):
        recurrence_groups(store)
    store.clear_recurrence()
    assert len(store._recurrence_identities) == 2
    assert pointers == {name:value.data_ptr() for name,value in store.named_buffers()}


def test_native_recurrence_capacity_defaults_to_concept_inventory(tmp_path):
    from test_output_path_supervised import _native_answer_model
    model = _native_answer_model(tmp_path, False)
    try:
        store = model.symbolSpace.ltm_store
        assert store.recurrence_capacity == model._concept_owner().nVectors
        assert store.recurrence_readings.shape == (store.capacity,)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
