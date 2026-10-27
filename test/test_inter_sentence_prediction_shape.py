"""Native activation shapes, source isolation and local training lifecycle."""
import os
import sys
import gc
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import matplotlib
matplotlib.use("Agg")
import Layers


def test_prediction_trains_predictor_with_detached_context_and_target():
    torch.manual_seed(321)
    layer = _make_layer(D=4)
    encoder = torch.nn.Linear(4, 4, bias=False)
    source = encoder(torch.tensor([[1., 2., 3., 4.]]))
    target = torch.nn.Parameter(torch.tensor([[4., 3., 2., 1.]]))
    layer.predict_and_observe_stm_end_state([1], [source])
    layer.predict_and_observe_stm_end_state([1], [target])
    loss = layer.consume_inter_loss()
    assert loss is not None
    loss.backward()
    assert encoder.weight.grad is None
    assert target.grad is None, "the observation must not move to meet its prediction"
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in layer._inter_predictor.parameters())
    assert all(not payload.requires_grad for _, payload, _ in layer.get_stm_chain())


def test_consuming_prediction_loss_never_retains_encoder_graph():
    torch.manual_seed(321)
    layer = _make_layer(D=4)
    encoder = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.Adam([*encoder.parameters(), *layer.parameters()], lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for sample in (torch.ones(1, 4), torch.arange(4.).reshape(1, 4)):
            layer.predict_and_observe_stm_end_state([1], [encoder(sample)])
        loss = layer.consume_inter_loss()
        assert loss is not None
        loss.backward()
        assert encoder.weight.grad is None
        assert any(p.grad is not None and p.grad.norm() > 0
                   for p in layer._inter_predictor.parameters())
        optimizer.step()


def test_prediction_context_is_row_local_and_reset_keeps_other_rows():
    torch.manual_seed(321)
    layer = _make_layer(D=4, batch=2)
    a, b = torch.ones(1, 4), torch.full((1, 4), 2.)
    layer.predict_and_observe_stm_end_state([1, 1], [a, b])
    before = layer.predict_next_end_state(1)[1].detach().clone()
    layer.observe_stm_end_state([1, 0], [a * 99, None], mask=[True, False])
    after = layer.predict_next_end_state(1)[1].detach()
    torch.testing.assert_close(after, before)
    layer.Reset(batch=0)
    assert not layer._inter_context[0] and layer._inter_context[1]
    assert layer.predict_next_end_state(0) is None


def test_consolidated_prediction_uses_observed_context_without_target_leak():
    layer = _make_layer(D=4)
    # A provisioned/global LTM row is not an external predecessor in this
    # document. The predictor view is populated only by observe, per row.
    from types import SimpleNamespace
    layer._ltm_store = SimpleNamespace(model_clock=0, when_start=None)
    layer.predict_and_observe_stm_end_state([1], [torch.ones(1, 4)])
    assert layer._inter_loss_count == 0
    assert not layer._stm_end_states[0], "durable LTM remains the only store"
    layer.predict_and_observe_stm_end_state([1], [torch.full((1, 4), 3.)])
    assert layer._inter_loss_count == 1
    layer.Reset()
    assert layer.predict_next_end_state() is None


def test_evaluation_does_not_accumulate_training_losses_with_autograd_enabled():
    layer = _make_layer(D=4)
    layer.eval()
    for value in (1., 2., 3.):
        layer.predict_and_observe_stm_end_state([1], [torch.full((1, 4), value)])
    assert layer.consume_inter_loss() is None
    assert not hasattr(layer, "consume_inter_contrastive_loss")


def _release_allocator_cache():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_layer(D=4, batch=1, ltm_capacity=1024, concept_dim=None):
    """An SymbolExpectation with the inter-level predictor built
    (``concept_dim`` set, defaulting to D so the predictor width matches
    the end-state payload width)."""
    layer = Layers.SymbolExpectation(
        n_symbols=4, max_depth=8, n_dim=D,
        batch=batch, ltm_capacity=ltm_capacity,
        concept_dim=(D if concept_dim is None else concept_dim),
    )
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    return layer


class _Base(unittest.TestCase):
    def tearDown(self):
        for attr in ("layer",):
            if hasattr(self, attr):
                setattr(self, attr, None)
        _release_allocator_cache()


class TestActivationShape(_Base):
    def test_native_bank_shape_independent_of_observation_depth(self):
        layer = _make_layer(D=4)
        layer.set_symbol_bank(torch.eye(4), orders=torch.arange(4))
        assert layer.expect_next_meaning() is None
        for depth in (3, 1, 2):
            layer.observe_stm_end_state([depth], [torch.eye(4)[:depth]])
            prediction = layer.expect_next_meaning()
            assert prediction.symbol_logits.shape == (4, 2)
            assert prediction.symbol_orders.tolist() == [0, 1, 2, 3]
            assert torch.isfinite(prediction.expected_activation).all()
            assert layer.predict_next_end_state()[0] == 1  # display adapter only

    def test_predictor_parameters_are_registered_once_and_word_shared(self):
        layer = _make_layer(D=4)
        assert layer.word_predictor is layer._inter_predictor
        assert {id(p) for p in layer.parameters()} == {id(p) for p in layer._inter_predictor.parameters()}

    def test_nonfinite_prediction_fails_loudly(self):
        layer = _make_layer(D=4)
        layer.observe_stm_end_state([1], [torch.ones(1, 4)])
        with torch.no_grad():
            next(layer._inter_predictor.parameters()).fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            layer.expect_next_meaning()

    def test_retired_contrastive_objective_is_rejected(self):
        layer = _make_layer(D=4)
        assert not hasattr(layer, 'set_inter_contrastive')
        assert not hasattr(layer, "consume_inter_contrastive_loss")
