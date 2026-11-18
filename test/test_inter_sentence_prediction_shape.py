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

    def test_predictor_parameters_are_registered_once_and_word_shared(self):
        layer = _make_layer(D=4)
        assert layer.word_predictor is layer._inter_predictor
        assert {id(p) for p in layer.parameters()} == {id(p) for p in layer._inter_predictor.parameters()}


    def test_retired_contrastive_objective_is_rejected(self):
        layer = _make_layer(D=4)
        assert not hasattr(layer, 'set_inter_contrastive')
        assert not hasattr(layer, "consume_inter_contrastive_loss")
