"""Regression: runtime-split ingestion via ``runEpoch(split="runtime")`` and
``store_truths`` must work end-to-end.

Two pre-existing bugs (each masked the next) blocked truth ingestion -- the
documented ``store_truths`` path
``with TheData.runtime_batch(texts): self.runEpoch(split="runtime")``:

  1. ``runEpoch``'s ``inference`` early-branch called ``runBatch`` WITHOUT a
     ``batch_override`` (the retired ARIR fast-path), so every
     ``runEpoch(split="runtime")`` raised "runBatch: no batch_override
     supplied". FIX: runtime shares the normal data-driven loop with the eval
     splits (training=False); ``data_loader`` maps ``runtime`` -> the
     ``train_input`` that ``runtime_batch`` stages.
  2. With (1) fixed, ``store_truths`` (which sets ``truthCriterion=0`` to record
     every gold truth) reached the WS truth-recording path, where the
     EVENT-width activation was handed to the CONTENT-width ``TruthLayer`` and
     size-mismatched ``_pending_truths``. FIX: conform the recorded vector to
     the TruthLayer width (slice the where/when tail) at the record site.

Run on CPU; MM_grammar is serial (sentenceProtocol ON), so store_truths
exercises the §6c prelude recording path that crashed.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
_DATA = os.path.join(_PROJECT, "data")

_CACHE = []


def _model():
    """Build the (serial, truth-layer-bearing) MM_grammar model once."""
    if not _CACHE:
        from util import init_config
        import Language
        import Models
        init_config(path=os.path.join(_DATA, "MM_grammar.xml"),
                    defaults_path=os.path.join(_DATA, "model.xml"))
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(
                os.path.join(_DATA, "MM_grammar.xml"))
        Models.TheData.load("xor")
        _CACHE.append((m, Models))
    return _CACHE[0]


class TestRuntimeSplitIngestion(unittest.TestCase):
    def test_runepoch_runtime_split_no_raise(self):
        # Bug 1: this raised "runBatch: no batch_override supplied".
        m, Models = _model()
        with torch.no_grad(), Models.TheData.runtime_batch(
                ["hello world", "hello there"]):
            out = m.runEpoch(batchSize=2, split="runtime")
        self.assertIsInstance(out, tuple)

    def test_data_loader_runtime_maps_to_train_input(self):
        # ``runtime`` has no ``runtime_input`` attr; it must map to the
        # ``train_input`` runtime_batch staged (was an AttributeError on
        # ``getattr(self, "runtime_input")`` once runtime reached data_loader).
        m, Models = _model()
        with Models.TheData.runtime_batch(["aa", "bb", "cc"]):
            loader = m.inputSpace.data.data_loader(
                split="runtime", num_streams=3)
            self.assertEqual(loader.dataset.num_streams, 3)

    def test_store_truths_records_truths(self):
        # Bug 2: the WS event-width activation size-mismatched the
        # content-width TruthLayer; store_truths now records end-to-end.
        m, Models = _model()
        tl = m.symbolSpace.truth_layer
        m.store_truths([{"content": "hello world", "trust": 0.9},
                        {"content": "loving there", "trust": 0.4}])
        self.assertGreater(int(tl.count.item()), 0)

    def test_store_truths_idempotent_clear_then_record(self):
        # store_truths clears + repopulates; a second call is independent.
        m, Models = _model()
        tl = m.symbolSpace.truth_layer
        m.store_truths([{"content": "hello world", "trust": 0.8}])
        first = int(tl.count.item())
        m.store_truths([{"content": "loving world", "trust": 0.6}])
        self.assertGreater(int(tl.count.item()), 0)
        self.assertGreater(first, 0)


if __name__ == "__main__":
    unittest.main()
