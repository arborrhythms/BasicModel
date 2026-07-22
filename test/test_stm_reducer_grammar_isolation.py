"""The bounded STM reducer belongs to its model, not the global XML parser."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(ROOT / "bin"))

import Language  # noqa: E402
from test_word_loop_buckets_and_orders import _grammar_model  # noqa: E402


def test_lazy_stm_reducer_ignores_later_global_grammar(monkeypatch):
    model = _grammar_model()
    expected = model.symbolSpace.languageLayer._binary_layers["CS"]

    # Another model's XML load mutates the process-global Grammar singleton.
    # Recreate that ordering without constructing a second heavyweight model.
    monkeypatch.setattr(Language.TheGrammar, "rules", [])
    object.__setattr__(model, "_stm_reducer_cached", None)

    reducer = model._stm_reducer()

    assert reducer is expected
    assert reducer is model._stm_reducer()
    assert reducer.op_names == expected.op_names
    assert "_stm_reducer_module" not in model._modules
    assert not any(
        key.startswith("_stm_reducer_") for key in model.state_dict())
