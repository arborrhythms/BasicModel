"""Regression tests for Inductor-safe serial radix part padding.

The raw constituent axis remains batch-dynamic, but it is identity-padded to
at least three positions so its declared dynamic range agrees with Inductor's
gather/set-fold guards.  Padding must never create another real constituent or
change the synthesized word.
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["BASICMODEL_DEVICE"] = "cpu"

import pytest
import torch


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "bin"))

import Language
import Models
from util import init_config, init_device


@pytest.fixture(scope="module")
def serial_model():
    """Build the smallest committed serial-radix integration fixture."""
    init_device("cpu")
    torch.manual_seed(0)
    cfg = _ROOT / "data" / "MM_mereology_serial.xml"
    init_config(path=str(cfg), defaults_path=str(_ROOT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(str(cfg))
    return model


def _stage_one_part_word(model):
    model._compiled_step = None
    batch = model.inputSpace.prepInput(["a"])
    with torch.no_grad():
        model._lex_embed_stem(batch)
    return batch


def test_single_real_part_is_identity_padded_to_three(serial_model):
    """P=1 becomes P=3 without changing the mask or synthesized word."""
    _stage_one_part_word(serial_model)
    input_space = serial_model.inputSpace
    part_space = serial_model.perceptualSpace

    ids = input_space._ar_word_part_ids[:, 0, :]
    mask = input_space._ar_word_part_mask[:, 0, :]
    offsets = input_space._ar_word_part_offsets[:, 0, :]

    assert ids.shape == mask.shape == offsets.shape == (1, 3)
    assert mask.tolist() == [[True, False, False]]
    assert offsets.tolist() == [[0, -1, -1]]

    with torch.no_grad():
        padded = part_space.synthesize_word_parts(ids, mask, offsets)
        real_only = part_space.synthesize_word_parts(
            ids[:, :1], mask[:, :1], offsets[:, :1])

    torch.testing.assert_close(padded, real_only, rtol=0, atol=0)
    torch.testing.assert_close(
        padded, input_space._ar_embedded_N[:, :1], rtol=0, atol=0)


def test_begin_step_declares_p3_and_flattened_3w_minima(
        serial_model, monkeypatch):
    """Every staged part view uses min P=3; flattened views use min 3*W."""
    calls = []

    def mark_dynamic_spy(tensor, axis, **bounds):
        calls.append((tensor, axis, bounds))

    monkeypatch.setattr(torch._dynamo, "mark_dynamic", mark_dynamic_spy)
    serial_model._compiled_step = lambda *args, **kwargs: None
    batch = serial_model.inputSpace.prepInput(["a"])
    with torch.no_grad():
        serial_model._begin_step(batch)

    input_space = serial_model.inputSpace
    forward_input = serial_model.perceptualSpace._forward_input
    word_width = int(input_space._active_word_bucket)

    part_views = (
        input_space._ar_word_part_ids,
        input_space._ar_word_part_mask,
        input_space._ar_word_part_offsets,
        forward_input["word_part_indices"],
        forward_input["word_part_mask"],
    )
    flat_views = tuple(forward_input[name] for name in (
        "indices", "seed_event", "word_groups", "part_spans",
        "percept_where",
    ))

    for expected in part_views:
        matching = [bounds for tensor, axis, bounds in calls
                    if tensor is expected and axis == 2]
        assert matching, f"part view {tuple(expected.shape)} was not marked"
        assert all(bounds["min"] == 3 for bounds in matching)

    for expected in flat_views:
        matching = [bounds for tensor, axis, bounds in calls
                    if tensor is expected and axis == 1]
        assert matching, f"flat view {tuple(expected.shape)} was not marked"
        assert all(bounds["min"] == 3 * word_width for bounds in matching)
