"""Overlapping PS/WS `.where` candidates in the subsymbolic pump."""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

import torch

from Layers import WhereTilingLayer
from eval_where_tiling import evaluate_records, load_jsonl


def _observe(parts, wholes):
    return WhereTilingLayer()(
        torch.tensor(parts, dtype=torch.float32),
        torch.tensor(wholes, dtype=torch.float32))


def test_many_parts_covering_one_whole_routes_sigma():
    out = _observe([[0, 1], [1, 2], [2, 3]], [[0, 3]])
    assert out["n_parts"].tolist() == [[3]]
    assert out["part_runs"].tolist() == [[1]]
    assert out["part_cover"].tolist() == [[True]]
    assert out["sigma_whole"].tolist() == [[True]]
    assert out["sigma_part"].tolist() == [[True, True, True]]


def test_one_part_covered_by_immediate_wholes_routes_pi():
    out = _observe([[0, 4]], [[0, 2], [2, 4]])
    assert out["n_wholes"].tolist() == [[2]]
    assert out["whole_runs"].tolist() == [[1]]
    assert out["whole_cover"].tolist() == [[True]]
    assert out["pi_part"].tolist() == [[True]]
    assert out["pi_whole"].tolist() == [[True, True]]


def test_exact_object_settles_and_remains_part_of_larger_whole():
    # The two children agree exactly at their own level and simultaneously
    # form the immediate parts of [0,4].  Settling is not a global stop.
    out = _observe(
        [[0, 2], [2, 4]],
        [[0, 2], [2, 4], [0, 4]])
    assert out["settled_part"].tolist() == [[True, True]]
    assert out["settled_whole"].tolist() == [[True, True, False]]
    assert out["sigma_whole"].tolist() == [[False, False, True]]
    assert out["sigma_part"].tolist() == [[True, True]]


def test_gap_raises_instead_of_inventing_boundary():
    out = _observe([[0, 1], [2, 3]], [[0, 3]])
    assert out["n_parts"].tolist() == [[2]]
    assert out["part_runs"].tolist() == [[2]]
    assert out["part_cover"].tolist() == [[False]]
    assert out["raise_whole"].tolist() == [[True]]
    assert out["sigma_whole"].tolist() == [[False]]


def test_routes_are_per_row_not_batch_amax():
    parts = torch.tensor([
        [[0, 1], [1, 2]],       # row 0: sigma
        [[0, 2], [0, 0]],       # row 1: exact/stable
    ], dtype=torch.float32)
    wholes = torch.tensor([
        [[0, 2]],
        [[0, 2]],
    ], dtype=torch.float32)
    out = WhereTilingLayer()(parts, wholes)
    assert out["sigma_whole"].tolist() == [[True], [False]]
    assert out["settled_whole"].tolist() == [[False], [True]]
    assert out["part_route"].tolist() == [
        [WhereTilingLayer.ROUTE_SIGMA, WhereTilingLayer.ROUTE_SIGMA],
        [WhereTilingLayer.ROUTE_SETTLED, WhereTilingLayer.ROUTE_NULL],
    ]


def test_refinement_schedule_reaches_exact_identity():
    layer = WhereTilingLayer()
    parts = torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 0]]],
                         dtype=torch.float32)
    wholes = torch.tensor([[[0, 3]]], dtype=torch.float32)
    schedule = layer.build_schedule(parts, wholes, passes=3)
    assert schedule[0]["sigma_whole"].tolist() == [[True]]
    assert schedule[1]["frontier_part_spans"][0, 0].tolist() == [0.0, 3.0]
    assert schedule[1]["settled_whole"].tolist() == [[True]]
    assert schedule[-1]["accepted_whole"].tolist() == [[True]]
    assert int(schedule[-1]["overflow"][0]) == 0


def test_curated_utf8_corpus_is_exact():
    records = load_jsonl(ROOT / "test" / "fixtures" /
                         "where_tiling_corpus.jsonl")
    result = evaluate_records(records, passes=3)
    assert result["f1"] == 1.0, result["failures"][:3]
    assert result["exact_sentence"] == 1.0, result["failures"][:3]
    assert result["overflow"] == 0


def _build_experiment():
    import Language
    import Models
    from util import init_config
    cfg = ROOT / "data" / "MM_overlap_tiling.xml"
    init_config(path=str(cfg), defaults_path=str(ROOT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(str(cfg))
    return model


def test_model_callosum_carries_tiling_to_final_cs():
    import Models
    model = _build_experiment()
    assert model.overlap_where_tiling
    Models.TheData.load("xor")
    loader = model.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = model.inputSpace.prepInput(items)
    with torch.no_grad():
        model.forward(x)
    last = getattr(model, "_combine_last_cs_sub", None)
    tiling = getattr(last, "_where_tiling", None)
    assert isinstance(tiling, dict)
    assert tiling["part_valid"].shape[0] == 4
    assert tiling["whole_valid"].shape[0] == 4
    assert torch.isfinite(last.materialize()).all()
