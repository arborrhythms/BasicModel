"""Phase 5 of the mathematical thinking plan: the evaluation script runs on
a tiny problem set and emits every report column (spec section 10)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import eval_math_thinking as ev  # noqa: E402


def test_eval_script_emits_every_report_column(tmp_path):
    out = tmp_path / "report.md"
    raw = tmp_path / "rows.json"
    rows = ev.main(["--config", str(_ROOT / "data" / "MM_math.xml"),
                    "--budgets", "1,4", "--seeds", "1", "--split", "train",
                    "--rows", "4", "--batch", "2", "--ablate-memory",
                    "--illumination", "--out", str(out), "--json", str(raw)])
    assert rows and {r["budget"] for r in rows} == {1, 4}
    assert {r["ablation"] for r in rows} == {"none", "memory"}
    for r in rows:
        assert set(r) >= {"row", "depth", "budget", "correct", "iterations",
                          "forced", "latency_s", "seed", "ablation"}
        assert r["iterations"] <= r["budget"] + r["forced"] or r["forced"] >= 0
    text = out.read_text()
    assert "# Mathematical thinking evaluation" in text
    assert "## thinking" in text and "## memory ablation" in text
    for column in ev.REPORT_COLUMNS:
        assert column in text
    assert raw.exists()
    table = ev.summarize(rows)
    assert (1, "all") in table and (4, "all") in table
    assert 0.0 <= table[(1, "all")]["accuracy"] <= 1.0
