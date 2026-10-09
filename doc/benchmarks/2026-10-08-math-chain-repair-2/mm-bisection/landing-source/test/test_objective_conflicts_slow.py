"""Production-batch objective measurements; opt in with RUN_SLOW=1.

The bounded slow runner uses a 24 GiB worker ceiling for this workload.
The ordinary sweep retains its 8 GiB ceiling and skips these measurements.
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


NATIVE_SLOW_MEMORY_GIB = 24
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_native_production_objective_measurements(tmp_path):
    arm = "ownership"
    """Measure each objective without changing native capacity or the batch."""
    output_root = Path(os.environ.get("OBJECTIVE_CONFLICTS_OUTPUT", str(tmp_path)))
    output = output_root / ("BasicModel_answers_tied_benchmark-" + arm)
    output.mkdir(parents=True, exist_ok=False)
    observer = ROOT / "test/objective_conflicts_probe.py"
    shutil.copyfile(observer, output / "observer-source.py.txt")
    env = dict(os.environ)
    env.pop("BASIC_SEED", None)
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, str(observer), "--config", "BasicModel_answers_tied_benchmark",
             "--arm", arm, "--output", str(output)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    assert result.returncode == 0, f"Objective measurement failed; see {output / 'run.log'}"
    complete = json.loads((output / "complete.json").read_text())
    assert complete["source_matched"]
    ownership = json.loads((output / "ownership.json").read_text())
    assert ownership["conflicts"] == 0
    assert all(not row["writers"] or row["writers"] == [row["owner"]] for row in ownership["parameters"])
    plan = json.loads((output / "plan.json").read_text())
    assert plan["effective"]["architecture"]["training"]["batchSize"] == 28
    assert plan["seed_applied"] is None
    outcome = json.loads((output / "outcome.json").read_text())
    assert outcome["training_batches"] > 0
    assert outcome["evaluation_batches"] > 0
    assert math.isfinite(outcome["last_batch"]["raw"]["totalLoss"])
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "batch_open" and e["size"] == 28 and e["supplied"] for e in events)
    assert any(e["kind"] == "selection" for e in events)
    displacements = [e for e in events if e["kind"] == "optimizer_displacement"]
    assert displacements and all((output / e["file"]).exists() for e in displacements)
    stability = json.loads((output / "derivation-stability.json").read_text())
    assert stability and all(0 < row["modal_fraction"] <= 1 for row in stability)
    if arm == "ownership":
        gradients = json.loads((output / "gradients.json").read_text())
        first = [g for g in gradients if g["scope"] == "trial" and g["pair"] == 0]
        assert {g["trial"] for g in first} == {"exploit", "explore"}
        assert len({g["version_sha256"] for g in first}) == 1
        for g in first:
            assert g["rng_unchanged"]
            assert {"perception", "codes", "chooser", "operators_and_tied_inverses",
                    "generate", "reading_map"} <= g["groups"].keys()
            assert all(math.isfinite(v) for v in g["costs"].values() if v is not None)
