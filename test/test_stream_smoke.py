"""End-to-end sanity that streaming keeps stream-slot -> sentence mapping."""
import os
from pathlib import Path

import pytest
import torch

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


def test_stream_slot_identity():
    """Run through every DataLoader step on a synthetic train_input and
    assert batch slot ``b`` always receives a sentence from its own
    contiguous slab (i.e. stream ``b``)."""
    import data as data_mod

    # 16 synthetic docs, B=4 -> stream_len=4. Slab 0 = [d0..d3],
    # slab 1 = [d4..d7], etc.
    td = data_mod.Data()
    td.train_input = [f"d{i}" for i in range(16)]
    td.train_output = [torch.zeros(1) for _ in td.train_input]
    loader = td.data_loader(split="train", num_streams=4, num_workers=0,
                            prefetch_factor=None)

    steps_seen = 0
    for step, (inp, _out) in enumerate(loader):
        for b, sentence in enumerate(inp):
            expected_idx = b * 4 + step
            assert sentence == f"d{expected_idx}", (
                f"step {step} row {b}: expected d{expected_idx}, "
                f"got {sentence}"
            )
        steps_seen += 1
    assert steps_seen == 4, f"expected 4 steps, saw {steps_seen}"


@pytest.mark.skipif(not _RUN_SLOW, reason="slow -- set RUN_SLOW=1")
def test_stream_smoke_runs_one_epoch(tmp_path):
    """Full training run of 1 epoch on stream_smoke.xml.

    Requires a downloaded shard. Skipped by default -- run with:
        RUN_SLOW=1 pytest test/test_stream_smoke.py
    """
    import subprocess

    proj = Path(__file__).resolve().parent.parent
    python = proj / ".venv" / "bin" / "python"
    if not python.exists():
        # Fall back to the repo-level venv (basicmodel/.venv can be missing
        # or corrupted; the project root usually has a working interpreter).
        alt = proj.parent / ".venv" / "bin" / "python"
        if alt.exists():
            python = alt
        else:
            pytest.skip("no usable python venv found for subprocess run")
    entry = proj / "bin" / "Models.py"
    config = proj / "data" / "stream_smoke.xml"

    env = {**os.environ, "PYTHONPATH": str(proj / "bin")}
    result = subprocess.run(
        [str(python), str(entry), str(config)],
        cwd=str(proj / "bin"), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"training exited {result.returncode}\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    # Cheap signal: the run must have printed at least one loss line.
    assert "loss" in result.stdout.lower()
