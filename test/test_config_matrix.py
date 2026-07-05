"""Task 7 canonical config-matrix SMOKE suite (plan
doc/plans/2026-07-03-reconstruction-fidelity-execution.md).

Seven configs span the reconstruction pipeline's control surface: the three
base paths (grammar meronomy/meronomy, xor parallel-mereology sO=0, legacy
bpe/byte) plus three data/matrix/ variants that each flip ONE architecture
flag off its parent (mereologyRaise off, readingAttention on, two-pass
learning+exploreTemperature on), plus the sO=3 sparse-concept wave (smoke
ONLY -- the wave is dark on prod xor). Each row is a fast SMOKE:
build the config, run one capped epoch, decode the reconstruction, and assert
finite losses + a clean decode path (no decode_note degradation, the Task-1
review caveat).

The exact-round-trip BAR lives in test_reconstruction_roundtrip.py and is
RUN_SLOW-gated there -- this file does not duplicate it (see the RUN_SLOW tier
note at the bottom). Grammar has NO round-trip pin at all: the grammatical
derivation round-trip is a DEFERRED design fork (plan Task 6 EXECUTION NOTES --
the serial single-S reduce caps the reverse at one slot; Alec's next design
decision), so grammar rides the matrix at the SMOKE tier only.
"""
import math
import os

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest

from recon_bench import DECODE_NOTE, run_config

# The xor dataset (4 rows) is ONE batch/epoch at every config's batchSize
# (plan Task 3 baselines), so max_batches=1 is a whole epoch for the
# xor-dataset configs and caps anything that would iterate further. The wall
# is build-bound (see the _SLOW note below for the per-tier timing).
SMOKE_BATCHES = 1

# Each config is a fresh build; the whole config matrix is dominated by
# build time (max_batches=1 makes the epoch itself ~one xor batch). Measured
# cpu/eager seed 0 on ArborBook: build-bound (grammar variants ~11-12s each
# under load); the FAST tier (5 rows below) stays under make test's budget.
# The two heaviest grammar variants (readingAttention, two-pass learning --
# the double-forward) gate behind RUN_SLOW: their serial-grammar variant
# BUILD is already smoked by the base grammar row, and make test_all runs
# the full matrix. See plan Task 7 EXECUTION NOTES for the per-config timing
# table.
_SLOW = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="heaviest grammar variant; make test_all (RUN_SLOW) runs it")

MATRIX = [
    pytest.param("data/MM_20M_grammar.xml", id="grammar"),                # predominant path (meronomy pair)
    pytest.param("data/MM_20M_xor.xml", id="xor"),                        # parallel mereology, sO=0
    pytest.param("data/MM_20M_legacy.xml", id="legacy"),                  # bpe/byte back-compat
    pytest.param("data/matrix/MM_20M_xor_noraise.xml", id="xor_noraise"),  # mereologyRaise off
    pytest.param("data/matrix/MM_20M_grammar_reading.xml", id="grammar_reading", marks=_SLOW),  # readingAttention on (RUN_SLOW)
    pytest.param("data/matrix/MM_20M_grammar_twopass.xml", id="grammar_twopass", marks=_SLOW),  # learning + exploreTemperature (RUN_SLOW)
    pytest.param("data/MM_sparse_concept.xml", id="sparse_concept"),      # sO=3 parallel (wave; smoke ONLY)
]


@pytest.mark.parametrize("cfg", MATRIX)
def test_config_builds_runs_and_reconstructs(cfg, tmp_path):
    """One capped epoch: builds, runs, finite losses, recon decodes cleanly."""
    rec = run_config(cfg, epochs=1, seed=0, out_dir=str(tmp_path),
                     max_batches=SMOKE_BATCHES)
    assert math.isfinite(rec.output_loss), (cfg, rec.output_loss)
    assert math.isfinite(rec.recon_loss), (cfg, rec.recon_loss)
    assert rec.exact_match_rate >= 0.0        # decode path executed
    # Decode did not degrade to the timing-only record (Task-1 review caveat).
    assert DECODE_NOTE not in rec.notes, (cfg, rec.notes.get(DECODE_NOTE))


# RUN_SLOW round-trip tier (plan Task 7.2): the exact-round-trip BAR is
# test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip (xor,
# EPOCHS_PINNED=38, RUN_SLOW-gated THERE) -- this matrix file deliberately
# does NOT duplicate it. There is NO grammar round-trip: the grammatical
# derivation round-trip is a DEFERRED design fork (plan Task 6 EXECUTION
# NOTES -- the serial single-S reduce caps the reverse at one slot, so no
# config choice reaches the bar; Alec's next design decision, the same
# family as the blind-tiling Gate-2b fork). Grammar rides this matrix at the
# SMOKE tier only.
