"""Tests for bin/recon_bench.py (2026-07-03 reconstruction-fidelity Task 1)."""
import json
from types import SimpleNamespace

from recon_bench import DECODE_NOTE, run_config, RunRecord, exact_match_rate


def test_exact_match_rate_is_1_on_identity():
    assert exact_match_rate(["a b c", "d e"], ["a b c", "d e"]) == 1.0
    assert exact_match_rate(["a b c"], ["a b X"]) == 0.0
    assert exact_match_rate([], []) == 0.0


def test_run_record_schema_smoke(tmp_path):
    """One epoch on a tiny fixture yields a complete run record."""
    rec = run_config("data/MM_xor_fixture.xml", epochs=1, seed=0,
                     out_dir=str(tmp_path))
    assert isinstance(rec, RunRecord)
    d = json.loads((tmp_path / rec.filename).read_text())
    for key in ("config", "seed", "epochs", "wall_s_per_epoch",
                "output_loss", "recon_loss", "exact_match_rate",
                "where_recovery", "channel_losses", "device",
                "compile_mode", "host", "timestamp"):
        assert key in d, key
    assert 0.0 <= d["exact_match_rate"] <= 1.0
    assert d["wall_s_per_epoch"] > 0
    assert isinstance(d["notes"], dict)
    assert d["notes"]["epoch_times_s"]


def _stub_model(batch_inputs, rendered):
    """Model stand-in exposing the two decode-path attributes."""
    return SimpleNamespace(
        inputSpace=SimpleNamespace(_last_sentences=list(batch_inputs)),
        perceptualSpace=SimpleNamespace(
            reconstruct_data=lambda text=False: list(rendered)))


def test_decode_rows_align_with_last_eval_batch(tmp_path, monkeypatch):
    """Targets and decoded rows correspond 1:1 (row order, not membership)."""
    import recon_bench

    # Identity pairing scores 1.0; permuting TARGETS drops it to 0.0.
    rows = ["hello there", "loving there"]
    t, d = recon_bench._decode_texts(_stub_model(rows, rows))
    assert exact_match_rate(t, d) == 1.0
    t, d = recon_bench._decode_texts(_stub_model(rows[::-1], rows))
    assert exact_match_rate(t, d) == 0.0

    # Force the 4-row fixture split to span two eval batches (cap=2): the
    # decode must cover exactly the last batch and say so in the note.
    monkeypatch.setattr(recon_bench, "_MAX_EVAL_BATCH", 2)
    rec = run_config("data/MM_xor_fixture.xml", epochs=1, seed=0,
                     out_dir=str(tmp_path))
    d = json.loads((tmp_path / rec.filename).read_text())
    notes = d["notes"]
    assert DECODE_NOTE not in notes, notes.get(DECODE_NOTE)
    assert "2 of 4 rows" in notes.get("exact_match_note", "")
    assert notes["decoded_rows"] == 2
    assert 0.0 <= d["exact_match_rate"] <= 1.0


def test_byte_lexer_config_decodes(tmp_path):
    """MM_20M_legacy (byte cursor) decodes via the _last_host_slab stash."""
    rec = run_config("data/MM_20M_legacy.xml", epochs=1, seed=0,
                     out_dir=str(tmp_path), max_batches=1)
    d = json.loads((tmp_path / rec.filename).read_text())
    notes = d["notes"]
    assert DECODE_NOTE not in notes, notes.get(DECODE_NOTE)
    assert notes["decoded_rows"] > 0
    assert 0.0 <= d["exact_match_rate"] <= 1.0
