"""Task 4 of the `.where`/`.when` encoding pass (2026-07-04 plan): BLIND
decode -- the tiling re-derived from the `.where` BAND, the forward
scaffold demoted to an explicit debug/fallback mode.

Mechanism (design sec 1.3): per active slot, decode the coarse `.where`
claim (magnitude floor gates pads -- a real stamp is ~1.0, pads carry
none); consecutive claim differences hypothesize tile SIZES; placement is
the running sum of emitted part sizes (the type-tiling reading) with the
absolute claim CONFIRMING; the two-arm association (size-restricted,
then unrestricted) is unchanged.

The SCAFFOLD-MASKING CURRICULUM (Alec 2026-07-04): ``decode_blind_rate``
masks that fraction of the scaffold tiles -- masked tiles decode BLIND
while the rest are given (the training bridge). rate 1.0 == fully blind;
rate None/0.0 == the 5c/5d scaffold path, byte-identical (the
content-identity regression pin keeps running there).
"""

import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch

import recon_bench
from recon_bench import run_config


def _seed(seed=0):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _trained_model(epochs=3, config="data/MM_20M_xor.xml"):
    _seed(0)
    model, dev, lr, bs = recon_bench._build_model(
        recon_bench._resolve_config(config))
    opt = model.getOptimizer(lr=lr)
    for _ in range(int(epochs)):
        model.runEpoch(optimizer=opt, batchSize=bs, split="train")
    ti, _ = model.inputSpace.getTestData()
    n = int(ti.shape[0]) if torch.is_tensor(ti) else len(ti)
    model.set_sigma(0)
    model.train(False)
    with torch.no_grad():
        model.runEpoch(batchSize=max(1, min(n, 512)), split="test")
    return model


def test_blind_rate_default_is_scaffold():
    """rate None (default) keeps the scaffold path: meta is identical to
    an explicit rate 0.0 decode (the 5c/5d content-identity regression)."""
    model = _trained_model()
    psp = model.perceptualSpace
    thunk = psp._recovered_input_thunk
    assert thunk is not None and thunk[0] == "radix"
    _tag, radix, event, sub = thunk
    assert getattr(psp, "decode_blind_rate", "MISSING") is None
    base = psp._decode_radix_meta(radix, event, sub)
    psp.decode_blind_rate = 0.0
    zero = psp._decode_radix_meta(radix, event, sub)
    assert base["tokens"] == zero["tokens"]
    assert base["offsets"] == zero["offsets"]


def _synthetic_event(psp, radix, sub, texts=("world", " ", "world")):
    """A crafted percept event with EXACT band stamps: the mechanism tests
    drive the decode with byte-exact `.where` claims (the trained-regime
    claim precision the design measured, 0.73 bytes at P=8192, arrives at
    the E~80 crossing -- the RUN_SLOW bar owns that; unit tests must not
    lean on undertrained claims)."""
    _tag, _r, event0, _s = psp._recovered_input_thunk
    if event0.dim() == 2:
        event0 = event0.unsqueeze(0)
    N, D = int(event0.shape[1]), int(event0.shape[-1])
    ev = torch.zeros(1, N, D)
    w_idx = [int(i) for i in sub.whereEncoding.resolve(D)]
    off = 0
    spans = []
    for n, t in enumerate(texts):
        bs = t.encode("utf-8")
        pid = radix.get_id(bs) if hasattr(radix, "get_id") else None
        vec = radix.vector_for(pid) if pid is not None else None
        assert vec is not None, f"radix store must know {t!r}"
        ev[0, n, :vec.shape[-1]] = vec
        ev[0, n, w_idx] = sub.whereEncoding.encode(float(off))
        spans.append((off, off + len(bs)))
        off += len(bs)
    fwd = {"indices": torch.zeros(1, N, dtype=torch.long),
           "tile_spans": [spans + [(-1, -1)] * (N - len(texts))],
           "percept_store": radix, "word_groups": None,
           "word_texts": [list(texts)], "tokens": [list(texts)]}
    return ev, fwd, [t for t in texts]


def test_blind_decode_ignores_scaffold_spans():
    """rate 1.0 re-derives the tiling from the BAND alone: the scaffold
    record may be poisoned and the decode must not read it; offsets are
    the running sum of emitted sizes, sized by claim diffs."""
    model = _trained_model()
    psp = model.perceptualSpace
    _tag, radix, _ev, sub = psp._recovered_input_thunk
    ev, fwd, texts = _synthetic_event(psp, radix, sub)
    saved = psp._forward_input
    poisoned = dict(fwd)
    poisoned["tile_spans"] = [[(-1, -1)] * len(fwd["tile_spans"][0])]
    psp._forward_input = poisoned
    psp.decode_blind_rate = 1.0
    try:
        blind = psp._decode_radix_meta(radix, ev, sub)
    finally:
        psp._forward_input = saved
        psp.decode_blind_rate = None
    assert blind["words"][0] == ["world", " ", "world"], blind["words"][0]
    assert blind["offsets"][0] == [0, 5, 6], blind["offsets"][0]


def test_blind_size_hypotheses_from_claim_diffs():
    """Consecutive `.where` claim differences hypothesize the tile sizes
    (all but the last tile), so arm (a) size-restricted association still
    fires blind; the last tile falls back to arm (b) (size=None)."""
    model = _trained_model()
    psp = model.perceptualSpace
    _tag, radix, _ev, sub = psp._recovered_input_thunk
    ev, fwd, _texts = _synthetic_event(psp, radix, sub)
    saved = psp._forward_input
    psp._forward_input = None          # no scaffold record at all
    psp.decode_blind_rate = 1.0
    calls = []
    orig = radix.associate_span

    def spy(content, size=None):
        calls.append(size)
        return orig(content, size=size)

    radix.associate_span = spy
    try:
        psp._decode_radix_meta(radix, ev, sub)
    finally:
        radix.associate_span = orig
        psp._forward_input = saved
        psp.decode_blind_rate = None
    # exact claims [0, 5, 6] -> diffs {5, 1}; the last tile has no
    # successor -> arm (b) unrestricted (None).
    assert calls[:3] == [5, 1, None], calls


def test_curriculum_fraction_masks_some_tiles():
    """0 < rate < 1 decodes the masked fraction blind (claim-diff sizes)
    and keeps the rest scaffold-fed: with exact claims the mixed decode
    reproduces the pure-scaffold rendering."""
    model = _trained_model()
    psp = model.perceptualSpace
    _tag, radix, _ev, sub = psp._recovered_input_thunk
    ev, fwd, _texts = _synthetic_event(psp, radix, sub)
    saved = psp._forward_input
    psp._forward_input = fwd
    try:
        psp.decode_blind_rate = 0.0
        scaffold = psp._decode_radix_meta(radix, ev, sub)
        psp.decode_blind_rate = 0.5
        mixed = psp._decode_radix_meta(radix, ev, sub)
    finally:
        psp._forward_input = saved
        psp.decode_blind_rate = None
    assert mixed["words"][0] == scaffold["words"][0]
    assert mixed["offsets"][0] == scaffold["offsets"][0]


def test_recon_bench_blind_flag(tmp_path):
    """recon_bench --blind computes the decode + where_recovery blind
    (default); --scaffold keeps the debug/fallback path."""
    rec_blind = run_config("data/MM_20M_xor.xml", epochs=3, seed=0,
                           out_dir=str(tmp_path), blind=True)
    assert rec_blind.notes.get("decode_mode") == "blind"
    rec_scaf = run_config("data/MM_20M_xor.xml", epochs=3, seed=0,
                          out_dir=str(tmp_path), blind=False)
    assert rec_scaf.notes.get("decode_mode") == "scaffold"
    # The scaffold run reproduces the Gate-A re-baseline point. RE-PINNED
    # 0.75 -> 1.0 (per-vector order-raise fix, 2026-07-07): the sigma/pi fold
    # now raises each word's order over its own D features independently (no
    # cross-word feature leak), which lifts E=3 exact match 0.75 -> 1.0. Same
    # value as the roundtrip trajectory pin in test_reconstruction_roundtrip.
    assert rec_scaf.exact_match_rate == 1.0
    assert rec_scaf.where_recovery == 1.0


@pytest.mark.xfail(reason="Gate-B blind round-trip DEFERRED (Alec 2026-07-05): "
                          "the closing knob (.where band training pressure vs "
                          "longer budgets vs accepting scaffold-fed) is Alec's "
                          "call; the mechanism is proven by the synthetic-stamp "
                          "tests above -- xfail until the band matures.",
                   strict=False)
@pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                    reason="~70s (build + pinned epochs) -- RUN_SLOW gates the bar")
def test_mm20m_xor_blind_roundtrip(tmp_path):
    """THE Gate-2b bar: scaffold OFF, tiling re-derived from the band,
    exact_match == 1.0 at the pinned budget.

    XFAIL (Alec 2026-07-05): deferred to Alec's band-precision knob (see
    the marker reason + the Gate-B EXECUTION NOTES). Kept RUN_SLOW-gated so
    it stays an explicit expected-failure of the bar, not a silent skip.

    STATUS: RED (Gate B, 2026-07-04) -- the bar's premise (the E~80
    byte-exact claim crossing, from the FIXTURE probes) does not hold in
    the full-model regime: measured claim errors vs true [0, 5, 6] are
    [8, 22, 34] bytes at E=80 and [7, 0, 4] at E=200 (converging, slow);
    the xor size inventory {1, 5, 6} needs sub-half-byte claims to
    separate 5 from 6, so arm (a) mis-restricts and exact stays 0.0
    (magnitude gating is PERFECT there -- real 0.997+ vs pads <=0.11).
    The mechanism itself is verified by the synthetic-stamp tests above.
    Deliberately left RED per the nWhere-fix precedent: the closing knob
    (more where-band training pressure vs longer budgets vs accepting
    scaffold-fed until the curriculum matures the band) is Alec's call --
    see the encoding plan's Gate-B EXECUTION NOTES."""
    from test_reconstruction_roundtrip import EPOCHS_PINNED
    rec = run_config("data/MM_20M_xor.xml", epochs=EPOCHS_PINNED, seed=0,
                     out_dir=str(tmp_path), blind=True)
    assert rec.exact_match_rate == 1.0
    assert rec.where_recovery == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
