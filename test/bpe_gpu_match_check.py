"""Keystone correctness check for the GPU BPE tokenizer.

Before the (intricate) _embed_bpe rewire, verify the foundational
piece: ``bpe_gpu.gpu_chunk_ids`` reproduces ``ChunkLayer.forward``'s
exact per-row chunk-id sequence on real frozen-MM_5M byte buffers.
``ChunkLayer.forward`` (the trie walk) is the ground truth; any
divergence here is a tokenization bug -> stop, do not proceed to the
rewire.

Frozen-vocab contract: force ``word_learning=0`` before the run so the
vocab cannot grow (train_step no-op), matching the CPU-pretrain ->
freeze -> GPU-train workflow the GPU path requires.

Usage: .venv/bin/python test/bpe_gpu_match_check.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "none"
os.environ["BASICMODEL_DEVICE"] = "cpu"

_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin"))
sys.path.insert(0, str(_p / "bin"))

import torch
from data import TheData
from Models import BaseModel
from util import init_config, init_device
import Spaces
from Layers import BPEGpuLayer


def main():
    init_device("cpu")
    cfg = str(_p / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=64)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    ps = m.perceptualSpace
    cl = ps.chunk_layer
    cl.word_learning = 0  # FREEZE (frozen-vocab GPU-train contract)

    tables = BPEGpuLayer.build_static_tables(cl, ps.subspace.what,
                                             torch.device("cpu"))
    print(f"[match-check] frozen vocab: V={tables['V']} "
          f"maxL={tables['maxL']}")

    captured = []
    orig = Spaces.PerceptualSpace._embed_bpe

    def _wrap(self, upstream_vspace):
        wb = upstream_vspace.materialize(mode="what")
        if wb is not None:
            bi = (wb[..., 0] if wb.dim() == 3 else wb).long()
            captured.append(bi.detach().clone())
        return orig(self, upstream_vspace)

    Spaces.PerceptualSpace._embed_bpe = _wrap
    try:
        opt = m.getOptimizer(lr=1e-4)
        m.runEpoch(optimizer=opt, batchSize=8, split="train",
                   max_batches=3)
    finally:
        Spaces.PerceptualSpace._embed_bpe = orig

    if not captured:
        print("[match-check] NO byte buffers captured -- abort")
        sys.exit(2)

    total_rows = 0
    mismatches = 0
    for cidx, byte_buf in enumerate(captured):
        # Ground truth: the trie walk.
        ref_chunks, _ = cl.forward(byte_buf)         # list[list[int]]
        bid, blen = BPEGpuLayer.gpu_longest_match(byte_buf, tables)
        gpu_ids, gpu_cnt = BPEGpuLayer.gpu_chunk_ids(byte_buf, bid, blen)
        B = byte_buf.shape[0]
        for b in range(B):
            total_rows += 1
            ref = list(ref_chunks[b])
            n = int(gpu_cnt[b].item())
            got = gpu_ids[b, :n].tolist()
            if got != ref:
                mismatches += 1
                if mismatches <= 5:
                    # First divergent position for debugging.
                    k = next((j for j in range(min(len(ref), len(got)))
                              if ref[j] != got[j]), min(len(ref), len(got)))
                    print(f"[match-check] MISMATCH call#{cidx} row#{b}: "
                          f"len ref={len(ref)} gpu={len(got)}; "
                          f"first diff @tok {k}: ref={ref[max(0,k-2):k+3]} "
                          f"gpu={got[max(0,k-2):k+3]}")

    if mismatches:
        print(f"[match-check] FAIL: {mismatches}/{total_rows} rows "
              f"diverge from the trie walk")
        sys.exit(1)
    print(f"[match-check] PASS: {total_rows} rows -- GPU longest-match "
          f"+ consumption bit-identical to ChunkLayer.forward")


if __name__ == "__main__":
    main()
