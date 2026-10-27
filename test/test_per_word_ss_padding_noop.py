"""Padding columns of the static per-word loop are no-ops.

Real lexical deposits are counted, excluding host closing ticks. Padding
never changes a chunk. The word-aligned symbolic handoff has zero presence
and evidence outside the active prefix.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.4-2.6.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("MODEL_DEBUG", "0")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import pytest
import torch


def _build_gate_model():
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device
    init_device("cpu")
    cfg = str(_root / "data" / "MM_ladder.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    return m.to("cpu")


@pytest.mark.slow
def test_stm_depth_tracks_valid_len_not_N(monkeypatch):
    """The active lexical leaves push; padded rows preserve every chunk field."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(["hello world"])
    in_sub = m._lex_embed_stem(inputTensor)
    L = int(isp._valid_len_host)
    N = int(isp.outputShape[0])
    if not (0 < L < N):
        pytest.skip(f"input lacks padding columns (L={L}, N={N}); "
                    "test needs an active prefix shorter than N")
    stm = m.conceptualSpace.stm
    if stm is None:
        pytest.skip("model has no STM")
    # A grammatical closing reduces the stack. Count the physical pushes
    # at the current tensor boundary, then check the concluded depth too.
    from Layers import ShortTermMemory
    original = ShortTermMemory.functional_push_step_masked
    pushes = []
    def observed(*args):
        gate = args[2].reshape(-1).bool()
        pushes.append(gate.clone())
        result = original(*args)
        if bool((~gate).any()):
            for before, after in zip(args[0], result):
                torch.testing.assert_close(after[~gate], before[~gate], rtol=0, atol=0)
        return result
    monkeypatch.setattr(ShortTermMemory, 'functional_push_step_masked', staticmethod(observed))
    m.forward(inputTensor)
    active = isp._word_active_mask
    # The current attention reader schedules lexical leaves and a separate
    # closing tick. The closing marker and unselected entries are not pushes.
    leaves = active & isp._ar_grammar_leaf_mask
    assert int(torch.stack(pushes).sum()) == int(leaves.sum())
    assert int(leaves.sum()) == 2  # the two words in this fixed input
    assert int(active.sum()) < active.numel()



@pytest.mark.slow
def test_concept_buf_zero_past_active_prefix():
    """The current word-aligned handoff is empty at every padded position."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(["hello world"])
    in_sub = m._lex_embed_stem(inputTensor)
    L = int(isp._valid_len_host)
    N = int(isp.outputShape[0])
    if not (0 < L < N):
        pytest.skip(f"input lacks padding columns (L={L}, N={N})")
    m._forward_body_per_word(in_sub)
    # The full W-wide concept history retired; SymbolSpace now owns its
    # word-aligned row/presence/evidence handoff, while STM owns the chunks.
    symbol = m.symbolSpace
    active = isp._word_active_mask
    presence = symbol._word_reference_presences
    evidence = symbol._word_reference_evidence
    assert presence.shape[:2] == active.shape
    assert bool((~active).any())
    assert torch.all(presence[~active] == 0)
    assert torch.all(evidence[~active] == 0)
    assert torch.all(symbol._word_reference_rows[~active] == -1)
