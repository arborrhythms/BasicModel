"""End-to-end SVO extraction on a real English sentence.

This test loads a trained LM_5M checkpoint, runs the model forward on
"the cat chased the mouse", extracts the chart's Viterbi derivation
trace, and verifies that the SVO triple matches the expected
(subject="cat", verb="chased", object="mouse").

It is xfail until two artifacts exist:
  - data/LM_5M.kv     (trained word vectors / lexicon)
  - data/LM_5M.ckpt   (trained model weights)

Both are produced by sufficiently long training runs of LM_5M.xml.
Until they exist, the chart's POS scoring is uniform-ish noise (the
lex_cat_scorer is randomly initialised; the codebook category_logits
buffer is zeros) so the Viterbi extract picks degenerate rules and
extract_svo returns None or the wrong tokens.

When the model is trained well enough that the chart commits to
``S = lift(NP, VP)`` over ``VP = intersection(V, O)`` for this
sentence, the test will start passing (xpass = unexpected pass), at
which point the @pytest.mark.xfail decorator should be removed.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "bin"))
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

CKPT = PROJECT / "data" / "LM_5M.ckpt"
KV = PROJECT / "data" / "LM_5M.kv"
CONFIG = PROJECT / "data" / "LM_5M.xml"

SENTENCE = "the cat chased the mouse"
EXPECTED_SUBJECT_TOKEN = "cat"
EXPECTED_VERB_TOKEN = "chased"
EXPECTED_OBJECT_TOKEN = "mouse"


def _trained_artifacts_present():
    """Both ckpt and kv must exist AND have non-trivial content.

    The ckpt is ~14 MB even from a smoke-run because the model itself
    has 5M params, so size doesn't distinguish trained-vs-not. The kv
    file IS a good proxy: a smoke-run leaves only the 257-entry BPE
    seed (~17 KB) plus maybe a NULL_PERCEPT slot; a real training run
    over even 1 fineweb shard produces thousands of vocabulary entries
    and a kv file in the multi-MB range.
    """
    if not (CKPT.exists() and KV.exists()):
        return False
    return KV.stat().st_size > 1_000_000  # > 1 MB kv → trained vocab


@pytest.mark.xfail(
    not _trained_artifacts_present(),
    reason=(
        "Requires a trained LM_5M.ckpt + LM_5M.kv with non-trivial "
        "vocabulary. The chart can't pick correct grammar rules until "
        "the lex_cat_scorer and category_logits have been updated by "
        "many batches of training. Remove the xfail marker after a real "
        "training run has produced these artifacts."
    ),
    strict=False,
)
def test_svo_extraction_on_real_sentence():
    """Forward "the cat chased the mouse" through trained LM_5M; verify
    that chart.extract_svo() returns operand tensors whose
    nearest-codebook decoding matches subject=cat, verb=chased, object=mouse."""
    from util import init_config, TheXMLConfig
    from data import TheData
    from Models import BaseModel

    init_config(path=str(CONFIG),
                defaults_path=str(PROJECT / "data" / "model.xml"))

    cfg = TheXMLConfig.data
    arch = cfg.get("architecture", {})
    dat = arch.get("data", {})
    TheData.load(dat.get("dataset"), num_shards=1, max_docs=10,
                 shard_dir=dat.get("shardDir"), dat=dat)

    m, _ = BaseModel.from_config(str(CONFIG), data=TheData)
    m.eval()

    batch = TheData.stringTensor(SENTENCE).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        m(batch)

    chart = m.wordSubSpace.chart
    svo = chart.extract_svo()
    assert svo is not None, (
        "chart.extract_svo() returned None — no S = lift(NP, VP) over "
        "VP = intersection(V, O) signature found in the derivation trace"
    )
    subj_vec, verb_vec, obj_vec = svo
    assert subj_vec.shape[0] == 1
    assert verb_vec.shape[0] == 1
    assert obj_vec.shape[0] == 1

    # Decode each operand to its nearest codebook atom, then resolve
    # that atom to a surface token via the InputSpace's word vectors.
    sym_sub = m.symbolicSpace.subspace
    cb_W = sym_sub.what.getW()
    assert cb_W is not None and torch.is_tensor(cb_W)

    def _nearest_token(vec):
        D_min = min(vec.shape[-1], cb_W.shape[-1])
        sims = (vec[0, 0, :D_min].unsqueeze(0)
                @ cb_W[:, :D_min].T).squeeze(0)
        atom_idx = int(sims.argmax().item())
        wv = getattr(m.inputSpace, 'wv', None)
        idx_to_key = getattr(wv, 'index_to_key', None) if wv else None
        if idx_to_key is not None and 0 <= atom_idx < len(idx_to_key):
            return str(idx_to_key[atom_idx])
        return f"atom_{atom_idx}"

    subj_tok = _nearest_token(subj_vec)
    verb_tok = _nearest_token(verb_vec)
    obj_tok  = _nearest_token(obj_vec)

    assert subj_tok == EXPECTED_SUBJECT_TOKEN, (
        f"SVO subject mismatch: expected {EXPECTED_SUBJECT_TOKEN!r}, "
        f"got {subj_tok!r}")
    assert verb_tok == EXPECTED_VERB_TOKEN, (
        f"SVO verb mismatch: expected {EXPECTED_VERB_TOKEN!r}, "
        f"got {verb_tok!r}")
    assert obj_tok == EXPECTED_OBJECT_TOKEN, (
        f"SVO object mismatch: expected {EXPECTED_OBJECT_TOKEN!r}, "
        f"got {obj_tok!r}")
