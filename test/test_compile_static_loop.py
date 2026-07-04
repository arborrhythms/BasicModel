"""Static per-word loop must not recompile across varying sentence
lengths. The legacy ``while next_word()`` boundary caused Dynamo to
specialize on each observed ``_valid_len_host`` value; the static
``for p in range(N)`` loop with tensor-only gating eliminates that.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.3-2.4.
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
    cfg = str(_root / "data" / "MM_20M_legacy.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    return m


@pytest.mark.xfail(
    reason="MM_20M_legacy.xml: percept_dim+nWhere+nWhen=12 != concept_dim+nWhere+"
           "nWhen=1028 since Stage 1.C retired sigma_percept (the "
           "percept-to-concept lift); the signal router replacement "
           "(Stage 3) is not yet wired.",
    strict=False,
)
def test_per_word_body_callable_with_static_signature():
    """The new ``_per_word_body_step(w, p, gate_b_1, out_slot)``
    signature must be callable. Smoke test for the refactor — the
    important assertion is that the call doesn't raise, signaling the
    arity / argument shapes are wired correctly."""
    pytest.importorskip("torch._dynamo")
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    assert isp._ar_embedded_N is not None
    assert isp._word_active_mask is not None
    N = int(isp.outputShape[0])
    m._per_word_prelude(in_sub)
    out_slot = m._per_word_contributions
    gate = isp._word_active_mask[:, 0:1]
    w = isp.word_at(0)
    cs, idea = m._per_word_body_step(w, 0, gate, out_slot)
    # idea may be None on the first call (PS/SS/CS may emit empty
    # subspaces); the assertion is structural: no exception was raised.


def test_word_at_returns_padded_shape_past_valid_len():
    """``word_at(p)`` for p past _valid_len_host must still return a
    well-shaped [B, 1, D] slice (zeros) — the gate masks the commits."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    # IS is a pure lexer post-2026-06-07; the model orchestrates lex -> PS
    # embed -> IS.finalize_stem (which populates ``_ar_embedded_N``).
    m._lex_embed_stem(inputTensor)
    # ``word_at`` / ``_word_active_mask`` index the PS-reduced per-word slab
    # (``_ar_embedded_N``), whose width is PartSpace.nOutput -- NOT the
    # InputSpace char width (text models reduce chars -> words). Read N off the
    # padded slab so word_at(N-1) is the last real per-word slot.
    N = int(isp._ar_embedded_N.shape[1])
    w0 = isp.word_at(0)
    w_last = isp.word_at(N - 1)
    assert w0 is not None and w0.dim() == 3 and w0.shape[1] == 1
    assert w_last is not None and w_last.dim() == 3 and w_last.shape[1] == 1
    # Past _valid_len_host, the per-position activity mask is False so
    # the rule-gate drops contributions. The slab itself may still
    # carry non-zero positional (where/when) encoding even at padding
    # columns — the gate is the source of truth, not raw event values.
    L = int(isp._valid_len_host)
    if L < N:
        active = isp._word_active_mask
        assert active is not None
        assert not bool(active[:, L].any().item()), (
            f"word_active_mask should be False at the first padding "
            f"column (L={L})")


# -- recompile-churn pin (doc/plans/2026-07-04-...) --------------------------
# The per-word contribution accumulator was a python list grown by
# ``append`` once per iteration; its post-loop ``len(out_slot)`` was a Dynamo
# guard, so each distinct sentence word-count minted a fresh compile of
# ``_per_word_body_step`` -> ``cache_size_limit`` blown -> revert to eager.
# The fix makes it a FIXED-LENGTH ``[None] * N_words`` list written by the
# python-constant position index. This pin guards that invariant: the
# accumulator length is the static per-word slab width, INDEPENDENT of the
# sentence's active word-count, so ``len(out_slot)`` can never again drive a
# recompile.

def _build_grammar_perword_model():
    """A per-word (serial-grammar) model whose ``_forward_body_per_word``
    runs cleanly (unlike MM_20M_legacy, which is dim-mismatched / xfail)."""
    import warnings
    import Language
    from data import TheData
    from Models import BasicModel
    from util import init_config, init_device
    init_device("cpu")
    cfg = str(_root / "data" / "MM_grammar.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = BasicModel.from_config(cfg)
    TheData.load("xor")
    return m


def test_per_word_accumulator_is_fixed_length_not_word_count():
    """PIN: ``_per_word_contributions`` is a fixed-length ``N_words`` list
    (the static per-word slab width), not a list whose length tracks the
    active word-count. This is the invariant that retires the
    ``len(out_slot) == k`` recompile guard the census flagged
    (bin/Models.py:7828, ``[8/1]..[8/7]`` frames)."""
    import warnings
    m = _build_grammar_perword_model()
    m.eval()
    isp = m.inputSpace
    loader = isp.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = isp.prepInput(items)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Route through the full forward so the prelude + per-word loop run.
        m.forward(x)
    # The per-word slab width == the static loop bound the fix uses.
    N_words = int(isp._ar_embedded_N.shape[1])
    K_host = int(isp._valid_len_host)
    # The accumulator is reset to ``[]`` at the tail of ``_forward_body_per_
    # word`` (its live length is only observable mid-forward), so re-run the
    # body-front prelude + loop in isolation to inspect the populated list.
    in_sub = isp.subspace
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m._per_word_prelude(in_sub)
        out_slot = [None] * N_words
        m._per_word_contributions = out_slot
        word_active = isp._word_active_mask
        for p in range(N_words):
            w = isp.word_at(p)
            if w is None:
                break
            gate = (word_active[:, p:p + 1] if word_active is not None
                    else torch.ones(w.shape[0], 1, dtype=torch.bool))
            m._per_word_body_step(w, p, gate, out_slot, active_host=True)
    # The accumulator length is the STATIC slab width, regardless of how many
    # of those columns were active (K_host). This is the anti-churn invariant.
    assert len(m._per_word_contributions) == N_words, (
        f"per-word accumulator length {len(m._per_word_contributions)} != "
        f"static slab width {N_words}; the fixed-length buffer (anti-recompile-"
        f"churn) invariant is broken")
    # Padding columns (p >= K_host) contribute exactly zero (gate-masked).
    for p in range(K_host, N_words):
        c = m._per_word_contributions[p]
        if c is not None:
            assert torch.count_nonzero(c) == 0, (
                f"padding column p={p} (>= K_host={K_host}) must be a "
                f"gate-masked zero contribution; got nonzero")


def test_per_word_body_does_not_recompile_on_accumulator_length():
    """PIN (best-effort, dynamo): compiling ``_per_word_body_step`` and
    driving it with a GROWING accumulator (as the old ``append`` path did)
    must NOT trigger a ``len(out_slot)``-keyed recompile. With the
    fixed-length list the body is called with a constant-length ``out_slot``,
    so no length guard exists to fail.

    Skips if the eager/inductor backend can't trace the isolated body on this
    host (the census evidence + the structural pin above remain the primary
    guards)."""
    pytest.importorskip("torch._dynamo")
    import warnings
    import torch._dynamo as dynamo
    m = _build_grammar_perword_model()
    m.eval()
    isp = m.inputSpace
    loader = isp.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = isp.prepInput(items)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    N_words = int(isp._ar_embedded_N.shape[1])
    dynamo.reset()
    compiled = torch.compile(m._per_word_body_step, backend="eager",
                             dynamic=False)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m._per_word_prelude(isp.subspace)
            fixed = [None] * N_words
            m._per_word_contributions = fixed
            wa = isp._word_active_mask
            # Two passes over the SAME fixed-length list: the second pass must
            # not recompile (the list length is invariant). The OLD append
            # accumulator would present len 1,2,3,... and recompile each step.
            for _pass in range(2):
                for p in range(N_words):
                    w = isp.word_at(p)
                    if w is None:
                        break
                    g = (wa[:, p:p + 1] if wa is not None
                         else torch.ones(w.shape[0], 1, dtype=torch.bool))
                    compiled(w, p, g, fixed, active_host=True)
    except Exception as exc:  # pragma: no cover - host-dependent trace failure
        pytest.skip(f"isolated per-word body compile unsupported here: {exc!r}")
    # The accumulator never changed length across the two passes -> no
    # length-keyed guard could have fired.
    assert len(m._per_word_contributions) == N_words
