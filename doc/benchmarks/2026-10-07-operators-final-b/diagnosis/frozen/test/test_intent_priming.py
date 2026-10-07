"""Intent priming over both codebook towers (GrammarOpsPass §5).

Author sign-off 2026-06-11: meaning-sensitivity enters as ATTENTION
over symbols — one current-intent code (the §6c pump-zero gist: the
parallel parse does the priming) produces boost-above-unity weights
per tower row by graded similarity (one matmul per tower), priming
recognition on the PS/extent tower alongside retrieval on the
SS/intent tower. Priming guides codebook focus only — never rule
dispatch. Pinned here:

  * the producer is monotone in similarity and identity (1.0) for
    dissimilar rows; ``None`` intent is the byte-identical off-path;
  * priming a tower biases recognition rankings MONOTONICALLY in the
    boost weight (the VQ row selection consults the boosts);
  * the same intent code moves BOTH towers' selection in the same
    semantic direction;
  * intent off => recognition byte-identical to today;
  * the SS boosts merge into the taxonomy priming buffer (the existing
    retrieval plumbing: priming_kwargs_for_slots -> recommender).
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_D = 8
_V = 5


def _rows(seed=3):
    torch.manual_seed(seed)
    W = torch.randn(_V, _D)
    return W / W.norm(dim=-1, keepdim=True)


# -- The producer ---------------------------------------------------------

def test_producer_monotone_in_similarity_identity_for_dissimilar():
    from Spaces import intent_priming_weights
    W = _rows()
    intent = W[2].clone()
    boosts = intent_priming_weights(intent, W, gain=1.0)
    assert boosts.shape == (_V,)
    assert torch.all(boosts >= 1.0)
    # The aligned row gets the top boost (1 + gain at perfect alignment).
    assert int(boosts.argmax()) == 2
    assert boosts[2] == pytest.approx(2.0, abs=1e-5)
    # An anti-aligned intent is identity for that row (clamp at 0).
    anti = intent_priming_weights(-W[2], W, gain=1.0)
    assert anti[2] == pytest.approx(1.0, abs=1e-6)
    # Gain scales the boost-above-unity part monotonically.
    g2 = intent_priming_weights(intent, W, gain=3.0)
    assert torch.all(g2 >= boosts)
    assert g2[2] == pytest.approx(4.0, abs=1e-4)


def test_producer_off_paths():
    from Spaces import intent_priming_weights
    W = _rows()
    assert intent_priming_weights(None, W) is None
    assert intent_priming_weights(W[0], None) is None
    assert intent_priming_weights(W[0], torch.zeros(0, _D)) is None


# -- Primed recognition: the VQ row selection -----------------------------

def _vq_with_rows(W):
    from Layers import VectorQuantize
    vq = VectorQuantize(dim=_D, codebook_size=_V)
    with torch.no_grad():
        vq.codebook = W.clone()
    vq.eval()
    return vq


def test_vq_selection_biased_monotonically_by_boost():
    """An input between two rows flips to the primed row as its boost
    grows; growth never un-flips it (monotone)."""
    torch.manual_seed(5)
    W = torch.zeros(_V, _D)
    W[0, 0] = 1.0
    W[1, 1] = 1.0
    for i in range(2, _V):
        W[i, i] = 1.0
    vq = _vq_with_rows(W)
    # Slightly nearer row 0 than row 1; far from the rest.
    x = torch.zeros(1, _D)
    x[0, 0] = 1.0
    x[0, 1] = 0.9

    def select(boost_for_row1):
        boosts = torch.ones(_V)
        boosts[1] = boost_for_row1
        vq.selection_boost_fn = (lambda V, device, _b=boosts:
                                 _b.to(device))
        try:
            out = vq(x)
        finally:
            vq.selection_boost_fn = None
        # quantized output identifies the selected row.
        q = out[0] if isinstance(out, tuple) else out
        return int((q.reshape(-1, _D) @ W.T).argmax(dim=-1)[0])

    unprimed = select(1.0)
    assert unprimed == 0
    picks = [select(b) for b in (1.0, 1.1, 1.5, 2.0, 4.0)]
    # Once the boost flips the selection to row 1, it stays flipped.
    flipped = [p == 1 for p in picks]
    assert flipped[-1], "a strong boost must win the selection"
    assert flipped == sorted(flipped), f"non-monotone selection: {picks}"


def test_vq_selection_byte_identical_with_no_intent():
    torch.manual_seed(9)
    W = _rows()
    x = torch.randn(4, _D)
    vq = _vq_with_rows(W)
    out_plain = vq(x)
    q_plain = out_plain[0] if isinstance(out_plain, tuple) else out_plain
    # Producer installed but returning None (no intent set).
    vq.selection_boost_fn = lambda V, device: None
    out_none = vq(x)
    q_none = out_none[0] if isinstance(out_none, tuple) else out_none
    assert torch.equal(q_plain, q_none)


# -- Space-level wiring ---------------------------------------------------

def _stub_space(W):
    """A minimal Space carrying the subspace->codebook->vq chain."""
    from Spaces import Space
    sp = Space.__new__(Space)
    vq = _vq_with_rows(W)
    cb = SimpleNamespace(vq=vq, getW=lambda: W)
    object.__setattr__(sp, 'subspace', SimpleNamespace(codebook=lambda: cb))
    return sp, vq


def test_space_set_intent_and_install():
    from Spaces import Space
    W = _rows(seed=11)
    sp, vq = _stub_space(W)
    assert Space.install_intent_priming(sp) is True
    assert Space.intent_boosts(sp) is None
    boosts = Space.set_intent(sp, W[3], gain=2.0)
    assert boosts is not None and int(boosts.argmax()) == 3
    # The installed producer serves the cached boosts sized to V.
    served = vq.selection_boost_fn(_V, torch.device('cpu'))
    assert torch.allclose(served, boosts)
    # Pads with identity when the VQ has more rows than the boosts.
    served_wide = vq.selection_boost_fn(_V + 3, torch.device('cpu'))
    assert served_wide.shape == (_V + 3,)
    assert torch.all(served_wide[_V:] == 1.0)
    # Clearing the intent restores the byte-identical off state.
    assert Space.set_intent(sp, None) is None
    assert vq.selection_boost_fn(_V, torch.device('cpu')) is None


def test_same_intent_moves_both_towers_same_direction():
    """One intent code, two towers: each tower's top-boosted row is the
    row semantically aligned with the intent."""
    from Spaces import Space
    torch.manual_seed(13)
    intent = torch.randn(_D)
    intent = intent / intent.norm()
    W_ps = _rows(seed=17)
    W_ss = _rows(seed=19)
    with torch.no_grad():
        W_ps[1] = intent           # PS row 1 carries the intent's extent
        W_ss[4] = intent           # SS row 4 carries the intent's word
    ps, _ = _stub_space(W_ps)
    ws, _ = _stub_space(W_ss)
    b_ps = Space.set_intent(ps, intent)
    b_ss = Space.set_intent(ws, intent)
    assert int(b_ps.argmax()) == 1
    assert int(b_ss.argmax()) == 4
    # Same direction: both towers' aligned rows get the SAME top boost.
    assert b_ps[1] == pytest.approx(float(b_ss[4]), abs=1e-5)


# -- SS retrieval plumbing: the taxonomy merge ----------------------------

def test_taxonomy_prime_with_weights_max_merges():
    from Language import Taxonomy
    tax = Taxonomy()
    tax.allocate_priming(batch_size=2, capacity=6, live=4)
    tax.prime([1], batch=0, boost=3.0)          # existing write: 4.0
    weights = torch.tensor([1.0, 2.0, 1.5, 1.0])
    tax.prime_with_weights(weights)             # merge into every row
    pm = tax.priming_mask()
    assert pm[0, 1] == pytest.approx(4.0)       # max keeps the stronger
    assert pm[1, 1] == pytest.approx(2.0)
    assert pm[0, 2] == pytest.approx(1.5)
    assert pm[0, 0] == pytest.approx(1.0)
    # Batch-targeted merge touches only that row.
    tax.prime_with_weights(torch.tensor([5.0, 1.0, 1.0, 1.0]), batch=1)
    pm = tax.priming_mask()
    assert pm[1, 0] == pytest.approx(5.0)
    assert pm[0, 0] == pytest.approx(1.0)
    # Unallocated buffer / None weights: silent no-ops.
    bare = Taxonomy()
    bare.prime_with_weights(weights)
    tax.prime_with_weights(None)
