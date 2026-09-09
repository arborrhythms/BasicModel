"""The successor as a VP (Alec 2026-09-09): ``next(one) = two``, addition
being iterated succession.  Two facts about the existing ``VerbLayer``
(``VP(NP) = tanh(e^w * atanh(NP))``, ``w`` a sparse readout of the verb
code): (1) SUFFICIENCY BY CONSTRUCTION -- with number codes on a geometric
progression per coordinate one fixed verb advances every noun to its
successor exactly, and counting by iterated application with a codebook
snap reaches every noun; (2) LEARNABILITY -- from the layer's real
zero-initialised readout, gradient descent finds the successor (the
soft threshold's dead zone used to block every gradient at init)."""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Language import VerbLayer  # noqa: E402

D, N = 16, 16


def _nearest(x, codes):
    return torch.cdist(x, codes).argmin(-1)


def _count(layer, codes, v, start=0):
    """Iterated succession WITH the codebook snap after every step."""
    x = codes[start:start + 1]
    path = []
    for _ in range(N - 1 - start):
        x = layer.apply_verb(x, v.unsqueeze(0))
        idx = int(_nearest(x, codes)[0])
        path.append(idx)
        x = codes[idx:idx + 1]                      # the snap: a discrete symbol
    return path


def test_successor_by_construction_on_the_real_verb_layer():
    layer = VerbLayer(D, D)
    with torch.no_grad():
        layer._verb_spec.weight.copy_(torch.eye(D))
        layer._verb_spec.bias.zero_()
    r, a = 1.45, 0.012
    codes = torch.tanh(torch.tensor([[a * (r ** n)] * D for n in range(N)]))
    v = torch.full((D,), math.log(r) + 0.1)        # readout = log r after the threshold
    with torch.no_grad():
        w = layer._verb_spectrum_w(v)
        assert torch.allclose(w, torch.full((D,), math.log(r)), atol=1e-6)
        out = layer.apply_verb(codes[:-1], v.unsqueeze(0).expand(N - 1, -1))
        assert _nearest(out, codes).tolist() == list(range(1, N))
        assert _count(layer, codes, v) == list(range(1, N))
        # exact inverse: the predecessor
        back = layer.unapply_verb(codes[1:], v.unsqueeze(0).expand(N - 1, -1))
        assert _nearest(back, codes).tolist() == list(range(0, N - 1))


def test_readout_receives_gradient_at_zero_init():
    layer = VerbLayer(D, D)
    assert float(layer._verb_spec.weight.abs().sum()) == 0.0    # the real init
    codes = torch.tanh(0.5 * torch.randn(N, D))
    v = 0.5 * torch.randn(D)
    out = layer.apply_verb(codes[:-1], v.unsqueeze(0).expand(N - 1, -1))
    ((out - codes[1:]) ** 2).mean().backward()
    assert float(layer._verb_spec.weight.grad.norm()) > 0.0


def test_successor_is_learned_from_zero_init():
    torch.manual_seed(1)
    layer = VerbLayer(D, D)
    codes_p = torch.nn.Parameter(0.5 * torch.randn(N, D))
    v_p = torch.nn.Parameter(0.5 * torch.randn(D))
    opt = torch.optim.Adam([codes_p, v_p] + list(layer._verb_spec.parameters()), lr=0.02)
    for _ in range(2000):
        opt.zero_grad()
        c = torch.tanh(codes_p)
        out = layer.apply_verb(c[:-1], v_p.unsqueeze(0).expand(N - 1, -1))
        loss = torch.nn.functional.cross_entropy(-torch.cdist(out, c) / 0.05,
                                                 torch.arange(1, N))
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = torch.tanh(codes_p)
        out = layer.apply_verb(c[:-1], v_p.unsqueeze(0).expand(N - 1, -1))
        assert _nearest(out, c).tolist() == list(range(1, N))
        assert _count(layer, c, v_p.detach()) == list(range(1, N))
        assert int((layer._verb_spectrum_w(v_p) != 0).sum()) > 0
