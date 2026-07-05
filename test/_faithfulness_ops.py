"""Faithfulness round-trips for the grammar ops Alec listed (idea-decoder, 2026-06-20):
lower (DET), not (negation), intersection (ADJ), union (ADV), conjunction (and),
disjunction (or). Run: .venv/bin/python test/_faithfulness_ops.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
sys.path.insert(0, _BIN)
import torch
torch.manual_seed(0)
try:
    from util import init_config
    _D = os.path.join(os.path.dirname(_BIN), "data")
    init_config(path=os.path.join(_D, "model.xml"), defaults_path=os.path.join(_D, "model.xml"))
except Exception as e:
    print("cfg note:", e)

import Language as L
from Layers import Ops

def mx(a, b): return (a - b).abs().max().item()
class Basis:                      # minimal codebook shim: reverse() calls basis.getW()
    def __init__(self, W): self._W = W
    def getW(self): return self._W

D = 8
def idea(): return torch.tanh(torch.randn(2, 1, D))

print("="*70); print("LOWER (DET)  — Pi fold, expected: exact-on-sum, balanced split"); print("="*70)
low = L.LowerLayer(nInput=D); low.eval()
NP, VP = idea(), idea()
p = low.forward(NP, VP); l, r = low.reverse(p); p2 = low.forward(l, r)
print(f"  forward(*reverse)==parent   max|Δ|={mx(p2,p):.2e}   (exact round-trip on the sum)")
print(f"  balanced split l==r         max|l-r|={mx(l,r):.2e}")
print(f"  recovers NP / VP?           |l-NP|={mx(l,NP):.2e}  |r-VP|={mx(r,VP):.2e}  (NP-VP={mx(NP,VP):.2e})")

print(); print("="*70); print("NOT (negation) — expected: exact involution"); print("="*70)
neg = L.NotLayer(); neg.eval()
x = torch.tanh(torch.randn(2, 3, 4))            # last dim >=2 (bivector + tail)
once = neg.forward(x); twice = neg.reverse(once)
print(f"  not(not(x))==x              max|Δ|={mx(twice,x):.2e}")
print(f"  not(x)!=x (actually flips)  max|Δ|={mx(once,x):.2e}")

# --- lossy set/logical ops: test (a) no-basis stub, (b) recommender reconstruction ---
K = 16
def run_lossy(name, layer, op_fn, monotonic):
    W = torch.rand(K, D)                          # nonneg codebook (clean same-sign min/max)
    a, b = W[3], W[9]
    parent = op_fn(a, b)
    # (a) no-basis path
    try:
        s1, s2 = layer.reverse(parent)
        stub = (mx(s1, parent) == 0.0 and mx(s2, parent) == 0.0)
    except Exception as e:
        stub = f"err: {e}"
    # (b) recommender path
    try:
        x1, x2 = layer.reverse(parent, basis=Basis(W))
        recon = mx(op_fn(x1, x2), parent)
        in_cb = any((x1 - W[k]).abs().max() < 1e-5 for k in range(K)) and \
                any((x2 - W[k]).abs().max() < 1e-5 for k in range(K))
        rec = f"reconstruction op(x1,x2) vs parent max|Δ|={recon:.2e}; operands are codebook rows={in_cb}"
    except Exception as e:
        rec = f"err: {e}"
    print(f"  no-basis -> (parent,parent) stub? {stub}")
    print(f"  with-basis recommender: {rec}")

print(); print("="*70); print("INTERSECTION (ADJ) — radmin; expected: stub w/o basis, recommender w/ basis"); print("="*70)
inter = L.IntersectionLayer(monotonic=True); inter.eval()
run_lossy("intersection", inter, lambda a, b: Ops.intersection(a, b, monotonic=True), True)

print(); print("="*70); print("UNION (ADV)"); print("="*70)
uni = L.JoinLayer(monotonic=True); uni.eval()
run_lossy("union", uni, lambda a, b: Ops.union(a, b, monotonic=True), True)

print(); print("="*70); print("CONJUNCTION (and) — monotonic min"); print("="*70)
conj = L.ConjunctionLayer(); conj.eval()
run_lossy("conjunction", conj, lambda a, b: torch.minimum(a, b), True)

print(); print("="*70); print("DISJUNCTION (or) — monotonic max"); print("="*70)
disj = L.DisjunctionLayer(); disj.eval()
run_lossy("disjunction", disj, lambda a, b: torch.maximum(a, b), True)
