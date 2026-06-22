"""Faithfulness round-trips for the REMAINING grammar ops (idea-decoder, 2026-06-20):
Space-role-A exact (non, tense, aspect, exist, sigma/pi unary, invertible-linear),
Space-role-C stubs (isEqual, isPart, part, query/queryPart/queryEqual),
Space-role-D carrier-dependent (morphology, preposition, symbolize, contextualBind),
legacy (equal, true, swap, copy). Run: .venv/bin/python test/_faithfulness_rest.py
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
from Layers import SigmaLayer, PiLayer

def mx(a, b): return (a - b).abs().max().item()
def ev(B=2, V=2, D=5): return torch.tanh(torch.randn(B, V, D))   # muxed-ish event in (-1,1)
def is_stub(l, r, parent):
    try: return mx(l, parent) == 0.0 and mx(r, parent) == 0.0
    except Exception: return "shape-mismatch"

def line(s): print(s)

print("="*72); print("SPACE_ROLE A — expected EXACT (involution / rotation / identity / LDU)"); print("="*72)

# non — pole complement involution
try:
    n = L.NonLayer(); x = ev();
    line(f"non:        non(non(x))==x  Δ={mx(n.reverse(n.forward(x)), x):.2e}   (exact involution)")
except Exception as e: line(f"non: ERR {e}")

# tense — phase rotation by ±delta (op must be known)
try:
    t = L.TenseLayer(); t.set_op('PAST'); x = ev()
    y = t.forward(x); x2 = t.reverse(y)
    line(f"tense:      reverse(forward(x))==x  Δ={mx(x2, x):.2e}   forward shifted .when Δ={mx(y, x):.2e}  (exact given op)")
except Exception as e: line(f"tense: ERR {e}")

# aspect — identity no-op
try:
    a = L.AspectLayer(); x = ev()
    line(f"aspect:     identity  fwd Δ={mx(a.forward(x), x):.2e}  rev Δ={mx(a.reverse(x), x):.2e}  (deliberate no-op)")
except Exception as e: line(f"aspect: ERR {e}")

# exist — identity wrapper
try:
    e_ = L.ExistLayer(); x = ev()
    line(f"exist:      identity  fwd Δ={mx(e_.forward(x), x):.2e}  rev Δ={mx(e_.reverse(x), x):.2e}  (exact)")
except Exception as e: line(f"exist: ERR {e}")

# sigma / pi unary — exact LDU inverse
for nm, Cls in (("sigma", SigmaLayer), ("pi", PiLayer)):
    try:
        op = Cls(nInput=8, nOutput=8, invertible=True, nonlinear=True); op.eval()
        x = torch.tanh(torch.randn(2, 8))
        line(f"{nm}(unary): reverse(forward(x))==x  Δ={mx(op.reverse(op.forward(x)), x):.2e}   (exact LDU inverse)")
    except Exception as e: line(f"{nm}: ERR {e}")

print(); print("="*72); print("SPACE_ROLE C — expected STUB: reverse = (parent,parent), forward lossy"); print("="*72)

for nm, mk in (
    ("isEqual",   lambda: L.IsEqualLayer()),
    ("isPart",    lambda: L.IsPartLayer()),
    ("part",      lambda: L.PartLayer()),
    ("query",     lambda: L.QueryLayer()),
    ("queryPart", lambda: L.QueryPartLayer()),
    ("queryEqual",lambda: L.QueryEqualLayer()),
):
    try:
        op = mk(); a, b = ev(), ev()
        parent = op.forward(a, b); l, r = op.reverse(parent)
        drop = "fwd=right (drops left)" if mx(parent, b) == 0.0 else ("fwd=max" if mx(parent, torch.maximum(a, b)) < 1e-6 else "fwd=lossy")
        line(f"{nm:10s}: reverse=(parent,parent)? {is_stub(l, r, parent)}   {drop}")
    except Exception as e: line(f"{nm}: ERR {e}")

print(); print("="*72); print("SPACE_ROLE D — carrier-dependent (behaviour COLD, i.e. no token/context/wiring)"); print("="*72)

# morphology — cold (no token) -> passthrough (tense never realized)
try:
    m = L.MorphologyLayer(); x = ev()
    line(f"morphology: cold fwd Δ={mx(m.forward(x), x):.2e}  rev Δ={mx(m.reverse(x), x):.2e}  (cold = identity passthrough; tense needs the surface token)")
except Exception as e: line(f"morphology: ERR {e}")

# preposition — content/.where rotation is exact; marker not recovered (returns x,x)
try:
    p = L.PrepositionLayer(nInput=8); marker = ev(D=8+4); phrase = ev(D=8+4)
    parent = p.forward(marker, phrase); l, r = p.reverse(parent)
    line(f"preposition: phrase recovered (content+.where un-rotated)? Δ(r,phrase)={mx(r, phrase):.2e}   marker slot==phrase (lost)? l==r {mx(l, r):.2e}")
except Exception as e: line(f"preposition: ERR {e}")

# symbolize — cold (unwired spaces) -> fwd=(a+b)/2, rev=balanced split (parent/2,parent/2)
try:
    s = L.SymbolizeLayer(nInput=8); a, b = torch.tanh(torch.randn(8)), torch.tanh(torch.randn(8))
    parent = s.forward(a, b); l, r = s.reverse(parent)
    line(f"symbolize:  cold fwd=(a+b)/2? Δ={mx(parent, (a+b)/2):.2e}   rev=balanced (parent/2)? Δ(l,parent/2)={mx(l, parent/2):.2e}")
except Exception as e: line(f"symbolize: ERR {e}")

# contextualBind — reverse = (parent,parent) stub
try:
    cb = L.ContextualBindLayer(); a, b = ev(), ev()
    parent = cb.forward(a, b); l, r = cb.reverse(parent)
    line(f"contextualBind: reverse=(parent,parent)? {is_stub(l, r, parent)}   (binding irrecoverable; needs parse context)")
except Exception as e: line(f"contextualBind: ERR {e}")

print(); print("="*72); print("LEGACY (Layers.py, dormant — not in role_collapsed.grammar)"); print("="*72)
try:
    from Layers import EqualLayer, TrueLayer, SwapLayer, CopyLayer
    a, b = ev(), ev()
    eq = EqualLayer(); pe = eq.forward(a, b); le, re = eq.reverse(pe)
    line(f"equal:  reverse=(parent,parent)? {is_stub(le, re, pe)}")
    tr = TrueLayer(); pt = tr.forward(a); line(f"true:   reverse identity stub? Δ={mx(tr.reverse(pt), pt):.2e} (pos-only kept, neg destroyed)")
    sw = SwapLayer(); ps = sw.forward(a, b); ls, rs = sw.reverse(ps)
    line(f"swap:   fwd=right? Δ={mx(ps, b):.2e}  reverse=(parent,parent)? {is_stub(ls, rs, ps)}")
    cp = CopyLayer(); pc = cp.forward(a, b); lc, rc = cp.reverse(pc)
    line(f"copy:   fwd=left? Δ={mx(pc, a):.2e}  reverse=(parent,parent)? {is_stub(lc, rc, pc)}")
except Exception as e: line(f"legacy: ERR {e}")
