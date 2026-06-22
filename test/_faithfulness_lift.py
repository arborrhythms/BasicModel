"""Ad-hoc faithfulness probe for LiftLayer (idea-decoder review, 2026-06-20).
Run: .venv/bin/python test/_faithfulness_lift.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
sys.path.insert(0, _BIN)
import torch
torch.manual_seed(0)

# minimal config so TheXMLConfig.get(...) works during layer construction
try:
    from util import init_config
    _DATA = os.path.join(os.path.dirname(_BIN), "data")
    init_config(path=os.path.join(_DATA, "model.xml"),
                defaults_path=os.path.join(_DATA, "model.xml"))
except Exception as e:
    print("config init note:", e)

from Language import LiftLayer

D = 8
def rand_idea(B=4):
    return torch.tanh(torch.randn(B, 1, D))     # valid atanh domain (-1,1)

def mx(a, b):
    return (a - b).abs().max().item()

print("=" * 64)
print("LIFT faithfulness (content-only, verbEigEdit OFF)")
print("=" * 64)
lift = LiftLayer(nInput=D); lift.eval()
NP, VP = rand_idea(), rand_idea()

# (1) forward then reverse, then forward again: the SUM round-trip
parent = lift.forward(NP, VP)
l, r = lift.reverse(parent)
parent2 = lift.forward(l, r)
print(f"(1) forward(*reverse(parent)) == parent ?   max|Δ| = {mx(parent2, parent):.2e}")
print(f"    reverse gives left==right (balanced)?   max|l-r| = {mx(l, r):.2e}")

# (2) reverse then forward: do we recover the DISTINCT operands NP, VP?
print(f"(2) reverse(forward(NP,VP)) recovers NP?    max|l-NP| = {mx(l, NP):.2e}")
print(f"    reverse(forward(NP,VP)) recovers VP?    max|r-VP| = {mx(r, VP):.2e}")
print(f"    (baseline: NP vs VP differ by          max|NP-VP|= {mx(NP, VP):.2e})")

print()
print("=" * 64)
print("LIFT with a NON-ZERO verb edit (design B: verb leaves a trace)")
print("=" * 64)
lift2 = LiftLayer(nInput=D); lift2.eval()
# force the edit on with a real (non-zero) projection so it actually perturbs
import torch.nn as nn
from Language import _make_lex_gate
object.__setattr__(lift2, "_verb_eig_edit", True)
edit = _make_lex_gate(D, D, seed=0xABCD, bias=0.0)
with torch.no_grad():
    edit.weight.normal_(0, 0.5)          # a real, trained-like edit
object.__setattr__(lift2, "_lex_edit", edit)

parentE = lift2.forward(NP, VP)                 # applies the verb edit
lE, rE = lift2.reverse(parentE)                 # reverse does NOT invert the edit
parentE2 = lift2.forward(lE, rE)
# how much does the un-inverted edit corrupt the round-trip?
print(f"(3) edit ON: forward(*reverse) == parent ?  max|Δ| = {mx(parentE2, parentE):.2e}")
# how big is the edit's footprint vs the plain fold (the 'trace' magnitude)?
plain = lift2._sigma.compose(NP, VP)
print(f"    verb-edit footprint on the result        max|Δ| = {mx(parentE, plain):.2e}")
print(f"    -> a non-zero footprint that reverse drops = the lost VP trace")
