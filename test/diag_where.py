#!/usr/bin/env python3
"""Diagnostic: trace where encoding through forward-reverse cycle in MM_xor."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
import Models
import Spaces
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from util import TheXMLConfig, TheDevice
from data import TheData

TheDevice.set("cpu")
TheData.load("xor")

model = Models.MentalModel()
model.create_from_config(config_path="data/MM_xor.xml", data=TheData)
model.eval()

we = model.inputSpace.subspace.whereEncoding
dt = we.div_term
print(f"WhereEncoding: maxVal={we.maxVal}, div_term={dt:.8f}, index={we.index}")
print(f"  offset 0 → sin={math.sin(0):.6f}, cos={math.cos(0):.6f}")
print(f"  offset 6 → sin={math.sin(6*dt):.6f}, cos={math.cos(6*dt):.6f}")
print(f"  angular diff = {6*dt:.6f} rad")
print()

# Run one forward pass
batch, _ = model.inputSpace.getBatch(0, batchSize=4, split="test")
test_tensor, output_tensor = batch

with torch.no_grad():
    # Hook into ConceptualSpace to capture pre/post sigma
    cs = model.conceptualSpace

    # Save original forward to wrap
    orig_fwd = cs.forward.__func__

    captured = {}
    def hooked_forward(self, vspace):
        x = self.forwardBegin(vspace, returnVectors=True)
        captured['cs_input'] = x.clone()
        if self.nonlinear:
            nW = self.subspace.nWhat
            x_what = torch.atanh(x[:, :, :nW] * (1 - 1e-6))
            x = torch.cat([x_what, x[:, :, nW:]], dim=-1)
        captured['cs_pre_sigma'] = x.clone()
        y = self.forwardSigma(x)
        captured['cs_post_sigma'] = y.clone()
        ws = getattr(self, 'wordSpace', None)
        c_sl = getattr(ws, 'conceptualSyntacticLayer', None) if ws is not None else None
        if c_sl is not None:
            y, self._last_svo = c_sl.compose(y, self.subspace, Spaces.TheGrammar)
            captured['cs_post_compose'] = y.clone()
        vspace = self.forwardEnd(y, returnVectors=True)
        vspace.normalize("concepts", target="what")
        vspace.normalize("concepts", target="activation")
        return vspace

    import types
    cs.forward = types.MethodType(hooked_forward, cs)

    input_state, symbols, outputData = model.forward(test_tensor)

    # Restore
    cs.forward = types.MethodType(orig_fwd, cs)

nWhat_in = model.inputSpace.subspace.nWhat
nWhere_in = model.inputSpace.subspace.nWhere

# Input vectors
input_vecs = model.inputSpace.subspace.materialize()
print(f"=== INPUT [B, N, D]={list(input_vecs.shape)} ===")
for v in range(6):
    norm = input_vecs[0, v].norm().item()
    where = input_vecs[0, v, nWhat_in:nWhat_in+nWhere_in].tolist()
    print(f"  vec{v}: norm={norm:.4f}, where={[f'{w:.6f}' for w in where]}")
print()

# ConceptualSpace input (percepts + symbol feedback)
ci = captured.get('cs_input')
if ci is not None:
    print(f"=== CS INPUT [B, N, D]={list(ci.shape)} ===")
    for v in range(ci.shape[1]):
        norm = ci[0, v].norm().item()
        print(f"  vec{v:2d}: norm={norm:.4f}")
    print()

# Post-sigma
ps = captured.get('cs_post_sigma')
if ps is not None:
    print(f"=== CS POST-SIGMA [B, N, D]={list(ps.shape)} ===")
    for v in range(ps.shape[1]):
        norm = ps[0, v].norm().item()
        print(f"  vec{v:2d}: norm={norm:.4f}")
    print()

# Post-compose
pc = captured.get('cs_post_compose')
if pc is not None:
    print(f"=== CS POST-COMPOSE [B, N, D]={list(pc.shape)} ===")
    for v in range(pc.shape[1]):
        norm = pc[0, v].norm().item()
        print(f"  vec{v:2d}: norm={norm:.4f}")
    print()

# Now reverse
with torch.no_grad():
    input_data, input_latent = model.reverse(symbols, outputData)

recon_vecs = model.inputs.materialize()
print(f"=== RECONSTRUCTED [B, N, D]={list(recon_vecs.shape)} ===")
for v in range(recon_vecs.shape[1]):
    norm = recon_vecs[0, v].norm().item()
    where = recon_vecs[0, v, nWhat_in:nWhat_in+nWhere_in].tolist()
    print(f"  vec{v}: norm={norm:.4f}, where={[f'{w:.6f}' for w in where]}")
