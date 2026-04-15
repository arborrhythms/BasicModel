#!/usr/bin/env python3
"""Debug: compare parameters collected by ergodic vs non-ergodic getOptimizer."""

import os, sys
_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import tempfile, xml.etree.ElementTree as ET

def create(ergodic):
    xml_path = os.path.join(_PROJECT, "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    erg_elem = root.find("architecture/ergodic")
    if erg_elem is None:
        erg_elem = ET.SubElement(root.find("architecture"), "ergodic")
    erg_elem.text = "true" if ergodic else "false"
    auto_elem = root.find("architecture/autoload")
    if auto_elem is None:
        auto_elem = ET.SubElement(root.find("architecture"), "autoload")
    auto_elem.text = "false"
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True)
    tmp.close()
    Models.TheData.load("xor")
    m = Models.BasicModel()
    m.create_from_config(tmp.name, data=Models.TheData)
    os.unlink(tmp.name)
    return m

# Create both
m_non = create(ergodic=False)
m_erg = create(ergodic=True)

# Non-ergodic: self.parameters()
non_params = {name: p.shape for name, p in m_non.named_parameters()}
print(f"Non-ergodic parameters: {len(non_params)} tensors, "
      f"{sum(p.numel() for p in m_non.parameters())} total elements")

# Ergodic: space.getParameters()
erg_params = []
for s in m_erg.spaces:
    sp = s.getParameters()
    print(f"  {s.__class__.__name__} getParameters: {len(sp)} tensors, "
          f"{sum(p.numel() for p in sp)} elements")
    erg_params.extend(sp)

erg_param_ids = {id(p) for p in erg_params}
print(f"\nErgodic collected: {len(erg_params)} tensors, "
      f"{sum(p.numel() for p in erg_params)} total elements")

# Find what's in non-ergodic but not in ergodic
print("\n--- Parameters in non-ergodic but NOT in ergodic getParameters: ---")
all_params_erg = {id(p): p for _, p in m_erg.named_parameters()}
erg_gp_ids = set(id(p) for p in erg_params)

for name, p in m_erg.named_parameters():
    if id(p) not in erg_gp_ids:
        print(f"  MISSING: {name} shape={list(p.shape)} numel={p.numel()}")

# Also check if self.params is set for each space
print("\n--- Space self.params check: ---")
for s in m_erg.spaces:
    has_params = hasattr(s, 'params')
    if has_params:
        print(f"  {s.__class__.__name__}: self.params has {len(s.params)} items")
    else:
        print(f"  {s.__class__.__name__}: NO self.params attribute!")

# Check the forward/reverse behavior difference
print("\n--- Forward pass bias/temp with ergodic=True, alpha=0: ---")
m_erg.set_sigma(0.0)
# Check what the SigmaLayer/PiLayer forward uses
for s in m_erg.spaces:
    if hasattr(s, 'ergodic'):
        print(f"  {s.__class__.__name__}.ergodic = {s.ergodic}")
    for layer_name in ['pi1', 'pi2', 'pi', 'sigma1', 'sigma2', 'sigma']:
        if hasattr(s, layer_name):
            layer = getattr(s, layer_name)
            if hasattr(layer, 'bias') and hasattr(layer, 'var'):
                b = layer.bias.item() if layer.bias.numel() == 1 else layer.bias
                v = layer.var.item() if layer.var.numel() == 1 else layer.var
                print(f"    {layer_name}: bias={b}, var={v}")
