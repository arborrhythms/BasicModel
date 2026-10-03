"""Saved pre-repair probe: every configured serial reading owns its inverse."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]

def test_default_xor_reconstructs_its_understanding():
    from test_mm_xor import _fresh_model
    import torch,util
    previous=util.TheCompileBackend;util.TheCompileBackend='none'
    model,_,_= _fresh_model(str(ROOT/'data/XOR_grammar.xml'))
    try:
        assert model.reconstruct_in_loop, 'serial reconstruction must not depend on an opt-in'
        with torch.no_grad():
            result=model.understand(model.inputSpace.prepInput(['hello world','hello there','loving world','loving there']))
        assert result.input_reconstruction is not None
        assert bool(torch.isfinite(result.input_reconstruction.byte_cost).all())
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset();util.TheCompileBackend=previous
