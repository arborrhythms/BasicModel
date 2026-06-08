"""Read-only harness: dump each Space's RESOLVED dims for a config.

Surfaces what nInputDim/nOutputDim/nDim/nVectors the 0/-1 sentinels
actually resolve to, so configs can be made fully explicit. Not a pytest
test (underscore prefix); run directly:

    python test/_introspect_dims.py data/MM_20M_grammar.xml
"""
import os, sys
from pathlib import Path

os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import torch  # noqa: E402
import Models  # noqa: E402


def dump(cfg_path):
    print(f"\n===== {cfg_path} =====")
    model, _ = Models.BasicModel.from_config(cfg_path)
    seen = []
    for sp in getattr(model, "spaces", []):
        name = type(sp).__name__
        iS = list(getattr(sp, "inputShape", []) or [])
        oS = list(getattr(sp, "outputShape", []) or [])
        niD = getattr(sp, "nInputDim", None)
        noD = getattr(sp, "nOutputDim", None)
        nV = getattr(sp, "nVectors", None)
        nDim = getattr(sp, "nDim", None)
        print(f"  {name:18s} inputShape={iS} outputShape={oS} "
              f"nInputDim={niD} nOutputDim={noD} nVectors={nV} nDim={nDim}")
        seen.append(name)
    print(f"  (spaces: {seen})")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["data/MM_20M_grammar.xml"]
    for t in targets:
        try:
            dump(t)
        except Exception as e:
            import traceback
            print(f"  !! {t} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
