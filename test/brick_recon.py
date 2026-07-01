"""Phase-1 recon: enumerate torch.compile graph breaks in the per-batch
forward and tag each grammar / non-grammar. NOT a pytest test (no
test_ prefix) -- an explicit recon tool.

Acceptance (design): the NON-GRAMMAR path must reach 0 breaks
(fullgraph). GRAMMAR-path breaks are deferrable with a recorded
reason. This harness produces the tagged backlog that drives the
elimination program.

Device-independent for break enumeration; forced CPU (MPS
torch.compile fake-tensor device propagation is incomplete).

Usage: .venv/bin/python test/brick_recon.py [MM_xor.xml]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "eager"
os.environ["BASICMODEL_DEVICE"] = "cpu"

_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin"))
sys.path.insert(0, str(_p / "bin"))

import torch
import torch._dynamo
from data import TheData
from Models import BaseModel
from util import init_config, init_device, TheXMLConfig

# Source markers that constitute the GRAMMAR path (deferrable). A break
# whose user stack touches any of these is tagged 'grammar'; everything
# else is 'non-grammar' (must-fix).
_GRAMMAR = ("SyntacticLayer", "_compose", "chart", "grammar", "Grammar",
            "_apply_rule", "derivation", "GrammarLayer", "partForward",
            "shift_reduce", "_reduce", "_shift")


def _tag(txt):
    return "grammar" if any(g in txt for g in _GRAMMAR) else "non-grammar"


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "MM_xor.xml"
    init_device("cpu")
    cfg = str(_p / "data" / cfg_name)
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    if cfg_name == "MM_20M_legacy.xml":
        # Bounded: break enumeration only needs one real batch through
        # every code path, not the full corpus.
        TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                     num_shards=1, max_docs=64)
    else:
        # Config-driven (mirrors ModelFactory.run / test_xor_grammar):
        # each config declares its own dataset under
        # <architecture><data>. Hardcoding "xor" broke grammar configs
        # (XOR_grammar's OutputSpace reshape mismatched the raw-xor
        # input). Bounded to one shard / 64 docs -- enumeration only
        # needs one real batch through every path.
        _arch = (TheXMLConfig.data or {}).get("architecture", {})
        _dat = _arch.get("data", {})
        _dataset = _dat.get("dataset", "xor")
        TheData.load(_dataset, num_shards=1, max_docs=64,
                     shard_dir=_dat.get("shardDir"), dat=_dat)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    opt = m.getOptimizer(lr=1e-4)

    # FAITHFUL GATE. The previous harness used ``torch._dynamo.explain``,
    # which traces *through* graph breaks and silently tolerates ops a
    # real ``fullgraph=True`` compile rejects (nn.Module ``__setattr__``,
    # SubSpace construction, ``copy_context`` plumbing). It reported
    # "0 breaks" for configs that a real fullgraph compile could not
    # capture -- the design doc's early "0 breaks" milestones were
    # measured with that unfaithful tool. This now exercises the EXACT
    # path training uses: ``enable_compiled_step`` (real
    # ``torch.compile(fullgraph=True)``) + a bounded ``runEpoch``.
    # Under ``fullgraph=True`` any break *raises* ``Unsupported``; a
    # non-raising run that captured >=1 graph is genuinely break-free.
    from torch._dynamo.exc import Unsupported
    m.enable_compiled_step()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    break_txt = None
    try:
        m.runEpoch(optimizer=opt, batchSize=2, split="train",
                   max_batches=2)
    except Unsupported as e:
        break_txt = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - report, don't mask
        break_txt = f"{type(e).__name__}: {e}"
    stats = dict(torch._dynamo.utils.counters.get("stats", {}))
    n_graphs = int(stats.get("unique_graphs", 0))
    n_calls = int(stats.get("calls_captured", 0))
    clean = (break_txt is None and n_graphs > 0)

    lines = [f"# Recon — REAL fullgraph={True} compile ({cfg_name})",
             "",
             f"status: {'FULLGRAPH-CLEAN' if clean else 'BROKEN'}",
             f"unique_graphs: {n_graphs}   calls_captured: {n_calls}",
             ""]
    if break_txt is not None:
        tag = _tag(break_txt)
        lines += [f"- [{tag}] first blocking break:",
                  f"    {break_txt[:1200]}"]
    else:
        lines += ["(no graph break: fullgraph=True captured the whole "
                  "forward without raising)"]

    out = _p / "doc" / "plans" / f"recon-breaks-{cfg_name.replace('.xml','')}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    print(f"[recon] {'FULLGRAPH-CLEAN' if clean else 'BROKEN'} | "
          f"unique_graphs={n_graphs} calls_captured={n_calls} | "
          f"wrote {out}")


if __name__ == "__main__":
    main()
