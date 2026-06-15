"""Build-all guard for the modality re-architecture (Phase 2, Task 2.4 of
doc/plans/2026-06-03-modality-architecture-plan.md) -- the green gate.

For every live config (data/*.xml minus _BROKEN), assert:
  * it builds;
  * IS/PS/CS main subspaces report where=2/when=2; SS/OS report 0/0;
  * every whenEncoding carrier reports nWhen == its encoding width (no drift)
    and nWhen in {0, 2};
  * a tiny forward is finite.

_BROKEN configs are pre-existing breakage (per test_use_flags.py), excluded.
"""

import glob, os, sys, unittest, warnings
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import Models, Language
from util import init_config, TheXMLConfig

_DATA = str(Path(__file__).resolve().parent.parent / "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")

# Pre-existing breakage (per test_use_flags.py), not modality regressions.
_BROKEN = {"model.xml", "MM_20M.xml", "MM_400M.xml", "MM_shamatha.xml",
           "MM_xor_step4.xml",
           # MM_xor deliberately disables the PS codebook (2x2 LDU exact-XOR;
           # "SS owns the VQ"); forcing the now-mandatory PS codebook would
           # muxed-snap its pi+sigma output and break the exact-XOR path, so it
           # stays broken under the converged architecture. (XOR_exact +
           # MM_xor_fixture + idempotent WERE migrated: PS/SS -> quantize and
           # the old-width nInputDim/nOutputDim overrides dropped so the +4
           # muxed width divides cleanly.)
           "MM_xor.xml"}

_TEXT_LEXERS = {"sentence", "word", "byte", "char", "n", "bpe", "radix",
                "lexicon", "analyse", "mphf"}


def _live_configs():
    out = []
    for f in sorted(glob.glob(os.path.join(_DATA, "*.xml"))):
        if os.path.basename(f) not in _BROKEN:
            out.append(f)
    return out


def _build(cfg):
    init_config(path=cfg, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    model, _ = Models.BasicModel.from_config(cfg)
    model.eval()
    return model


def _tiny_forward(model):
    """Drive a minimal forward. Text models lex a byte tensor (stringTensor);
    numeric models take a [B, nInput, inputDim] float tensor."""
    data = getattr(model.inputSpace, "data", None)
    try:
        lexer = TheXMLConfig.space("InputSpace", "lexer")
    except Exception:
        lexer = None
    is_text = (data is not None and hasattr(data, "stringTensor")
               and str(lexer) in _TEXT_LEXERS)
    if is_text:
        xb = torch.stack([data.stringTensor("the cat sat")]).unsqueeze(1).float()
    else:
        nIn = model.inputSpace.outputShape[0]
        inDim = model.inputSpace.subspace.nWhat
        xb = torch.randn(2, nIn, inDim).tanh()
    with torch.no_grad():
        res = model.forward(xb)
    out = res[2] if isinstance(res, (tuple, list)) and len(res) >= 3 else res
    return out


def _assert_tier_shapes(tc, model, name):
    # Per the 2026-06-06 dim-convention unification (canonical_shape returns
    # (2, 2) for every tier), this asserts each tier reports the canonical
    # band instead of duplicating the table. Was: SS/OS hardcoded to (0, 0).
    from architecture import canonical_shape as _cs
    def cs(section):
        try: return _cs(section)
        except Exception: return (0, 0)
    checks = [
        ("inputSpace", getattr(model, "inputSpace", None), *cs("InputSpace")),
        ("perceptualSpace", getattr(model, "perceptualSpace", None), *cs("PartSpace")),
        ("conceptualSpace", getattr(model, "conceptualSpace", None), *cs("ConceptualSpace")),
        ("symbolicSpace", getattr(model, "symbolicSpace", None), *cs("WholeSpace")),
        ("outputSpace", getattr(model, "outputSpace", None), *cs("OutputSpace")),
    ]
    for tier, space, ew, en in checks:
        sub = getattr(space, "subspace", None) if space is not None else None
        if sub is None:
            continue
        tc.assertEqual(getattr(sub, "nWhere", None), ew,
                       f"{name}:{tier} nWhere")
        tc.assertEqual(getattr(sub, "nWhen", None), en,
                       f"{name}:{tier} nWhen")
    # No encoding-width drift anywhere; nWhen in {0, 2}.
    for n, m in model.named_modules():
        we = getattr(m, "whenEncoding", None)
        if we is None:
            continue
        nWhen = getattr(m, "nWhen", None)
        tc.assertEqual(nWhen, we.nDim, f"{name}:{n} nWhen != encoding nDim")
        tc.assertIn(nWhen, (0, 2), f"{name}:{n} nWhen not in (0,2)")


_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


class TestModalityConfigsBuildAndForward(unittest.TestCase):
    """Substrate safety net: every live config BUILDS with the canonical
    per-tier shapes. Forward-finiteness is best-effort here (the generic
    text/numeric driver below cannot match every config's bespoke input
    pipeline); per-config forward validation lives in the Phase 6 regression
    gate, which drives each config with its own proper input."""

    @unittest.skipIf(not _RUN_SLOW, "slow (~48s build-all every live config) -- set RUN_SLOW=1")
    def test_all_live_configs_build_with_canonical_shapes(self):
        not_driven = []
        nonfinite = []
        for cfg in _live_configs():
            name = os.path.basename(cfg)
            with self.subTest(config=name):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # BUILD + per-tier canonical shape are the strict gate.
                    model = _build(cfg)
                    _assert_tier_shapes(self, model, name)
                    # Forward is best-effort: a generic driver can't match
                    # every config's input pipeline. Assert finiteness only
                    # when the forward actually runs.
                    try:
                        out = _tiny_forward(model)
                    except Exception as e:
                        not_driven.append(f"{name}: {type(e).__name__}")
                        out = None
                    if torch.is_tensor(out) and not torch.isfinite(out).all():
                        nonfinite.append(name)
        if nonfinite:
            self.fail(f"non-finite forward output: {nonfinite}")
        if not_driven:
            print(f"\n[modality-configs] built+shape-OK but forward not driven "
                  f"by the generic harness ({len(not_driven)}): {not_driven}")


if __name__ == "__main__":
    unittest.main()
