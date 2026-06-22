"""Parse-tree-deleted decode -- ``<ideaDecode>`` step 1
(doc/plans/2026-06-19-grammar-inverses-handoff.md "Goal 2"; reading-attention.md
"(C) Idea decoding").

Goal 2 decodes an idea into a surface with the parse tree DELETED: the reverse
path must not rebuild ``generate_rules`` from the chart, so the decode is driven
by the primed symbolic space instead. Step 1 (this slice) adds the
``<ideaDecode>`` flag and gates the reverse-leg chart fire
(``_chart_generate_from_stm``) so the path runs chart-free.

DARK by default: with the flag off the gate is inert -- the chart fire happens
exactly as before (byte-identical; the full suite is the off-path witness). These
tests cover (a) the flag defaults off, and (b) the gate actually SKIPS the
``symbolSpace.generate`` rule rebuild when on (and fires it when off, so the
skip is meaningful rather than a no-op because the STM happened to be empty).
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import pytest
import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def _count_generate_fires(m):
    """Replace ``symbolSpace.generate`` with a counting wrapper; return the
    box whose ``n`` is bumped each time the rule rebuild fires (covers both the
    default-only inline path and the ``_ss_generate_eager`` island, since both
    dispatch through this same bound method)."""
    box = {"n": 0}
    orig = m.symbolSpace.generate

    def _wrapped(*a, **k):
        box["n"] += 1
        return orig(*a, **k)

    m.symbolSpace.generate = _wrapped
    return box


def test_idea_decode_defaults_off():
    # MM_mereology does not set <ideaDecode> -> the gate is inert; the
    # rule-driven reverse path is untouched (byte-identical).
    m = _build("MM_mereology.xml")
    assert getattr(m, "idea_decode", None) is False


def test_gate_skips_generate_rebuild_when_on():
    m = _build("MM_mereology.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        m.forward(x)  # populate the C-tier STM so the snapshot is non-None

    # OFF: the chart fire rebuilds the rules (the snapshot is non-None, so this
    # confirms the skip below is meaningful, not a vacuous empty-STM no-op).
    box = _count_generate_fires(m)
    m.idea_decode = False
    m._chart_generate_from_stm()
    assert box["n"] >= 1, "off path must fire the rule rebuild"

    # ON: the parse tree is deleted -> the rebuild is skipped entirely.
    box["n"] = 0
    m.idea_decode = True
    m._chart_generate_from_stm()
    assert box["n"] == 0, "idea_decode must skip the generate_rules rebuild"
