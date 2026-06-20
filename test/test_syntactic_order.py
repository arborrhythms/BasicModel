"""syntacticOrder (doc/specs/orders.md, NEW 2026-06-19): the parse-tree DEPTH
cap on the serial grammatical reduction.

  * read from <architecture><syntacticOrder>, default 0 (unbounded/inert);
  * 0 collapses the STM to a single S (byte-identical to before the knob);
  * a positive value caps the NULL-seal reduce sweep to that many fold levels,
    handing on a partially-composed forest (depth > 1);
  * the <= word-count bound holds STRUCTURALLY (a reduce micro-step no-ops once
    a row reaches depth 1), so a cap above the live depth still collapses fully.
"""
import os, sys, warnings, tempfile
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


def _fill_stm(m, depth):
    """Seed the STM with ``depth`` distinct non-trivial constituents (newest at
    slot 0) on a single batch row, then return (cap, dim)."""
    stm = m.conceptualSpace.stm
    cap = int(stm.capacity)
    dim = int(stm.concept_dim)
    stm.begin_forward(1, device=torch.device("cpu"))
    buf = torch.zeros(1, cap, dim)
    for k in range(depth):
        buf[0, k, :] = float(k + 1) * 0.1   # distinct, non-zero per slot
    stm._buffer = buf
    stm._depth = torch.tensor([depth], dtype=torch.long)
    return cap, dim


def test_syntactic_order_defaults_zero_and_reads_config():
    # Default (no knob) is 0 = unbounded; an explicit value is read.
    m = _build("XOR_grammar.xml")
    assert int(getattr(m, "syntacticOrder", 0)) == 0

    # A temp config that sets the knob is honoured.
    src = open(os.path.join(_DATA, "XOR_grammar.xml")).read()
    assert "<symbolicOrder>" in src
    src2 = src.replace("<symbolicOrder>",
                       "<syntacticOrder>2</syntacticOrder>\n    <symbolicOrder>", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA)
    tmp.write(src2); tmp.close()
    try:
        import Models, Language
        from util import init_config
        init_config(path=tmp.name, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m2, _ = Models.BasicModel.from_config(tmp.name)
        assert int(m2.syntacticOrder) == 2
    finally:
        os.unlink(tmp.name)


def test_syntactic_order_zero_collapses_to_single_s():
    # syntacticOrder=0 (default) runs the full cap-1 sweep -> depth 1.
    m = _build("XOR_grammar.xml")
    m.syntacticOrder = 0
    cap, _ = _fill_stm(m, depth=6)
    with torch.no_grad():
        _S, post_depth = m._stm_reduce_to_single_S()
    assert int(post_depth.max()) == 1, (
        f"unbounded sweep must collapse to a single S, got depth "
        f"{int(post_depth.max())}")


def test_syntactic_order_caps_parse_tree_depth():
    # A positive cap stops the sweep early -> a partially-composed forest
    # (depth > 1) survives. With depth=6 and cap k=2, two folds leave depth 4.
    m = _build("XOR_grammar.xml")
    start_depth = 6
    m.syntacticOrder = 2
    _fill_stm(m, depth=start_depth)
    with torch.no_grad():
        _S, post_depth = m._stm_reduce_to_single_S()
    d = int(post_depth.max())
    assert d == start_depth - 2, (
        f"syntacticOrder=2 must fold exactly 2 levels (depth {start_depth} -> "
        f"{start_depth - 2}), got {d}")
    assert d > 1, "the capped sweep must leave a partial forest (depth > 1)"


def test_syntactic_order_negative_rejected():
    # A negative value is rejected (the XSD's nonNegativeInteger restriction
    # fires at load; the code guard mirrors symbolicOrder's for non-XML paths).
    from util import init_config
    src = open(os.path.join(_DATA, "XOR_grammar.xml")).read().replace(
        "<symbolicOrder>",
        "<syntacticOrder>-1</syntacticOrder>\n    <symbolicOrder>", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA)
    tmp.write(src); tmp.close()
    try:
        with pytest.raises(ValueError):
            init_config(path=tmp.name, defaults_path=_DEFAULTS)
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
