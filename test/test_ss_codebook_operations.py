"""Insert grammar OPERATIONS into the SS codebook (Phase 2, amended).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md + user
steering (2026-06-02): operators are kept in the codebook so the
soft-superposition over the operator-prefixed parse tree can resolve them,
but they are removed from the STM idea space -- the operator defines HOW
meanings combine and contributes no meaning of its own, so it is a distinct
codebook kind ('op'), not a meaning-bearing symbol ('ws') and not an STM
idea vector.
"""

import copy
import os
import sys
import warnings

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


@pytest.fixture(autouse=True)
def _restore_global_singletons():
    """Snapshot + restore the process-global config / grammar singletons.

    ``_make_model`` calls ``init_config(MM_xor_fixture.xml)`` which mutates
    ``TheXMLConfig._data`` and reconfigures ``TheGrammar`` -- without restore
    that leaks into any later test in the same process (the conftest autouse
    reset clears ``_configured`` but not ``_data``). Mirrors the
    snapshot/restore in test_xor_grammar.py.
    """
    import Language
    from util import TheXMLConfig
    snap = {
        "data": copy.deepcopy(TheXMLConfig._data),
        "sources": list(TheXMLConfig._sources),
        "requirements": list(TheXMLConfig._requirements),
        "grammar": copy.deepcopy(Language.TheGrammar.__dict__),
    }
    try:
        yield
    finally:
        TheXMLConfig._data = copy.deepcopy(snap["data"])
        TheXMLConfig._sources = list(snap["sources"])
        TheXMLConfig._requirements = list(snap["requirements"])
        Language.TheGrammar.__dict__.clear()
        Language.TheGrammar.__dict__.update(copy.deepcopy(snap["grammar"]))
        Language.TheGrammar._configured = False


def _make_model():
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    m.eval()
    return m


def test_operations_inserted_into_ws_codebook():
    """Every grammar operation gets its own SS codebook row, distinct and
    resolvable, tagged as an operation rather than a meaning symbol."""
    import Language
    m = _make_model()
    ws = m.wholeSpace
    g = Language.TheGrammar

    ops = {r.method_name for r in g.rules if r.method_name}
    assert ops, "expected the configured grammar to declare operations"

    pos_map = ws.insert_operations(g)
    assert set(pos_map) == ops, (set(pos_map), ops)

    positions = list(pos_map.values())
    # Distinct, positive op-codebook positions.
    assert len(set(positions)) == len(positions), positions
    for p in positions:
        assert isinstance(p, int) and p > 0, p
    # Each operator has a resolvable identity vector in the dedicated
    # operator codebook (separate from the symbol codebook).
    for name in ops:
        v = ws.operation_vector(name)
        assert v is not None and int(v.numel()) == int(ws.nDim), (name, v)
    # Resolvable for operator dispatch / soft-superposition.
    for name, p in pos_map.items():
        assert ws.operation_position(name) == p
    assert ws.operation_position("definitely_not_an_op") is None
    assert ws.operation_vector("definitely_not_an_op") is None


def test_build_auto_inserts_operations():
    """Building a model auto-inserts the grammar operations into the SS
    codebook (wired in SymbolSubSpace.__init__) -- the operator-prefixed
    tree's operation nodes are codebook-resolvable without a manual call."""
    import Language
    m = _make_model()
    ws = m.wholeSpace
    g = Language.TheGrammar
    ops = {r.method_name for r in g.rules if r.method_name}
    assert ops
    for op in ops:
        assert ws.operation_position(op) is not None, (
            f"build did not insert operation {op!r} into the SS codebook")


def test_insert_operations_is_idempotent():
    """Re-inserting keeps each operation's existing codebook position."""
    import Language
    m = _make_model()
    ws = m.wholeSpace
    g = Language.TheGrammar
    first = ws.insert_operations(g)
    second = ws.insert_operations(g)
    assert first == second


def test_operations_are_distinct_from_meaning_symbols():
    """An operation row is tagged 'op', a freshly inserted meaning symbol
    is tagged 'ws' -- the STM idea space (meanings) and the operator
    codebook entries are different kinds."""
    import Language
    m = _make_model()
    ws = m.wholeSpace
    g = Language.TheGrammar
    pos_map = ws.insert_operations(g)
    op_name = next(iter(pos_map))
    # Operators live in their OWN codebook (operation_vector), not the
    # symbol codebook: inserting them creates no "op"-tagged positions in
    # the symbol position namespace.
    assert ws.operation_vector(op_name) is not None
    assert "op" not in set(ws._pos_kind.values())
    # The symbol codebook still allocates meaning-bearing "ws" rows.
    sym_pos = ws.insert_symbol()
    assert ws._pos_kind.get(sym_pos) == "ws"
