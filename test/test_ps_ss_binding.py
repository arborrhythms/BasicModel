"""PS-to-SS binding hook (Phase 7).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("PS to SS
Binding"): a new perceptual terminal emits NULL_SEM before a binding
exists; repeated stable exposure promotes it into an SS codebook row.
"""

import copy
import os
import sys
import warnings

import pytest
import torch

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


def test_ps_to_ss_null_before_binding():
    """A first-seen PS terminal emits NULL_SEM (zeros), ungrounded."""
    ss = _make_model().symbolicSpace
    vec, pos, grounded = ss.resolve_ps_terminal(4242, promote_threshold=2)
    assert grounded is False
    assert pos == -1
    assert torch.allclose(vec, torch.zeros_like(vec))


def test_ps_to_ss_binding_after_repetition():
    """Repeated stable exposure promotes the PS terminal to an SS row."""
    ss = _make_model().symbolicSpace
    pid = 4242
    _, _, g1 = ss.resolve_ps_terminal(pid, promote_threshold=2)
    assert g1 is False
    vec2, pos2, g2 = ss.resolve_ps_terminal(pid, promote_threshold=2)
    assert g2 is True and pos2 > 0
    assert float(torch.linalg.vector_norm(vec2)) > 0  # a real SS row
    # Now bound: same position, still grounded.
    vec3, pos3, g3 = ss.resolve_ps_terminal(pid, promote_threshold=2)
    assert g3 is True and pos3 == pos2
    # A different PS terminal is independent (still ungrounded on first sight).
    _, _, g_other = ss.resolve_ps_terminal(9999, promote_threshold=2)
    assert g_other is False


def test_byte_fallback_terminal_is_never_grounded():
    """An unidentified terminal (ps_id < 0) is never bound."""
    ss = _make_model().symbolicSpace
    for _ in range(5):
        vec, pos, grounded = ss.resolve_ps_terminal(-1, promote_threshold=2)
        assert grounded is False and pos == -1
        assert torch.allclose(vec, torch.zeros_like(vec))
