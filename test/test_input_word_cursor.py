"""InputSpace's word cursor returns staged inputs and stops at their boundary.

The construction-time cursor flag mirrors ``model.word_brackets``, derived
from the native word/object carrier. The model-free cases check exact slices,
valid length and disabled behavior. The isolated model probes check both
sides of that construction contract without sharing grammar singletons.
"""
import os

os.environ["BASICMODEL_DEVICE"] = "cpu"

import types

import pytest
import torch

# ``test_phase2a_labor_division`` was retired with the dissolve-
# SentenceState plan (doc/plans/2026-05-21-dissolve-sentencestate-
# wordsubspace.md). The model-level test below depends on its
# ``_build_gate_model`` helper; guard the import so module collection
# survives, and gate the dependent test on the helper being available.
try:
    from test.test_phase2a_labor_division import _build_gate_model
except ImportError:
    _build_gate_model = None


def _build_nongrammar_model():
    """Sibling of ``_build_gate_model`` for the non-grammar path.

    ``data/model.xml``'s shipped default grammar is the default-only
    unary ``sigma``/``pi`` folds (compose + generate) at every space_role --
    the one grammar shape ``_derive_use_grammar`` maps to
    ``useGrammar='none'`` (every other config, including any with an
    empty ``<grammar>``, inherits the ``not(S)`` NOOP fallback and
    derives ``'all'``). Mirrors ``_build_gate_model`` exactly but on
    the non-grammar config + its native ``xor`` dataset, so the 2a
    wiring is exercised on a genuinely grammar-disabled model.
    """
    from pathlib import Path
    _p = Path(__file__).resolve().parents[1]
    from util import init_config, init_device
    from data import TheData
    from Models import BaseModel
    init_device("cpu")
    cfg = str(_p / "data" / "model.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("xor")
    m, _ = BaseModel.from_config(cfg, data=TheData)
    return m


def _bare_inputspace():
    """A minimally-initialized InputSpace stand-in.

    ``next_word`` only touches ``_per_word_enabled``,
    ``_per_word_cursor``, ``_ar_embedded`` and ``_peer_perceptual`` --
    so we exercise the *unbound* method against a tiny namespace seeded
    with exactly those attributes (the production-default lifecycle
    values). This keeps the unit truly model-free.
    """
    from Spaces import InputSpace

    self = types.SimpleNamespace(
        _per_word_enabled=False,   # production default (inert)
        _per_word_cursor=0,        # __init__ default
        _ar_embedded=None,         # Start()/Reset() default
        _peer_perceptual=None,     # no BPE peer -> nonzero-vector path
        # Cached host-int lexed length: live code populates this ONCE
        # per forward (D8 piece 2); the bare unit replays the same
        # contract by hand via ``_seed``.
        _valid_len_host=0,
    )
    return InputSpace.next_word, self


def _compute_valid_len_host(buf, peer):
    """Mirror ``InputSpace.forward``'s per-forward cache: highest
    occupied T-index + 1, validity source = peer BPE word mask if
    available, else nonzero-vector. Lives next to ``_seed`` so the
    bare-mock unit replays the same contract as the live IS forward
    boundary (``_valid_len_host`` is now read by ``next_word``)."""
    if buf is None:
        return 0
    T = buf.shape[1]
    valid_pos = buf.abs().sum(dim=-1) > 0
    any_pos = valid_pos.any(dim=0)
    if any_pos.any().item():
        return int(any_pos.nonzero().max().item()) + 1
    return 0


def _seed(self, B, T, D, valid_len):
    """Left-aligned [B,T,D] buffer: rows 0..valid_len-1 nonzero, the
    rest exact zeros (the NULL/end-of-valid sentinel ``forward`` uses
    via ``abs().sum(-1) > 0``)."""
    buf = torch.zeros(B, T, D)
    if valid_len > 0:
        buf[:, :valid_len, :] = torch.randn(B, valid_len, D)
        # Guard against a randn slot summing to exactly 0.
        buf[:, :valid_len, 0] += 1.0
    self._ar_embedded = buf
    self._valid_len_host = _compute_valid_len_host(
        buf, getattr(self, "_peer_perceptual", None))
    return buf


def test_disabled_by_default_returns_none_and_no_advance():
    """Production path: ``_per_word_enabled`` False => never taken,
    cursor untouched, even with a fully valid buffer present."""
    next_word, self = _bare_inputspace()
    buf = _seed(self, B=2, T=5, D=4, valid_len=5)
    assert self._per_word_enabled is False
    for _ in range(3):
        assert next_word(self) is None
    assert self._per_word_cursor == 0, "disabled path must not advance"
    # Buffer object is untouched (no in-place mutation by the feed).
    assert self._ar_embedded is buf


def test_none_buffer_returns_none_even_when_enabled():
    next_word, self = _bare_inputspace()
    self._per_word_enabled = True
    self._ar_embedded = None
    assert next_word(self) is None
    assert self._per_word_cursor == 0


def test_yields_words_one_per_call_in_T_order_then_null_ends():
    """Enabled: one ground-truth [B,1,D] slice per call, in T-order,
    advancing the cursor, then None at the NULL sentinel (end of valid
    lexed content -- here T-positions >= valid_len are all-zero)."""
    next_word, self = _bare_inputspace()
    self._per_word_enabled = True
    B, T, D, valid_len = 3, 8, 6, 5
    buf = _seed(self, B, T, D, valid_len)

    for p in range(valid_len):
        w = next_word(self)
        assert w is not None, f"word {p} should be a real slice"
        assert w.shape == (B, 1, D), w.shape
        # It is the ground-truth lexed slice -- byte-identical to the
        # buffer's T-position (NOT a model reconstruction).
        torch.testing.assert_close(w, buf[:, p:p + 1, :])
        assert self._per_word_cursor == p + 1, "cursor advances by 1"

    # Cursor reached the NULL/end sentinel -> endings with None, idempotent.
    for _ in range(3):
        assert next_word(self) is None
    assert self._per_word_cursor == valid_len, "no advance past the closing"


def test_full_buffer_walks_all_T_then_ends():
    """No padding: valid_len == T. Cursor must walk all T positions
    then closing (end of valid content == end of buffer)."""
    next_word, self = _bare_inputspace()
    self._per_word_enabled = True
    B, T, D = 2, 4, 3
    buf = _seed(self, B, T, D, valid_len=T)
    seen = []
    while True:
        w = next_word(self)
        if w is None:
            break
        seen.append(w)
        assert len(seen) <= T, "must not run past the buffer"
    assert len(seen) == T
    torch.testing.assert_close(torch.cat(seen, dim=1), buf)


def test_empty_valid_content_ends_immediately():
    """All-zero buffer = no valid lexed content => first call is the
    NULL closing (None), cursor never advances."""
    next_word, self = _bare_inputspace()
    self._per_word_enabled = True
    _seed(self, B=2, T=5, D=4, valid_len=0)
    assert next_word(self) is None
    assert self._per_word_cursor == 0






def test_nongrammar_config_disables_cursor_and_next_word_is_none():
    """The genuine DISABLED path: a non-grammar config
    (``useGrammar='none'`` -- default-only unary sigma/pi grammar,
    i.e. ``data/model.xml``) leaves ``_per_word_enabled is False`` and
    ``next_word`` returns ``None`` and never advances the cursor (the
    flag short-circuits before the buffer is touched)."""
    m = _build_nongrammar_model()
    isp = m.inputSpace

    assert m.useGrammar == "none", (
        "model.xml default-only sigma/pi grammar must derive useGrammar"
        " == 'none'")
    assert isp._per_word_enabled is False, (
        "2a wiring: _per_word_enabled = (useGrammar != 'none') => False "
        "for the non-grammar config")
    assert isp._per_word_cursor == 0

    for _ in range(3):
        assert isp.next_word() is None, "disabled path must return None"
    assert isp._per_word_cursor == 0, "disabled path must not advance"


def _probe_flag_in_subprocess(config_rel, dataset, dat_inline="None"):
    """Build ``config_rel`` in a clean subprocess and return
    ``(useGrammar, word_brackets, _per_word_enabled)``.

    The grammar / XML-config singletons (``TheGrammar._configured``,
    ``TheXMLConfig``) are process-wide and ``init_config`` does NOT
    re-arm ``TheGrammar`` -- so two *different* configs cannot be built
    back-to-back in one interpreter (the first config's grammar would
    leak into the second). The codebase's established isolation idiom
    for cross-config model builds is the subprocess (see
    ``test_explicit_dimensions._run_cli``); we mirror it so this single
    test can legitimately pin BOTH sides of the predicate.
    """
    import subprocess
    import sys

    from pathlib import Path
    _p = Path(__file__).resolve().parents[1]

    code = (
        "import os; os.environ['BASICMODEL_DEVICE']='cpu'\n"
        "import sys\n"
        "sys.path.insert(0, 'test')\n"
        "sys.path.insert(0, 'bin')\n"
        "from pathlib import Path\n"
        "_p = Path.cwd()\n"
        "from util import init_config, init_device\n"
        "from data import TheData\n"
        "from Models import BaseModel\n"
        "init_device('cpu')\n"
        f"cfg=str(_p/'data'/'{config_rel}')\n"
        "init_config(path=cfg, defaults_path=str(_p/'data'/'model.xml'))\n"
        f"TheData.load({dataset}, dat={dat_inline})\n"
        "m,_=BaseModel.from_config(cfg, data=TheData)\n"
        "print('RESULT', repr(m.useGrammar), "
        "bool(m.word_brackets), m.inputSpace._per_word_enabled)\n"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_p), env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"subprocess build of {config_rel} failed:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    line = [ln for ln in proc.stdout.splitlines()
            if ln.startswith("RESULT ")]
    assert line, f"no RESULT line in:\n{proc.stdout}"
    _, ug_repr, brackets, flag = line[-1].split(maxsplit=3)
    return ug_repr.strip("'\""), brackets == "True", flag.strip() == "True"


@pytest.mark.skipif(
    os.getenv("RUN_SLOW") != "1",
    reason="slow: builds native and dense models in separate subprocesses; set RUN_SLOW=1")
def test_per_word_enabled_predicate_tracks_word_objects():
    """The cursor flag exactly mirrors native word-bracket availability."""
    ug_g, brackets_g, flag_g = _probe_flag_in_subprocess(
        "MM_ladder.xml", dataset="'math'")
    assert ug_g != "none", f"native fixture must include grammar, got {ug_g!r}"
    assert brackets_g is True
    assert flag_g is True, "native word objects enable the word cursor"

    ug_n, brackets_n, flag_n = _probe_flag_in_subprocess(
        "model.xml", dataset="'xor'")
    assert ug_n == "none", f"dense fixture has no grammar, got {ug_n!r}"
    assert brackets_n is False
    assert flag_n is False, "no native word objects leaves the cursor disabled"

    assert flag_g is brackets_g
    assert flag_n is brackets_n
