"""The §6c sentence protocol: parallel pump first (GrammarOpsPass §6c;
author sign-off 2026-06-11).

In serial mode, every sentence gets an independent PARALLEL prelude of
``subsymbolicOrder`` pumps — pump zero — seeding the codebook towers
(EMA on: the word-learning guarantee) and producing the gist that IS
the §5 single intent (intent-only commit: nothing enters the
workspace), then the serial per-word task runs under the §6d serial
partition (``serial_pump`` is a PER-PUMP property). The gist re-pumps
on preemption ONLY (rising edge of the conflict latch; threshold +
hysteresis on the absolute set's per-dimension-max conflict mass).

The protocol is config-gated (``<architecture><sentenceProtocol>``).
DEFAULT CUTOVER (2026-06-18): default is now **ON in SERIAL mode**
(``symbolicOrder >= 1``) so the whole-sentence gist conditions each
word's parts/wholes (the context the per-word hard mask drops re-enters
via the gist/intent); **OFF in parallel** (the prelude is only invoked
from the serial per-word body). Explicit ``<sentenceProtocol>`` overrides.
"""

import os
import sys
import warnings

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_model():
    """Cheap PS/CS/SS boot from MM_xor_loopback.xml (the serial-mode
    fixture the Stage-1.* test files share)."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return model


def _xor_input():
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ).float().unsqueeze(1)


def test_protocol_on_by_default_in_serial():
    """DEFAULT CUTOVER (2026-06-18): the legacy serial fixture now has the
    protocol ON by default — pump zero runs and sets the gist WITHOUT any
    explicit ``<sentenceProtocol>`` in the config."""
    m = _make_model()
    assert m.sentence_protocol is True               # serial default = ON
    m.forward(_xor_input())
    assert int(getattr(m, '_prelude_pumps', 0)) == 1
    assert getattr(m, '_last_gist', None) is not None


def test_protocol_off_is_dark_when_disabled():
    """Forcing the protocol OFF restores the dark path: no prelude pumps,
    no intent, no per-pump stamp — the legacy serial path."""
    m = _make_model()
    m.sentence_protocol = False                       # explicit override
    m.forward(_xor_input())
    assert int(getattr(m, '_prelude_pumps', 0)) == 0
    assert getattr(m, '_last_gist', None) is None
    for sp in (m.perceptualSpace, m.wholeSpace):
        assert getattr(sp, 'serial_pump', None) is None
        assert sp.intent_boosts() is None


def test_protocol_off_by_default_in_parallel():
    """Parallel mode (symbolicOrder==0) keeps the protocol OFF by default —
    the prelude is a serial-only pass."""
    init_config(path=os.path.join(_DATA_DIR, "MM_xor.xml"),
                defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        m, cfg = Models.BasicModel.from_config(
            os.path.join(_DATA_DIR, "MM_xor.xml"))
    assert int(cfg["architecture"].get("symbolicOrder", 0)) == 0
    assert m.sentence_protocol is False               # parallel default = OFF


def test_protocol_runs_pump_zero_and_sets_intent():
    """Protocol ON: pump zero runs once per sentence, the gist exists
    and feeds the §5 single intent, and the per-pump stamp is cleared
    at sentence end."""
    m = _make_model()
    m.sentence_protocol = True
    m.forward(_xor_input())
    assert int(getattr(m, '_prelude_pumps', 0)) == 1
    assert getattr(m, '_last_gist', None) is not None
    # One intent, both towers: any tower with a codebook carries the
    # boosts (the off-state for a codebook-less tower is None).
    boosted = [sp for sp in (m.perceptualSpace, m.wholeSpace)
               if sp.intent_boosts() is not None]
    assert boosted, "the gist must prime at least one codebook tower"
    for b in (sp.intent_boosts() for sp in boosted):
        assert torch.all(b >= 1.0)
    # Sentence end: the per-pump mode is un-stamped (the §6d law falls
    # back to the legacy read between sentences).
    for sp in (m.perceptualSpace, m.wholeSpace):
        assert getattr(sp, 'serial_pump', 'missing') is None


def test_preemption_rising_edge_repumps_once():
    """Gist refresh on preemption ONLY: a contested absolute corpus
    fires the latch at the first tick; the re-pump fires on the RISING
    edge and the latch prevents per-word thrash."""
    m = _make_model()
    m.sentence_protocol = True
    tl = m._get_truth_layer()
    assert tl is not None
    D = int(tl.nDim)
    hot = torch.zeros(D)
    hot[0] = 0.9
    tl.record(hot, degree=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.record(-hot, degree=1.0)         # sharply contested witness
    assert tl.conflict_mass() == pytest.approx(0.9, abs=1e-5)
    m.forward(_xor_input())
    # Pump zero + exactly ONE preemption re-pump (rising edge), no
    # matter how many words the sentence carries.
    assert int(getattr(m, '_prelude_pumps', 0)) == 2


def test_protocol_word_learning_pumps_parallel_partition():
    """Pump zero is a PARALLEL pump for the §6d law: during the
    prelude the spaces carry serial_pump=False; during the per-word
    ticks, True. (Observed via the stamp the prelude leaves while
    running — here we pin the epilogue contract: ticks ran serial.)"""
    m = _make_model()
    m.sentence_protocol = True
    seen = {}
    orig = m._sentence_prelude

    def spy(in_event, word_carrier):
        out = orig(in_event, word_carrier)
        # The prelude's finally-block hands the sentence to the serial
        # ticks: the stamp must be True right after pump zero returns.
        seen['after_prelude'] = (
            getattr(m.perceptualSpace, 'serial_pump', None),
            getattr(m.wholeSpace, 'serial_pump', None))
        return out

    m._sentence_prelude = spy
    m.forward(_xor_input())
    assert seen['after_prelude'] == (True, True)
