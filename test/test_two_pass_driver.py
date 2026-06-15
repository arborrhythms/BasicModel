"""Two-pass soft-superposition learning at the model level.

When <architecture><learning>true, each training batch is run TWICE as two
separate trials -- pass A at superposition temperature 0 (sharp, recorded)
and pass B at <exploreTemperature> (flatter, exploration, NOT recorded).
Both passes are ordinary differentiable forward/loss/backward steps over the
soft-superposition route; the chooser is in the gradient path in both.
These tests check the flag plumbing and that the runBatch / runEpoch
two-pass steps complete end-to-end. Default (learning off) is one pass.
"""

import os
import re
import sys
import tempfile
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

_DATA = os.path.join(_PROJECT, "data")
_GRAMMAR_CONFIG = os.path.join(_DATA, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(extra=""):
    import Models, Language
    from util import init_config, init_device
    init_device("cpu")
    torch.manual_seed(0)
    with open(_GRAMMAR_CONFIG) as f:
        text = f.read()
    text = re.sub(r"\s*<learning>[^<]*</learning>\s*\n", "\n", text)
    if extra:
        text = text.replace("<architecture>", f"<architecture>\n    {extra}", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA)
    tmp.write(text)
    tmp.close()
    try:
        init_config(path=tmp.name, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(tmp.name)
        import Models as _M
        _M.TheData.load("xor")
        m.train()
        return m
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _batch(m):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


def test_learning_flags_parsed():
    m = _build("<learning>true</learning>")
    assert m.two_pass_learning is True
    assert abs(m.explore_temperature - 0.5) < 1e-9
    assert _build().two_pass_learning is False          # default off


def test_set_superposition_temperature_reaches_structured_layers():
    m = _build("<learning>true</learning>")
    m._set_superposition_temperature(0.0)
    ll = m.wordSubSpace.languageLayer
    layers = list(ll._unary_layers.values()) + list(ll._binary_layers.values())
    assert layers, "no structured layers attached"
    assert all(l.superposition_temperature == 0.0 for l in layers)
    m._set_superposition_temperature(None)
    assert all(l.superposition_temperature is None for l in layers)


def test_runbatch_soft_superposition_step_runs():
    # A single batch at superposition temperature 0 must complete the whole
    # training step (forward / loss / backward / step) and reset the temp.
    m = _build("<learning>true</learning>")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)       # lr=0: exercise, no drift
    batch = _batch(m)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.runBatch(train=True, batchSize=4, optimizer=opt,
                   batch_override=batch, superposition_temperature=0.0)
    ll = m.wordSubSpace.languageLayer
    layers = list(ll._unary_layers.values()) + list(ll._binary_layers.values())
    assert all(l.superposition_temperature is None for l in layers)  # reset


def test_runepoch_two_pass_runs():
    # <learning>true makes runEpoch run each batch twice (temp 0 + explore).
    m = _build("<learning>true</learning>")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)


def test_pass_b_does_not_advance_per_sentence_state():
    # Pass B (explore_trial=True) must NOT advance per-sentence state: the
    # model clock and the periodic-checkpoint step counter tick ONCE per
    # sentence (pass A only), not twice. Regression for the two-pass
    # side-effect double-commit.
    m = _build("<learning>true</learning>")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    clock0 = m.present()
    steps0 = int(getattr(m, "_training_step_count", 0) or 0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    assert m.present() - clock0 == 1                       # not 2
    assert int(m._training_step_count) - steps0 == 1       # not 2


def test_explore_trial_flag_resets_after_runbatch():
    # runBatch must clear the _exploration_trial instance flag (finally), so a
    # subsequent ordinary forward is never mistaken for an explore trial.
    m = _build("<learning>true</learning>")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    batch = _batch(m)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.runBatch(train=True, batchSize=4, optimizer=opt,
                   batch_override=batch, superposition_temperature=0.5,
                   exploration_trial=True)
    assert getattr(m, "_exploration_trial", False) is False
