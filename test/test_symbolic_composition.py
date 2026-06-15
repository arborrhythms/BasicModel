"""SymbolicComposition cutover smoke test (Architecture.md "three
cognitive operations", op 2).

When <architecture><symbolicComposition>true</symbolicComposition> is set,
the subsymbolicOrder CS->PS loop re-feeds the prior pass's symbolic carrier
(cs._subspaceForSS) to PartSpace at t>0 so SigmaLayer composes higher-order
symbols. Default off re-feeds the stage-0 percept every pass (unchanged).

These are smoke tests: with the flag on, a parallel multi-order forward
(MM_symbolic_iter: subsymbolicOrder=2, symbolicOrder=0) must build and run
finite. Semantic validation of the composed symbols is the user's training
concern; here we pin that the gated path is wired and runs.
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
_BASE = os.path.join(_DATA, "MM_symbolic_iter.xml")     # subsym=2, sym=0
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(symbolic_composition):
    import Models, Language
    from util import init_config, init_device
    init_device("cpu")
    torch.manual_seed(0)
    with open(_BASE) as f:
        text = f.read()
    text = re.sub(
        r"\s*<symbolicComposition>[^<]*</symbolicComposition>\s*\n", "\n", text)
    if symbolic_composition:
        inject = "<symbolicComposition>true</symbolicComposition>"
        text = text.replace("<architecture>", f"<architecture>\n    {inject}", 1)
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
        m.eval()
        return m
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def test_flag_parsed_onto_model():
    assert _build(symbolic_composition=True).symbolic_composition is True
    assert _build(symbolic_composition=False).symbolic_composition is False


def test_symbolic_composition_forward_runs_finite():
    m = _build(symbolic_composition=True)
    x = _batch(m)
    with torch.no_grad():
        out = m.forward(x)
    # forward returns a tuple; the first element is the forward input /
    # primary output -- just assert the pass produced finite tensors.
    fwd = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.is_tensor(fwd) and torch.isfinite(fwd).all()


def test_flag_off_also_runs_finite():
    m = _build(symbolic_composition=False)
    x = _batch(m)
    with torch.no_grad():
        out = m.forward(x)
    fwd = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.is_tensor(fwd) and torch.isfinite(fwd).all()
