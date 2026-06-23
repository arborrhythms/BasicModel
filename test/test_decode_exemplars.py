"""Smoke coverage for the idea-decode exemplar configs (data/MM_decode.xml and
data/MM_phrase_decode.xml). These keep the decode exemplars referenced and
verify they schema-validate, build, and wire the <ideaDecode> + <symbolTower>
decode path (a tall WholeSpace so symbol_dim == concept_dim and the seed-swap
drives). See doc/old/2026-06-20-idea-decoder.md.
"""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models
    import Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_mm_decode_builds_and_wires_idea_decode():
    m = _build("MM_decode.xml")
    assert getattr(m, "idea_decode", False) is True


def test_mm_phrase_decode_builds_and_wires_idea_decode():
    m = _build("MM_phrase_decode.xml")
    assert getattr(m, "idea_decode", False) is True


def test_mm_decode_forward_smoke():
    import Models
    from util import TheXMLConfig
    m = _build("MM_decode.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)        # runs without error (decode path off-by-default behavior)
