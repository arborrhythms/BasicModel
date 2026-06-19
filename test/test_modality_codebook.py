"""PartSpace / WholeSpace codebook tiering (asymmetric VQ).

2026-06-09 (asymmetric-VQ plan §7 task 7) made the ``<codebook>`` knob
asymmetric per tier:

  * PartSpace is SUBSYMBOLIC: its ``<codebook>`` element was retired
    from the schema entirely. PS is hardwired to ``"none"`` -- a continuous
    ``.event`` passthrough; the percept prototypes live on the ``.what``
    Embedding, never on a VQ snap. There is no config knob and no way to turn
    a codebook on for PS.
  * WholeSpace is SYMBOLIC: a symbol qua symbol quantizes onto a codebook,
    so its ``<codebook>`` DEFAULTS to ``"quantize"`` when a config omits it. An
    explicit ``<codebook>none</codebook>`` on SS is still honored (e.g.
    data/XOR_exact.xml builds a full-width invertible passthrough for the
    exact-XOR reconstruction fixture).
"""

import os, sys, tempfile, unittest, warnings
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import Models, Language
from util import init_config

_DATA = str(Path(__file__).resolve().parent.parent / "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(cfg):
    init_config(path=cfg, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    model, _ = Models.BasicModel.from_config(cfg)
    model.eval()
    return model


class TestCodebookTiering(unittest.TestCase):

    def test_ps_subsymbolic_ws_quantized_by_default(self):
        """MentalModel sets no <codebook> on PS/SS. PartSpace is
        subsymbolic (always resolves to "none"); WholeSpace inherits the
        "quantize" default (non-"none")."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _build(os.path.join(_DATA, "MentalModel.xml"))
        self.assertEqual(model.perceptualSpace.codebook_mode, "none",
                         "PartSpace is subsymbolic: codebook is hardwired "
                         "to none")
        self.assertNotEqual(model.wholeSpace.codebook_mode, "none",
                            "WholeSpace codebook defaults to quantize")

    def test_ws_codebook_none_opt_out_honored(self):
        """data/XOR_exact.xml explicitly resolves WholeSpace to
        <codebook>none</codebook> (a full-width invertible passthrough) and
        must build without raising. PartSpace is subsymbolic regardless,
        so it is always "none"."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _build(os.path.join(_DATA, "XOR_exact.xml"))
        self.assertEqual(model.perceptualSpace.codebook_mode, "none",
                         "PartSpace is subsymbolic: always none")
        self.assertEqual(model.wholeSpace.codebook_mode, "none",
                         "explicit WholeSpace codebook=none should be honored")


if __name__ == "__main__":
    unittest.main()
