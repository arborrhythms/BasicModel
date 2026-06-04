"""Mandatory PerceptualSpace / SymbolicSpace codebooks (Phase 2, Task 2.3 of
doc/plans/2026-06-03-modality-architecture-plan.md).

In the converged modality architecture a percept and a symbol qua symbol must
quantize onto a codebook (architecture.MANDATORY_CODEBOOK_TIERS). The model.xml
defaults for these tiers are <codebook>quantize</codebook>; a config that
explicitly resolves either to "none" is a loud build error.
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


class TestMandatoryCodebooks(unittest.TestCase):

    def test_ps_ss_codebooks_present(self):
        """MentalModel's PerceptualSpace and SymbolicSpace resolve to a
        non-"none" codebook mode (inheriting the quantize default)."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _build(os.path.join(_DATA, "MentalModel.xml"))
        self.assertNotEqual(model.perceptualSpace.codebook_mode, "none",
                            "PerceptualSpace codebook is mandatory")
        self.assertNotEqual(model.symbolicSpace.codebook_mode, "none",
                            "SymbolicSpace codebook is mandatory")

    def test_symbolicspace_codebook_none_raises(self):
        """An explicit <codebook>none</codebook> on SymbolicSpace is a loud
        build error, not a silent passthrough."""
        tree = ET.parse(os.path.join(_DATA, "MentalModel.xml"))
        root = tree.getroot()
        ss = root.find("SymbolicSpace")
        cb = ss.find("codebook")
        if cb is None:
            cb = ET.SubElement(ss, "codebook")
        cb.text = "none"
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with self.assertRaises(ValueError):
                    _build(tmp.name)
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
