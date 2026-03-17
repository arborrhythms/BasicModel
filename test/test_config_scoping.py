"""Tests for space-scoped XML config resolution."""
import os, sys, unittest

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from BasicModel import BasicModelFactory


class TestGetSpaceParam(unittest.TestCase):
    """get_space_param resolves space -> architecture fallback."""

    def test_space_overrides_architecture(self):
        cfg = {
            "architecture": {"hasAttention": True},
            "PerceptualSpace": {"hasAttention": False},
        }
        result = BasicModelFactory.get_space_param(cfg, "PerceptualSpace", "hasAttention")
        self.assertFalse(result)

    def test_falls_back_to_architecture(self):
        cfg = {
            "architecture": {"invertible": True},
            "PerceptualSpace": {"nActive": 4},
        }
        result = BasicModelFactory.get_space_param(cfg, "PerceptualSpace", "invertible")
        self.assertTrue(result)

    def test_raises_when_missing_everywhere(self):
        cfg = {"architecture": {}, "PerceptualSpace": {}}
        with self.assertRaises(KeyError):
            BasicModelFactory.get_space_param(cfg, "PerceptualSpace", "hasNorm")

    def test_space_section_missing_entirely(self):
        cfg = {"architecture": {"invertible": True}}
        result = BasicModelFactory.get_space_param(cfg, "ConceptualSpace", "invertible")
        self.assertTrue(result)


from BasicModel import BaseModel


class TestDefaultsXml(unittest.TestCase):
    """defaults.xml has space-scoped sections."""

    @classmethod
    def setUpClass(cls):
        defaults_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "defaults.xml")
        cls.cfg = BaseModel.load_config(defaults_path)

    def test_has_space_sections(self):
        for name in ["InputSpace", "PerceptualSpace", "ConceptualSpace",
                      "SymbolicSpace", "OutputSpace"]:
            self.assertIn(name, self.cfg, f"Missing <{name}> section")

    def test_space_has_nActive(self):
        for name in ["InputSpace", "PerceptualSpace", "ConceptualSpace",
                      "SymbolicSpace", "OutputSpace"]:
            self.assertIn("nActive", self.cfg[name],
                          f"<{name}> missing nActive")

    def test_space_has_nDim(self):
        for name in ["InputSpace", "PerceptualSpace", "ConceptualSpace",
                      "SymbolicSpace", "OutputSpace"]:
            self.assertIn("nDim", self.cfg[name], f"<{name}> missing nDim")

    def test_architecture_has_model_wide_only(self):
        arch = self.cfg["architecture"]
        for old_key in ["nInput", "nPercepts", "nConcepts", "nSymbols", "nOutput",
                        "inputDim", "perceptDim", "conceptDim", "symbolDim", "outputDim",
                        "perceptPassThrough", "symbolPassThrough",
                        "perceptPrototypes", "conceptPrototypes",
                        "perceptHasAttention", "conceptHasAttention"]:
            self.assertNotIn(old_key, arch,
                             f"'{old_key}' should be in a space section, not architecture")

    def test_architecture_keeps_model_wide(self):
        arch = self.cfg["architecture"]
        for key in ["reversible", "reshape", "conceptualOrder", "symbolicOrder",
                    "ergodic", "certainty",
                    "processSymbols",
                    "dataset", "modelType", "numTrials", "numEpochs",
                    "batchSize", "learningRate", "pretrained",
                    "weightsPath", "autoload", "autosave"]:
            self.assertIn(key, arch, f"architecture missing model-wide key '{key}'")


import tempfile


class TestCreateFromConfig(unittest.TestCase):
    """create_from_config reads space-scoped XML."""

    def _write_xml(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_reads_space_nActive(self):
        xml = self._write_xml("""<?xml version="1.0" ?>
<model>
  <architecture>
    <reversible>true</reversible>
    <reshape>true</reshape>
    <dataset>xor</dataset>
    <modelType>embedding</modelType>
    <pretrained>false</pretrained>
  </architecture>
  <InputSpace><nActive>2</nActive><nDim>1</nDim></InputSpace>
  <PerceptualSpace>
    <nActive>4</nActive><nDim>1</nDim>
    <passThrough>true</passThrough>
    <hasAttention>false</hasAttention>
  </PerceptualSpace>
  <ConceptualSpace>
    <nActive>3</nActive><nDim>1</nDim>
    <invertible>true</invertible>
  </ConceptualSpace>
  <SymbolicSpace>
    <nActive>3</nActive><nDim>1</nDim>
    <passThrough>true</passThrough>
  </SymbolicSpace>
  <OutputSpace><nActive>1</nActive><nDim>1</nDim></OutputSpace>
</model>""")
        try:
            from BasicModel import BasicModel, TheData
            TheData.load("xor")
            m = BasicModel()
            m.create_from_config(xml, data=TheData)
            self.assertEqual(m.nInput, 2)
            self.assertEqual(m.nPercepts, 4)
            self.assertEqual(m.nConcepts, 3)
            self.assertEqual(m.nSymbols, 3)
            self.assertEqual(m.nOutput, 1)
        finally:
            os.unlink(xml)


class TestValidateConfig(unittest.TestCase):
    """validate_config reads space-scoped keys."""

    def test_attention_reshape_incompatible(self):
        cfg = {
            "architecture": {"reshape": True},
            "PerceptualSpace": {"hasAttention": True, "invertible": False,
                                "passThrough": False, "nActive": 4, "nDim": 1},
            "ConceptualSpace": {"hasAttention": False},
            "SymbolicSpace": {"passThrough": False},
        }
        with self.assertRaises(ValueError) as ctx:
            BasicModelFactory.validate_config(cfg)
        self.assertIn("hasAttention", str(ctx.exception))

    def test_attention_reshape_ok_when_false(self):
        cfg = {
            "architecture": {"reshape": True},
            "PerceptualSpace": {"hasAttention": False, "invertible": False,
                                "passThrough": False, "nActive": 4, "nDim": 1},
            "ConceptualSpace": {"hasAttention": False},
            "SymbolicSpace": {"passThrough": False},
        }
        # Should not raise
        BasicModelFactory.validate_config(cfg)

