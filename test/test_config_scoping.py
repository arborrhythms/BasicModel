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
            BasicModelFactory.get_space_param(cfg, "PerceptualSpace", "nonExistentKey")

    def test_space_section_missing_entirely(self):
        cfg = {"architecture": {"invertible": True}}
        result = BasicModelFactory.get_space_param(cfg, "ConceptualSpace", "invertible")
        self.assertTrue(result)


from BasicModel import BaseModel, BasicModel


class TestDefaultsXml(unittest.TestCase):
    """model.xml has space-scoped sections."""

    @classmethod
    def setUpClass(cls):
        defaults_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "model.xml")
        cls.cfg = BaseModel.load_config(defaults_path)

    def test_has_space_sections(self):
        for name in ["InputSpace", "PerceptualSpace", "ConceptualSpace",
                      "SymbolicSpace", "OutputSpace"]:
            self.assertIn(name, self.cfg, f"Missing <{name}> section")

    def test_space_has_nOutput(self):
        for name in ["InputSpace", "PerceptualSpace", "ConceptualSpace",
                      "SymbolicSpace", "OutputSpace"]:
            self.assertIn("nOutput", self.cfg[name],
                          f"<{name}> missing nOutput")

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
        for key in ["reconstruct", "conceptualOrder", "symbolicOrder",
                    "ergodic", "certainty", "processSymbols"]:
            self.assertIn(key, arch, f"architecture missing model-wide key '{key}'")
        trn = arch.get("training", {})
        for key in ["numTrials", "numEpochs", "batchSize", "learningRate",
                    "autoload", "autosave"]:
            self.assertIn(key, trn, f"architecture.training missing key '{key}'")
        # weightsPath and embeddingPath are at architecture level, not training
        for key in ["weightsPath", "embeddingPath"]:
            self.assertIn(key, arch, f"architecture missing key '{key}'")
        dat = arch.get("data", {})
        self.assertIn("dataset", dat, "architecture.data missing 'dataset'")


import tempfile


class TestCreateFromConfig(unittest.TestCase):
    """create_from_config reads space-scoped XML."""

    def _write_xml(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_reads_space_nOutput(self):
        xml = self._write_xml("""<?xml version="1.0" ?>
<model>
  <architecture>
    <reconstruct>symbols</reconstruct>
    <modelType>embedding</modelType>
    <data><dataset>xor</dataset></data>
    <training><autoload>false</autoload></training>
  </architecture>
  <InputSpace><nOutput>2</nOutput><nDim>1</nDim></InputSpace>
  <PerceptualSpace>
    <nOutput>4</nOutput><nDim>1</nDim>
    <passThrough>true</passThrough>
    <hasAttention>false</hasAttention>
  </PerceptualSpace>
  <ConceptualSpace>
    <nOutput>3</nOutput><nDim>1</nDim>
    <invertible>true</invertible>
  </ConceptualSpace>
  <SymbolicSpace>
    <nOutput>3</nOutput><nDim>1</nDim><nVectors>3</nVectors>
    <passThrough>true</passThrough>
  </SymbolicSpace>
  <OutputSpace><nOutput>1</nOutput><nDim>1</nDim></OutputSpace>
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
            "architecture": {},
            "PerceptualSpace": {"hasAttention": True, "invertible": False,
                                "passThrough": False, "nActive": 4, "nDim": 1, "flatten": True},
            "ConceptualSpace": {"hasAttention": False, "flatten": False},
            "SymbolicSpace": {"passThrough": False},
        }
        with self.assertRaises(ValueError) as ctx:
            BasicModelFactory.validate_config(cfg)
        self.assertIn("hasAttention", str(ctx.exception))

    def test_attention_reshape_ok_when_false(self):
        cfg = {
            "architecture": {},
            "PerceptualSpace": {"hasAttention": False, "invertible": False,
                                "passThrough": False, "nActive": 4, "nDim": 1, "flatten": True},
            "ConceptualSpace": {"hasAttention": False, "flatten": False},
            "SymbolicSpace": {"passThrough": False},
        }
        # Should not raise
        BasicModelFactory.validate_config(cfg)

    def test_arir_requires_reconstruction(self):
        cfg = {
            "architecture": {
                "reconstruct": "NONE",
                "training": {"maskedPrediction": "ARIR"},
            },
            "PerceptualSpace": {"hasAttention": False, "invertible": False,
                                "passThrough": False, "nActive": 4, "nDim": 1, "flatten": False},
            "ConceptualSpace": {"hasAttention": False, "flatten": False},
            "SymbolicSpace": {"passThrough": False},
        }
        with self.assertRaises(ValueError) as ctx:
            BasicModelFactory.validate_config(cfg)
        self.assertIn("maskedPrediction=ARIR", str(ctx.exception))


class TestInferValidation(unittest.TestCase):
    def test_arir_infer_requires_reversible(self):
        model = BasicModel()
        model.reversible = False
        with self.assertRaises(ValueError) as ctx:
            model.infer("hello", max_length=1, mode="ARIR")
        self.assertIn("reversible=True", str(ctx.exception))
