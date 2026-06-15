"""Tests for space-scoped XML config resolution."""
import os, sys, unittest

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")



class TestGetSpaceParam(unittest.TestCase):
    """get_space_param resolves space -> architecture fallback."""

    def test_space_overrides_architecture(self):
        cfg = {
            "architecture": {"hasAttention": True},
            "PartSpace": {"hasAttention": False},
        }
        result = Models.BasicModelFactory.get_space_param(cfg, "PartSpace", "hasAttention")
        self.assertFalse(result)

    def test_falls_back_to_architecture(self):
        cfg = {
            "architecture": {"invertible": True},
            "PartSpace": {"nActive": 4},
        }
        result = Models.BasicModelFactory.get_space_param(cfg, "PartSpace", "invertible")
        self.assertTrue(result)

    def test_raises_when_missing_everywhere(self):
        cfg = {"architecture": {}, "PartSpace": {}}
        with self.assertRaises(KeyError):
            Models.BasicModelFactory.get_space_param(cfg, "PartSpace", "nonExistentKey")

    def test_space_section_missing_entirely(self):
        cfg = {"architecture": {"invertible": True}}
        result = Models.BasicModelFactory.get_space_param(cfg, "ConceptualSpace", "invertible")
        self.assertTrue(result)




class TestDefaultsXml(unittest.TestCase):
    """model.xml has space-scoped sections."""

    @classmethod
    def setUpClass(cls):
        defaults_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "model.xml")
        cls.cfg = Models.BaseModel.load_config(defaults_path)

    def test_has_space_sections(self):
        for name in ["InputSpace", "PartSpace", "ConceptualSpace",
                      "WholeSpace", "OutputSpace"]:
            self.assertIn(name, self.cfg, f"Missing <{name}> section")

    def test_space_has_nOutput(self):
        for name in ["InputSpace", "PartSpace", "ConceptualSpace",
                      "WholeSpace", "OutputSpace"]:
            self.assertIn("nOutput", self.cfg[name],
                          f"<{name}> missing nOutput")

    def test_space_has_nDim(self):
        for name in ["InputSpace", "PartSpace", "ConceptualSpace",
                      "WholeSpace", "OutputSpace"]:
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
        # ``reconstruct`` was retired (A1); reconstruction is now
        # unconditionally concepts-seeded.
        for key in ["subsymbolicOrder",
                    "ergodic", "processSymbols"]:
            self.assertIn(key, arch, f"architecture missing model-wide key '{key}'")
        trn = arch.get("training", {})
        for key in ["numTrials", "numEpochs", "batchSize", "learningRate",
                    "autoload", "autosave", "certainty"]:
            self.assertIn(key, trn, f"architecture.training missing key '{key}'")
        # weightsPath is at architecture level, not training.
        # (embeddingPath retired 2026-05-12 -- integrated-weights bundle
        # carries embeddings inside the .ckpt.)
        for key in ["weightsPath"]:
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
        # Stage 1.D flat-slab invariant: <modelType>passthrough</modelType>
        # so the validator doesn't fire on the deliberately-mismatched
        # per-space nOutput values (the point of this test is that
        # nOutput is read per-space, not architectural correctness).
        xml = self._write_xml("""<?xml version="1.0" ?>
<model>
  <architecture>
    <modelType>passthrough</modelType>
    <data><dataset>xor</dataset></data>
    <training><autoload>false</autoload></training>
  </architecture>
  <InputSpace><nOutput>2</nOutput><nDim>5</nDim></InputSpace>
  <PartSpace>
    <nOutput>4</nOutput><nDim>5</nDim>

    <hasAttention>false</hasAttention>
  </PartSpace>
  <ConceptualSpace>
    <nOutput>3</nOutput><nDim>5</nDim>
    <invertible>true</invertible>
  </ConceptualSpace>
  <WholeSpace>
    <nOutput>3</nOutput><nDim>5</nDim><nVectors>3</nVectors>

  </WholeSpace>
  <OutputSpace><nOutput>1</nOutput><nDim>1</nDim></OutputSpace>
</model>""")
        try:
            Models.TheData.load("xor")
            m = Models.BasicModel()
            m.create_from_config(xml, data=Models.TheData)
            self.assertEqual(m.nInput, 2)
            self.assertEqual(m.nPercepts, 4)
            self.assertEqual(m.nConcepts, 3)
            self.assertEqual(m.nSymbols, 3)
            self.assertEqual(m.nOutput, 1)
        finally:
            os.unlink(xml)


class TestValidateConfig(unittest.TestCase):
    """validate_config reads space-scoped keys."""

    def test_attention_reshape_guard_retired(self):
        """RETIRED (plan 2026-06-06-symbolic-heat-retrieval.md §Handoff
        addendum): the old ``hasAttention``-vs-reshape guard was removed
        with the QKV ``AttentionLayer`` enlistment. ``<attention>`` is now a
        symbolic-retrieval mode (not tensor self-attention), so
        ``hasAttention=True`` together with a ``flatten``/``nInputDim`` reshape
        is now ACCEPTED (must NOT raise).
        """
        cfg = {
            "architecture": {},
            "PartSpace": {"hasAttention": True, "invertible": False,
                                "nActive": 4, "nDim": 1, "flatten": True},
            "WholeSpace": {},
            "ConceptualSpace": {"hasAttention": True, "flatten": True},
        }
        # Should not raise: the reshape-vs-QKV incompatibility no longer exists.
        Models.BasicModelFactory.validate_config(cfg)

    def test_attention_reshape_ok_when_false(self):
        """Post-guard-retirement smoke check: validate_config runs clean on
        the minimal config with hasAttention=False.  Kept (not a duplicate of
        test_attention_reshape_guard_retired) to confirm the validator itself
        remains functional after the reshape-vs-QKV guard was removed."""
        cfg = {
            "architecture": {},
            "PartSpace": {"hasAttention": False, "invertible": False,
                                "nActive": 4, "nDim": 1, "flatten": True},
            "WholeSpace": {},
            "ConceptualSpace": {"hasAttention": False, "flatten": False},
        }
        # Should not raise
        Models.BasicModelFactory.validate_config(cfg)

    def test_arir_requires_reconstruction_retired(self):
        """Retired 2026-05-14: ``<maskedPrediction>`` is gone, so the
        ARIR/reconstruct coupling validation is gone too.  Within-
        sentence training is IR-only; ``<reconstruct>`` is now an
        independent forward-only loss selector.
        """
        return  # retired check; see plan §1

    def test_symbol_dim_must_match_concept_dim(self):
        """Post 2026-05 ownership rule: WholeSpace owns no SigmaLayer,
        so symbol_dim must match the ConceptualSpace effective output dim.
        """
        cfg = {
            "architecture": {
                "reconstruct": "symbols",
            },
            "InputSpace": {"nOutput": 8, "nDim": 4},
            "PartSpace": {"nOutput": 8, "nDim": 4, "hasAttention": False,
                                "invertible": True},
            "WholeSpace": {"nOutput": 5, "nDim": 3},
            "ConceptualSpace": {"nOutput": 8, "nDim": 4, "hasAttention": False},
            "OutputSpace": {"nOutput": 1, "nDim": 1},
        }
        with self.assertRaises(ValueError) as ctx:
            Models.BasicModelFactory.validate_config(cfg)
        # C1 handoff invariant (2026-06): the SS<->CS dim-match check now
        # raises on ``SS.nWhat == CS.nWhat`` (was the older ``symbol_dim``
        # message).
        self.assertIn("SS.nWhat == CS.nWhat", str(ctx.exception))


class TestInferValidation(unittest.TestCase):
    def test_non_ir_infer_modes_raise_not_implemented(self):
        """ARIR/AR were retired 2026-05-14 in favour of
        ``BasicModel.generate_sentence`` for sentence-level
        generation.  ``infer(mode='ARIR')`` (or anything other than
        'IR') now raises ``NotImplementedError`` rather than running
        the legacy state machine.
        """
        model = Models.BasicModel()
        with self.assertRaises(NotImplementedError):
            model.infer("hello", max_length=1, mode="ARIR")
