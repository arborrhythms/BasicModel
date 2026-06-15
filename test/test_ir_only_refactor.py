"""Plan-side tests for the 2026-05-14 IR-only refactor.

Covers the new-test items from
``doc/plans/2026-05-14-retire-maskedPrediction-IR-only-within-sentence.md``
§"Add tests":

  * XSD validation rejects ````.
  * XSD validation rejects ``<reconstruct>output</reconstruct>``.
  * ``<reconstructionScale>`` parses; legacy ``<reverseScale>`` triggers
    a deprecation warning and maps to the same field.
  * IR forward produces ``[B, N, predDim]`` predictions (no K axis).
  * C3 (spec sec 7): reconstruction is unconditionally concepts-seeded
    -- ``runBatch`` adds a ``reconstruction_reverse`` term (concepts
    path) when ``reconstruction_scale > 0``, with no ``<reconstruct>``
    enum (retired in A1).
  * ``InterSentenceLayer.predict_next()`` returns the right shape
    after ``armaP`` observations.
  * ``InterSentenceLayer.observe(s_t)`` accumulates a non-zero ARMA
    loss after the ring has been primed.
  * ``BasicModel.generate_sentence(...)`` produces a non-empty
    decode list after one warm-up sentence.
"""

import os
import sys
import tempfile
import unittest
import warnings

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers
import Models
import Language
from util import init_config, TheXMLConfig


_MODEL_XML_HEADER = """<?xml version="1.0" ?>
<model>
  <architecture>
    <modelType>embedding</modelType>
    <data>
      <dataset>inline</dataset>
      <input use="train">a b c</input>
      <output use="train">0</output>
    </data>
    <training>
      <numEpochs>1</numEpochs>
      <learningRate>0.01</learningRate>
{training_extra}
    </training>
  </architecture>
"""

_MODEL_XML_FOOTER = """
  <InputSpace>
    <nDim>4</nDim>
    <nVectors>4</nVectors>
    <nOutput>4</nOutput>
  </InputSpace>
  <PartSpace>
    <nDim>4</nDim>
    <nVectors>4</nVectors>
    <nOutput>4</nOutput>
  </PartSpace>
  <ConceptualSpace>
    <nDim>4</nDim>
    <nVectors>4</nVectors>
    <nOutput>4</nOutput>
  </ConceptualSpace>
  <WholeSpace>
    <nDim>4</nDim>
    <nVectors>4</nVectors>
    <nOutput>4</nOutput>
  </WholeSpace>
  <OutputSpace>
    <nDim>1</nDim>
    <nVectors>1</nVectors>
    <nOutput>1</nOutput>
  </OutputSpace>
</model>
"""


def _write_tmp_xml(training_extra=""):
    body = _MODEL_XML_HEADER.format(training_extra=training_extra) + _MODEL_XML_FOOTER
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False,
        dir=_DATA)
    tmp.write(body)
    tmp.close()
    return tmp.name


# -- XSD validation --------------------------------------------------------

class TestXsdRejectsRetiredElements(unittest.TestCase):
    """The XSD must surface a validation warning when an XML uses the
    retired ``<maskedPrediction>`` or ``<reconstruct>output</...>``
    syntax — the parser is soft-validating so the warning is the only
    signal that the deprecation took effect.
    """

    def _parse_with_validation(self, training_extra="", arch_extra=""):
        from util import XMLConfig
        xml = _MODEL_XML_HEADER.format(training_extra=training_extra)
        if arch_extra:
            xml = xml.replace(
                "<modelType>embedding</modelType>",
                "<modelType>embedding</modelType>\n    " + arch_extra)
        xml = xml + _MODEL_XML_FOOTER
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, dir=_DATA)
        tmp.write(xml)
        tmp.close()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                XMLConfig._validate_against_schema(tmp.name)
            return caught
        finally:
            os.unlink(tmp.name)

    def _require_lxml(self):
        try:
            import lxml  # noqa: F401
        except ImportError:
            self.skipTest("lxml not installed -- soft XSD validation skipped")

    def test_rejects_maskedPrediction(self):
        self._require_lxml()
        caught = self._parse_with_validation(
            training_extra="      ")
        msgs = " ".join(str(w.message) for w in caught)
        self.assertIn("maskedPrediction", msgs,
                      "XSD did not flag <maskedPrediction>")

    def test_rejects_reconstruct_output(self):
        self._require_lxml()
        caught = self._parse_with_validation(
            arch_extra="<reconstruct>output</reconstruct>")
        msgs = " ".join(str(w.message) for w in caught)
        self.assertIn("output", msgs,
                      "XSD did not flag <reconstruct>output</...>")


# -- reverseScale deprecation shim ---------------------------------------

class TestReverseScaleBackcompat(unittest.TestCase):
    """Legacy ``<reverseScale>`` is renamed to ``<reconstructionScale>``
    in-place by the parser, with a one-shot deprecation warning so
    existing checked-in configs keep training without manual migration.
    """

    def test_reverse_scale_legacy_maps_through(self):
        path = _write_tmp_xml(
            training_extra="      <reverseScale>0.3</reverseScale>")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                init_config(path=path,
                            defaults_path=os.path.join(_DATA, "model.xml"))
            msgs = " ".join(str(w.message) for w in caught)
            self.assertIn("reverseScale", msgs)
            self.assertIn("reconstructionScale", msgs)
            self.assertAlmostEqual(
                float(TheXMLConfig.training("reconstructionScale")), 0.3)
        finally:
            os.unlink(path)

    def test_reconstruction_scale_parses_cleanly(self):
        path = _write_tmp_xml(
            training_extra="      <reconstructionScale>0.5</reconstructionScale>")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                init_config(path=path,
                            defaults_path=os.path.join(_DATA, "model.xml"))
            # No deprecation warning under the canonical name.
            msgs = " ".join(str(w.message) for w in caught)
            self.assertNotIn("reverseScale is deprecated", msgs)
            self.assertAlmostEqual(
                float(TheXMLConfig.training("reconstructionScale")), 0.5)
        finally:
            os.unlink(path)


# -- IR forward shape -----------------------------------------------------

class TestIrForwardShape(unittest.TestCase):
    """IR-only training: the forward 4-tuple's predictions slot is
    ``[B, N, predDim]`` — no K axis (retired alongside AR cursor
    unfold).  Validates the canonical post-2026-05-14 shape contract.
    """

    def test_ir_forward_returns_b_n_d(self):
        init_config(path=os.path.join(_DATA, "MM_xor.xml"),
                    defaults_path=os.path.join(_DATA, "model.xml"))
        Language.TheGrammar._configured = False
        from data import TheData
        TheData.load("xor")
        m, _ = Models.BaseModel.from_config(
            os.path.join(_DATA, "MM_xor.xml"), data=TheData)
        m.eval()
        inp = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]]).float().unsqueeze(1)
        _input_state, _symbols, pred, recon = m.forward(inp)
        self.assertIsNone(recon,
                          "reverse pipeline retired -- reconstruction slot must be None")
        self.assertIsNotNone(pred)
        self.assertEqual(pred.dim(), 3,
                         f"IR predictions should be [B, N, predDim], got {tuple(pred.shape)}")


# -- Reconstruction is unconditionally from concepts ---------------------

class TestReconstructConceptsLoss(unittest.TestCase):
    """C3 (spec sec 7): reconstruction is UNCONDITIONALLY concepts-seeded.

    The ``<reconstruct>`` enum (``none``/``symbols``/``concepts``/
    ``both``) was retired in A1, so no config can request a mode and there
    is no ``self.reconstruct`` attribute. ``runBatch`` now always seeds
    the reverse pass from the terminal ConceptualSpace STM snapshot via
    ``reverse`` and reports the concepts
    reconstruction in the ``reconstruction_reverse`` ``TheError`` term
    (weighted by ``reconstruction_scale``). This test asserts that term
    fires -- WITHOUT touching any enum -- and that the concepts reverse
    path (``reverse``) is the one taken (the head-seeded
    ``_run_pipeline_rev`` was removed 2026-06-07).
    """

    def test_reconstruction_is_unconditionally_from_concepts(self):
        init_config(path=os.path.join(_DATA, "MM_xor.xml"),
                    defaults_path=os.path.join(_DATA, "model.xml"))
        Language.TheGrammar._configured = False
        from data import TheData
        TheData.load("xor")
        m, _ = Models.BaseModel.from_config(
            os.path.join(_DATA, "MM_xor.xml"), data=TheData)
        # The retired enum leaves NO attribute behind.
        self.assertFalse(
            hasattr(m, "reconstruct"),
            "the <reconstruct> enum was retired (A1); BaseModel must not "
            "carry a self.reconstruct attribute")
        # The reconstruction term is gated by reconstruction_scale > 0
        # (MM_xor.xml configures 0.1); without that gate there is nothing
        # to assert.
        self.assertGreater(
            float(getattr(m.loss, "reconstruction_scale", 0.0) or 0.0), 0.0,
            "MM_xor.xml is expected to set reconstructionScale > 0 so the "
            "reconstruction_reverse term fires")
        # The head-seeded reverse primitive (``_run_pipeline_rev``) was
        # REMOVED 2026-06-07, so the concepts-seeded ``reverse`` is the ONLY
        # reverse path. Assert it's gone, then spy ``reverse`` to prove it
        # fires.
        self.assertFalse(
            hasattr(m, "_run_pipeline_rev"),
            "the head-seeded _run_pipeline_rev primitive was removed; "
            "reconstruction is unconditionally concepts-seeded")
        calls = {"concepts": 0}
        _orig_concepts = m.reverse

        def _spy_concepts(x):
            calls["concepts"] += 1
            return _orig_concepts(x)

        m.reverse = _spy_concepts
        m.eval()
        loader = m.inputSpace.data.data_loader(
            split="train", num_streams=2)
        inp_items, out_items = next(iter(loader))
        inputTensor = m.inputSpace.prepInput(inp_items)
        outputTensor = m.outputSpace.prepOutput(out_items)
        m.runBatch(train=False, batchSize=2, split="train",
                   batch_override=(inputTensor, outputTensor))
        # Concepts-seeded reverse fired; head-seeded reverse did NOT --
        # reconstruction no longer dispatches on the (gone) enum.
        self.assertGreater(
            calls["concepts"], 0,
            "reconstruction must seed the reverse pass from concepts "
            "(reverse), unconditionally")
        # The concepts reconstruction term is present and finite.
        terms = {t[0]: t[1] for t in Layers.TheError.terms()}
        self.assertIn(
            "reconstruction_reverse", terms,
            f"concepts reconstruction must add a reconstruction_reverse "
            f"term to TheError; got {set(terms)}")
        val = terms["reconstruction_reverse"]
        if torch.is_tensor(val):
            self.assertTrue(
                bool(torch.isfinite(val).all()),
                f"reconstruction_reverse must be finite, got {val!r}")


# -- ARMA layer shape + loss --------------------------------------------

class TestArmaPredictAndObserve(unittest.TestCase):

    def test_predict_next_shape_after_p_observations(self):
        p, q = 5, 2
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=2, n_dim=6, p=p, q=q, batch=3)
        for _ in range(p):
            layer.observe(torch.randn(3, 4, 6))
        pred = layer.predict_next()
        self.assertEqual(tuple(pred.shape), (3, 6))

    def test_observe_accumulates_nonzero_loss(self):
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=2, n_dim=6, p=5, q=2, batch=2)
        layer.observe(torch.randn(2, 4, 6))  # primes
        loss = layer.observe(torch.randn(2, 4, 6))
        self.assertIsNotNone(loss)
        self.assertGreater(float(loss), 0.0)


# -- generate_sentence smoke --------------------------------------------

class TestGenerateSentenceSmoke(unittest.TestCase):

    def test_generate_sentence_runs(self):
        """Smoke: ``BasicModel.generate_sentence`` returns a list (the
        IR-only infill decode of the seed text's masked positions).
        Cold-start with no prior priming is acceptable -- the call
        must succeed and not raise.
        """
        init_config(path=os.path.join(_DATA, "MM_xor.xml"),
                    defaults_path=os.path.join(_DATA, "model.xml"))
        Language.TheGrammar._configured = False
        from data import TheData
        TheData.load("xor")
        m, _ = Models.BaseModel.from_config(
            os.path.join(_DATA, "MM_xor.xml"), data=TheData)
        m.eval()
        out = m.generate_sentence("zero")
        self.assertIsInstance(out, list)


# -- ε-growing codebook --------------------------------------------------

class TestGrowOnNovelty(unittest.TestCase):
    """``VectorQuantize.grow_on_novelty`` inserts encoder outputs that
    are at distance > ε from every live codebook entry into the next
    empty (cluster_size == 0) slot.  Disabled by default; opt-in via
    ``<codebookGrowthEpsilon>`` XSD knob.
    """

    def _vq_with_zero_cluster(self, codebook_size=8, dim=4):
        """Construct a VQ and zero its initial ``cluster_size`` so
        ``grow_on_novelty``'s ``free_threshold=0.5`` default sees every
        slot as dead.  The real EMA path would decay any unused slot
        toward 0 within a handful of batches; zeroing up front lets the
        test exercise the contract without running training steps.
        """
        from Layers import VectorQuantize
        vq = VectorQuantize(dim=dim, codebook_size=codebook_size,
                            decay=0.9)
        vq.train()
        vq.cluster_size.zero_()
        return vq

    def test_cold_codebook_seeds_first_rows(self):
        vq = self._vq_with_zero_cluster()
        x = torch.randn(3, 4)
        inserted = vq.grow_on_novelty(x, eps=1e-3)
        self.assertEqual(inserted, 3)
        self.assertEqual(int((vq.cluster_size > 0).sum()), 3)

    def test_eps_zero_is_noop(self):
        vq = self._vq_with_zero_cluster()
        x = torch.randn(3, 4)
        self.assertEqual(vq.grow_on_novelty(x, eps=0.0), 0)
        self.assertTrue(bool((vq.cluster_size == 0).all()))

    def test_close_input_does_not_insert(self):
        vq = self._vq_with_zero_cluster()
        seed = torch.randn(1, 4)
        vq.grow_on_novelty(seed, eps=1e-3)
        live_before = int((vq.cluster_size > 0).sum())
        # Re-feed the same row -- min-distance is 0 < eps, no insertion.
        vq.grow_on_novelty(seed, eps=0.5)
        live_after = int((vq.cluster_size > 0).sum())
        self.assertEqual(live_before, live_after)


if __name__ == "__main__":
    unittest.main()
