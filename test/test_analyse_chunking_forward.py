"""analyse migration contracts (Phase 4b, analysis/synthesis dual-input
plan rev. 2026-06-09).

The meronymic analyzer is top-down ANALYSIS and lives on WholeSpace
(``<analysis>analyse``, consuming the unity view): PS-side
``<synthesis>analyse`` is REJECTED loudly (schema + reader), the lexicon
synthesis path keeps the surface word resolution PS analyse used to
provide, and the SS analysis cut shapes the stage-0 evidence
(boundaries-define-parts; per-part coarse means).

The standalone ``chunk_static(..., "analyse")`` byte-terminal learning
machinery is covered by test_chunk_static_analyse.py /
test_analyse_word_learning.py (knob-free analyzer plumbing).
"""

import os
import sys
import tempfile
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _write_xml(tmpdir, *, synthesis="lexicon", lexer="byte", analysis=None,
               n_vectors=512):
    analysis_elem = (f"\n    <analysis>{analysis}</analysis>"
                     if analysis else "")
    xml = f"""<?xml version='1.0'?>
<model>
  <architecture>
    <subsymbolicOrder>2</subsymbolicOrder>
    <nWhere>0</nWhere>
    <nWhen>0</nWhen>
    <processSymbols>false</processSymbols>
    <ergodic>false</ergodic>
    <modelType>embedding</modelType>
    <data><dataset>xor</dataset></data>
    <training>
      <numTrials>1</numTrials>
      <numEpochs>1</numEpochs>
      <batchSize>1</batchSize>
      <learningRate>0.01</learningRate>
      <autoload>false</autoload>
      <autosave>false</autosave>
    </training>
  </architecture>
  <InputSpace>
    <!-- Uniform (2,2): IS/PS/CS/SS are muxed tiers, so nDim is the EVENT
         width = content(4) + .where/.when band(4) = 8. OS is the only
         (0,0) tier and keeps the bare content width (1). -->
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <nOutput>32</nOutput>
  </InputSpace>
  <PartSpace>
    <nInput>32</nInput>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
    <nVectors>{n_vectors}</nVectors>
    <synthesis>{synthesis}</synthesis>
  </PartSpace>
  <ConceptualSpace>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </ConceptualSpace>
  <WholeSpace>
    <nOutput>32</nOutput>
    <!-- Uniform (2,2): SS.nWhat must equal CS.nWhat (handoff invariant);
         nDim = content(4) + band(4) = 8 (was 4 under the retired SS=(0,0)). -->
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
    <lexer>{lexer}</lexer>{analysis_elem}
  </WholeSpace>
  <OutputSpace>
    <nOutput>1</nOutput>
    <nDim>1</nDim>
    <nVectors>1</nVectors>
  </OutputSpace>
</model>
"""
    path = os.path.join(tmpdir, "mm_analyse_test.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


class TestAnalyseMigration(unittest.TestCase):

    def test_ps_synthesis_analyse_rejected_loudly(self):
        # Hard cut: analyse is no longer a synthesis value. The schema
        # rejects it at validation (or, were validation bypassed, the
        # reader raises) -- either way, building must fail LOUDLY.
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_xml(tmp, synthesis="analyse")
            with self.assertRaises((ValueError, KeyError)):
                BaseModel.from_config(config_path=path)

    def test_is_side_lexer_rejected_loudly(self):
        # <lexer> moved to WholeSpace; an InputSpace-side <lexer> fails
        # validation (schema) and the runtime reader (resolve_lexer).
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_xml(tmp)
            with open(path) as f:
                xml = f.read()
            xml = xml.replace("<nOutput>32</nOutput>\n  </InputSpace>",
                              "<nOutput>32</nOutput>\n"
                              "    <lexer>byte</lexer>\n  </InputSpace>")
            with open(path, "w") as f:
                f.write(xml)
            with self.assertRaises((ValueError, KeyError)):
                BaseModel.from_config(config_path=path)

    def test_ws_analysis_knob_accepted(self):
        # <analysis>analyse on WholeSpace builds; mode is stashed.
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_xml(tmp, analysis="analyse")
            model, _cfg = BaseModel.from_config(config_path=path)
            self.assertEqual(model.wholeSpace.analysis_mode, "analyse")

    def _tokens(self, synthesis, lexer="byte"):
        import torch
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_xml(tmp, synthesis=synthesis, lexer=lexer)
            model, _cfg = BaseModel.from_config(config_path=path)
            inp = model.inputSpace.prepInput(["hello world foo"])
            with torch.no_grad():
                model.forward(inp)
            return model.perceptualSpace._forward_input["tokens"]

    def test_lexicon_synthesis_owns_full_surface_lexing(self):
        """The lexicon synthesis path self-lexes the whole-line surface
        (the word resolution PS analyse used to provide): word runs are
        NOT truncated to the compatibility byte-buffer token width."""
        lexicon = [t for t in self._tokens("lexicon", lexer="word")[0] if t]
        self.assertEqual(lexicon, ["hello", " ", "world", " ", "foo"])


if __name__ == "__main__":
    unittest.main()
