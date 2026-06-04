"""Live 'analyse' chunking through the model forward (Phase R3-live, model
path).

The meronymic analyzer is wired as a real PerceptualSpace chunking mode. The
live front end owns tokenization: InputSpace passes the unanalyzed host
surface, then PS applies the space-lexer and optional learned merges. The
standalone ``chunk_static(..., "analyse")`` path covers the cold byte-terminal
learning model.
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


def _write_analyse_xml(tmpdir, n_vectors=512):
    xml = f"""<?xml version='1.0'?>
<model>
  <architecture>
    <conceptualOrder>2</conceptualOrder>
    <nWhere>0</nWhere>
    <nWhen>0</nWhen>
    <processSymbols>false</processSymbols>
    <type>mental</type>
    <ergodic>false</ergodic>
    <modelType>embedding</modelType>
    <maskedPrediction>NONE</maskedPrediction>
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
    <!-- "6+2+2": IS/PS/CS are (2,2) muxed tiers, so nDim is the EVENT
         width = content(4) + .where/.when band(4) = 8. SS/OS are (0,0)
         content tiers and keep the bare content width (4 / 1). -->
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <nOutput>32</nOutput>
    <lexer>byte</lexer>
  </InputSpace>
  <PerceptualSpace>
    <nInput>32</nInput>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
    <nVectors>{n_vectors}</nVectors>
    <codebook>true</codebook>
    <chunking>analyse</chunking>
  </PerceptualSpace>
  <ConceptualSpace>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </ConceptualSpace>
  <SymbolicSpace>
    <nOutput>32</nOutput>
    <nDim>4</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </SymbolicSpace>
  <OutputSpace>
    <nOutput>1</nOutput>
    <nDim>1</nDim>
    <nVectors>1</nVectors>
  </OutputSpace>
  <WordSpace><useGrammar>none</useGrammar></WordSpace>
</model>
"""
    path = os.path.join(tmpdir, "mm_analyse_test.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


class TestAnalyseChunkingForward(unittest.TestCase):

    def test_init_reads_analyse_mode(self):
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_analyse_xml(tmp)
            model, _cfg = BaseModel.from_config(config_path=path)
            self.assertEqual(model.perceptualSpace.chunking_mode, "analyse")

    def _tokens(self, chunk, lexer="byte"):
        import torch
        from Models import BaseModel
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_analyse_xml(tmp)
            with open(path) as f:
                xml = f.read()
            xml = xml.replace("<lexer>byte</lexer>", f"<lexer>{lexer}</lexer>")
            with open(path, "w") as f:
                f.write(xml.replace(
                    "<chunking>analyse</chunking>",
                    f"<chunking>{chunk}</chunking>"))
            model, _cfg = BaseModel.from_config(config_path=path)
            inp = model.inputSpace.prepInput(["hello world foo"])
            with torch.no_grad():
                model.forward(inp)
            return model.perceptualSpace._forward_input["tokens"]

    def test_forward_analyse_space_lexer_owns_full_surface_lexing(self):
        """C: InputSpace hands PS the UNANALYZED whole-line surface and the
        analyzer's space-lexer owns tokenization, even when the upstream
        compatibility lexer is byte-level."""
        analyse = [t for t in self._tokens("analyse")[0] if t]
        self.assertEqual(analyse, ["hello", " ", "world", " ", "foo"])

    def test_forward_analyse_is_not_limited_by_token_byte_width(self):
        """The live analyzer reconstructs the host surface before PS lexing,
        so word runs are not truncated to the compatibility byte-buffer
        token width used by the legacy lexicon path in this tiny fixture."""
        analyse = [t for t in self._tokens("analyse", lexer="byte")[0] if t]
        lexicon = [t for t in self._tokens("lexicon", lexer="word")[0] if t]
        self.assertEqual(analyse, ["hello", " ", "world", " ", "foo"])
        self.assertEqual(lexicon, ["hel", " ", "wor", " ", "foo"])


if __name__ == "__main__":
    unittest.main()
