"""End-to-end BPE tests for PerceptualSpace.

Covers the config-consolidation + forward-wiring change documented in
doc/specs/2026-04-23-perceptualspace-bpe-chunking-design.md.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _write_minimal_bpe_xml(tmpdir, n_vectors=512):
    """Write a tiny XML config that exercises the bpe chunking path."""
    import os
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
    <nDim>4</nDim>
    <nVectors>8</nVectors>
    <!-- nOutput sized to fit the test's "hello world foo" input
         (15 chars under the BPE pre-chunking byte stream + 1 EOS
         slot). Was 8 under the legacy silent-truncation path;
         raised to 32 when §8g of the brick-vectorization handoff
         replaced the truncation with an assert. -->
    <nOutput>32</nOutput>
    <lexer>byte</lexer>
  </InputSpace>
  <PerceptualSpace>
    <nInput>32</nInput>
    <nOutput>32</nOutput>
    <nDim>4</nDim>
    <nVectors>{n_vectors}</nVectors>
    <codebook>true</codebook>
    <chunking>bpe</chunking>
    <wordLearning>2</wordLearning>
  </PerceptualSpace>
  <ConceptualSpace>
    <nOutput>32</nOutput>
    <nDim>4</nDim>
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
    path = os.path.join(tmpdir, "mm_bpe_test.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


class TestPerceptualSpaceBPE(unittest.TestCase):

    def test_init_reads_new_config_fields(self):
        """Task 3: PerceptualSpace reads <chunking>, <nVectors>, <wordLearning>."""
        import tempfile
        from util import init_config
        import Spaces

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=512)
            init_config(path=path,
                        defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
            self.assertEqual(
                Spaces.TheXMLConfig.space("PerceptualSpace", "chunking"), "bpe")
            self.assertEqual(
                int(Spaces.TheXMLConfig.space("PerceptualSpace", "nVectors")), 512)
            self.assertEqual(
                int(Spaces.TheXMLConfig.space("PerceptualSpace", "wordLearning")), 2)

    def test_init_rejects_bpe_with_nVectors_below_256(self):
        """Task 4: bpe mode requires nVectors>=256."""
        import tempfile
        from Models import BaseModel

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=128)
            with self.assertRaises(ValueError) as ctx:
                BaseModel.from_config(config_path=path)
            self.assertIn("nVectors", str(ctx.exception))
            self.assertIn("256", str(ctx.exception))

    def test_init_accepts_bpe_with_nVectors_256(self):
        """Task 4: nVectors>=256 is accepted; chunk_layer is built in bpe mode."""
        import tempfile
        from Models import BaseModel

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=512)
            model, cfg = BaseModel.from_config(config_path=path)
            ps = model.perceptualSpace
            self.assertEqual(ps.chunking_mode, "bpe")
            self.assertEqual(ps.word_learning, 2)
            self.assertTrue(ps.chunk_layer.bpe)
            self.assertEqual(ps.chunk_layer.n_vectors, 512)
            self.assertEqual(ps.chunk_layer.word_learning, 2)

    def test_forward_bpe_emits_word_level_vectors(self):
        """Task 5: BPE forward pass emits [B, nOutput, nDim] with one position per whitespace word."""
        import tempfile
        import torch
        from Models import BaseModel

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=512)
            model, cfg = BaseModel.from_config(config_path=path)
            ps = model.perceptualSpace

            input_text = ["hello world foo"]
            inp_tensor = model.inputSpace.prepInput(input_text)
            with torch.no_grad():
                _ = model.forward(inp_tensor)

            ps_event = ps.subspace.event.getW()
            self.assertIsNotNone(ps_event, "PerceptualSpace.subspace.event must be populated")
            self.assertEqual(ps_event.shape[0], 1)
            self.assertEqual(ps_event.shape[-1], 4)
            non_zero_rows = (ps_event[0].abs().sum(dim=-1) > 0).sum().item()
            self.assertGreaterEqual(non_zero_rows, 1,
                "At least one word position must have a non-zero vector")
            self.assertLessEqual(non_zero_rows, 3,
                "At most 3 word positions should have non-zero vectors for 3 input words")


    def test_max_fusion_elementwise_max(self):
        """MAX fusion of two sub-tokens equals elementwise max of their embeddings."""
        import tempfile
        import torch
        from Models import BaseModel

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=512)
            model, cfg = BaseModel.from_config(config_path=path)
            ps = model.perceptualSpace
            codebook = ps.subspace.what

            for ch in ('a', 'b'):
                if ch not in codebook.pretrain.key_to_index:
                    codebook.insert(ch)
            with torch.no_grad():
                idx_a = codebook._token_to_index('a')
                idx_b = codebook._token_to_index('b')
                v_a = torch.tensor([1.0, 0.0, 0.5, 0.0])
                v_b = torch.tensor([0.0, 1.0, 0.3, 0.0])
                codebook.wv._vectors.data[idx_a] = v_a
                codebook.wv._vectors.data[idx_b] = v_b
                codebook.wv._normed = None

            fused = ps._max_fuse_subtokens([(97,), (98,)], codebook)
            expected = torch.max(v_a, v_b)
            self.assertTrue(torch.allclose(fused, expected, atol=1e-6),
                f"MAX fusion mismatch: got {fused}, expected {expected}")

    def test_train_step_respects_nVectors_cap(self):
        """Task 7: train_step stops growing vocab once len(vocab) == n_vectors."""
        import torch
        from Layers import ChunkLayer

        layer = ChunkLayer(nDim=4, bpe=True,
                           n_vectors=260, word_learning=1)
        layer.train()
        batch = torch.tensor(
            [list(b"abababababababababab") + [0] * 4],
            dtype=torch.long,
        )
        for _ in range(100):
            layer.train_step(batch, k_merges=4)
        self.assertLessEqual(len(layer.vocab), 260,
            f"vocab overflow: got {len(layer.vocab)}, cap was 260")

    def test_mm_5m_xml_loads_and_forwards(self):
        """Task 11: MM_5M.xml can be loaded and runs one forward pass.

        Forces ``autoload=false`` so a stale on-disk checkpoint
        (``data/MM_5M.ckpt``) doesn't block this smoke test -- we're
        verifying the XML loads and forward runs, not weight loading.
        """
        import torch, tempfile
        import xml.etree.ElementTree as ET
        from Models import BaseModel

        cfg_path = os.path.join(_PROJECT, "data", "MM_5M.xml")
        tree = ET.parse(cfg_path)
        root = tree.getroot()
        arch = root.find("architecture")
        training = arch.find("training")
        if training is None:
            training = ET.SubElement(arch, "training")
        auto = training.find("autoload")
        if auto is None:
            auto = ET.SubElement(training, "autoload")
        auto.text = "false"
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()
        cfg_path = tmp.name
        model, cfg = BaseModel.from_config(config_path=cfg_path)
        ps = model.perceptualSpace
        self.assertEqual(ps.chunking_mode, "bpe")
        self.assertEqual(ps.nVectors, 4096)

        input_text = ["hello world foo"]
        inp_tensor = model.inputSpace.prepInput(input_text)
        with torch.no_grad():
            _ = model.forward(inp_tensor)
        event = ps.subspace.event.getW()
        self.assertIsNotNone(event)
        self.assertTrue(torch.isfinite(event).all(),
            "PerceptualSpace output must be finite")


if __name__ == "__main__":
    unittest.main()
