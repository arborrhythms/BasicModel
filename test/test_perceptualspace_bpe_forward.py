"""Percept-store permanence and standalone ChunkLayer byte lookup.

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


def _write_minimal_bpe_xml(tmpdir, n_vectors=512, synthesis="meronomy"):
    """Write a tiny XML config that exercises the bpe/mphf chunking path."""
    import os
    xml = f"""<?xml version='1.0'?>
<model>
  <architecture>
    <subsymbolicOrder>2</subsymbolicOrder>
    <nWhere>0</nWhere>
    <nWhen>0</nWhen>
    <processSymbols>false</processSymbols>
    <ergodic>false</ergodic>
    <data><dataType>embedding</dataType><dataset>xor</dataset></data>
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
    <!-- Uniform-band convention: EVERY interior space_role has the same
         band, so nDim = nWhat + .where(4) + .when(4). nWhat=4 everywhere
         => nDim=12 on IS/PS/CS AND SS (2026-07-09 multi-rung pass: .where
         widened 2->4). OS is (0,0), content-only nDim=1. -->
    <nDim>12</nDim>
    <nVectors>8</nVectors>
    <!-- nOutput sized to fit the test's "hello world foo" input
         (15 chars under the BPE pre-chunking byte stream + 1 EOS
         slot). Was 8 under the legacy silent-truncation path;
         raised to 32 when §8g of the brick-vectorization handoff
         replaced the truncation with an assert. -->
    <nOutput>32</nOutput>
  </InputSpace>
  <PartSpace>
    <nInput>32</nInput>
    <nOutput>32</nOutput>
    <nDim>12</nDim>
    <nVectors>{n_vectors}</nVectors>
    <synthesis>{synthesis}</synthesis>
    <wordLearning>2</wordLearning>
  </PartSpace>
  <ConceptualSpace>
    <nOutput>32</nOutput>
    <nDim>12</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </ConceptualSpace>
  <WholeSpace>
    <nOutput>32</nOutput>
    <nDim>12</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
    <!-- Phase 4b: <lexer> lives on WholeSpace (analytic cutting). -->
    <lexer>byte</lexer>
  </WholeSpace>
  <OutputSpace>
    <nOutput>1</nOutput>
    <nDim>1</nDim>
    <nVectors>1</nVectors>
  </OutputSpace>
</model>
"""
    path = os.path.join(tmpdir, "mm_bpe_test.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path



class TestSharedByteStore(unittest.TestCase):
    """Task 4 (2026-06-09 build-batch plan): ONE shared byte/percept
    codebook across the chunking front ends. bpe/mphf construct a
    PerceptStore, mirror their vocabulary into it in chunk-id order
    (``percept_id == chunk_id``), and resolve byte identity through the
    SAME reverse surface (``bytes_for``) that radix uses;
    ``chunk_layer.id_to_bytes`` is demoted to the segmentation-side
    mirror."""

    def _build_ps(self, synthesis="meronomy", n_vectors=512):
        import tempfile
        from Models import BaseModel
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _write_minimal_bpe_xml(
            tmp.name, n_vectors=n_vectors, synthesis=synthesis)
        model, _ = BaseModel.from_config(config_path=path)
        return model.perceptualSpace



    def test_store_codebook_is_permanent_parameter(self):
        import torch.nn as nn
        ps = self._build_ps("meronomy")
        W = ps.percept_store._basis._parameters["W"]
        self.assertIsInstance(
            W, nn.Parameter,
            "the shared store's W must be a Parameter (permanent + "
            "persisted), mirroring the radix recipe")
        # Permanence: the runtime-clear idiom must preserve it.
        ps.percept_store._basis.setW(None)
        self.assertIsNotNone(ps.percept_store._basis.getW())



    def test_bytes_for_unwired_falls_back_to_private_table(self):
        from Layers import ChunkLayer
        cl = ChunkLayer(8, bpe=True, n_vectors=512)
        self.assertIsNone(cl.percept_store)
        self.assertEqual(cl.bytes_for(104), b"h")
        self.assertIsNone(cl.bytes_for(99999))


if __name__ == "__main__":
    unittest.main()
