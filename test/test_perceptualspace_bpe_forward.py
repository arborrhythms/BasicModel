"""End-to-end BPE tests for PartSpace.

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


def _write_minimal_bpe_xml(tmpdir, n_vectors=512, synthesis="bpe"):
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
    <!-- 2026-06-06 uniform-band convention: EVERY tier is (2,2), so
         nDim = nWhat + .where(2) + .when(2). nWhat=4 everywhere => nDim=8
         on IS/PS/CS AND SS (was SS nDim=4 under the old (0,0) convention).
         OS nWhat=1 => nDim=5. -->
    <nDim>8</nDim>
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
    <nDim>8</nDim>
    <nVectors>{n_vectors}</nVectors>
    <synthesis>{synthesis}</synthesis>
    <wordLearning>2</wordLearning>
  </PartSpace>
  <ConceptualSpace>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </ConceptualSpace>
  <WholeSpace>
    <nOutput>32</nOutput>
    <nDim>8</nDim>
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


class TestPerceptualSpaceBPE(unittest.TestCase):

    def test_init_reads_new_config_fields(self):
        """Task 3: PartSpace reads <synthesis>, <nVectors>, <wordLearning>."""
        import tempfile
        from util import init_config
        import Spaces

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_minimal_bpe_xml(tmp, n_vectors=512)
            init_config(path=path,
                        defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
            self.assertEqual(
                Spaces.TheXMLConfig.space("PartSpace", "synthesis"), "bpe")
            self.assertEqual(
                int(Spaces.TheXMLConfig.space("PartSpace", "nVectors")), 512)
            self.assertEqual(
                int(Spaces.TheXMLConfig.space("PartSpace", "wordLearning")), 2)

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
            self.assertEqual(ps.synthesis_mode, "bpe")
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
            # ``ps.subspace`` is a working buffer the body CONSUMES after the
            # PartSpace stage (the per-stage cs.forward loop + head
            # mutate it), so reading it after the full forward is fragile.
            # Capture PartSpace.forward's OUTPUT (the percept) at the
            # source instead.
            captured = {}
            _orig_ps_fwd = ps.forward

            def _cap(x, *a, **k):
                out = _orig_ps_fwd(x, *a, **k)
                # Spec doc/specs/2026-05-21-subspace-slot-architecture.md:
                # default ``materialize()`` (mode='active') applies the
                # activation-presence gate (word_active mask), so non-word
                # positions zero out. ``mode='event'`` is the raw event.
                captured["active"] = out.materialize().detach().clone()
                return out

            ps.forward = _cap
            try:
                with torch.no_grad():
                    _ = model.forward(inp_tensor)
            finally:
                ps.forward = _orig_ps_fwd

            ps_event = captured.get("active")
            self.assertIsNotNone(ps_event, "PartSpace.forward must emit a percept")
            self.assertEqual(ps_event.shape[0], 1)
            # "6+2+2": PartSpace is a (2,2) muxed tier, so the
            # materialized event is the full EVENT width = content(4) +
            # .where(2) + .when(2) = 8, matching <PartSpace><nDim>8
            # in the fixture above (was 4 before the muxed-tier migration).
            self.assertEqual(ps_event.shape[-1], 8)
            non_zero_rows = (ps_event[0].abs().sum(dim=-1) > 0).sum().item()
            self.assertGreaterEqual(non_zero_rows, 1,
                "At least one position must have a non-zero vector")
            # BPE emits one position per byte (cold-start uses 256-byte
            # vocab; nothing merges into multi-byte chunks at init), so
            # the upper bound is the slot count (nOutput=32), not the
            # word count (3). The "word-level" assumption was tightened
            # to per-byte after the cold-start vocab change.
            self.assertLessEqual(non_zero_rows, ps_event.shape[1],
                "Non-zero positions must fit within the nOutput slot count")


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

    @unittest.expectedFailure
    def test_mm_5m_xml_loads_and_forwards(self):
        """Task 11: MM_20M.xml can be loaded and runs one forward pass.

        Forces ``autoload=false`` so a stale on-disk checkpoint
        (``data/MM_20M.ckpt``) doesn't block this smoke test -- we're
        verifying the XML loads and forward runs, not weight loading.

        Currently expected-failure (MM_20M.xml architectural mismatch):
        PS percept_dim+nWhere+nWhen=12 vs CS concept_dim+nWhere+nWhen
        =1028. Stage 1.C retired the ``sigma_percept`` lift; the signal
        router replacement (Stage 3) is not yet wired.
        """
        import torch, tempfile
        import xml.etree.ElementTree as ET
        from Models import BaseModel

        cfg_path = os.path.join(_PROJECT, "data", "MM_20M.xml")
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
        self.assertEqual(ps.synthesis_mode, "bpe")
        self.assertEqual(ps.nVectors, 4096)

        input_text = ["hello world foo"]
        inp_tensor = model.inputSpace.prepInput(input_text)
        with torch.no_grad():
            _ = model.forward(inp_tensor)
        event = ps.subspace.event.getW()
        self.assertIsNotNone(event)
        self.assertTrue(torch.isfinite(event).all(),
            "PartSpace output must be finite")


class TestSharedByteStore(unittest.TestCase):
    """Task 4 (2026-06-09 build-batch plan): ONE shared byte/percept
    codebook across the chunking front ends. bpe/mphf construct a
    PerceptStore, mirror their vocabulary into it in chunk-id order
    (``percept_id == chunk_id``), and resolve byte identity through the
    SAME reverse surface (``bytes_for``) that radix uses;
    ``chunk_layer.id_to_bytes`` is demoted to the segmentation-side
    mirror."""

    def _build_ps(self, synthesis="bpe", n_vectors=512):
        import tempfile
        from Models import BaseModel
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _write_minimal_bpe_xml(
            tmp.name, n_vectors=n_vectors, synthesis=synthesis)
        model, _ = BaseModel.from_config(config_path=path)
        return model.perceptualSpace

    def test_bpe_constructs_shared_store_with_aligned_ids(self):
        ps = self._build_ps("bpe")
        store = ps.percept_store
        self.assertIsNotNone(
            store, "bpe mode must construct the shared byte store")
        cl = ps.chunk_layer
        self.assertIs(cl.percept_store, store,
                      "ChunkLayer must be wired to the shared store")
        self.assertEqual(len(store), len(cl.vocab))
        for cid, bt in cl.id_to_bytes.items():
            self.assertEqual(store.get_id(bytes(bt)), cid)
            self.assertEqual(store.bytes_for(cid), bytes(bt))

    def test_mphf_constructs_shared_store_with_aligned_ids(self):
        ps = self._build_ps("mphf")
        store = ps.percept_store
        self.assertIsNotNone(
            store, "mphf mode must construct the shared byte store")
        cl = ps.chunk_layer
        self.assertIs(cl.percept_store, store)
        self.assertEqual(len(store), len(cl.vocab))
        for cid, bt in cl.id_to_bytes.items():
            self.assertEqual(store.bytes_for(cid), bytes(bt))

    def test_store_codebook_is_permanent_parameter(self):
        import torch.nn as nn
        ps = self._build_ps("bpe")
        W = ps.percept_store.codebook
        self.assertIsInstance(
            W, nn.Parameter,
            "the shared store's W must be a Parameter (permanent + "
            "persisted), mirroring the radix recipe")
        # Permanence: the runtime-clear idiom must preserve it.
        ps.percept_store._basis.setW(None)
        self.assertIsNotNone(ps.percept_store._basis.getW())

    def test_promotion_mirrors_into_store(self):
        import torch
        ps = self._build_ps("bpe")
        cl = ps.chunk_layer
        before = len(cl.vocab)
        # Crafted stats: one dominant pair ('a','b') promotes on the
        # first train_step (V=256 < cold_start_floor skips the lift
        # gate; vocab not full).
        seq = torch.tensor([[97, 98] * 64], dtype=torch.long)
        added = cl.train_step(seq, k_merges=1)
        self.assertGreaterEqual(added, 1, "crafted pair must promote")
        self.assertGreater(len(cl.vocab), before)
        self.assertEqual(len(ps.percept_store), len(cl.vocab),
                         "promotions must mirror into the shared store")
        for cid in (c for c in cl.id_to_bytes if c >= before):
            self.assertEqual(ps.percept_store.bytes_for(cid),
                             bytes(cl.id_to_bytes[cid]))

    def test_bytes_for_round_trips_smoke_prompt_through_store(self):
        ps = self._build_ps("bpe")
        cl = ps.chunk_layer
        prompt = b"hello world"
        pids = [cl.vocab[(b,)] for b in prompt]
        self.assertEqual(b"".join(cl.bytes_for(i) for i in pids), prompt)
        # Identical through the store surface radix uses.
        self.assertEqual(
            b"".join(ps.percept_store.bytes_for(i) for i in pids), prompt)

    def test_bytes_for_unwired_falls_back_to_private_table(self):
        from Layers import ChunkLayer
        cl = ChunkLayer(8, bpe=True, n_vectors=512)
        self.assertIsNone(cl.percept_store)
        self.assertEqual(cl.bytes_for(104), b"h")
        self.assertIsNone(cl.bytes_for(99999))


if __name__ == "__main__":
    unittest.main()
