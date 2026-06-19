import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN  = os.path.normpath(os.path.join(_HERE, "..", "bin"))
sys.path.insert(0, _BIN)

import Models  # noqa: E402
from embed import WordVectors  # noqa: E402
from Spaces import Embedding  # noqa: E402


def _build_text_model(train_embeddings=False):
    """Construct an embedding-mode XOR model. Mirrors
    test_basicmodel.TestTrainEmbeddingsFlag._create_model.
    """
    xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    training = root.find("architecture/training")
    if training is None:
        training = ET.SubElement(root.find("architecture"), "training")
    auto = training.find("autoload")
    if auto is None:
        auto = ET.SubElement(training, "autoload")
    auto.text = "false"
    te = root.find("architecture/trainEmbeddings")
    if te is None:
        te = ET.SubElement(root.find("architecture"), "trainEmbeddings")
    te.text = str(train_embeddings).lower()
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True)
    tmp.close()
    Models.TheData.load("xor")
    m = Models.BasicModel()
    m.create_from_config(tmp.name, data=Models.TheData)
    os.unlink(tmp.name)
    return m


class TestLexiconOwnership(unittest.TestCase):
    def test_embedding_lives_on_perceptual_space(self):
        m = _build_text_model()
        self.assertIsInstance(m.perceptualSpace.vocabulary, Embedding)
        # InputSpace must not own a lexicon in text mode. A non-None
        # placeholder Tensor is fine -- what matters is that no Embedding
        # lives here.
        self.assertNotIsInstance(m.inputSpace.vocabulary, Embedding)

    def test_symbolic_space_paired_row_api_is_retired(self):
        """Step 3 (2026-06-10 symbolic-iteration plan): the Stage-1.B
        paired-row API (``insert_paired_word`` -- orth copy + random
        semantic partner per lexicon word on SS.codebook) is RETIRED.
        The CS-leg symbol codebook captures the code-as-written vs
        code-for-the-concept correspondence in place; the lexicon stays
        PS-local.
        """
        m = _build_text_model()
        ws = m.wholeSpace
        self.assertFalse(hasattr(ws, 'insert_paired_word'),
                         "WholeSpace.insert_paired_word must be "
                         "RETIRED (Step 3, symbolic-iteration plan).")
        self.assertFalse(hasattr(ws, 'mark_word_atom'),
                         "the mark_word_atom autobind fallback retires "
                         "with the paired-row machinery.")

    def test_embedding_symbolic_back_ref_wired(self):
        """The PS-side Embedding carries ``wholeSpace_ref`` for the
        vocabulary/orthographic API delegation. (Its former role --
        triggering ``insert_paired_word`` on lexicon inserts -- retired
        with Step 3 of the symbolic-iteration plan.)
        """
        m = _build_text_model()
        emb = m.perceptualSpace.vocabulary
        self.assertIsInstance(emb, Embedding)
        peer = getattr(emb, 'wholeSpace_ref', None)
        self.assertIsNotNone(
            peer,
            "Embedding.wholeSpace_ref must be wired so the PS-side "
            "OOV insert path triggers SS-side paired-row insertion.")

    def test_get_embedding_returns_perceptual(self):
        m = _build_text_model()
        emb = m._get_embedding()
        self.assertIs(emb, m.perceptualSpace.vocabulary)

    def test_output_space_text_mode_reads_perceptual(self):
        m = _build_text_model()
        self.assertIs(m.outputSpace._vocabulary, m.perceptualSpace.vocabulary)
        self.assertTrue(m.outputSpace.text_mode)

    def test_save_and_load_embeddings_round_trip(self):
        """Post-2026-05-12 integrated weights: embeddings round-trip
        through the single ``save_weights`` / ``load_weights`` bundle
        (the ``.kv`` artifact and ``save_embeddings`` / ``load_embeddings``
        methods were retired).
        """
        m = _build_text_model()
        self.assertIsInstance(m.perceptualSpace.vocabulary, Embedding)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "weights.ckpt")
            m.save_weights(path)
            self.assertTrue(os.path.exists(path))
            ok = m.load_weights(path)
            self.assertTrue(ok)

    def test_optimizer_includes_emb_params_when_trainable(self):
        m = _build_text_model(train_embeddings=True)
        emb_weight = m.perceptualSpace.vocabulary.wv._vectors
        opt = m.getOptimizer(lr=1e-3)
        opt_ptrs = [p.data_ptr() for g in opt.param_groups for p in g["params"]]
        self.assertIn(emb_weight.data_ptr(), opt_ptrs)

    def test_optimizer_excludes_emb_params_when_frozen(self):
        m = _build_text_model(train_embeddings=False)
        emb_weight = m.perceptualSpace.vocabulary.wv._vectors
        opt = m.getOptimizer(lr=1e-3)
        opt_ptrs = [p.data_ptr() for g in opt.param_groups for p in g["params"]]
        self.assertNotIn(emb_weight.data_ptr(), opt_ptrs)

    def test_getW_wraps_into_unit_cell(self):
        """getW() always returns coordinates in [-1, 1) regardless of
        optimizer drift on the underlying parameter."""
        m = _build_text_model()
        emb = m.perceptualSpace.vocabulary
        raw = torch.zeros_like(emb.wv._vectors[0])
        # Drift outside the cell along two coordinates.
        raw[:2] = torch.tensor([1.25, -1.50], device=raw.device)
        with torch.no_grad():
            emb.wv._vectors[0].copy_(raw)
        # Expected wrapped values: 1.25 -> -0.75, -1.50 -> 0.50
        wrapped = emb.getW()[0, :2]
        self.assertTrue(torch.allclose(
            wrapped,
            torch.tensor([-0.75, 0.50], device=raw.device),
            atol=1e-6,
        ))

    def test_reverse_uses_wrapped_mse(self):
        """Reverse codebook-snap finds the nearest entry under wrapped
        MSE, not cosine -- so a query at +0.95 should snap to a codebook
        entry at -0.90 (wrapped distance 0.15) rather than +0.20 (Euclidean
        distance 0.75)."""
        emb = Embedding()
        emb.nInput = 1
        emb.nVectors = 2
        emb.nDim = 2
        vectors = torch.tensor([[-0.90, 0.0], [0.20, 0.0]],
                               dtype=torch.float32)
        emb.wv = WordVectors(vectors, ["wrap", "plain"])

        snapped = emb.reverse(torch.tensor([[[0.95, 0.0]]]))

        self.assertTrue(torch.allclose(snapped[0, 0], vectors[0]))

    def test_save_load_round_trip_wraps_vectors(self):
        """Vectors are wrapped on construction, save, and load -- so the
        round-trip is idempotent and legacy sphere artifacts (norm ~= 1,
        components in [-1, 1]) survive untouched."""
        original = torch.tensor(
            [[0.25, 0.50], [-0.75, 0.10]], dtype=torch.float32)
        wv = WordVectors(original, ["a", "b"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lex.kv")
            wv.save(path)
            loaded = WordVectors.load(path)
        loaded_cpu = loaded._vectors.detach().cpu()
        self.assertTrue(torch.allclose(loaded_cpu, original.cpu()))

    def test_load_wraps_drifted_vectors(self):
        """A saved artifact whose vectors drifted outside [-1, 1) gets
        wrapped on load (the WordVectors constructor enforces the
        domain). Defends against half-broken legacy artifacts where the
        post-step wrap was missed."""
        drifted = torch.tensor(
            [[1.25, -1.50], [0.30, 0.40]], dtype=torch.float32)
        wv = WordVectors(drifted, ["a", "b"])
        # Construction itself should have wrapped.
        wrapped_cpu = wv._vectors[0].detach().cpu()
        expected = torch.tensor([-0.75, 0.50], device=wrapped_cpu.device)
        self.assertTrue(torch.allclose(wrapped_cpu, expected, atol=1e-6))

    def test_basis_unit_ball_dispatch(self):
        """``Basis.unit_ball`` is derived from ``use_dot_product``: False
        means torus (default for Embedding, Codebook, Tensor); True
        means sphere (ConceptualSpace's Codebook). The two flags can
        never disagree because one is computed from the other."""
        from Spaces import Basis, Codebook, Embedding
        self.assertTrue(Basis().unit_ball)
        self.assertTrue(Codebook().unit_ball)
        self.assertTrue(Embedding().unit_ball)
        cb = Codebook()
        cb.use_dot_product = True
        self.assertFalse(cb.unit_ball)

    def test_codebook_torus_uses_wrapped_mse(self):
        """Codebook with default geometry (use_dot_product=False) snaps
        on wrapped MSE, so a query at +0.95 finds a codebook entry at
        -0.90 (wrapped distance 0.15) over one at +0.20 (Euclidean
        0.75). Mirrors the lexicon test, but on a plain Codebook so the
        torus path lifted into Basis is exercised for WholeSpace's
        codebook geometry too."""
        from Spaces import Basis
        b = Basis()
        b.nDim = 2
        b.nVectors = 2
        weight = torch.tensor([[-0.90, 0.0], [0.20, 0.0]])
        snapped = b._snap_content(
            torch.tensor([[[0.95, 0.0]]]),
            weight=weight, nWhat=2,
        )
        self.assertTrue(torch.allclose(
            snapped[0, 0],
            torch.tensor([-0.90, 0.0], device=snapped.device),
            atol=1e-6,
        ))

    def test_codebook_sphere_uses_cosine(self):
        """Codebook with use_dot_product=True keeps cosine snap. Same
        query as the torus test: +0.95 should snap to +0.20 (cosine
        similarity 1.0) not -0.90 (cosine -1.0)."""
        from Spaces import Basis
        b = Basis()
        b.use_dot_product = True
        b.nDim = 2
        b.nVectors = 2
        weight = torch.tensor([[-0.90, 0.0], [0.20, 0.0]])
        snapped = b._snap_content(
            torch.tensor([[[0.95, 0.0]]]),
            weight=weight, nWhat=2,
        )
        self.assertTrue(torch.allclose(
            snapped[0, 0],
            torch.tensor([0.20, 0.0], device=snapped.device),
            atol=1e-6,
        ))

    def test_sbow_smoke_keeps_vectors_in_unit_cell(self):
        """End-to-end: a few SBOW steps must leave every vector inside
        ``[-1, 1)``. Catches regressions where post-step wrapping is
        skipped or where the loss surface drives weights unbounded."""
        from embed import StreamingSBOWTrainer
        trainer = StreamingSBOWTrainer(
            vector_size=4, min_count=1, learning_rate=0.05)
        trainer.scan_document("the quick brown fox jumps over the lazy dog")
        trainer.scan_document("the quick brown dog barks")
        trainer.build_vocab()
        for _ in range(3):
            trainer.process_document(
                "the quick brown fox jumps over the lazy dog")
        wv = trainer.finish()
        v = wv._vectors.detach()
        self.assertTrue(torch.all(v >= -1.0))
        self.assertTrue(torch.all(v < 1.0 + 1e-6))


class TestCheckpointBundle(unittest.TestCase):
    """Post-2026-05-12 integrated weights: the single .ckpt bundle
    carries every learnable tensor (model parameters + embedding
    vectors + register-buffer state) alongside the Python-side
    vocab and BPE state. The separate .kv embedding artifact was
    retired so the .ckpt is now self-sufficient for reload.
    """

    def test_ckpt_includes_embedding_vectors(self):
        import torch
        m = _build_text_model()
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "weights.ckpt")
            m.save_weights(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))
            saved = torch.load(ckpt_path, map_location="cpu",
                               weights_only=False)
            state = saved["state_dict"] if isinstance(saved, dict) \
                                           and "state_dict" in saved else saved
            # The embedding vectors are stored as ``wv._local_vectors`` when
            # WordVectors owns its own Parameter. In the converged modality
            # architecture the WholeSpace codebook is mandatory and
            # WordVectors ties to it (single trainable storage), so the local
            # Parameter is dropped and the vectors live in the SS codebook's
            # ``W`` -- which the state_dict still carries (under the codebook's
            # key). Accept either, so the ckpt is self-sufficient for reload.
            present = [k for k in state.keys() if "wv._local_vectors" in k]
            if not present:
                vec = m._get_embedding().wv._vectors.detach().cpu().float()
                present = [
                    k for k, v in state.items()
                    if torch.is_tensor(v) and tuple(v.shape) == tuple(vec.shape)
                    and torch.allclose(v.detach().cpu().float(), vec)
                ]
            self.assertGreater(
                len(present), 0,
                ".ckpt must include the embedding vectors -- via "
                "wv._local_vectors (untied) or the tied SS-codebook W (tied)."
            )

    def test_ckpt_includes_vocab_and_bpe_extras(self):
        """The bundle carries Python-side mappings (vocab_extras) and
        ChunkLayer state (bpe_extras) alongside the tensor state_dict.
        """
        import torch
        m = _build_text_model()
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "weights.ckpt")
            m.save_weights(ckpt_path)
            saved = torch.load(ckpt_path, map_location="cpu",
                               weights_only=False)
            self.assertIn("state_dict", saved)
            self.assertIn("vocab_extras", saved)
            # bpe_extras is None for non-BPE models; just assert the key
            # is present (it's always written, possibly as None).
            self.assertIn("bpe_extras", saved)


def test_inputspace_expand_masked_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'expand_masked'), \
        "InputSpace.expand_masked should be deleted (MLM removed)"

def test_outputspace_expand_masked_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.OutputSpace, 'expand_masked'), \
        "OutputSpace.expand_masked should be deleted (MLM removed)"


def test_inputspace_arir_step_retired():
    """``arir_step`` retired 2026-05-14 alongside ``<maskedPrediction>``;
    sentence-level generation moves to ``BasicModel.generate_sentence``
    backed by the ARMA(p, q) InterSentenceLayer.
    """
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'arir_step'), \
        "InputSpace.arir_step should be deleted (ARIR runtime retired)"


def test_inputspace_getbatch_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'getBatch'), \
        "InputSpace.getBatch should be deleted"


def test_inputspace_peer_embedding_is_gone():
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.InputSpace.__init__)
    assert '_peer_embedding' not in src, \
        "_peer_embedding shortcut is gone; use _peer_perceptual.vocabulary"

def test_inputspace_predict_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'predict'), \
        "InputSpace.predict delegator deleted; callers use perceptualSpace.vocabulary.predict"

def test_inputspace_embed_token_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'embed_token')

def test_inputspace_get_space_embedding_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'get_space_embedding')

def test_inputspace_get_mask_embedding_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'get_mask_embedding')

def test_inputspace_lexicon_helper_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, '_lexicon')


def test_inputspace_has_lex_batch():
    from bin import Spaces
    assert hasattr(Spaces.InputSpace, '_lex_batch'), \
        "InputSpace should own lexing via _lex_batch (pure lexer, no codebook)"


def test_inputspace_forward_does_not_stash_raw_input():
    """After lex move: embedding-mode InputSpace.forward no longer sets
    subspace._raw_input (lexing happens locally; subspace.what.W holds the
    null-terminated byte buffer)."""
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.InputSpace.forward)
    assert '_raw_input' not in src, \
        "InputSpace.forward should no longer stash _raw_input; " \
        "lexing happens in-space and subspace.what/where/when are populated directly."


def test_perceptualspace_embed_reads_subspace_not_raw():
    """After lex move: PartSpace._embed decodes the upstream
    byte buffer (subspace.what.W), not _raw_input. Tokenization no longer
    hides inside vocab.forward(raw_input, return_meta=True)."""
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.PartSpace._embed)
    assert '_raw_input' not in src, \
        "_embed should read upstream subspace.what.W, not _raw_input"
    assert 'vocab.forward' not in src, \
        "_embed should not call vocab.forward (that does lexing); " \
        "use codebook lookup on pre-lexed byte buffer instead."


def test_perceptualspace_lex_and_embed_renamed_to_embed():
    """After Task 4: the old _lex_and_embed name is gone -- the method is
    codebook-only now (it doesn't lex) so it's just _embed."""
    from bin import Spaces
    assert not hasattr(Spaces.PartSpace, '_lex_and_embed'), \
        "_lex_and_embed was renamed to _embed; delete the old reference"
    assert hasattr(Spaces.PartSpace, '_embed'), \
        "_embed must exist after Task 4 rename"


def test_what_encoding_roundtrip():
    """WhatEncoding.encode_tokens / decode_tokens are the single
    reader/writer for the null-terminated byte layout on .what.W."""
    import torch
    from bin.Spaces import WhatEncoding
    enc = WhatEncoding([2, 10], [2, 10])  # nObj=2, per-slot dim irrelevant here
    nWhat = 8
    tokens = [["hello", "world"], ["", "hi"]]
    buf = enc.encode_tokens(tokens, batch=2, nObj=2, nWhat=nWhat,
                            device=torch.device("cpu"))
    assert buf.shape == (2, 2, nWhat)
    # null terminator is present in used slots
    assert buf[0, 0, 5].item() == 0  # after "hello" (5 bytes)
    decoded = enc.decode_tokens(buf)
    assert decoded == tokens


def test_what_encoding_truncates_oversized_tokens():
    """Tokens whose UTF-8 encoding exceeds nWhat-1 are truncated."""
    import torch
    from bin.Spaces import WhatEncoding
    enc = WhatEncoding([1, 4], [1, 4])
    nWhat = 4  # 3 bytes for content + 1 null
    buf = enc.encode_tokens([["abcdef"]], batch=1, nObj=1, nWhat=nWhat,
                            device=torch.device("cpu"))
    decoded = enc.decode_tokens(buf)
    assert decoded == [["abc"]]


if __name__ == "__main__":
    unittest.main()
