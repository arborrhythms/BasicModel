import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN  = os.path.normpath(os.path.join(_HERE, "..", "bin"))
sys.path.insert(0, _BIN)

import Models  # noqa: E402
from Spaces import Embedding  # noqa: E402


def _build_text_model(train_embeddings=False):
    """Construct an embedding-mode XOR model. Mirrors
    test_basicmodel.TestTrainEmbeddingsFlag._create_model.
    """
    xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    auto = root.find("architecture/autoload")
    if auto is None:
        auto = ET.SubElement(root.find("architecture"), "autoload")
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

    def test_get_embedding_returns_perceptual(self):
        m = _build_text_model()
        emb = m._get_embedding()
        self.assertIs(emb, m.perceptualSpace.vocabulary)

    def test_output_space_text_mode_reads_perceptual(self):
        m = _build_text_model()
        self.assertIs(m.outputSpace._vocabulary, m.perceptualSpace.vocabulary)
        self.assertTrue(m.outputSpace.text_mode)

    def test_save_and_load_embeddings_round_trip(self):
        m = _build_text_model()
        self.assertIsInstance(m.perceptualSpace.vocabulary, Embedding)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lex.kv")
            m.save_embeddings(path)
            self.assertTrue(os.path.exists(path))
            ok = m.load_embeddings(path)
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


class TestWeightSpacePartition(unittest.TestCase):
    """The .ckpt (weights) and .kv (embeddings) artifacts must partition
    the weight space: no wv._vectors key may leak into .ckpt, and the
    Embedding weights must be recoverable from .kv alone.
    """

    def test_ckpt_excludes_embedding_vectors(self):
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
            leaked = [k for k in state.keys() if "wv._vectors" in k]
            self.assertEqual(
                leaked, [],
                f".ckpt must not contain embedding vectors; found: {leaked}"
            )

    def test_kv_contains_embedding_vectors(self):
        m = _build_text_model()
        with tempfile.TemporaryDirectory() as td:
            kv_path = os.path.join(td, "lex.kv")
            m.save_embeddings(kv_path)
            self.assertTrue(os.path.exists(kv_path))
            # File must be non-trivially sized (at minimum has some vectors).
            self.assertGreater(os.path.getsize(kv_path), 0)


def test_inputspace_expand_masked_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'expand_masked'), \
        "InputSpace.expand_masked should be deleted (MLM removed)"

def test_outputspace_expand_masked_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.OutputSpace, 'expand_masked'), \
        "OutputSpace.expand_masked should be deleted (MLM removed)"


def test_inputspace_arir_step_is_public():
    from bin import Spaces
    assert hasattr(Spaces.InputSpace, 'arir_step'), \
        "InputSpace.arir_step should be a public method (promoted from _getBatch_arir)"


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
    """After lex move: PerceptualSpace._embed decodes the upstream
    byte buffer (subspace.what.W), not _raw_input. Tokenization no longer
    hides inside vocab.forward(raw_input, return_meta=True)."""
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.PerceptualSpace._embed)
    assert '_raw_input' not in src, \
        "_embed should read upstream subspace.what.W, not _raw_input"
    assert 'vocab.forward' not in src, \
        "_embed should not call vocab.forward (that does lexing); " \
        "use codebook lookup on pre-lexed byte buffer instead."


def test_perceptualspace_lex_and_embed_renamed_to_embed():
    """After Task 4: the old _lex_and_embed name is gone -- the method is
    codebook-only now (it doesn't lex) so it's just _embed."""
    from bin import Spaces
    assert not hasattr(Spaces.PerceptualSpace, '_lex_and_embed'), \
        "_lex_and_embed was renamed to _embed; delete the old reference"
    assert hasattr(Spaces.PerceptualSpace, '_embed'), \
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
