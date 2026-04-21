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


if __name__ == "__main__":
    unittest.main()
