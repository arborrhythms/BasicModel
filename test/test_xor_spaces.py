"""XOR_spaces testpoint -- inline dataset, tokenizer, and space prediction.

Exercises:
  - XML parser: ``<input use="train">`` / ``<output use="train">`` attributes
  - TheData.loadInline(): pipe-separated sentence/label parsing
  - Embedding: space character (chr(32)) in vocab via ASCII bootstrap
  - Forward pass on whole-sentence XOR input
  - reconstruct_buffer: tokens joined with ``"".join()`` not ``" ".join()``
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "XOR_spaces.xml")

_TEST = os.path.dirname(os.path.abspath(__file__))
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

import Models

from test_basicmodel import _populate_test_config, _obj_size

# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestXORSpacesConfigParsing(unittest.TestCase):
    """_parse_element handles <input use="..."> attributes and repeated tags."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Models.BaseModel.load_config(_CONFIG)

    def test_config_loads(self):
        self.assertIn("architecture", self.cfg)
        self.assertIn("InputSpace", self.cfg)

    def test_dataset_is_inline(self):
        dat = self.cfg["architecture"]["data"]
        self.assertEqual(dat.get("dataset"), "inline")

    def test_input_elements_parsed(self):
        """<input use="train"> is parsed with 'use' and '_' keys."""
        dat = self.cfg["architecture"]["data"]
        inputs = dat.get("input")
        # Two <input> elements -> list
        self.assertIsInstance(inputs, list, "Expected list of input entries")
        self.assertEqual(len(inputs), 2)
        uses = {item["use"] for item in inputs}
        self.assertIn("train", uses)
        self.assertIn("test", uses)

    def test_output_elements_parsed(self):
        """<output use="train"> is parsed with 'use' and '_' keys."""
        dat = self.cfg["architecture"]["data"]
        outputs = dat.get("output")
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        uses = {item["use"] for item in outputs}
        self.assertIn("train", uses)
        self.assertIn("test", uses)

    def test_train_input_contains_sentences(self):
        dat = self.cfg["architecture"]["data"]
        inputs = dat["input"]
        train_text = next(i["_"] for i in inputs if i["use"] == "train")
        sentences = [s.strip() for s in train_text.split("|") if s.strip()]
        self.assertEqual(len(sentences), 4)
        # Each sentence has at least one space
        for s in sentences:
            self.assertIn(" ", s, f"Sentence has no spaces: {s!r}")

    def test_train_output_labels(self):
        dat = self.cfg["architecture"]["data"]
        outputs = dat["output"]
        train_labels = next(o["_"] for o in outputs if o["use"] == "train")
        labels = [v.strip() for v in train_labels.split("|") if v.strip()]
        self.assertEqual(len(labels), 4)
        self.assertEqual([float(l) for l in labels], [0.0, 1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Inline data loading
# ---------------------------------------------------------------------------

class TestLoadInline(unittest.TestCase):
    """TheData.loadInline() correctly parses pipe-separated sentences."""

    def test_four_training_examples(self):
        dat = {
            "input": [
                {"_": "zero xor zero|zero xor one|one xor zero|one xor one", "use": "train"},
                {"_": "zero xor zero|one xor one",                           "use": "test"},
            ],
            "output": [
                {"_": "0|1|1|0", "use": "train"},
                {"_": "0|0",     "use": "test"},
            ],
        }
        data = Models.Data()
        data.loadInline(dat)
        # processLM stores string tensors for train_input
        self.assertEqual(len(data.train_input), 4)
        self.assertEqual(len(data.test_input), 2)

    def test_labels_as_float_tensors(self):
        import torch
        dat = {
            "input": [
                {"_": "zero xor zero|zero xor one", "use": "train"},
                {"_": "zero xor zero|zero xor one", "use": "test"},
            ],
            "output": [
                {"_": "0|1", "use": "train"},
                {"_": "0|1", "use": "test"},
            ],
        }
        data = Models.Data()
        data.loadInline(dat)
        # Each output is a 1-D float tensor
        for t in data.test_output:
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.dtype, torch.float32)

    def test_single_input_element_not_list(self):
        """A single <input> (no duplication) is normalised to a list."""
        # When only one element, parser returns a dict (not a list)
        dat = {
            "input": {"_": "hello world|foo bar", "use": "train"},
            "output": {"_": "0|1", "use": "train"},
        }
        data = Models.Data()
        data.loadInline(dat)
        self.assertEqual(len(data.train_input), 2)


# ---------------------------------------------------------------------------
# Model creation and forward pass
# ---------------------------------------------------------------------------

class TestXORSpacesModel(unittest.TestCase):
    """BasicModel creates and runs on XOR_spaces inline data."""

    @classmethod
    def setUpClass(cls):
        import torch
        # Load config, extract inline data into TheData singleton
        cfg = Models.BasicModel.load_config(_CONFIG)
        arch = cfg.get("architecture", {})
        dat = arch.get("data", {})

        Models.TheData.load("inline", dat=dat)
        cls.data = Models.TheData

        cls.model = Models.BasicModel()
        cls.model.create_from_config(_CONFIG, data=Models.TheData)

    def test_model_has_input_space(self):
        self.assertTrue(hasattr(self.model, "inputSpace"))

    def test_space_char_in_vocab(self):
        """ASCII bootstrap ensures space chr(32) is always in vocabulary."""
        emb = self.model.perceptualSpace.vocabulary
        self.assertIsInstance(emb, Models.Embedding,
                              "InputSpace should use Embedding for text model")
        self.assertIn(" ", emb.pretrain.key_to_index,
                      "Space character must be in vocabulary (ASCII bootstrap)")

    def test_xor_words_in_vocab(self):
        """XOR sentence words are added to vocabulary during data loading."""
        emb = self.model.perceptualSpace.vocabulary
        # Words appear in training data; they should be added via forward passes
        # or they may still be pending -- we only assert ASCII chars are present
        for ch in "zero one xor":
            if ch == " ":
                continue  # already tested above
            # Individual characters must always be there (ASCII bootstrap)
            self.assertIn(ch, emb.pretrain.key_to_index,
                          f"ASCII char {ch!r} missing from vocabulary")

    def test_forward_pass_runs(self):
        """Forward pass on the first XOR training sentence completes without error."""
        import torch
        self.model.eval()
        loader = self.model.inputSpace.data.data_loader(split="train", num_streams=1)
        inp_items, out_items = next(iter(loader))
        inputTensor = self.model.inputSpace.prepInput(inp_items)
        outputTensor = (self.model.outputSpace.prepOutput(out_items)
                        if out_items is not None else None)
        batch = (inputTensor, outputTensor)
        inp, _ = batch
        with torch.no_grad():
            out = self.model.forward(inp)
        # forward() returns (input_state, symbols, output, reconstruction)
        self.assertEqual(len(out), 4)

    def test_output_shape(self):
        """Output tensor has shape [batch, nOutput, outputDim]."""
        import torch
        self.model.eval()
        loader = self.model.inputSpace.data.data_loader(split="train", num_streams=1)
        inp_items, out_items = next(iter(loader))
        inputTensor = self.model.inputSpace.prepInput(inp_items)
        outputTensor = (self.model.outputSpace.prepOutput(out_items)
                        if out_items is not None else None)
        batch = (inputTensor, outputTensor)
        inp, _ = batch
        with torch.no_grad():
            _, _, output, _ = self.model.forward(inp)
        self.assertEqual(output.shape[0], 1)   # batch
        self.assertEqual(output.shape[1], self.model.nOutput)


# ---------------------------------------------------------------------------
# Space prediction in reconstruct_buffer
# ---------------------------------------------------------------------------

class TestSpacePrediction(unittest.TestCase):
    """Space character is predicted as a token, not auto-inserted by join."""

    def test_consecutive_join_no_auto_spaces(self):
        """reconstruct_buffer consecutive path uses ''.join(), not ' '.join().

        Verify that two adjacent tokens are concatenated without an extra space
        being inserted between them -- spaces must come from predicted tokens.
        """
        import torch

        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nPercepts=nInput, nConcepts=nInput,
                              nSymbols=nInput, nWords=nInput, nOutput=nInput,
                              nWhere=2, nWhen=2, flatten=True)
        Models.TheData.load("xor")

        _pdim = Models.TheXMLConfig.space("PerceptualSpace", "nDim")
        _pvec = Models.TheXMLConfig.space("PerceptualSpace", "nVectors")
        _obj = _obj_size("PerceptualSpace")
        psp = Models.PerceptualSpace([nInput, _pdim], [_pvec, _pdim], [nInput, _pdim + _obj],
                         model_type="embedding")
        emb = psp.vocabulary
        self.assertIsInstance(emb, Models.Embedding)

        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [4, _odim], [4, _odim])
        os_.set_text_mode(psp)

        codebook = emb.wv._vectors.detach()
        words_list = emb.wv.index_to_key
        embSize = psp.muxedSize
        nWhat = psp.nWhat  # content dims only

        # Find two non-[MASK] tokens that are NOT the space or \x00 sentinel
        usable = [j for j, w in enumerate(words_list)
                  if w not in ("[MASK]", " ", "\x00")]

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize])
        expected = []
        for slot, j in enumerate(usable[:nVec]):
            # Assign only content dims; positional slots remain 0 -> consecutive mode
            vectors[0, slot, :nWhat] = codebook[j][:nWhat]
            expected.append(words_list[j])

        result = os_.reconstruct_buffer(vectors)
        text = result[0]

        # The two tokens should appear concatenated without an extra space
        # (unless the predicted tokens themselves contain spaces)
        self.assertEqual(text, "".join(expected),
                         f"Expected '{''.join(expected)}', got {text!r}")

    def test_space_token_predicted_by_reverse(self):
        """When a space vector is placed in output, it decodes to ' '."""
        import torch

        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nPercepts=nInput, nConcepts=nInput,
                              nSymbols=nInput, nWords=nInput, nOutput=nInput,
                              nWhere=2, nWhen=2, flatten=True)
        Models.TheData.load("xor")

        _pdim = Models.TheXMLConfig.space("PerceptualSpace", "nDim")
        _pvec = Models.TheXMLConfig.space("PerceptualSpace", "nVectors")
        _obj = _obj_size("PerceptualSpace")
        psp = Models.PerceptualSpace([nInput, _pdim], [_pvec, _pdim], [nInput, _pdim + _obj],
                         model_type="embedding")
        emb = psp.vocabulary
        self.assertIn(" ", emb.pretrain.key_to_index)

        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [4, _odim], [4, _odim])
        os_.set_text_mode(psp)

        space_idx = emb.pretrain.key_to_index[" "]
        codebook = emb.wv._vectors.detach()
        embSize = psp.muxedSize
        nWhat = psp.nWhat  # content dims only

        vectors = torch.zeros([1, 1, embSize])
        # Assign only the content dims; positional slots remain 0 -> consecutive mode
        vectors[0, 0, :nWhat] = codebook[space_idx][:nWhat]

        result = os_.reconstruct_buffer(vectors)
        self.assertEqual(result[0], " ",
                         f"Space vector should decode to ' ', got {result[0]!r}")


# ---------------------------------------------------------------------------
# NULL/EOS termination for clean reconstruction
# ---------------------------------------------------------------------------

class TestNullEOS(unittest.TestCase):
    """NULL character (\x00) acts as an EOS sentinel in reconstruction.

    The design:
    - The Embedding vocabulary contains \x00 mapped to the zero vector.
    - Padding slots in the input produce zero vectors; the model learns to
      predict zero for those slots.
    - reconstruct_buffer returns a null-terminated buffer, so decoding stops
      at the first \x00 without trailing padding garbage.
    """

    @classmethod
    def setUpClass(cls):
        import torch

        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nPercepts=nInput, nConcepts=nInput,
                              nSymbols=nInput, nWords=nInput, nOutput=nInput,
                              nWhere=2, nWhen=2, flatten=True)
        Models.TheData.load("xor")

        _pdim = Models.TheXMLConfig.space("PerceptualSpace", "nDim")
        _pvec = Models.TheXMLConfig.space("PerceptualSpace", "nVectors")
        cls.psp = Models.PerceptualSpace([nInput, _pdim], [_pvec, _pdim], [nInput, _pdim],
                             model_type="embedding")
        cls.emb = cls.psp.vocabulary
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        cls.os_ = Models.OutputSpace([nInput, _sdim], [4, _odim], [4, _odim])
        cls.os_.set_text_mode(cls.psp)

    def test_null_char_in_embedding_vocab(self):
        """\\x00 must be registered in the Embedding vocabulary.

        This is the EOS/padding sentinel -- its presence guarantees that the
        nearest-neighbour lookup in reconstruct_buffer can resolve the zero
        vector to '\\x00' rather than an arbitrary ASCII character.
        """
        self.assertIn(
            "\x00", self.emb.pretrain.key_to_index,
            "Null byte (EOS sentinel) must be in Embedding vocabulary",
        )

    def test_null_embedding_is_regular_vector(self):
        """The \\x00 codebook entry has a regular nonzero embedding.

        \\x00 participates in the vocabulary like any other token -- it is NOT
        the zero vector.  Zero-vector slots are detected in reconstruct_data by
        the abs-sum threshold check (< 1e-8) and mapped directly to '\\x00'
        before the nearest-neighbour cosine lookup.
        """
        import torch
        self.assertIn("\x00", self.emb.pretrain.key_to_index)
        idx = self.emb.pretrain.key_to_index["\x00"]
        vec = self.emb.wv._vectors[idx].detach()
        self.assertGreater(
            vec.norm().item(), 1e-6,
            "\\x00 embedding must be a regular nonzero vector",
        )

    def test_reconstruct_buffer_stops_at_null_token(self):
        """reconstruct_buffer (consecutive) stops at the first \\x00 token.

        Sequence layout:  [word_A, \\x00, word_B]
        Expected output:   word_A + \\x00  (stopped before B, buffer terminated)
        """
        import torch

        emb = self.emb
        self.assertIn("\x00", emb.pretrain.key_to_index)

        codebook = emb.wv._vectors.detach()
        words_list = emb.wv.index_to_key
        embSize = self.psp.muxedSize
        # The reconstruction content-match (``_reverse_text_vectors``) splits the
        # query at ``emb.content_dim`` (the embedding's full width, which for a
        # muxed embedding includes the where/when band baked into each codebook
        # row), so copy that full width -- copying only ``psp.nWhat`` leaves the
        # where/when dims zero and the nearest-row match can drift. (2026-06-07
        # .when redesign shrank the .when magnitude, exposing this slicing gap.)
        nWhat = emb.content_dim

        null_idx = emb.pretrain.key_to_index["\x00"]

        # Pick a real word that is not [MASK] or \x00
        word_idx = next(
            j for j, w in enumerate(words_list) if w not in ("[MASK]", "\x00", " ")
        )
        word_text = words_list[word_idx]

        # Build 3-slot sequence: [word, \x00, word] over the full content width.
        batch, nVec = 1, 3
        vectors = torch.zeros([batch, nVec, embSize])
        vectors[0, 0, :nWhat] = codebook[word_idx][:nWhat]   # slot 0: real word
        vectors[0, 1, :nWhat] = codebook[null_idx][:nWhat]   # slot 1: \x00  (zero vec)
        vectors[0, 2, :nWhat] = codebook[word_idx][:nWhat]   # slot 2: same real word

        result = self.os_.reconstruct_buffer(vectors)
        text = result[0]

        # Should see word_text once, followed by a single NULL terminator.
        self.assertEqual(
            text, word_text + "\x00",
            f"Expected null-terminated buffer {(word_text + chr(0))!r}, got {text!r}",
        )

    def test_trailing_null_produces_clean_reconstruction(self):
        """A sequence ending with \\x00 vectors produces text without trailing garbage.

        Mirrors the XOR reconstruction scenario: the sentence occupies the
        first N slots and the rest are padding (zero vectors -> \\x00).
        """
        import torch

        emb = self.emb
        self.assertIn("\x00", emb.pretrain.key_to_index)

        codebook = emb.wv._vectors.detach()
        words_list = emb.wv.index_to_key
        embSize = self.psp.muxedSize
        # Copy the full content width the reconstruction content-match consumes
        # (``emb.content_dim``); see the note in
        # test_reconstruct_buffer_stops_at_null_token.
        nWhat = emb.content_dim

        null_idx = emb.pretrain.key_to_index["\x00"]

        usable = [j for j, w in enumerate(words_list)
                  if w != "\x00"]
        word_idx = usable[0]
        word_text = words_list[word_idx]

        # Two real words followed by two \x00 (terminator) slots
        batch, nVec = 1, 4
        vectors = torch.zeros([batch, nVec, embSize])
        vectors[0, 0, :nWhat] = codebook[word_idx][:nWhat]
        vectors[0, 1, :nWhat] = codebook[word_idx][:nWhat]
        # slots 2 and 3 use the actual \x00 codebook embedding (not zeros)
        vectors[0, 2, :nWhat] = codebook[null_idx][:nWhat]
        vectors[0, 3, :nWhat] = codebook[null_idx][:nWhat]

        result = self.os_.reconstruct_buffer(vectors)
        text = result[0]

        expected = word_text + word_text + "\x00"
        self.assertEqual(
            text, expected,
            f"Trailing \\x00 slots should collapse to one terminator; expected {expected!r}, got {text!r}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
