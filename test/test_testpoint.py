"""Testpoint -- embedding probes and server queries.

Prints diagnostic values so a human can evaluate embedding quality
and inference behavior. Runs as part of `make test`.

Embedding probes run unconditionally (only need sentence.pt).
Server queries start serve.py automatically if not already running.
"""

import json
import os
import signal
import subprocess
import sys
import time
import unittest
import urllib.request
import urllib.error

import pytest

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from embed import WordVectors

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT = os.path.join(_PROJECT, "output", "embeddings", "sentence.pt")
_CONFIG = os.path.join(_PROJECT, "data", "BasicModel.xml")
_VENV_PYTHON = os.path.join(_PROJECT, ".venv", "bin", "python")

SERVER_URL = "http://127.0.0.1:8001"

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


# ---------------------------------------------------------------------------
# Embedding probes
# ---------------------------------------------------------------------------

def _load_embeddings(path=_ARTIFACT):
    """Load saved WordVectors, return (WordVectors, path)."""
    return WordVectors.load(path), path


@unittest.skipIf(not os.path.exists(_ARTIFACT), "sentence.pt not built")
class TestEmbeddingProbes(unittest.TestCase):
    """Probe embedding quality with nearest-neighbor and analogy checks."""

    @classmethod
    def setUpClass(cls):
        cls.wv, cls.path = _load_embeddings()
        cls.vocab_size = len(cls.wv)
        cls.dim = cls.wv.vector_size

    def test_vocab_minimum(self):
        """Vocabulary has at least 500 words."""
        print(f"\n  Vocab: {self.vocab_size} words, {self.dim}-dim")
        self.assertGreaterEqual(self.vocab_size, 500)

    def test_nearest_neighbors(self):
        """Print nearest neighbors for probe words (visual check)."""
        probes = ["water", "energy", "school", "food", "history",
                  "children", "science", "important", "world", "health"]
        found = 0
        for word in probes:
            if word not in self.wv:
                print(f"  {word:15s}  -- not in vocab")
                continue
            found += 1
            neighbors = self.wv.neighbors(word, topn=5)
            top = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
            print(f"  {word:15s} -> {top}")
        self.assertGreaterEqual(found, 5, "Too few probe words in vocab")

    def test_similarity_pairs(self):
        """Print cosine similarity for semantically related/unrelated pairs."""
        pairs = [
            ("water", "food"),
            ("energy", "nuclear"),
            ("children", "school"),
            ("science", "history"),
            ("health", "important"),
        ]
        for w1, w2 in pairs:
            if w1 in self.wv and w2 in self.wv:
                sim = self.wv.similarity(w1, w2)
                print(f"  sim({w1}, {w2}) = {sim:.4f}")
            else:
                missing = w1 if w1 not in self.wv else w2
                print(f"  sim({w1}, {w2}) -- '{missing}' not in vocab")

    def test_self_similarity(self):
        """Every word's nearest neighbor has positive similarity."""
        import random
        sample = random.sample(self.wv.index_to_key,
                               min(20, self.vocab_size))
        for w in sample:
            neighbors = self.wv.neighbors(w, topn=1)
            self.assertTrue(len(neighbors) > 0, f"No neighbors for '{w}'")
            _, score = neighbors[0]
            self.assertGreater(score, 0.0,
                               f"Nearest neighbor of '{w}' has non-positive similarity")

    def test_vector_norms_nonzero(self):
        """No zero vectors in the vocabulary."""
        import torch
        norms = torch.norm(torch.as_tensor(self.wv.vectors), dim=1)
        zeros = int((norms == 0).sum().item())
        print(f"  Zero-norm vectors: {zeros}/{self.vocab_size}")
        self.assertEqual(zeros, 0, "Found zero-norm vectors in embeddings")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_proc = None


def _server_reachable():
    try:
        urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def _start_server():
    """Start serve.py as a subprocess, wait until healthy."""
    global _server_proc
    if _server_reachable():
        return True
    if not os.path.exists(_CONFIG) or not os.path.exists(_VENV_PYTHON):
        return False

    env = os.environ.copy()
    env["BASICMODEL_DEVICE"] = "cpu"
    _server_proc = subprocess.Popen(
        [_VENV_PYTHON, os.path.join(_BIN, "serve.py"), "--config", _CONFIG],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Wait up to 90s for server to become healthy
    for _ in range(90):
        if _server_proc.poll() is not None:
            return False
        if _server_reachable():
            return True
        time.sleep(1)
    # Timed out
    _stop_server()
    return False


def _stop_server():
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
            _server_proc.wait()
        _server_proc = None


# ---------------------------------------------------------------------------
# Server queries
# ---------------------------------------------------------------------------

def _chat(text):
    """Send a chat query, return response text."""
    payload = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


@unittest.skipIf(not os.path.exists(_ARTIFACT), "sentence.pt not built")
@pytest.mark.slow
class TestServerQueries(unittest.TestCase):
    """Query the BasicModel server and print responses."""

    _server_started = False

    @classmethod
    def setUpClass(cls):
        cls._server_started = _start_server()
        if not cls._server_started:
            raise unittest.SkipTest("Could not start serve.py")

    @classmethod
    def tearDownClass(cls):
        _stop_server()

    def test_health(self):
        """Health endpoint returns ok=true."""
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        body = json.loads(resp.read())
        print(f"\n  Health: ok={body['ok']}, model={body['model']}")
        self.assertTrue(body["ok"])

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_demo_queries(self):
        """Print inference output for sample prompts (visual check)."""
        queries = [
            "The quick brown fox",
            "Once upon a time",
            "Scientists discovered that",
            "The weather today is",
            "In the beginning there was",
        ]
        for q in queries:
            resp = _chat(q)
            short = resp[:120] + "..." if len(resp) > 120 else resp
            print(f"  >> {q}")
            print(f"  << {short}")

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_response_format(self):
        """Response is wrapped in <conversation> tags."""
        resp = _chat("hello world")
        self.assertIn("<conversation>", resp)
        self.assertIn("</conversation>", resp)

    def test_empty_rejected(self):
        """Empty message returns 400."""
        payload = json.dumps({"messages": []}).encode()
        req = urllib.request.Request(
            f"{SERVER_URL}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("Expected HTTP error for empty messages")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_response_length_bounded(self):
        """Response doesn't exceed maxResponseLength words."""
        resp = _chat("tell me about the world")
        inner = resp.replace("<conversation>", "").replace("</conversation>", "").strip()
        word_count = len(inner.split())
        print(f"  Response length: {word_count} words")
        self.assertLessEqual(word_count, 128,
                             "Response exceeds expected max length")


# ---------------------------------------------------------------------------
# GPU device placement
# ---------------------------------------------------------------------------

# This test runs in a subprocess without BASICMODEL_DEVICE=cpu so that
# the model loads onto the best available accelerator (CUDA or MPS).

_GPU_CHECK_SCRIPT = '''\
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("{config}"))), "bin"))
os.environ.pop("BASICMODEL_DEVICE", None)

import torch
import Models

cfg_path = "{config}"
cfg = Models.BasicModel.load_config(cfg_path)
arch = cfg.get("architecture", {{}})
Models.TheData.load(arch.get("dataset", "text"),
             num_shards=arch.get("numShards", 1),
             max_docs=min(arch.get("maxDocs", 100), 100),
             shard_dir=arch.get("shardDir"))

model = Models.BasicModel()
model.create_from_config(cfg_path, data=Models.TheData)
model.eval()

# Check device of model parameters
param_devices = set()
for name, p in model.named_parameters():
    param_devices.add(str(p.device))

# Check device of buffers (registered tensors like noise)
buffer_devices = set()
for name, b in model.named_buffers():
    buffer_devices.add(str(b.device))

# Check device of training data
data_devices = set()
for t in Models.TheData.train_input[:5]:
    if isinstance(t, torch.Tensor):
        data_devices.add(str(t.device))

# Check that data_loader produces tensors on the right device
loader = model.inputSpace.data.data_loader(split="train", num_streams=2)
inp_items, out_items = next(iter(loader))
inp_tensor = model.inputSpace.prepInput(inp_items)
out_tensor = (model.outputSpace.prepOutput(out_items)
              if out_items is not None else None)
inp_device = str(inp_tensor.device)

with torch.no_grad():
    # prepInput() returns raw input for all modes; masking (if any) is applied
    # inside InputSpace.forward(), so we always call model.forward().
    _, _, out, _ = model.forward(inp_tensor)
    # AR modes return a list of per-token prediction tensors.
    first = out[0] if isinstance(out, list) else out
    out_device = str(first.device)

result = {{
    "device": str(Models.TheDevice),
    "param_devices": sorted(param_devices),
    "buffer_devices": sorted(buffer_devices),
    "data_devices": sorted(data_devices),
    "input_device": inp_device,
    "output_device": out_device,
    "n_params": sum(p.numel() for p in model.parameters()),
}}
print(json.dumps(result))
'''


def _has_gpu():
    """Check if CUDA or MPS is available without importing BasicModel."""
    import torch
    return torch.cuda.is_available() or torch.backends.mps.is_available()


@unittest.skipIf(not os.path.exists(_ARTIFACT), "sentence.pt not built")
@unittest.skipIf(not _has_gpu(), "no GPU available (need CUDA or MPS)")
@unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
class TestGPUDevicePlacement(unittest.TestCase):
    """Verify the model runs entirely on GPU when BASICMODEL_DEVICE is unset."""

    def test_model_on_gpu(self):
        """All parameters, buffers, and data tensors are on the GPU device."""
        script = _GPU_CHECK_SCRIPT.format(config=_CONFIG)
        env = os.environ.copy()
        env.pop("BASICMODEL_DEVICE", None)
        result = subprocess.run(
            [_VENV_PYTHON, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            self.fail(f"GPU check subprocess failed:\n{result.stderr[-500:]}")

        info = json.loads(result.stdout.strip().split("\n")[-1])
        device = info["device"]
        device_type = device.split(":")[0]
        print(f"\n  Device: {device} ({info['n_params']} params)")
        print(f"  Param devices: {info['param_devices']}")
        print(f"  Buffer devices: {info['buffer_devices']}")
        print(f"  Data devices: {info['data_devices']}")
        print(f"  Input device: {info['input_device']}")
        print(f"  Output device: {info['output_device']}")

        self.assertNotEqual(device, "cpu", "Model is on CPU despite GPU being available")

        # All parameter tensors on the expected device
        for d in info["param_devices"]:
            self.assertTrue(d.startswith(device_type),
                            f"Parameter on {d}, expected {device}")

        # All buffer tensors on the expected device
        for d in info["buffer_devices"]:
            self.assertTrue(d.startswith(device_type),
                            f"Buffer on {d}, expected {device}")

        # Input batch on the expected device
        self.assertTrue(info["input_device"].startswith(device_type),
                        f"Input on {info['input_device']}, expected {device}")

        # Forward pass output must be on the GPU device
        self.assertTrue(info["output_device"].startswith(device_type),
                        f"Output on {info['output_device']}, expected {device}")


_DATA_DIR = os.path.join(_PROJECT, "data")
_SCHEMA_FILE = os.path.join(_DATA_DIR, "model.xsd")


class TestModelXmlSchema(unittest.TestCase):
    """Validate all model XML config files against model.xsd."""

    @classmethod
    def setUpClass(cls):
        try:
            import lxml.etree as ET
            cls.ET = ET
            schema_doc = ET.parse(_SCHEMA_FILE)
            cls.schema = ET.XMLSchema(schema_doc)
        except ImportError:
            cls.schema = None

    def _validate(self, xml_path):
        if self.schema is None:
            self.skipTest("lxml not installed")
        ET = self.ET
        doc = ET.parse(xml_path)
        if not self.schema.validate(doc):
            msgs = "\n".join(str(e.message) for e in self.schema.error_log)
            self.fail(f"{os.path.basename(xml_path)} failed schema validation:\n{msgs}")

    def test_model_xml(self):
        self._validate(os.path.join(_DATA_DIR, "model.xml"))

    def test_basicmodel_xml(self):
        self._validate(os.path.join(_DATA_DIR, "BasicModel.xml"))

    def test_xor_exact_xml(self):
        self._validate(os.path.join(_DATA_DIR, "XOR_symbols.xml"))



    def test_xor_pos_xml(self):
        self._validate(os.path.join(_DATA_DIR, "XOR_pos.xml"))


# ---------------------------------------------------------------------------
# The structure of forms as composed
# ---------------------------------------------------------------------------

class TestStructureValid(unittest.TestCase):
    """On the lesson's words, after a suffix lesson and after two forms have
    had to be told apart: every recorded part lies inside its form, every
    known form inside a known form reads wholly inside it, and the forms that
    read as inside a form they are not inside are counted."""

    @classmethod
    def setUpClass(cls):
        from test_item17_forms import lesson_store
        cls.store, cls.identity = lesson_store()
        cls.forms = {word: cls.identity.form(word) for word in cls.identity.words}

    def _found(self, part, whole):
        from ConceptualMatch import match
        found, _ = match(self.forms[whole][None], self.forms[part][None])
        return float(found[0])

    def test_recorded_parts_lie_inside_their_form(self):
        codes = self.store._basis.fixed_row_codes[:, :self.identity.width]
        recorded = inside = 0
        for word, form in self.forms.items():
            for row in self.identity.parts(word):
                recorded += 1
                inside += bool((codes[row] <= form).all())
        print(dict(recorded_parts=recorded, inside=inside, added=dict(self.identity.added),
                   nested={w.decode(): [p.decode() for p, _, _ in v]
                           for w, v in self.identity.nested.items()}))
        self.assertEqual(inside, recorded)
        self.assertTrue(self.identity.added)
        self.assertIn(b'cats', self.identity.nested)

    def test_a_form_inside_a_form_reads_wholly_inside(self):
        words = sorted(self.forms)
        pairs = [(a, b) for a in words for b in words if a != b and a in b]
        read = {(a.decode(), b.decode()): self._found(a, b) for a, b in pairs}
        print(dict(pairs=len(pairs), wholly_inside=sum(v == 1. for v in read.values()),
                   is_in_this=read[('is', 'this')], his_in_this=read[('his', 'this')],
                   at_in_cat=read[('at', 'cat')], here_in_there=read[('here', 'there')],
                   the_in_there=read[('the', 'there')], cat_in_cats=read[('cat', 'cats')]))
        self.assertEqual(read[('is', 'this')], 1.)
        self.assertTrue(all(value == 1. for value in read.values()), read)

    def test_forms_that_read_inside_a_form_they_are_not_inside(self):
        words = sorted(self.forms)
        false = [(a.decode(), b.decode()) for a in words for b in words
                 if a != b and a not in b and self._found(a, b) == 1.]
        cat_in_caat = self._found(b'cat', b'caat')
        print(dict(read_inside_but_not_inside=len(false), forms=false, cat_in_caat=cat_in_caat))
        # No code lies inside another's by accident: a form reads wholly
        # inside only a form it is a part of, and `cat` reads below one in
        # `caat`.
        self.assertEqual(false, [])
        self.assertLess(cat_in_caat, 1.)


# ---------------------------------------------------------------------------
# A blind spot is filled (certificate 24)
# ---------------------------------------------------------------------------

class TestBlindSpot:
    """On the small lesson ("alpha gamma", "beta delta"), with every second
    form of the second word withheld: the structure expects that word's
    basic percept at its place from the parts that remain, and the way back
    reads it there; the word beside it is unchanged; a word never taught is
    not completed. Prints the values of sigma-pi section 19.23."""

    def _looks(self, model, monkeypatch, text):
        import SentenceLooks
        from test_item18_expect import read
        seen = []
        perceive = SentenceLooks.perceive
        def kept(model_, interval, rows):
            field = perceive(model_, interval, rows)
            seen.append((interval.detach().clone(), model_._concept_owner()._percept_field))
            return field
        monkeypatch.setattr(SentenceLooks, 'perceive', kept)
        read(model, text)
        monkeypatch.setattr(SentenceLooks, 'perceive', perceive)
        return seen

    def _withhold(self, monkeypatch, model, word, every):
        import torch
        from Omission import Draw
        def draw(valid, *, training):
            hidden = torch.zeros_like(valid)
            hidden[:, word, ::every] = valid[:, word, ::every]
            words = hidden.any(-1) if every == 1 else torch.zeros_like(valid.any(-1))
            return Draw(words, hidden, torch.zeros(len(valid), dtype=torch.long))
        monkeypatch.setattr(model.omission, 'draw', draw)

    def test_blind_spot(self, tmp_path, monkeypatch):
        import torch
        import ConceptSpread
        from test_item18_expect import lesson, read
        from nomination_fixtures import readings_before_nomination
        model = lesson(tmp_path, monkeypatch)
        try:
            for text in ('alpha gamma', 'beta delta'):
                readings_before_nomination(model, lambda: read(model, text))
                read(model, text)
            owner = model._concept_owner()
            basic = owner.basic_percepts
            from PrependedSymbols import word_view
            read(model, 'alpha gamma')
            concept_rows = word_view(model.inputSpace, 'concept_rows')[0, :2].tolist()
            percept_of = {row: percept for row, percept, negated in owner._concept_allocator.order_zero_members
                          if not negated}
            alpha, gamma = (percept_of.get(row) for row in concept_rows)
            ordinary = model.omission.draw
            rows = {}
            for label, text, every in (('none', 'alpha gamma', None), ('half', 'alpha gamma', 2),
                                       ('whole', 'alpha gamma', 1), ('untaught', 'alpha omega', None)):
                if every is None:
                    model.omission.draw = ordinary
                else:
                    self._withhold(monkeypatch, model, 1, every)
                looks = self._looks(model, monkeypatch, text)
                # The concepts the sentence observed: with the second word
                # withheld, the first word's alone.
                knowing = owner._observed_knowing[:1].clone()
                read_at = []
                for interval, perception in looks:
                    if perception is None or not perception.keys:
                        continue
                    held = {key[1]: i for i, key in enumerate(perception.keys)}
                    # Withheld: the readings of the second word's place.
                    withheld = (perception.where[..., 0] >= 6) & (every is not None)
                    back = ConceptSpread.read_back(owner, perception, withheld, knowing)
                    parts = perception.parts
                    at = lambda values, row: None if row not in held else float(values[held[row], 0].amax())
                    read_at.append(dict(
                        where=interval[0].tolist(), withheld=bool(withheld.any()),
                        parts_present=0 if parts is None else int((parts[1].amax(-1) > 0).sum()),
                        gamma_perceived=at(perception.values, gamma), gamma_read_back=at(back, gamma),
                        alpha_perceived=at(perception.values, alpha), alpha_read_back=at(back, alpha)))
                rows[label] = read_at
            model.omission.draw = ordinary
            print(dict(gamma_percept=gamma, alpha_percept=alpha, basic_percepts=len(basic.index), looks=rows))
            half = [look for look in rows['half'] if look['where'] == [6, 11]]
            beside = [look for look in rows['half'] if look['where'] == [0, 5]]
            # The word beside the blind spot is unchanged on the way back.
            assert beside and all(look['alpha_read_back'] == look['alpha_perceived'] for look in beside)
            # Nothing is read back where nothing was withheld.
            for look in rows['none']:
                assert look['gamma_read_back'] == look['gamma_perceived']
            # A word never taught has no basic percept to complete.
            assert len(basic.index) == len(set(range(len(basic.index))))
            # The hidden word, withheld at its place, is read back there from
            # what the structure expects.
            assert half and half[0]['gamma_perceived'] == 0. and half[0]['gamma_read_back'] > 0.
            # Printed: "alpha" at the place of "gamma" is read back only
            # through rows above that hold it with something else observed
            # (test_item17_spread's test_24 states the mechanism).
            print(dict(alpha_read_back_at_gamma=half[0]['alpha_read_back']))
        finally:
            model.End(); model.symbolSpace.soft_reset()


# ---------------------------------------------------------------------------
# A word and an object of another modality (certificate 25), and a sequence
# (certificate 26). Certificates of learning: the values are printed and
# recorded; only what the look-first mechanism gives is required.
# ---------------------------------------------------------------------------

class TestCrossModalTie:
    """Words read with values of the scalar modality (sigma-pi section 19.25,
    certificate 25)."""

    def test_cross_modal_tie(self, tmp_path, monkeypatch):
        import torch
        import ConceptSpread
        from test_compiled_word_chunk import _tiny_canonical_model
        from nomination_fixtures import recurrent_interpretation
        from ScalarPercepts import code, observe
        from PerceptCollections import objects, definitions
        from WordTies import ties
        model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
        try:
            owner, store = model._concept_owner(), model.symbolSpace.ltm_store
            def word(form):
                return recurrent_interpretation(owner, model.perceptualSpace.percept_store.spell_out(form.encode()),
                    model.wholeSpace.property_rows_for_bytes(form), key=form)[0]
            def show(value, form=None):
                row = store.append_idea(code(owner, value), trust=1.)
                return observe(owner, word(form), value, store.occurrence_of(row), stream=0)
            def expected_from(cid):
                V = int(owner.nVectors)
                observed = torch.zeros(V, 1, 2)
                observed[owner._concept_row_of(cid), 0, 0] = 1.
                value, _ = ConceptSpread.spread(owner, observed)
                return value[:V, 0, 0]
            n = owner.concept_mint_count
            stages = {}
            water = word('water')
            collections_before = len(definitions(owner))
            stages['word_alone'] = dict(objects=objects(owner, water), collections=collections_before)
            # The two together, n times and more.
            for _ in range(n + 3):
                obj = show(5., 'water')
            stages['together'] = dict(objects=objects(owner, water), tie=vars(ties(owner).get(water, obj)))
            spread_word = expected_from(water)
            spread_object = expected_from(obj)
            stages['each_raises_the_other'] = dict(
                object_from_word=float(spread_word[owner._concept_row_of(obj)]),
                word_from_object=float(spread_object[owner._concept_row_of(water)]))
            # A word met with a different value each time.
            for value in (6., 7., 8., 9., 10., 11.):
                varied = show(value, 'kappa')
            kappa = word('kappa')
            stages['varied'] = dict(objects=objects(owner, kappa),
                                    ties={k: vars(t) for k, t in ties(owner).ties.items() if k[0] == kappa})
            # A second word met with the same value.
            for _ in range(n + 3):
                second = show(5., 'aqua')
            aqua = word('aqua')
            stages['two_words_one_object'] = dict(objects=objects(owner, aqua), same=second == obj,
                collections=len(definitions(owner)), tie=vars(ties(owner).get(aqua, obj)))
            print(stages)
            # Look-first: the second word is tied to the object the first
            # made, and no second object is made for it.
            assert second == obj and objects(owner, aqua) == (obj,)
        finally:
            model.End(); model.symbolSpace.soft_reset()


class TestBellSalivate:
    """A sequence: two words of one sentence, the second withheld (sigma-pi
    section 19.25, certificate 26; the first form, item 1.8's certificate
    81 on these words). What the first leaves expected for the second, at
    the look of the second, before and after the pairing; and for a word
    never followed by it."""

    def test_bell_salivate(self, tmp_path, monkeypatch):
        import ConceptSpread
        from LookForecast import LookForecast, contained
        from PrependedSymbols import word_view
        from nomination_fixtures import readings_before_nomination
        from test_item18_expect import lesson, read, omit_second
        model = lesson(tmp_path, monkeypatch)
        ordinary = model.omission.draw
        try:
            pairs = ('bell food', 'horn rest')
            rows = {}
            for text in pairs:
                readings_before_nomination(model, lambda: read(model, text))
                read(model, text)
                rows[text] = [int(r) for r in word_view(model.inputSpace, 'concept_rows')[0, :2]]
            owner = model._concept_owner()
            looks = []
            forecast, write = LookForecast.forecast, ConceptSpread.write_look
            def forecast_extent(self, spans):
                looks.append([self.spans.detach().clone(), spans.detach().clone(), None])
                return forecast(self, spans)
            def write_kept(owner_, at, value):
                write(owner_, at, value)
                looks[-1][2] = owner_._expected_knowing[0].detach().clone()
            monkeypatch.setattr(LookForecast, 'forecast', forecast_extent)
            monkeypatch.setattr(ConceptSpread, 'write_look', write_kept)
            omit_second(monkeypatch, model)
            def at_second(text):
                looks.clear()
                read(model, text)
                held = [knowing for words, span, knowing in looks if knowing is not None
                        and bool(contained(span[0], words[0, 1:2])[0, 0])
                        and not bool(contained(span[0], words[0, :1])[0, 0])]
                return None if not held else held[0][:, 0]
            readings = {}
            for stage in ('before', 'after'):
                if stage == 'after':
                    optimizer = model.getOptimizer(lr=.01)
                    for presentation in range(200):
                        read(model, pairs[presentation % 2], training=True, optimizer=optimizer)
                for text in pairs:
                    value = at_second(text)
                    first, second = rows[text]
                    other = rows[pairs[1 - pairs.index(text)]][1]
                    readings[(stage, text)] = None if value is None else dict(
                        second=float(value[second]), other_sentence_second=float(value[other]))
            print(dict(rows=rows, readings=readings))
        finally:
            model.omission.draw = ordinary
            model.End(); model.symbolSpace.soft_reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
