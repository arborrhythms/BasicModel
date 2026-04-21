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

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from embed import WordVectors

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT = os.path.join(_PROJECT, "output", "embeddings", "sentence.pt")
_CONFIG = os.path.join(_PROJECT, "data", "BasicModel.xml")
_VENV_PYTHON = os.path.join(_PROJECT, ".venv", "bin", "python")

SERVER_URL = "http://127.0.0.1:8001"


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

# Check that getBatch produces tensors on the right device
batch, _ = model.inputSpace.getBatch(0, batchSize=2)
inp_tensor, out_tensor = batch
inp_device = str(inp_tensor.device)

with torch.no_grad():
    # getBatch() returns raw input for all modes; masking (if any) is applied
    # inside InputSpace.forward(), so we always call model.forward().
    _, _, out, _ = model.forward(inp_tensor)
    out_device = str(out.device)

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
        self._validate(os.path.join(_DATA_DIR, "XOR_exact.xml"))

    def test_xor_spaces_xml(self):
        self._validate(os.path.join(_DATA_DIR, "XOR_spaces.xml"))

    def test_xor_recon_xml(self):
        self._validate(os.path.join(_DATA_DIR, "XOR_recon.xml"))

    def test_xor_pos_xml(self):
        self._validate(os.path.join(_DATA_DIR, "XOR_pos.xml"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
