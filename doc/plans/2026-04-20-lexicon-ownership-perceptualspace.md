# Lexicon Ownership Move: InputSpace → PerceptualSpace

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PerceptualSpace` the home of the token lexicon (the `Embedding` Basis, its `.kv` persistence, its training methods, and the text-mode forward/reverse pipeline) so that words are uniformly treated as a kind of percept.

**Architecture:** Today the `Embedding` Basis is built by `InputSpace._build_what_basis()` (Spaces.py:3861) and stored on `inputSpace.subspace.what`. Every consumer (`Model._get_embedding`, `save_embeddings`, optimizer wiring, `OutputSpace(... vectors=...)`, `Model.reconstruct_*`) reads it through `self.inputSpace.vocabulary`. We move the Basis to `perceptualSpace.subspace.what`, move the text-mode `forward`/`reverse`, the SBOW/CBOW training helpers, the `doc_spans`/`doc_sources`/`embedding_path` accounting, and the `reconstruct_data` / `reconstruct_to_buffer` / `get_recovered_word` accessors to PerceptualSpace. Non-text model_types (`vq`, `passthrough`, `simple`) keep their Basis on InputSpace unchanged. After the move, `inputSpace.vocabulary` returns `None` for text models and `InputSpace.forward` becomes a thin pass-through that just stages raw input for PerceptualSpace.

**Tech Stack:** Python 3.12 + PyTorch (run via `basicmodel/.venv/bin/python`), `unittest`, `TheXMLConfig` (XML config), `Embedding`/`Codebook`/`Tensor` Basis hierarchy, `SubSpace` muxed event encoding.

**Out of scope:**
- Reshaping the XML config schema (the `<lexer>`, `<embeddingPath>`, `<minFrequency>` keys stay where they are; PerceptualSpace just reads them).
- Changing `whereEncoding`/`whenEncoding` semantics or moving positional-encoding ownership.
- Touching Codebook/Tensor (non-text) Basis paths beyond a smoke check that they still work.
- Any change to BPE/grammar lexer modes beyond what falls out of the move.

**Conventions for the engineer:**
- Run all tests with `basicmodel/.venv/bin/python -m unittest …` (per project memory `reference_python_env`). Never run `make train`.
- This user manages their own commits. The "Commit" steps below are checkpoints — pause and let the user commit (or skip, if they want to bundle).
- File line numbers are accurate as of `2026-04-20`. If a step's line number is off by a few because of preceding edits, search for the surrounding code rather than trusting the number.
- Search the codebase before editing: `inputSpace.vocabulary` and `subspace.what` appear in many files; this plan enumerates the production sites but tests in `basicmodel/test/` may have additional uses.

---

## File Map

**Modify:**
- `basicmodel/bin/Spaces.py` — relocate `_build_what_basis` Embedding branch, text-mode `forward`/`reverse`, `train_embeddings`/`sbow_loss`, `_snapshot_embeddings`, `set_embedding_sigma`, `reconstruct_data`/`reconstruct_to_buffer`/`get_recovered_word`, and the `doc_spans`/`doc_sources`/`embedding_path` attributes from `InputSpace` → `PerceptualSpace`.
- `basicmodel/bin/Models.py` — repoint `_get_embedding` (line 409), `save_embeddings` (814), `load_embeddings` (863), `_restore_vocab` callsite (885), optimizer params (211–217, 259–263), `OutputSpace(..., vectors=…)` (1347, 2683), the lm_labels prep block (1331–1336), and the reconstruction accessor calls (1049–1093).
- `basicmodel/test/test_basicmodel.py` — update assertions at lines 2546, 2547, 2556, 2557, 2566, 2825 plus any other `inputSpace.vocabulary` references.
- `basicmodel/test/test_xor_spaces.py` — update lines 176, 184.

**Create:**
- `basicmodel/test/test_lexicon_ownership.py` — pinning tests that lock in the new contract (perceptualSpace owns the Embedding, persistence round-trips through perceptualSpace, optimizer collects perceptualSpace embedding params, OutputSpace text-mode reads perceptualSpace).

**Read-only (search to confirm no other callers):**
- `basicmodel/bin/data.py`, `basicmodel/bin/Layers.py`, `basicmodel/bin/Language.py`, `basicmodel/bin/embed.py`, `basicmodel/bin/lex.py` — grep for `inputSpace.vocabulary` and resolve any stragglers in the relevant tasks.

---

## Task 1: Pinning tests for the new contract

**Files:**
- Create: `basicmodel/test/test_lexicon_ownership.py`

- [ ] **Step 1: Write the failing tests**

The existing canonical text-mode fixture lives in `test_basicmodel.py:2516-2541` (the `_create_model(train_embeddings)` helper inside `TestTrainEmbeddingsFlag`). It loads `basicmodel/data/XOR_exact.xml` (which has `<modelType>embedding</modelType>` per line 6) and applies the requested `trainEmbeddings` flag. Reuse that helper directly via copy — do not invent a new XML config.

```python
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
        self.assertIsNone(m.inputSpace.vocabulary)

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


if __name__ == "__main__":
    unittest.main()
```

Before continuing, open `basicmodel/test/test_basicmodel.py` near line 2540 and paste the actual fixture body into `_build_text_model()`. Do not invent a config — reuse what exists.

- [ ] **Step 2: Run the new tests to verify they all fail**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_lexicon_ownership -v
```

Expected: every test fails. The first three should fail because `perceptualSpace.vocabulary` is currently `None` (`_build_what_basis` returns `None` on the base `Space`). The last two should fail or error for the same reason.

If any test *passes* before any production code changes, the fixture is wrong — debug the fixture before moving on. (A green test pre-implementation is a false positive that defeats the purpose of TDD here.)

- [ ] **Step 3: Commit (checkpoint)**

```
git add basicmodel/test/test_lexicon_ownership.py
git diff --stat HEAD
```

Pause for the user to commit. Suggested message:

```
test: pin lexicon-ownership contract on PerceptualSpace (failing)
```

---

## Task 2: Build the Embedding on PerceptualSpace

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — add `PerceptualSpace._build_what_basis` (insert near line 4663, before `_register_requirements`).

- [ ] **Step 1: Read the current Embedding-construction block**

Open `basicmodel/bin/Spaces.py:3861-3876` and copy the `if self.model_type == "embedding":` branch — that is the body to relocate. Note the inputs it depends on: `self.inputShape[0]`, `self.outputShape[0]`, `self.nDim`, `self.embedding_path`, `self.embedding_source`, `self.min_frequency`, `self.neg_samples`, `self.byte_mode`, `self.ergodic`.

PerceptualSpace currently does not store any of those, so it needs to read them from config (or from `architecture` for `embeddingPath`) at `__init__` time.

- [ ] **Step 2: Read the relevant config in PerceptualSpace.__init__**

Modify `PerceptualSpace.__init__` (Spaces.py:4636) to read and stash the Embedding-construction inputs *before* calling `super().__init__(...)` (because `_build_what_basis` runs inside `Space.__init__`). Mirror the InputSpace block at Spaces.py:3905-3922. Add fields to `self`:

```python
# Inside PerceptualSpace.__init__, BEFORE super().__init__():
self.model_type = TheXMLConfig.get("architecture.modelType")
self.embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
# lexer/min_frequency/neg_samples currently live under <InputSpace> in XML.
# Keep reading them from there to avoid a config-schema change.
self.lexer = TheXMLConfig.space("InputSpace", "lexer")
self.byte_mode = (self.lexer == "byte")
self.min_frequency = float(TheXMLConfig.data_param("minFrequency", 0.0))
self.neg_samples = int(TheXMLConfig.training("negSamples", 64))
self.embedding_source = TheData.train_input if TheData.train_input else None
```

(Per memory `feedback_no_defensive_getattr`: initialize these unconditionally in `__init__`, do not paper over with `getattr`.)

- [ ] **Step 3: Add the override**

Add this method on `PerceptualSpace` (place it right after `_register_requirements`, around Spaces.py:4685):

```python
def _build_what_basis(self):
    """Lexicon home: build the Embedding when running in text mode."""
    if self.model_type != "embedding":
        return None
    basis = Embedding()
    basis.ergodic = self.ergodic
    basis.create(
        self.inputShape[0],
        self.outputShape[0],
        self.nDim,
        embedding_path=self.embedding_path,
        source=self.embedding_source,
        min_frequency=self.min_frequency,
        neg_samples=self.neg_samples,
        byte_mode=self.byte_mode,
    )
    return basis
```

- [ ] **Step 4: Run the first pinning test**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_lexicon_ownership.TestLexiconOwnership.test_embedding_lives_on_perceptual_space -v
```

Expected: the `assertIsInstance(m.perceptualSpace.vocabulary, Embedding)` half passes; the `assertIsNone(m.inputSpace.vocabulary)` half **still fails** because InputSpace also still constructs an Embedding. That is the next task.

- [ ] **Step 5: Commit (checkpoint)**

```
git add basicmodel/bin/Spaces.py
```

Suggested message: `feat(spaces): build Embedding on PerceptualSpace (still dual-owned)`.

---

## Task 3: Stop building the Embedding on InputSpace

**Files:**
- Modify: `basicmodel/bin/Spaces.py:3861-3876` — remove the `if self.model_type == "embedding":` branch from `InputSpace._build_what_basis`.

- [ ] **Step 1: Strip the embedding branch**

Edit `InputSpace._build_what_basis` to drop the embedding case. The remaining body should look like:

```python
def _build_what_basis(self):
    """InputSpace .what holds non-lexical bases (Codebook/Tensor only).

    The Embedding (lexicon) is owned by PerceptualSpace -- see
    PerceptualSpace._build_what_basis.
    """
    if self.model_type == "embedding":
        return None  # owned by PerceptualSpace.subspace.what
    if self.model_type == "vq":
        basis = Codebook()
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
            customVQ=self.customVQ,
            passThrough=False,
        )
        return basis
    if self.model_type in ("passthrough", "simple"):
        basis = Tensor()
        basis.create(
            self.inputShape[0],
            self.outputShape[0],
            self.nDim,
            passThrough=True,
        )
        return basis
    raise RuntimeError("Unexpected model_type")
```

- [ ] **Step 2: Strip the doc_spans/whereEncoding adjustments that depend on the Embedding being on InputSpace**

In `InputSpace.__init__` at Spaces.py:3934-3956, the block `lexical_basis = self.subspace.what` and the `if isinstance(lexical_basis, Embedding):` adjustments to `doc_spans`/`doc_sources`/`whereEncoding.maxVal` no longer apply — `self.subspace.what` is `None` in text mode now. Cut the block. (We will replicate the equivalent adjustment on PerceptualSpace in Task 4.)

Replace lines 3934-3956 with:

```python
self.doc_spans = []
self.doc_sources = []
```

(They are still referenced by `InputSpace.forward`'s non-text path indirectly via `self.subspace.what`. They will become dead in InputSpace once Task 5 lands. Leave the empty defaults for now so attribute access does not break.)

- [ ] **Step 3: Run the embedding-ownership tests**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_lexicon_ownership.TestLexiconOwnership.test_embedding_lives_on_perceptual_space -v
```

Expected: PASS. Both halves now hold.

- [ ] **Step 4: Run the existing non-text suite to confirm Codebook/Tensor paths still work**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_xor_spaces -v
```

Expected: all tests pass (xor uses Codebook/Tensor, which were untouched).

If any text-mode tests in `test_basicmodel.py` are run at this point they will fail — that is expected; subsequent tasks fix them.

- [ ] **Step 5: Commit (checkpoint)**

Suggested message: `refactor(inputspace): drop Embedding construction; perceptualSpace owns it`.

---

## Task 4: Mirror Embedding-dependent setup on PerceptualSpace

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — extend `PerceptualSpace.__init__` to take over the `doc_spans`/`doc_sources`/`whereEncoding.maxVal` setup that previously lived on InputSpace at lines 3934-3953.

- [ ] **Step 1: Append the post-super setup to PerceptualSpace.__init__**

After `super().__init__(inputShape, spaceShape, outputShape)` in `PerceptualSpace.__init__` (around Spaces.py:4644), insert:

```python
lexical_basis = self.subspace.what
if isinstance(lexical_basis, Embedding):
    self.doc_spans = lexical_basis.doc_spans
    self.doc_sources = lexical_basis.doc_sources
    data = TheData
    if data.train_input and self.subspace.whereEncoding.nDim > 0:
        if (isinstance(data.train_input, list) and data.train_input
                and isinstance(data.train_input[0], str)):
            actual_max = max(len(s.encode('utf-8'))
                             for s in data.train_input)
            maxP = max(self.subspace.whereEncoding.maxVal,
                       actual_max * 2)
        else:
            maxP = max(self.subspace.whereEncoding.maxVal,
                       data.inputLength)
        self.subspace.whereEncoding.maxVal = maxP
        self.subspace.whereEncoding.div_term = 2 * math.pi / maxP
else:
    self.doc_spans = []
    self.doc_sources = []
```

(`math` and `TheData` are already imported at the top of Spaces.py — verify by `grep -n "^import math\|^from .* import.*TheData" basicmodel/bin/Spaces.py`. If not, add the imports.)

- [ ] **Step 2: Run the pinning tests for ownership**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_lexicon_ownership.TestLexiconOwnership.test_embedding_lives_on_perceptual_space -v
```

Expected: PASS.

- [ ] **Step 3: Commit (checkpoint)**

Suggested message: `refactor(perceptualspace): take over doc_spans + whereEncoding sizing`.

---

## Task 5: Move text-path forward into PerceptualSpace

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — split `InputSpace.forward` (Spaces.py:4053-4128) into a thin pass-through; absorb the embedding/lex/positional logic into `PerceptualSpace.forward` (Spaces.py:4738).

This is the largest single change in the plan. Read both methods end-to-end before editing.

- [ ] **Step 1: Add a text-mode branch to PerceptualSpace.forward**

`PerceptualSpace.forward` currently assumes its input is the upstream `vspace` (a `SubSpace`) and calls `forwardBegin` / attention / VQ / sparsity / `forwardEnd`. We need it to additionally handle the text-mode case where the upstream `inputSpace.subspace` carries raw byte indices and the lexicon lookup happens *here*.

At the top of `PerceptualSpace.forward`, before the existing `if self.passThrough: return vspace`, insert a text-mode branch:

```python
def forward(self, vspace, wordSpace=None, quantize=True):
    """Perception: map input vectors to percepts via attention + VQ + chunking.

    When this PerceptualSpace owns the lexicon (text mode), the upstream
    InputSpace passes raw input through; the lex + embedding lookup +
    positional encoding happens here, then we materialize a percept
    SubSpace and run the standard PiLayer/attention/VQ pipeline.
    """
    if isinstance(self.subspace.what, Embedding):
        vspace = self._lex_and_embed(vspace)
    if self.passThrough:
        return vspace
    # ... existing body unchanged from here ...
```

- [ ] **Step 2: Add the `_lex_and_embed` helper**

Port the text-path body from `InputSpace.forward` (Spaces.py:4087-4126). Place it as a method on `PerceptualSpace`:

```python
def _lex_and_embed(self, upstream_vspace):
    """Run vocabulary.forward on the upstream raw input, populate this
    space's subspace with what/where/when indices, and materialize.
    """
    raw_input = upstream_vspace._raw_input  # InputSpace stages this
    batch = raw_input.shape[0]
    nObj = self.outputShape[0]
    vocab = self.subspace.what
    dev = TheDevice.get()

    self.subspace.whereEncoding.p = 0
    what, meta = vocab.forward(raw_input, return_meta=True)
    self._forward_input = meta
    self._last_tokens = [
        [tok for tok, _ in row] for row in meta.get('tokens', [])
    ]
    what_indices = meta['indices']
    if self.nWhere > 0:
        where_indices = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
        for b, batch_tokens in enumerate(meta['tokens']):
            for i, (_, start) in enumerate(batch_tokens):
                where_indices[b, i] = start
            final_offset = meta['final_offsets'][b]
            for i in range(len(batch_tokens), nObj):
                where_indices[b, i] = final_offset + (i - len(batch_tokens))
    else:
        where_indices = None
    if self.nWhen > 0:
        when_indices = torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1)
    else:
        when_indices = None
    self.subspace.set_forward_content(what_indices, where_indices, when_indices)
    self.subspace.normalize("input", target="what", normalize=True)
    return self.subspace
```

(The reference to `upstream_vspace._raw_input` is the contract InputSpace will publish in Step 3.)

- [ ] **Step 3: Slim InputSpace.forward to a pass-through that stages raw input**

Replace `InputSpace.forward` (Spaces.py:4053-4128) with:

```python
def forward(self, input, mask=None):
    # ARIR-inference cache bypass (unchanged).
    if self._cached_embedding is not None:
        self.input = self._cached_embedding
        self._cached_embedding = None
        self._forward_input = None
        self.subspace.set_event(self.input)
        return self.subspace

    self.subspace.whereEncoding.p = 0
    vocab = self.subspace.what
    if vocab is None:
        # Text mode -- PerceptualSpace owns the lexicon and will lex+embed.
        # Stage the raw input on the subspace for the downstream space to read.
        self.subspace._raw_input = input
        return self.subspace

    # Non-text path retained: vocab is Codebook/Tensor.
    batch = input.shape[0]
    nObj = self.outputShape[0]
    dev = TheDevice.get()
    assert list(input.shape) == [batch, self.inputShape[0], self.inputShape[1]]
    what = vocab.forward(input)
    self._forward_input = None
    self.subspace.set_what(what)
    if self.nWhere > 0:
        positions = torch.arange(nObj, dtype=torch.float32, device=dev).unsqueeze(0).expand(batch, -1)
        self.subspace.set_where(self.subspace.whereEncoding.encode(positions))
    if self.nWhen > 0:
        timesteps = torch.arange(nObj, dtype=torch.float32, device=dev).unsqueeze(0).expand(batch, -1)
        self.subspace.set_when(self.subspace.whenEncoding.encode(timesteps))
    self.input = self.subspace.materialize()
    self.subspace.normalize("input", target="what", normalize=True)
    self.input = self.subspace.materialize()
    return self.subspace
```

- [ ] **Step 4: Run the smallest text-mode integration test**

Pick a small text-mode test from `test_basicmodel.py` — search for `model_type="embedding"` and find one that exercises `model.Start(...)` or `model.run(...)`. Run it:

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_basicmodel.<TextModeTestClass>.<smallest_method> -v
```

Expected: PASS. If it fails on a missing attribute (`_raw_input`, `doc_spans`, etc.), debug the bridge between InputSpace and PerceptualSpace before moving on. Do *not* paper over with `getattr` (per memory `feedback_no_defensive_getattr`).

- [ ] **Step 5: Commit (checkpoint)**

Suggested message: `refactor(spaces): move text-mode forward (lex+embed) to PerceptualSpace`.

---

## Task 6: Move text-path reverse and reconstruction accessors

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — port `InputSpace.reverse` (Spaces.py:4260-4296) and the three `reconstruct_data`/`reconstruct_to_buffer`/`get_recovered_word` accessors (Spaces.py:4298-4320) to `PerceptualSpace`.

- [ ] **Step 1: Add a text-mode reverse path to PerceptualSpace.reverse**

`PerceptualSpace.reverse` currently does the inverse of its forward. Extend it so that, when this space owns the lexicon, it also produces the recovered text state:

```python
def reverse(self, vspace, wordSpace=None):
    if self.passThrough and not isinstance(self.subspace.what, Embedding):
        return vspace
    # ... existing reverse body unchanged ...

    if isinstance(self.subspace.what, Embedding):
        self._reverse_text(vspace)
    return vspace
```

Then add `_reverse_text`, ported from `InputSpace.reverse` lines 4263-4296, but operating on `self` (PerceptualSpace) instead of InputSpace. The body is mostly mechanical: rename `self.subspace.what` references etc. to PerceptualSpace's own subspace.

- [ ] **Step 2: Strip the text-mode body from InputSpace.reverse**

`InputSpace.reverse` becomes a no-op when `self.subspace.what is None` (text mode). For non-text Codebook/Tensor cases, keep the existing body. Wrap the existing body in a guard:

```python
def reverse(self, vspace):
    if self.subspace.what is None:
        # Text mode: PerceptualSpace already produced the recovered text state.
        return self.subspace
    # ... existing body unchanged ...
```

- [ ] **Step 3: Move reconstruction accessors**

Move `reconstruct_data`, `reconstruct_to_buffer`, and `get_recovered_word` from `InputSpace` (Spaces.py:4298-4320) to `PerceptualSpace`. They reference `self._recovered_input` and `self.subspace.vocabulary`, both of which now live on PerceptualSpace.

Delete the InputSpace copies. Do not leave forwarding shims.

- [ ] **Step 4: Update Model-level callers in Models.py**

Lines 1049-1093 currently call `self.inputSpace.get_recovered_word(...)`, `self.inputSpace.reconstruct_data(...)`, `self.inputSpace.reconstruct_to_buffer(...)`. Change all three to `self.perceptualSpace.…`.

- [ ] **Step 5: Run a text-mode reconstruction test**

Find the test in `test_basicmodel.py` that exercises `reconstruct_data` or `reconstruct_to_buffer` (search for those symbols). Run it:

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_basicmodel.<TextReverseTestClass>.<method> -v
```

Expected: PASS.

- [ ] **Step 6: Commit (checkpoint)**

Suggested message: `refactor(spaces): move text-mode reverse + reconstruct accessors to PerceptualSpace`.

---

## Task 7: Move SBOW/CBOW training helpers and embedding accessors

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — move `train_embeddings` (4326), `sbow_loss` (4333), `_snapshot_embeddings` (4340), `set_embedding_sigma` (4347) from `InputSpace` to `PerceptualSpace`.
- Modify: `basicmodel/bin/Models.py:1855-1875` — change `trainEmbeddings` to call `self.perceptualSpace.sbow_loss(...)` / `self.perceptualSpace.train_embeddings(...)`.

- [ ] **Step 1: Move the four methods**

Cut the four methods from `InputSpace` (Spaces.py:4326-4360 — verify the exact range) and paste them as-is onto `PerceptualSpace`. They reference `self.subspace.vocabulary`, which now resolves correctly on PerceptualSpace.

- [ ] **Step 2: Update Models.trainEmbeddings**

In `Models.py:1855-1875`, replace `self.inputSpace.vocabulary`, `self.inputSpace.sbow_loss(words)`, and `self.inputSpace.train_embeddings(words, method=method)` with the perceptualSpace equivalents. The byte-mode skip check (`self.lexer in ('byte', 'bytes')`) does not change.

- [ ] **Step 3: Run a CBOW/SBOW training test**

Find the relevant test (search `test_basicmodel.py` for `train_embeddings` or `sbow_loss`). Run it.

Expected: PASS.

- [ ] **Step 4: Commit (checkpoint)**

Suggested message: `refactor(spaces): move SBOW/CBOW helpers to PerceptualSpace`.

---

## Task 8: Repoint Model persistence and OutputSpace wiring

**Files:**
- Modify: `basicmodel/bin/Models.py` — update `_get_embedding` (409), `save_embeddings` (814 — only the path-defaulting reference), `load_embeddings` (863), the optimizer wiring (211-217 and 259-263), the `OutputSpace(... vectors=…)` call sites (1347 and 2683), and the lm_labels prep block (1331-1336).

- [ ] **Step 1: Repoint `_get_embedding`**

Edit Models.py:409-413:

```python
def _get_embedding(self):
    """Return the Embedding instance if this model uses one, else None."""
    if hasattr(self, 'perceptualSpace') and isinstance(self.perceptualSpace.vocabulary, Embedding):
        return self.perceptualSpace.vocabulary
    return None
```

`save_embeddings` and `load_embeddings` already route through `self._get_embedding()` for the runtime object — no change needed beyond this.

- [ ] **Step 2: Repoint optimizer wiring (creation site)**

Edit Models.py:211-217. Replace every `self.inputSpace.vocabulary` with `self.perceptualSpace.vocabulary`, and re-attach the embedding params to `self.perceptualSpace.params` instead of `self.inputSpace.params`:

```python
if self.optimize_embedding and isinstance(self.perceptualSpace.vocabulary, Embedding):
    emb_params = self.perceptualSpace.vocabulary.embedding_parameters()
    self.perceptualSpace.params = self.perceptualSpace.params + emb_params
self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
if isinstance(self.perceptualSpace.vocabulary, Embedding):
    self.perceptualSpace.vocabulary.optimize_embedding = self.optimize_embedding
    object.__setattr__(self.perceptualSpace.vocabulary, "_model", self)
```

- [ ] **Step 3: Repoint optimizer-exclude path**

Edit Models.py:259-263 (inside `getOptimizer`):

```python
if hasattr(self, 'perceptualSpace') and isinstance(self.perceptualSpace.vocabulary, Embedding):
    for p in self.perceptualSpace.vocabulary.embedding_parameters():
        exclude.add(p.data_ptr())
```

- [ ] **Step 4: Reorder model construction so PerceptualSpace exists before lm_labels prep**

Edit Models.py:1327-1347. The current order is:

```
inputSpace = self._make_input_space(...)
[ if data._lm_labels: data.prepare_lm_targets(self.inputSpace.vocabulary) ]
perceptualSpace = self._make_perceptual_space(...)
... 
outputSpace = OutputSpace(..., vectors=self.inputSpace.vocabulary)
```

Change to:

```python
self.inputSpace      = self._make_input_space(rawInputShape, spaceShape_input, inputShape,
                                              model_type=model_type)
self.perceptualSpace = self._make_perceptual_space(inputShape, spaceShape_percept, perceptShape)
if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
    embedding = self.perceptualSpace.vocabulary
    if embedding is not None and hasattr(embedding, 'pretrain'):
        data.prepare_lm_targets(embedding)
        data.toDevice()
self.conceptualSpace = ConceptualSpace(perceptShape, spaceShape_concept, conceptShape)
self.symbolicSpace   = SymbolicSpace(conceptShape, spaceShape_symbol, symbolShape,
                                     conceptualSpace=self.conceptualSpace)
self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])
```

- [ ] **Step 5: Repoint OutputSpace `vectors=` at both call sites**

- Models.py:1345-1347: `vectors=self.perceptualSpace.vocabulary`
- Models.py:2681-2683: `vectors=self.perceptualSpace.vocabulary`

- [ ] **Step 6: Run the persistence + OutputSpace pinning tests**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_lexicon_ownership -v
```

Expected: all five tests PASS.

- [ ] **Step 7: Commit (checkpoint)**

Suggested message: `refactor(models): point persistence + optimizer + OutputSpace at perceptualSpace.vocabulary`.

---

## Task 9: Update existing tests

**Files:**
- Modify: `basicmodel/test/test_basicmodel.py` (lines 2546, 2547, 2556, 2557, 2566, 2825 plus any others surfaced by grep).
- Modify: `basicmodel/test/test_xor_spaces.py` (lines 176, 184).

- [ ] **Step 1: Inventory remaining usages**

```
basicmodel/.venv/bin/python -m grep ...   # OR: rely on the grep below
```

Use the project's Grep tool to enumerate `inputSpace.vocabulary` references across `basicmodel/test/`. There may be more than the lines listed above; treat the grep output as authoritative.

- [ ] **Step 2: Flip the assertions**

For each text-mode test, change `m.inputSpace.vocabulary` → `m.perceptualSpace.vocabulary`. For non-text tests (xor), confirm whether the test cares about the *model_type* — if it asserts the basis under model_type=`vq`/`passthrough`, leave it on InputSpace; if it merely wants the codebook reference and is text-mode, move it.

`test_xor_spaces.py:176` and `:184` — read the surrounding context. If the model under test is xor (`vq` mode), the assertion stays on inputSpace. If it is text mode, move it. Do not blindly substitute.

- [ ] **Step 3: Run both test files end-to-end**

```
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_basicmodel -v
basicmodel/.venv/bin/python -m unittest basicmodel.test.test_xor_spaces -v
```

Expected: all PASS.

- [ ] **Step 4: Commit (checkpoint)**

Suggested message: `test: point lexicon assertions at perceptualSpace.vocabulary`.

---

## Task 10: Audit non-test consumers and full-suite verification

**Files:**
- Read: `basicmodel/bin/Layers.py`, `basicmodel/bin/Language.py`, `basicmodel/bin/data.py`, `basicmodel/bin/embed.py`, `basicmodel/bin/lex.py`.
- Modify: any of the above whose grep hit on `inputSpace.vocabulary` is genuinely a *consumer* (not just a comment / docstring).

- [ ] **Step 1: Grep for stragglers**

Use the Grep tool against `basicmodel/bin/` for `inputSpace.vocabulary`. For each hit:
- If it is a comment or docstring describing the *old* design, update the prose (or delete if it is no longer load-bearing — see project conventions on docstrings).
- If it is a runtime reference, retarget to `perceptualSpace.vocabulary` and add the corresponding test coverage to `test_lexicon_ownership.py` if missing.

- [ ] **Step 2: Run the full unit-test suite**

```
basicmodel/.venv/bin/python -m unittest discover -s basicmodel/test -v
```

Expected: every test PASSes. If something fails that did not fail before this plan, debug — the plan covers the known consumer set, but a stray reference might have been missed.

- [ ] **Step 3: Verify the docs**

Skim `basicmodel/doc/Spaces.md`, `basicmodel/doc/Training.md`, `basicmodel/doc/Language.md`. Update any prose that asserts InputSpace owns the lexicon (look for `inputSpace.vocabulary` in those `.md` files specifically). Keep edits minimal — fix the inaccurate sentence; do not restructure the doc.

- [ ] **Step 4: Confirm `.kv` and `.ckpt` round-trip with a fresh model**

Build a small text-mode model interactively (or via a one-shot script) and confirm:
1. `model.save_embeddings("/tmp/lex.kv")` writes a file.
2. `Models.print_weights_info("/tmp/model.ckpt")` does NOT list any `wv._vectors` keys (these are filtered in `save_weights` at Models.py:809-812; that filter still works because `wv._vectors` lives on the Embedding wherever it is stored — the filter is name-based, not path-based).
3. Loading both back into a new model reproduces the same `perceptualSpace.vocabulary` weights and metadata.

This is the practical confirmation that the user's `.kv` vs `.ckpt` separation is preserved.

- [ ] **Step 5: Commit (final checkpoint)**

Suggested message: `chore: complete lexicon-ownership move to PerceptualSpace`.

---

## Self-Review Checklist (run before handing off)

- [ ] Every `inputSpace.vocabulary` reference in `basicmodel/bin/` and `basicmodel/test/` is either gone, retargeted, or knowingly kept (Codebook/Tensor non-text cases).
- [ ] `_get_embedding`, `save_embeddings`, `load_embeddings`, `_restore_vocab` callsite, `OutputSpace(..., vectors=…)` (both call sites), and the optimizer wiring all read `perceptualSpace.vocabulary`.
- [ ] PerceptualSpace owns: the Embedding Basis, the text-path forward (`_lex_and_embed`), the text-path reverse, `train_embeddings`, `sbow_loss`, `_snapshot_embeddings`, `set_embedding_sigma`, `reconstruct_data`, `reconstruct_to_buffer`, `get_recovered_word`, `doc_spans`, `doc_sources`.
- [ ] InputSpace.forward is a thin pass-through in text mode; non-text Codebook/Tensor paths are unchanged.
- [ ] `test_lexicon_ownership.py` passes; `test_basicmodel.py` and `test_xor_spaces.py` pass; full discover-run is green.
- [ ] No `getattr` was added as defensive padding (per `feedback_no_defensive_getattr`).
- [ ] No XML config keys were renamed (we read `<InputSpace><lexer>` etc. from PerceptualSpace).

## Follow-up (out of scope; suggest as a separate plan if needed)

- Move `<lexer>`, `<minFrequency>`, etc. from `<InputSpace>` to a new `<PerceptualSpace>` sub-section in the XML schema, with a deprecation shim that reads the old keys.
- Decide whether `InputSpace` should exist at all once it has nothing to do in text mode, or whether the raw-input staging role (AR sliding buffer, `prepInput`, `_cached_embedding`) justifies keeping it.
- Audit `Language.py` / `WordSpace` for any analogous moves (truth layer, discourse predictor) that should also follow the lexicon to PerceptualSpace.
