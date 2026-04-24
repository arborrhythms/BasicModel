# PerceptualSpace BPE Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `<chunking>bpe</chunking>` end-to-end on `PerceptualSpace` and consolidate the XML config surface by removing `<chunkBPE>`, `<chunkTargetVocabSize>`, `raw`, and the vestigial 256-prototype table in `ChunkLayer`.

**Architecture:** `ChunkLayer` becomes a pure symbolic BPE tokenizer (vocab dict + merges list + `forward()` / `train_step()`). `PerceptualSpace.forward` gains a new `_embed_bpe` method that (1) whitespace-splits the upstream byte buffer into words, (2) BPE-tokenizes each word via `ChunkLayer.forward`, (3) inserts each byte-tuple chunk as a latin-1 key into the `Embedding` codebook (reusing the OOV insert path), (4) MAX-fuses sub-token embeddings within each word, and (5) writes one `[nDim]` word-vector per position into `self.subspace.event`. `nVectors` on `PerceptualSpace` is the single source of truth for BPE vocab size.

**Tech stack:** Python 3.12, PyTorch 2.x. Tests run with `basicmodel/.venv/bin/python -m pytest` from the `basicmodel/` directory. User manages git commits — do not `git add` / `git commit`; end-of-task "commit checkpoints" are advisory.

**Spec:** [doc/specs/2026-04-23-perceptualspace-bpe-chunking-design.md](../specs/2026-04-23-perceptualspace-bpe-chunking-design.md)

**Files touched.**
- Modify: `basicmodel/bin/Layers.py` — `ChunkLayer.__init__` (rename args, drop prototype params), remove `score_pair`/`encode`/`decode`/`should_merge`.
- Modify: `basicmodel/bin/Spaces.py` — `PerceptualSpace.__init__` (new config read), `PerceptualSpace._embed_bpe` (new method), `PerceptualSpace.forward` (dispatch update), `PerceptualSpace.chunk_static` (drop `raw`), `Embedding._token_stream` (drop `raw`).
- Modify: `basicmodel/data/model.xsd` — remove 3 elements, add 1, drop `raw` enum.
- Modify: `basicmodel/data/MM_5M.xml`, `basicmodel/data/MM_bpe.xml`, `basicmodel/data/model.xml`.
- Modify: `basicmodel/test/test_chunk_layer_bpe.py`, `basicmodel/test/test_perceptual_chunking.py`.
- Add: `basicmodel/test/test_perceptualspace_bpe_forward.py`.

---

## Phase 0: Baseline verification

### Task 0: Verify pre-change test baseline

**Files:** none modified.

- [ ] **Step 1: Run the BPE-related test suites and capture baseline**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py test/test_perceptual_chunking.py -v 2>&1 | tail -60
```

Expected: Record which tests PASS and which FAIL. `test_chunk_layer_bpe.py` is expected all-PASS. `test_perceptual_chunking.py` should also be all-PASS — if any test is red here, investigate before proceeding (the plan assumes a green baseline).

- [ ] **Step 2: Confirm no external callers of soon-removed `ChunkLayer` methods**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
grep -rnE "\.(score_pair|should_merge|encode|decode)\(" bin/ test/ 2>/dev/null | grep -vE "/.claude/|^Binary"
```

Expected: Only hits inside `bin/Layers.py` itself (the methods' own definitions and internal calls between them). If any external file calls these, **stop and widen the plan** — removal is not safe as specified.

- [ ] **Step 3: Confirm no external callers use `chunkBPE` / `chunkTargetVocabSize` / `chunkMinPairFrequency` outside this repo's XMLs and the single test**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
grep -rnE "chunkBPE|chunkTargetVocabSize|chunkMinPairFrequency" bin/ test/ data/ doc/ 2>/dev/null | grep -vE "/.claude/"
```

Expected: Hits only in `bin/Spaces.py:5107-5121`, `data/model.xml`, `data/MM_bpe.xml`, `data/MM_5M.xml`, `data/model.xsd`, `test/test_chunk_layer_bpe.py`, and `doc/` (informational). No surprise consumers.

---

## Phase 1: ChunkLayer surface updates (TDD)

### Task 1: Test for renamed ChunkLayer constructor args

**Files:**
- Test: `basicmodel/test/test_chunk_layer_bpe.py`

- [ ] **Step 1: Add a failing test for the new constructor signature**

Edit `basicmodel/test/test_chunk_layer_bpe.py`. Add a new test method inside `TestChunkLayerBPE` (alongside `test_cold_start_vocab_has_bytes`):

```python
def test_constructor_accepts_new_arg_names(self):
    """After Task 1, ChunkLayer takes n_vectors / chunking_frequency."""
    from Layers import ChunkLayer
    layer = ChunkLayer(nDim=8, bpe=True,
                       n_vectors=1024, chunking_frequency=2)
    self.assertEqual(layer.n_vectors, 1024)
    self.assertEqual(layer.chunking_frequency, 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py::TestChunkLayerBPE::test_constructor_accepts_new_arg_names -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'n_vectors'`.

- [ ] **Step 3: Rename args in `ChunkLayer.__init__`**

Edit `basicmodel/bin/Layers.py` at the `ChunkLayer.__init__` signature (around line 3783). Replace the constructor's arg names and the corresponding attribute assignments:

Before:
```python
def __init__(self, nDim, nChunks=256, bpe=False,
             target_vocab_size=1024, min_pair_frequency=2):
    super().__init__(nDim, nDim)
    self.nDim = nDim
    self.nChunks = nChunks
    # Split prototypes: what each chunk decodes to  [nChunks, 2, nDim]
    self.split = nn.Parameter(torch.randn(nChunks, 2, nDim) * 0.02)
    # Merge prototypes: the single vector a chunk becomes  [nChunks, nDim]
    self.merge = nn.Parameter(torch.randn(nChunks, nDim) * 0.02)
    # Learned threshold -- merge only when score exceeds this
    self.threshold = nn.Parameter(torch.zeros(1))
    # -- BPE state (active only when ``bpe`` is True) ------------------
    self.bpe = bool(bpe)
    self.target_vocab_size = int(target_vocab_size)
    self.min_pair_frequency = int(min_pair_frequency)
```

After:
```python
def __init__(self, nDim, bpe=False,
             n_vectors=1024, chunking_frequency=2):
    super().__init__(nDim, nDim)
    self.nDim = nDim
    # -- BPE state (active only when ``bpe`` is True) ------------------
    self.bpe = bool(bpe)
    self.n_vectors = int(n_vectors)
    self.chunking_frequency = int(chunking_frequency)
```

Also update internal references from `self.target_vocab_size` → `self.n_vectors` and `self.min_pair_frequency` → `self.chunking_frequency`. Specifically:

- In `train_step` ([Layers.py:3950](../../bin/Layers.py)): `if len(self.vocab) >= self.target_vocab_size` → `if len(self.vocab) >= self.n_vectors`. Same at line 3964.
- In `train_step` ([Layers.py:3962](../../bin/Layers.py)): `if freq < self.min_pair_frequency` → `if freq < self.chunking_frequency`.
- Update the docstring references as well.

Leave `self.merges`, `self.vocab`, `self.id_to_bytes`, `self._next_id`, `self._max_merge_len` in place (they are still needed).

**Do not delete** `self.split` / `self.merge` / `self.threshold` yet — that happens in Task 2 so we can test each change independently.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py::TestChunkLayerBPE::test_constructor_accepts_new_arg_names -v
```

Expected: PASS.

- [ ] **Step 5: Update all existing tests in `test_chunk_layer_bpe.py` that still use old arg names**

Search for `target_vocab_size=` and `min_pair_frequency=` in `test/test_chunk_layer_bpe.py` and rename each to `n_vectors=` and `chunking_frequency=` respectively. Also update the `test_mm_bpe_config_drives_chunk_layer_flags` test body to read `chunking` / `nVectors` / `chunkingFrequency` from the config — **but defer touching the XML config read** to Task 10; for now just rename the ChunkLayer constructor call:

Before (at `test/test_chunk_layer_bpe.py:167-172`):
```python
layer = ChunkLayer(
    nDim=8,
    bpe=bool(cfg.space("PerceptualSpace", "chunkBPE")),
    target_vocab_size=target,
    min_pair_frequency=min_freq,
)
```

After:
```python
layer = ChunkLayer(
    nDim=8,
    bpe=bool(cfg.space("PerceptualSpace", "chunkBPE")),
    n_vectors=target,
    chunking_frequency=min_freq,
)
```

Same pattern for all other `ChunkLayer(...)` constructions in this file: replace `target_vocab_size=` with `n_vectors=` and `min_pair_frequency=` with `chunking_frequency=`.

- [ ] **Step 6: Run the full `test_chunk_layer_bpe.py` to verify everything still passes**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py -v
```

Expected: all tests PASS (the `test_mm_bpe_config_drives_chunk_layer_flags` test still reads the old XML field names — this is fine because we haven't changed the XML yet).

- [ ] **Step 7: Commit checkpoint (user action)**

Suggested commit message: `refactor(ChunkLayer): rename constructor args to n_vectors / chunking_frequency`. **Do not run `git commit` yourself** — the user manages commits.

---

### Task 2: Remove vestigial 256-prototype table

**Files:**
- Modify: `basicmodel/bin/Layers.py` — `ChunkLayer` class.
- Test: `basicmodel/test/test_chunk_layer_bpe.py`.

The `self.split` / `self.merge` / `self.threshold` parameters and the `score_pair` / `encode` / `decode` / `should_merge` methods were auxiliary to a pair-scoring path never called from `PerceptualSpace.forward`. Task 0 Step 2 verified no external callers. Remove them.

- [ ] **Step 1: Add a failing test that asserts the prototype attrs are gone**

Add to `test/test_chunk_layer_bpe.py`:

```python
def test_no_prototype_table(self):
    """Task 2: ChunkLayer no longer carries the unused 256-prototype table."""
    from Layers import ChunkLayer
    layer = ChunkLayer(nDim=8, bpe=True, n_vectors=1024, chunking_frequency=2)
    self.assertFalse(hasattr(layer, 'split'),
                     "self.split should be removed")
    self.assertFalse(hasattr(layer, 'merge'),
                     "self.merge should be removed (distinct from self.merges)")
    self.assertFalse(hasattr(layer, 'threshold'),
                     "self.threshold should be removed")
    self.assertFalse(hasattr(layer, 'score_pair'))
    self.assertFalse(hasattr(layer, 'encode'))
    self.assertFalse(hasattr(layer, 'decode'))
    self.assertFalse(hasattr(layer, 'should_merge'))
    # merges list (the BPE merges, not prototype merge tensor) stays:
    self.assertTrue(hasattr(layer, 'merges'))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py::TestChunkLayerBPE::test_no_prototype_table -v
```

Expected: FAIL because `hasattr(layer, 'split')` returns True.

- [ ] **Step 3: Delete the prototype parameters from `ChunkLayer.__init__`**

Edit `basicmodel/bin/Layers.py`. Remove these three lines from `ChunkLayer.__init__` (they are the assignments we left in place during Task 1):

```python
self.split = nn.Parameter(torch.randn(nChunks, 2, nDim) * 0.02)
self.merge = nn.Parameter(torch.randn(nChunks, nDim) * 0.02)
self.threshold = nn.Parameter(torch.zeros(1))
```

- [ ] **Step 4: Delete the unused methods**

In `basicmodel/bin/Layers.py`, delete these method definitions in `ChunkLayer`:

- `score_pair(self, v1, v2)` ([Layers.py:3812-3827](../../bin/Layers.py))
- `encode(self, v1, v2)` ([Layers.py:3829-3837](../../bin/Layers.py))
- `decode(self, chunk_id)` ([Layers.py:3839-3847](../../bin/Layers.py))
- `should_merge(self, v1, v2)` ([Layers.py:3849-3852](../../bin/Layers.py))

After deletion the class should go directly from `__init__` (ending after `self._max_merge_len = 1`) to the `BOUNDARY_BYTES = frozenset(...)` constant and `is_word_boundary` method.

- [ ] **Step 5: Run the test and the full BPE suite**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py -v
```

Expected: all tests PASS, including the new `test_no_prototype_table`.

- [ ] **Step 6: Commit checkpoint**

Suggested: `refactor(ChunkLayer): drop vestigial 256-prototype parameters and pair-scoring methods`.

---

## Phase 2: PerceptualSpace config read

### Task 3: Read new config fields

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — `PerceptualSpace.__init__`.
- Test: `basicmodel/test/test_perceptualspace_bpe_forward.py` (new file).

- [ ] **Step 1: Create the new test file with a failing test**

Create `basicmodel/test/test_perceptualspace_bpe_forward.py`:

```python
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
    <useButterflies>false</useButterflies>
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
    <nOutput>8</nOutput>
    <lexer>byte</lexer>
  </InputSpace>
  <PerceptualSpace>
    <nInput>8</nInput>
    <nOutput>8</nOutput>
    <nDim>4</nDim>
    <nVectors>{n_vectors}</nVectors>
    <codebook>true</codebook>
    <chunking>bpe</chunking>
    <chunkingFrequency>2</chunkingFrequency>
  </PerceptualSpace>
  <ConceptualSpace>
    <nOutput>8</nOutput>
    <nDim>4</nDim>
    <nVectors>8</nVectors>
    <codebook>true</codebook>
  </ConceptualSpace>
  <SymbolicSpace>
    <nOutput>8</nOutput>
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
        """Task 3: PerceptualSpace reads <chunking>, <nVectors>, <chunkingFrequency>."""
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
                int(Spaces.TheXMLConfig.space("PerceptualSpace", "chunkingFrequency")), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test; it should currently fail on the XSD validation**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_init_reads_new_config_fields -v
```

Expected: FAIL — the XSD does not yet define `chunkingFrequency`, so `init_config` raises a schema validation error, or `TheXMLConfig.space("PerceptualSpace", "chunkingFrequency")` raises `KeyError`.

- [ ] **Step 3: Update the XSD to allow `chunkingFrequency` (leave old fields for now to stay incremental)**

Edit `basicmodel/data/model.xsd` at the `perceptualSpaceType` complexType (around line 288). Add the new element alongside the existing ones:

```xml
<xs:element name="chunkingFrequency" type="xs:positiveInteger" minOccurs="0"/>
```

Place it adjacent to `chunkMinPairFrequency` — both are valid during this transitional task. They'll be consolidated in Task 10.

- [ ] **Step 4: Re-run the test**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_init_reads_new_config_fields -v
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Suggested: `feat(schema): add <chunkingFrequency> element to PerceptualSpace XSD`.

---

### Task 4: Rewrite PerceptualSpace.__init__ config block

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — `PerceptualSpace.__init__` at [Spaces.py:5097-5141](../../bin/Spaces.py).
- Test: `basicmodel/test/test_perceptualspace_bpe_forward.py`.

- [ ] **Step 1: Add a failing test for construction validation**

Append to `TestPerceptualSpaceBPE` in `test/test_perceptualspace_bpe_forward.py`:

```python
def test_init_rejects_bpe_with_nVectors_below_256(self):
    """Task 4: bpe mode requires nVectors>=256."""
    import tempfile
    from util import init_config
    from Models import ModelFactory

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_minimal_bpe_xml(tmp, n_vectors=128)
        init_config(path=path,
                    defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
        with self.assertRaises(ValueError) as ctx:
            ModelFactory.create()
        self.assertIn("nVectors", str(ctx.exception))
        self.assertIn("256", str(ctx.exception))

def test_init_accepts_bpe_with_nVectors_256(self):
    """Task 4: nVectors>=256 is accepted; chunk_layer is built in bpe mode."""
    import tempfile
    from util import init_config
    from Models import ModelFactory

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_minimal_bpe_xml(tmp, n_vectors=512)
        init_config(path=path,
                    defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
        model = ModelFactory.create()
        ps = model.perceptualSpace
        self.assertEqual(ps.chunking_mode, "bpe")
        self.assertEqual(ps.chunking_frequency, 2)
        self.assertTrue(ps.chunk_layer.bpe)
        self.assertEqual(ps.chunk_layer.n_vectors, 512)
        self.assertEqual(ps.chunk_layer.chunking_frequency, 2)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py -v
```

Expected: `test_init_rejects_bpe_with_nVectors_below_256` fails (no `ValueError` raised); `test_init_accepts_bpe_with_nVectors_256` fails (ChunkLayer does not have `.n_vectors` attr after rename, or `ps.chunking_frequency` is unset).

- [ ] **Step 3: Replace the config block in `PerceptualSpace.__init__`**

Edit `basicmodel/bin/Spaces.py` starting at the `try:` block at [Spaces.py:5097](../../bin/Spaces.py). Replace lines 5097-5141 (the entire current block including `chunking_mode` / `chunkBPE` / `chunkTargetVocabSize` / `chunkMinPairFrequency` reads and the `ChunkLayer` construction) with:

```python
try:
    self.chunking_mode = str(
        TheXMLConfig.space(section, "chunking") or "lexicon"
    )
except KeyError:
    self.chunking_mode = "lexicon"
if self.chunking_mode not in ("bpe", "lexicon"):
    raise ValueError(
        f"PerceptualSpace.chunking must be bpe|lexicon, "
        f"got {self.chunking_mode!r}")
if isinstance(lexical_basis, Embedding):
    lexical_basis.chunking_mode = self.chunking_mode
    lexical_basis.lexer_mode = self.lexer
try:
    self.chunking_frequency = int(
        TheXMLConfig.space(section, "chunkingFrequency") or 2)
except (KeyError, TypeError, ValueError):
    self.chunking_frequency = 2
if self.chunking_mode == "bpe":
    if self.nVectors < 256:
        raise ValueError(
            f"PerceptualSpace.chunking='bpe' requires nVectors>=256 "
            f"(to seed the byte range); got nVectors={self.nVectors}")
    if self.model_type != "embedding":
        raise ValueError(
            "PerceptualSpace.chunking='bpe' requires "
            "<modelType>embedding</modelType>")
self._recovered_input = None
self._embedded_input = None
if passThrough:
    return
input = self.subspace.getEncodedInputSize()
self.attention = AttentionLayer(input, input, type="transformer")
self.subspace._nWordSlots = outputShape[0]
self.params = []
self.layers = nn.ModuleList()
# Symbolic BPE tokenizer. Built lazily later if lexical_basis is
# missing; here we always build so attribute access is uniform.
self.chunk_layer = ChunkLayer(
    self.nDim,
    bpe=(self.chunking_mode == "bpe"),
    n_vectors=self.nVectors,
    chunking_frequency=self.chunking_frequency,
)
```

**Important:** the lines `self._recovered_input = None` through the final `self.chunk_layer = ...` assignment are already present in the current code — the change is replacing the chunkBPE / chunkTargetVocabSize / chunkMinPairFrequency read blocks (lines 5107-5123 in the current file) with the new single `chunking_frequency` read, and tightening the ChunkLayer constructor call.

Keep the earlier `chunking_mode` read (lines 5098-5105) in its new form shown above. Merge the two reads into one consecutive block.

- [ ] **Step 4: Run the two new tests and verify they pass**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py -v
```

Expected: both new tests PASS. The earlier `test_init_reads_new_config_fields` should also still PASS.

- [ ] **Step 5: Run the full `test_chunk_layer_bpe.py` and `test_perceptual_chunking.py`**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py test/test_perceptual_chunking.py -v
```

Expected: `test_mm_bpe_config_drives_chunk_layer_flags` in `test_chunk_layer_bpe.py` still PASSES because it reads the XML field name `chunkBPE` directly from `TheXMLConfig`, which MM_bpe.xml still has. `test_perceptual_chunking.py` should still all-pass because forward-path dispatch is unchanged so far.

- [ ] **Step 6: Commit checkpoint**

Suggested: `feat(PerceptualSpace): read <chunking>/<nVectors>/<chunkingFrequency>, validate bpe requirements`.

---

## Phase 3: BPE forward path

### Task 5: Implement `_embed_bpe` (failing test first)

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — add `PerceptualSpace._embed_bpe`, update `PerceptualSpace.forward` dispatch.
- Test: `basicmodel/test/test_perceptualspace_bpe_forward.py`.

- [ ] **Step 1: Add a failing end-to-end test**

Append to `TestPerceptualSpaceBPE` in `test/test_perceptualspace_bpe_forward.py`:

```python
def test_forward_bpe_emits_word_level_vectors(self):
    """Task 5: BPE forward pass emits [B, nOutput, nDim] with one position per whitespace word."""
    import tempfile
    import torch
    from util import init_config
    from Models import ModelFactory

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_minimal_bpe_xml(tmp, n_vectors=512)
        init_config(path=path,
                    defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
        model = ModelFactory.create()
        ps = model.perceptualSpace

        # Feed a simple 3-word byte buffer through the full stem.
        input_text = ["hello world foo"]
        out = model.forward(input_text)

        ps_event = ps.subspace.event.getW()
        self.assertIsNotNone(ps_event, "PerceptualSpace.subspace.event must be populated")
        # Shape: [B=1, nOutput=8, nDim=4]
        self.assertEqual(ps_event.shape[0], 1)
        self.assertEqual(ps_event.shape[-1], 4)
        # First 3 positions (one per word) should be non-zero; trailing positions zero-padded.
        non_zero_rows = (ps_event[0].abs().sum(dim=-1) > 0).sum().item()
        self.assertGreaterEqual(non_zero_rows, 1,
            "At least one word position must have a non-zero vector")
        self.assertLessEqual(non_zero_rows, 3,
            "At most 3 word positions should have non-zero vectors for 3 input words")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_forward_bpe_emits_word_level_vectors -v
```

Expected: FAIL with `NotImplementedError: PerceptualSpace chunking='bpe' not yet wired; use 'lexicon'.` (raised at [Spaces.py:5438-5440](../../bin/Spaces.py)).

- [ ] **Step 3: Add `_embed_bpe` method to `PerceptualSpace`**

Edit `basicmodel/bin/Spaces.py`. Add a new method `_embed_bpe` immediately after the existing `_embed` method (which ends around [Spaces.py:5366](../../bin/Spaces.py)):

```python
def _embed_bpe(self, upstream_vspace):
    """BPE chunking path. Decode the upstream byte buffer, whitespace-split
    into words, BPE-tokenize each word via ChunkLayer, look up (and insert
    as OOV) each byte-tuple chunk in the codebook, then MAX-fuse sub-token
    vectors within each word.  Emit one [nDim] vector per word.

    Upstream layout (same as _embed):
      - what.W:  [B, N, nWhat] null-terminated UTF-8 byte buffer (or [B, N] byte indices)
      - where.W: [B, N] byte offsets (long)
      - when.W:  [B, N] sequential positions (long)
    """
    what_buf = upstream_vspace.what.getW()
    if what_buf is None:
        raise RuntimeError(
            "PerceptualSpace._embed_bpe: upstream subspace.what is empty. "
            "InputSpace.forward must lex into subspace.what.W before "
            "PerceptualSpace.forward runs.")

    dev = TheDevice.get()
    batch = what_buf.shape[0]
    nObj = self.outputShape[0]
    codebook = self.subspace.what
    boundary = self.chunk_layer.BOUNDARY_BYTES

    # Normalize to a [B, N_bytes] long tensor of byte values.
    if what_buf.dim() == 3:
        byte_indices = what_buf[..., 0].long()
    else:
        byte_indices = what_buf.long()

    # Run BPE training-step before segmentation when model is training
    # (mirrors the hard_merge_spans pattern in Layers.py).
    if self.chunk_layer.bpe and self.training:
        self.chunk_layer.train_step(byte_indices)

    # Greedy longest-match BPE over the whole byte row. Word boundaries
    # are filtered out separately below, so the chunk list may include
    # whitespace-backed chunks that we'll skip.
    all_chunks, all_spans = self.chunk_layer.forward(byte_indices)

    null_idx = codebook.wv.key_to_index.get("\x00", 0)
    what_indices = torch.full(
        (batch, nObj), null_idx, dtype=torch.long, device=dev)
    word_vectors = torch.zeros(batch, nObj, self.nDim, device=dev)

    for b in range(batch):
        chunks = all_chunks[b]
        spans = all_spans[b]
        if not chunks:
            continue
        word_idx = 0
        word_subtokens = []  # byte-tuples for current word
        for (chunk_id, (start, end, key)) in zip(chunks, spans):
            # Determine whether this chunk is entirely made of boundary
            # bytes (i.e., whitespace).  If so, it closes the current word.
            is_boundary = all(bv in boundary for bv in key)
            if is_boundary:
                if word_subtokens and word_idx < nObj:
                    word_vectors[b, word_idx] = self._max_fuse_subtokens(
                        word_subtokens, codebook)
                    what_indices[b, word_idx] = self._chunk_to_codebook_idx(
                        word_subtokens, codebook)
                    word_idx += 1
                word_subtokens = []
            else:
                word_subtokens.append(key)
        # Flush trailing word (no terminal whitespace)
        if word_subtokens and word_idx < nObj:
            word_vectors[b, word_idx] = self._max_fuse_subtokens(
                word_subtokens, codebook)
            what_indices[b, word_idx] = self._chunk_to_codebook_idx(
                word_subtokens, codebook)

    # where / when passthrough from upstream
    where_raw = upstream_vspace.where.getW()
    when_raw = upstream_vspace.when.getW()
    where_indices = (where_raw[:, :nObj].long()
                     if (self.nWhere > 0 and where_raw is not None)
                     else (torch.zeros(batch, nObj, dtype=torch.long, device=dev)
                           if self.nWhere > 0 else None))
    when_indices = (when_raw[:, :nObj].long()
                    if (self.nWhen > 0 and when_raw is not None)
                    else (torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1)
                          if self.nWhen > 0 else None))

    self.subspace.whereEncoding.p = 0
    self.subspace.set_forward_content(what_indices, where_indices, when_indices)
    # Override the event tensor with our MAX-fused word vectors.
    self.subspace.event.setW(word_vectors)
    self._embedded_input = word_vectors
    return self.subspace

def _chunk_key_to_latin1(self, byte_tuple):
    """Convert a byte-tuple key (e.g., (104, 101)) to its latin-1 string ('he')."""
    return "".join(chr(int(b) & 0xFF) for b in byte_tuple)

def _chunk_to_codebook_idx(self, word_subtokens, codebook):
    """Resolve a list of byte-tuple sub-tokens to a single codebook index
    for the fused word.  Inserts OOV keys as needed using the existing
    Embedding.insert() path.  Returns the index of the first sub-token
    (used only for bookkeeping; the real word vector comes from MAX fusion)."""
    keys = [self._chunk_key_to_latin1(bt) for bt in word_subtokens]
    for key in keys:
        if (key and key not in codebook.pretrain.key_to_index
                and not getattr(codebook, 'byte_mode', False)):
            codebook.insert(key)
    return codebook._token_to_index(keys[0]) if keys else 0

def _max_fuse_subtokens(self, word_subtokens, codebook):
    """MAX-fuse sub-token vectors into a single [nDim] word vector."""
    if not word_subtokens:
        return torch.zeros(self.nDim, device=TheDevice.get())
    vecs = []
    for bt in word_subtokens:
        key = self._chunk_key_to_latin1(bt)
        if key and key in codebook.pretrain.key_to_index:
            idx = codebook._token_to_index(key)
            vecs.append(codebook.wv._vectors[idx])
    if not vecs:
        return torch.zeros(self.nDim, device=TheDevice.get())
    stacked = torch.stack(vecs, dim=0)   # [k, nDim]
    return stacked.max(dim=0).values     # [nDim]
```

- [ ] **Step 4: Wire the dispatch in `PerceptualSpace.forward`**

Edit `basicmodel/bin/Spaces.py` at [Spaces.py:5431-5443](../../bin/Spaces.py). Replace the `elif mode == "bpe":` branch body and the stale `elif mode == "cached":` branch with the new dispatch:

Before:
```python
if isinstance(self.subspace.what, Embedding) and not vspace.stem_embedded:
    mode = self.chunking_mode
    if mode == "lexicon":
        vspace = self._embed(vspace)
    elif mode == "cached":
        raise NotImplementedError(
            "PerceptualSpace chunking='cached' not yet wired; use 'lexicon'.")
    elif mode == "bpe":
        raise NotImplementedError(
            "PerceptualSpace chunking='bpe' not yet wired; use 'lexicon'.")
    else:
        raise ValueError(
            f"PerceptualSpace chunking must be lexicon|cached|bpe, got {mode!r}")
```

After:
```python
if isinstance(self.subspace.what, Embedding) and not vspace.stem_embedded:
    mode = self.chunking_mode
    if mode == "lexicon":
        vspace = self._embed(vspace)
    elif mode == "bpe":
        vspace = self._embed_bpe(vspace)
    else:
        raise ValueError(
            f"PerceptualSpace chunking must be bpe|lexicon, got {mode!r}")
```

- [ ] **Step 5: Run the new end-to-end test**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_forward_bpe_emits_word_level_vectors -v
```

Expected: PASS. If it fails with a shape-mismatch or device error, inspect which downstream consumer of `subspace.event` is misinterpreting the new MAX-fused layout and iterate.

- [ ] **Step 6: Run full perceptual-chunking test suite to catch regressions**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptual_chunking.py test/test_chunk_layer_bpe.py -v
```

Expected: all still PASS.

- [ ] **Step 7: Commit checkpoint**

Suggested: `feat(PerceptualSpace): wire _embed_bpe with word-boundary MAX fusion over BPE sub-tokens`.

---

### Task 6: MAX-fusion correctness unit test

**Files:**
- Test: `basicmodel/test/test_perceptualspace_bpe_forward.py`.

- [ ] **Step 1: Add a targeted MAX-fusion test**

Append to `TestPerceptualSpaceBPE`:

```python
def test_max_fusion_elementwise_max(self):
    """MAX fusion of two sub-tokens equals elementwise max of their embeddings."""
    import tempfile
    import torch
    from util import init_config
    from Models import ModelFactory

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_minimal_bpe_xml(tmp, n_vectors=512)
        init_config(path=path,
                    defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
        model = ModelFactory.create()
        ps = model.perceptualSpace
        codebook = ps.subspace.what

        # Pin known vectors for byte 'a' (97) and 'b' (98) in the codebook.
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

        # With no merges learned, "ab" tokenizes to two single-byte chunks.
        fused = ps._max_fuse_subtokens([(97,), (98,)], codebook)
        expected = torch.max(v_a, v_b)  # [1.0, 1.0, 0.5, 0.0] — normalized identically
        # The stored codebook vectors are L2-normalized on read via getW(),
        # but _max_fuse_subtokens reads raw ._vectors. Assert raw max matches.
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6),
            f"MAX fusion mismatch: got {fused}, expected {expected}")
```

- [ ] **Step 2: Run it**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_max_fusion_elementwise_max -v
```

Expected: PASS (builds directly on the `_max_fuse_subtokens` from Task 5). If FAIL, inspect normalization — if `codebook.wv._vectors` is being auto-normalized during `insert`, the assertion needs to compare against the normalized pair.

- [ ] **Step 3: Commit checkpoint**

Suggested: `test(PerceptualSpace): verify MAX fusion is elementwise max of sub-token embeddings`.

---

### Task 7: Vocab cap test

**Files:**
- Test: `basicmodel/test/test_perceptualspace_bpe_forward.py`.

- [ ] **Step 1: Add the vocab cap test**

Append to `TestPerceptualSpaceBPE`:

```python
def test_train_step_respects_nVectors_cap(self):
    """Task 7: train_step stops growing vocab once len(vocab) == n_vectors."""
    import torch
    from Layers import ChunkLayer

    layer = ChunkLayer(nDim=4, bpe=True,
                       n_vectors=260, chunking_frequency=1)
    layer.train()
    batch = torch.tensor(
        [list(b"abababababababababab") + [0] * 4],
        dtype=torch.long,
    )
    for _ in range(100):
        layer.train_step(batch, k_merges=4)
    self.assertLessEqual(len(layer.vocab), 260,
        f"vocab overflow: got {len(layer.vocab)}, cap was 260")
```

- [ ] **Step 2: Run it**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_train_step_respects_nVectors_cap -v
```

Expected: PASS (this is a direct rename of an existing passing test in `test_chunk_layer_bpe.py`; the behavior is already there, just under new names).

- [ ] **Step 3: Commit checkpoint**

Suggested: `test(ChunkLayer): regression-guard n_vectors cap in train_step`.

---

## Phase 4: Drop `raw` mode

### Task 8: Remove `raw` branch from `chunk_static` and `Embedding._token_stream`

**Files:**
- Modify: `basicmodel/bin/Spaces.py` — `PerceptualSpace.chunk_static` ([Spaces.py:5242-5261](../../bin/Spaces.py)), `Embedding._token_stream` ([Spaces.py:1809-1820](../../bin/Spaces.py)).
- Modify: `basicmodel/test/test_perceptual_chunking.py` — remove two `raw`-specific tests.
- Modify: `basicmodel/data/model.xsd` — drop `raw` enum value.

- [ ] **Step 1: Delete `test_chunking_mode_raw_returns_bytes`**

Edit `basicmodel/test/test_perceptual_chunking.py`. Delete the function at [test_perceptual_chunking.py:5-9](../../test/test_perceptual_chunking.py):

```python
def test_chunking_mode_raw_returns_bytes():
    byte_stream = b"Hello, world."
    units = PerceptualSpace.chunk_static(byte_stream, mode="raw")
    assert units == [b"H", b"e", b"l", b"l", b"o", b",", b" ",
                     b"w", b"o", b"r", b"l", b"d", b"."]
```

- [ ] **Step 2: Delete `test_embedding_token_stream_honors_raw_chunking_mode`**

In the same file, delete the function at [test_perceptual_chunking.py:45-50](../../test/test_perceptual_chunking.py):

```python
def test_embedding_token_stream_honors_raw_chunking_mode():
    emb = Embedding()
    emb.byte_mode = False
    emb.chunking_mode = "raw"

    assert emb._token_stream("Az") == [("A", 0), ("z", 1)]
```

- [ ] **Step 3: Update `chunk_static` to drop the `raw` branch**

Edit `basicmodel/bin/Spaces.py` at [Spaces.py:5242-5261](../../bin/Spaces.py). Replace:

```python
@staticmethod
def chunk_static(stream: bytes, mode: str) -> list:
    """Three-way chunking switch: raw | bpe | lexicon.

    - raw: split into single bytes.
    - lexicon: split on whitespace (word-level).
    - bpe: cold-start BPE (byte-level fallback when no trained merges).
    """
    if mode == "raw":
        return [bytes([b]) for b in stream]
    if mode == "lexicon":
        return stream.split()
    if mode == "bpe":
        # Cold-start BPE: no merges table available here -> fall back to
        # single bytes. The real BPE path runs through ChunkLayer (see
        # basicmodel/bin/Layers.py) once merges have been learned.
        return [bytes([b]) for b in stream]
    raise ValueError(
        f"chunking mode must be raw|bpe|lexicon, got {mode!r}"
    )
```

with:

```python
@staticmethod
def chunk_static(stream: bytes, mode: str) -> list:
    """Two-way chunking switch: bpe | lexicon.

    - lexicon: split on whitespace (word-level).
    - bpe: cold-start BPE (byte-level fallback when no trained merges).
    """
    if mode == "lexicon":
        return stream.split()
    if mode == "bpe":
        return [bytes([b]) for b in stream]
    raise ValueError(
        f"chunking mode must be bpe|lexicon, got {mode!r}"
    )
```

- [ ] **Step 4: Update `Embedding._token_stream` to drop `raw`**

Edit `basicmodel/bin/Spaces.py` at [Spaces.py:1814-1820](../../bin/Spaces.py). Replace:

```python
mode = getattr(self, 'chunking_mode', 'lexicon')
if mode in ('raw', 'bpe'):
    # Both raw and cold-start BPE fall back to per-character
    # tokenization. The real BPE path runs through ChunkingLayer
    # once merges have been learned.
    return self._char_stream(text)
return parse(self._to_text(text), lex='words')
```

with:

```python
mode = getattr(self, 'chunking_mode', 'lexicon')
if mode == 'bpe':
    # Cold-start BPE falls back to per-character tokenization here;
    # the real BPE path runs through ChunkLayer in PerceptualSpace.
    return self._char_stream(text)
return parse(self._to_text(text), lex='words')
```

- [ ] **Step 5: Drop `raw` from the XSD enum**

Edit `basicmodel/data/model.xsd` at [model.xsd:291-299](../../data/model.xsd). Replace:

```xml
<xs:element name="chunking" minOccurs="0">
  <xs:simpleType>
    <xs:restriction base="xs:string">
      <xs:enumeration value="raw"/>
      <xs:enumeration value="bpe"/>
      <xs:enumeration value="lexicon"/>
    </xs:restriction>
  </xs:simpleType>
</xs:element>
```

with:

```xml
<xs:element name="chunking" minOccurs="0">
  <xs:simpleType>
    <xs:restriction base="xs:string">
      <xs:enumeration value="bpe"/>
      <xs:enumeration value="lexicon"/>
    </xs:restriction>
  </xs:simpleType>
</xs:element>
```

- [ ] **Step 6: Run `test_perceptual_chunking.py` and confirm the remaining tests still pass**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptual_chunking.py -v
```

Expected: the remaining 5 tests (`test_chunking_mode_lexicon_splits_on_spaces`, `test_chunking_mode_bpe_returns_learned_segments`, `test_chunking_invalid_mode_raises`, `test_perceptual_space_exposes_chunking_mode_attribute`, `test_embedding_token_stream_honors_bpe_fallback_mode`) all PASS. Note that `test_chunking_invalid_mode_raises` now also covers `"raw"` as invalid — no test-code change needed.

- [ ] **Step 7: Commit checkpoint**

Suggested: `refactor: drop 'raw' chunking mode from runtime, schema, and tests`.

---

## Phase 5: XML migration

### Task 9: Migrate `model.xml` (the defaults)

**Files:**
- Modify: `basicmodel/data/model.xml`.

`model.xml` is the defaults file; it currently has `<chunkBPE>false</chunkBPE>` and `<chunkTargetVocabSize>1024</chunkTargetVocabSize>` in PerceptualSpace. Remove them.

- [ ] **Step 1: Edit `basicmodel/data/model.xml`**

At [model.xml:150-154](../../data/model.xml), replace:

```xml
<chunking>lexicon</chunking>
<chunkBPE>false</chunkBPE>
<chunkTargetVocabSize>1024</chunkTargetVocabSize>
<chunkMinPairFrequency>2</chunkMinPairFrequency>
```

with:

```xml
<chunking>lexicon</chunking>
<chunkingFrequency>2</chunkingFrequency>
```

- [ ] **Step 2: Run the perceptual-chunking tests to confirm no regression**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptual_chunking.py test/test_perceptualspace_bpe_forward.py -v
```

Expected: all PASS.

---

### Task 10: Migrate `MM_bpe.xml` and update its test

**Files:**
- Modify: `basicmodel/data/MM_bpe.xml`.
- Modify: `basicmodel/test/test_chunk_layer_bpe.py` — `test_mm_bpe_config_drives_chunk_layer_flags`.

- [ ] **Step 1: Edit `basicmodel/data/MM_bpe.xml`**

At [MM_bpe.xml:60-62](../../data/MM_bpe.xml), replace:

```xml
<chunkBPE>true</chunkBPE>
<chunkTargetVocabSize>512</chunkTargetVocabSize>
<chunkMinPairFrequency>2</chunkMinPairFrequency>
```

with:

```xml
<chunking>bpe</chunking>
<chunkingFrequency>2</chunkingFrequency>
```

Also verify `<nVectors>` on the PerceptualSpace block. `MM_bpe.xml` currently has `<nVectors>8</nVectors>` ([MM_bpe.xml:55](../../data/MM_bpe.xml)) which is below the 256 minimum for BPE. Update it to 512 (the old `chunkTargetVocabSize`):

At [MM_bpe.xml:55](../../data/MM_bpe.xml), replace `<nVectors>8</nVectors>` with `<nVectors>512</nVectors>`.

- [ ] **Step 2: Update `test_mm_bpe_config_drives_chunk_layer_flags`**

Edit `basicmodel/test/test_chunk_layer_bpe.py` at [test_chunk_layer_bpe.py:151-175](../../test/test_chunk_layer_bpe.py). Replace:

```python
def test_mm_bpe_config_drives_chunk_layer_flags(self):
    """MM_bpe.xml config should produce a BPE-mode ChunkLayer."""
    from util import init_config
    import Spaces
    from Layers import ChunkLayer

    cfg_path = os.path.join(_PROJECT, "data", "MM_bpe.xml")
    init_config(
        path=cfg_path,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    cfg = Spaces.TheXMLConfig
    self.assertTrue(bool(cfg.space("PerceptualSpace", "chunkBPE")),
                    "MM_bpe.xml must set chunkBPE=true")
    target = int(cfg.space("PerceptualSpace", "chunkTargetVocabSize"))
    min_freq = int(cfg.space("PerceptualSpace", "chunkMinPairFrequency"))
    layer = ChunkLayer(
        nDim=8,
        bpe=bool(cfg.space("PerceptualSpace", "chunkBPE")),
        n_vectors=target,
        chunking_frequency=min_freq,
    )
    self.assertTrue(layer.bpe)
    self.assertEqual(layer.target_vocab_size, target)
    self.assertEqual(layer.min_pair_frequency, min_freq)
```

with:

```python
def test_mm_bpe_config_drives_chunk_layer_flags(self):
    """MM_bpe.xml config should produce a BPE-mode ChunkLayer."""
    from util import init_config
    import Spaces
    from Layers import ChunkLayer

    cfg_path = os.path.join(_PROJECT, "data", "MM_bpe.xml")
    init_config(
        path=cfg_path,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    cfg = Spaces.TheXMLConfig
    self.assertEqual(cfg.space("PerceptualSpace", "chunking"), "bpe",
                     "MM_bpe.xml must set chunking=bpe")
    n_vec = int(cfg.space("PerceptualSpace", "nVectors"))
    freq = int(cfg.space("PerceptualSpace", "chunkingFrequency"))
    layer = ChunkLayer(
        nDim=8,
        bpe=(cfg.space("PerceptualSpace", "chunking") == "bpe"),
        n_vectors=n_vec,
        chunking_frequency=freq,
    )
    self.assertTrue(layer.bpe)
    self.assertEqual(layer.n_vectors, n_vec)
    self.assertEqual(layer.chunking_frequency, freq)
```

- [ ] **Step 3: Update the file-header docstring in `test_chunk_layer_bpe.py`**

At [test_chunk_layer_bpe.py:13-14](../../test/test_chunk_layer_bpe.py), replace:

```
  5. Loading ``MM_bpe.xml`` propagates ``chunkBPE`` / ``chunkTargetVocabSize``
     / ``chunkMinPairFrequency`` into a ``ChunkLayer`` instance.
```

with:

```
  5. Loading ``MM_bpe.xml`` propagates ``chunking`` / ``nVectors``
     / ``chunkingFrequency`` into a ``ChunkLayer`` instance.
```

- [ ] **Step 4: Run the BPE test suite**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py -v
```

Expected: all PASS.

---

### Task 11: Migrate `MM_5M.xml`

**Files:**
- Modify: `basicmodel/data/MM_5M.xml`.

- [ ] **Step 1: Edit PerceptualSpace block in MM_5M.xml**

At [MM_5M.xml:66-81](../../data/MM_5M.xml), the current PerceptualSpace block is:

```xml
<PerceptualSpace>
    <nInput>1024</nInput>
    <nVectors>8192</nVectors>
    <nDim>6</nDim>
    <nWhere>2</nWhere>
    <nWhen>0</nWhen>
    <nOutput>1024</nOutput>
    <passThrough>false</passThrough>
    <hasAttention>false</hasAttention>
    <invertible>false</invertible>
    <codebook>false</codebook>
    <chunking>bpe</chunking>
    <chunkTargetVocabSize>4096</chunkTargetVocabSize>
    <chunkMinPairFrequency>2</chunkMinPairFrequency>

</PerceptualSpace>
```

Replace with (note `<nVectors>` drops from 8192 → 4096 to match the old vocab target; remove the now-redundant fields; rename the frequency tag):

```xml
<PerceptualSpace>
    <nInput>1024</nInput>
    <nVectors>4096</nVectors>
    <nDim>6</nDim>
    <nWhere>2</nWhere>
    <nWhen>0</nWhen>
    <nOutput>1024</nOutput>
    <passThrough>false</passThrough>
    <hasAttention>false</hasAttention>
    <invertible>false</invertible>
    <codebook>false</codebook>
    <chunking>bpe</chunking>
    <chunkingFrequency>2</chunkingFrequency>
</PerceptualSpace>
```

- [ ] **Step 2: Verify MM_5M.xml loads cleanly**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -c "
import sys; sys.path.insert(0, 'bin')
from util import init_config
import Spaces
init_config(path='data/MM_5M.xml', defaults_path='data/model.xml')
cfg = Spaces.TheXMLConfig
print('chunking =', cfg.space('PerceptualSpace', 'chunking'))
print('nVectors =', cfg.space('PerceptualSpace', 'nVectors'))
print('chunkingFrequency =', cfg.space('PerceptualSpace', 'chunkingFrequency'))
"
```

Expected:
```
chunking = bpe
nVectors = 4096
chunkingFrequency = 2
```

No XSD validation errors.

- [ ] **Step 3: MM_5M smoke test**

Append to `TestPerceptualSpaceBPE` in `test/test_perceptualspace_bpe_forward.py`:

```python
def test_mm_5m_xml_loads_and_forwards(self):
    """Task 11: MM_5M.xml can be loaded and runs one forward pass."""
    import torch
    from util import init_config
    from Models import ModelFactory

    cfg_path = os.path.join(_PROJECT, "data", "MM_5M.xml")
    defaults = os.path.join(_PROJECT, "data", "model.xml")
    init_config(path=cfg_path, defaults_path=defaults)
    model = ModelFactory.create()
    ps = model.perceptualSpace
    self.assertEqual(ps.chunking_mode, "bpe")
    self.assertEqual(ps.nVectors, 4096)

    # One forward pass on a tiny ASCII buffer.
    out = model.forward(["hello world foo"])
    event = ps.subspace.event.getW()
    self.assertIsNotNone(event)
    self.assertTrue(torch.isfinite(event).all(),
        "PerceptualSpace output must be finite")
```

- [ ] **Step 4: Run the smoke test**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py::TestPerceptualSpaceBPE::test_mm_5m_xml_loads_and_forwards -v
```

Expected: PASS. MM_5M is a large config; construction may take a few seconds. If it fails on unrelated downstream shape issues (not BPE-related), narrow the plan and investigate before continuing.

- [ ] **Step 5: Commit checkpoint**

Suggested: `feat(config): migrate MM_5M.xml to unified <chunking>bpe</chunking> + <nVectors>4096</nVectors>`.

---

## Phase 6: Schema cleanup

### Task 12: Remove legacy elements from XSD

**Files:**
- Modify: `basicmodel/data/model.xsd`.

All XMLs have been migrated off the old fields. Now remove them from the schema.

- [ ] **Step 1: Edit `basicmodel/data/model.xsd`**

At [model.xsd:288-290](../../data/model.xsd), delete the three legacy element declarations:

```xml
<xs:element name="chunkBPE" type="xs:boolean" minOccurs="0"/>
<xs:element name="chunkTargetVocabSize" type="xs:positiveInteger" minOccurs="0"/>
<xs:element name="chunkMinPairFrequency" type="xs:positiveInteger" minOccurs="0"/>
```

The `chunkingFrequency` element added in Task 3 remains.

- [ ] **Step 2: Verify the full test suite still passes**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/test_chunk_layer_bpe.py test/test_perceptual_chunking.py test/test_perceptualspace_bpe_forward.py -v
```

Expected: all PASS.

- [ ] **Step 3: Check for stray references to removed elements anywhere in the repo**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
grep -rnE "chunkBPE|chunkTargetVocabSize|chunkMinPairFrequency" bin/ test/ data/ 2>/dev/null | grep -vE "/.claude/"
```

Expected: **no output** (empty result). Any hit is a leftover reference to remove before marking this task done.

- [ ] **Step 4: Commit checkpoint**

Suggested: `chore(schema): remove legacy <chunkBPE>/<chunkTargetVocabSize>/<chunkMinPairFrequency> elements`.

---

## Phase 7: Wider regression sweep

### Task 13: Full BasicModel test suite

**Files:** none modified.

- [ ] **Step 1: Run the full basicmodel test suite**

Run:
```
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
.venv/bin/python -m pytest test/ -v 2>&1 | tail -80
```

Expected: same pass/fail status as the pre-change baseline (captured in Task 0 Step 1) modulo the new passing tests this plan added. No previously-green test goes red.

- [ ] **Step 2: If any test regresses, diagnose**

For each regression:
1. Read the failing test's code.
2. Look at the diff the plan introduced in that area.
3. Either fix the implementation (if the test encodes a requirement we missed) or update the test (if the test encoded a stale assumption).
4. Re-run.

Do not declare completion until this step yields a clean run.

- [ ] **Step 3: Final commit checkpoint (user action)**

After the user reviews the green suite, they can squash/commit the full change.

---

## Summary of Changes

| File | Change Type | Summary |
|------|-------------|---------|
| `basicmodel/bin/Layers.py` | Modify | Drop 256-prototype table from `ChunkLayer`; rename constructor args |
| `basicmodel/bin/Spaces.py` | Modify | New `_embed_bpe` method; rewrite `__init__` config block; drop `raw` from `chunk_static` and `Embedding._token_stream`; simplify forward dispatch |
| `basicmodel/data/model.xsd` | Modify | Remove 3 legacy elements; add `chunkingFrequency`; drop `raw` enum |
| `basicmodel/data/model.xml` | Modify | Remove legacy fields; add `chunkingFrequency` |
| `basicmodel/data/MM_bpe.xml` | Modify | Switch to `<chunking>bpe</chunking>` + `<nVectors>512</nVectors>` |
| `basicmodel/data/MM_5M.xml` | Modify | Drop legacy fields; set `<nVectors>4096</nVectors>` |
| `basicmodel/test/test_chunk_layer_bpe.py` | Modify | Rename ChunkLayer args in all constructions; rewrite `test_mm_bpe_config_drives_chunk_layer_flags` |
| `basicmodel/test/test_perceptual_chunking.py` | Modify | Delete 2 raw-specific tests |
| `basicmodel/test/test_perceptualspace_bpe_forward.py` | Add | New end-to-end test file (5 tests) |
