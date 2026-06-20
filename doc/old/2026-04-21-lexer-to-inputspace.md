# Move Lexing to InputSpace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reverse today's lexing-in-PerceptualSpace arrangement. After this refactor, `InputSpace` is a **pure lexer** (byte-stream / word-stream tokenization via `<lexer>` XML param), and `PerceptualSpace` owns the **codebook** (OOV discovery, insertion, index resolution, embedding lookup, chunking via `<chunking>` XML param). The interface between the two is `InputSpace.subspace` with `.what` carrying token text, `.where` carrying byte offsets, `.when` carrying sequential positions.

**Architecture:**
- `InputSpace.forward(input)` in embedding mode: lex → pack tokens as null-terminated UTF-8 bytes into `subspace.what.W` (`[B, N, nWhat]` long tensor), `subspace.where.W = [B, N]` byte offsets, `subspace.when.W = [B, N]` sequential indices. No codebook access, no OOV work, no index resolution, no `_raw_input` stashing.
- `PerceptualSpace.forward(vspace)`: decodes `vspace.what.getW()` byte buffer → tokens, does OOV discovery + insert + index resolution + embedding lookup + chunking on its own codebook, populates its own subspace via `set_forward_content`.

**Byte buffer layout:** Each `what_buf[b, n, :nWhat]` slot holds up to `nWhat-1` UTF-8 bytes of one token, followed by a null (`0`) byte. Tokens longer than `nWhat-1` bytes are truncated. Empty slots (beyond `len(tokens)`) are all-zero (decode to empty string).
- `Embedding` keeps its tokenization methods for its own internal use (`train_step`, `insert`, etc.) — we **don't** deduplicate yet; that's a separate cleanup.

**Architectural principle:** InputSpace is a lexer; it does not touch the codebook. PerceptualSpace owns the codebook and all operations on it (OOV discovery, OOV insertion, index resolution, embedding lookup).

**Tech Stack:** Python 3.12, PyTorch, pytest. Run commands using `basicmodel/.venv/bin/python`. Working directory: `/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel`. All file paths below are relative to that working directory.

---

## Preconditions

- Today's next-percept-decoupling plan is complete (MLM gone, `getBatch` gone, lexicon shim gone, `arir_step` public).
- `_peer_perceptual` back-reference exists on InputSpace (kept in Task 4 of previous plan for exactly this kind of access).
- Full suite passes modulo 3 pre-existing `TestServerQueries` HTTP 500 failures (live-server tests, unrelated).
- User manages commits. **Do not `git commit` without permission.** Stage with `git add` and hand back.

## File Structure

| File | What changes |
|------|--------------|
| `bin/Spaces.py` | `InputSpace`: add `_lex_batch(input)` helper (lex → OOV insert via peer → resolve indices → return `(what_idx, where_idx, when_idx, meta)`); rewrite embedding-mode branch of `InputSpace.forward` to call `_lex_batch` and populate `subspace.set_forward_content(what, where, when)`. `PerceptualSpace`: rename `_lex_and_embed` → `_embed` and simplify it to read the pre-populated upstream subspace (no more tokenization, no more raw-byte path). |
| `test/test_lexicon_ownership.py` | Add tests confirming lexing happens in InputSpace. |
| Test files | Run regression — no rewires expected since public API (forward → subspace) is unchanged. |

## Testing Commands

- Targeted: `.venv/bin/python -m pytest test/test_lexicon_ownership.py test/test_basicmodel.py test/test_mm_xor.py test/test_streaming_ar_training.py -q --tb=short`
- Full: `.venv/bin/python -m pytest test/ -q` — takes ~40 min, expect ~825 passing (3 pre-existing TestServerQueries failures).

---

## Task 1: Add `InputSpace._lex_batch()` — lexing-in-InputSpace primitive

**Context:** Extract only the tokenization step from `Embedding.forward` onto InputSpace. `Embedding.forward` stays intact (for its own internal callers like `train_step`). The new method reaches through `_peer_perceptual.vocabulary._token_stream` to lex (a temporary coupling; later InputSpace can own its own Lex) and packs the resulting tokens as null-terminated UTF-8 bytes. **No** codebook access, **no** OOV discovery/insert, **no** index resolution — those live entirely on PerceptualSpace.

**Files:**
- Modify: `bin/Spaces.py` — add `InputSpace._lex_batch(self, input)` method.
- Append tests to: `test/test_lexicon_ownership.py`.

- [ ] **Step 1: Append a failing test**

```python
def test_inputspace_has_lex_batch():
    from bin import Spaces
    assert hasattr(Spaces.InputSpace, '_lex_batch'), \
        "InputSpace should own lexing via _lex_batch"
```

- [ ] **Step 2: Verify it fails**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_has_lex_batch -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_lex_batch`**

In `bin/Spaces.py`, inside class `InputSpace`, add this method (place it near `prepInput`, around line 4003). It tokenizes via the peer's `_token_stream` (the only peer reach-through) and packs tokens as null-terminated UTF-8 bytes — no codebook access:

```python
def _lex_batch(self, input):
    """Tokenize a raw byte tensor into null-terminated UTF-8 byte slots.

    Pure lexer — no codebook access, no OOV discovery, no index resolution.
    Those all live on PerceptualSpace (see PerceptualSpace._embed).

    Returns: (what_buf, where_idx, when_idx)
      - what_buf: [B, nObj, nWhat] long tensor of UTF-8 bytes, null-terminated.
        Each slot holds one token's bytes followed by a null terminator.
        Tokens longer than nWhat-1 bytes are truncated.
      - where_idx: [B, nObj] long tensor of byte offsets into the source buffer.
      - when_idx: [B, nObj] long tensor of sequential positions.

    Requires self._peer_perceptual to be wired (BasicModel/MentalModel do this)
    because the tokenizer (_token_stream) currently lives on the peer's
    vocabulary. Future: InputSpace owns its own Lex.
    """
    assert self._peer_perceptual is not None, \
        "InputSpace._lex_batch requires _peer_perceptual (lexer owner for now)"
    vocab = self._peer_perceptual.vocabulary
    dev = TheDevice.get()

    if input.dim() == 3:
        input = input.squeeze(1)
    if input.dim() == 1:
        input = input.unsqueeze(0)
    batch = input.shape[0]
    nObj = self.outputShape[0]
    nWhat = self.subspace.nWhat

    what_buf = torch.zeros(batch, nObj, nWhat, dtype=torch.long, device=dev)
    where_idx = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
    when_idx = torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1).contiguous()

    for b in range(batch):
        stream = vocab._token_stream(input[b])
        n_tokens = min(len(stream), nObj)
        for i in range(n_tokens):
            token_text, start = stream[i]
            raw = token_text.encode('utf-8')[: nWhat - 1]  # reserve 1 byte for null
            for j, byte in enumerate(raw):
                what_buf[b, i, j] = byte
            # remaining bytes in slot are already zero (null-terminator + padding)
            where_idx[b, i] = start
        if stream[:n_tokens]:
            last_text, last_start = stream[n_tokens - 1]
            final_offset = last_start + len(last_text.encode('utf-8'))
        else:
            final_offset = 0
        for i in range(n_tokens, nObj):
            where_idx[b, i] = final_offset + (i - n_tokens)

    return what_buf, where_idx, when_idx
```

- [ ] **Step 4: Verify the test passes**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_has_lex_batch -v`
Expected: PASS.

- [ ] **Step 5: Run full lexicon_ownership suite (sanity)**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py -v`
Expected: all PASS.

- [ ] **Step 6: Stage and hand back**
```bash
git add bin/Spaces.py test/test_lexicon_ownership.py
git status
```

Do not commit. Report: "Task 1 complete. `_lex_batch` lives on InputSpace; existing `Embedding.forward` untouched. Ready for Task 2."

---

## Task 2: Switch `InputSpace.forward` embedding mode to populate subspace

**Context:** Today the embedding-mode branch in `InputSpace.forward` just stashes `input` on `self.subspace._raw_input` and returns. That's because tokenization was happening downstream in `PerceptualSpace._lex_and_embed`. After Task 1, `_lex_batch` is available locally. Switch the branch to call it and populate `self.subspace.set_forward_content(what, where, when)`.

**Files:**
- Modify: `bin/Spaces.py` — `InputSpace.forward` embedding branch at ~line 4074.
- Append to: `test/test_lexicon_ownership.py`.

- [ ] **Step 1: Append a failing test**

```python
def test_inputspace_forward_populates_subspace():
    """After lex move: InputSpace.forward in embedding mode populates
    subspace.what (indices) and subspace.where (offsets), NOT _raw_input."""
    from bin import Models, Spaces
    from bin.Config import TheXMLConfig
    import torch
    # Smoke test: a BasicModel in text mode; call inputSpace.forward and
    # check subspace state.
    # (Use existing fixture pattern from test_lexicon_ownership.py's other tests.)
    # This test passes once InputSpace.forward populates subspace instead of _raw_input.
    # Skip shell — flesh out once Task 2 implementation begins, using the
    # TestLexiconOwnership fixture style above.
    import pytest
    pytest.skip("Fleshed out when Task 2 impl lands")
```

Actually: write a direct introspection test instead — it doesn't need a full model:

```python
def test_inputspace_forward_does_not_stash_raw_input():
    """After lex move: embedding-mode InputSpace.forward no longer
    sets subspace._raw_input (lexing happens locally, subspace is
    populated with what/where indices directly)."""
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.InputSpace.forward)
    assert '_raw_input' not in src, \
        "InputSpace.forward should no longer stash _raw_input; " \
        "lexing happens in-space and subspace.what/where are populated directly."
```

- [ ] **Step 2: Verify the test fails**

- [ ] **Step 3: Rewrite the embedding-mode branch in `InputSpace.forward`**

Current code at `bin/Spaces.py:4074-4085`:
```python
        if self.model_type == "embedding":
            # Text mode: the lexicon lives on PerceptualSpace. Stash the
            # raw buffer for PerceptualSpace._lex_and_embed (which runs
            # the embedding forward on the single downstream path) and
            # return. Running the embedding here too would create a
            # second graph through the same Parameter, whose saved
            # tensors leak across training iterations because nothing
            # backward()s through this branch and the references aren't
            # GC'd until the next forward overwrites them.
            self.subspace._raw_input = input
            self._forward_input = None
            return self.subspace
```

Replace with:
```python
        if self.model_type == "embedding":
            # Text mode: InputSpace is a pure lexer. Pack tokens as
            # null-terminated UTF-8 bytes into subspace.what.W, byte
            # offsets into subspace.where.W, sequential positions into
            # subspace.when.W. PerceptualSpace reads these, handles
            # OOV / codebook / chunking on its own codebook.
            what_buf, where_idx, when_idx = self._lex_batch(input)
            self.subspace.whereEncoding.p = 0
            self.subspace.what.setW(what_buf)
            self.subspace.where.setW(where_idx)
            self.subspace.when.setW(when_idx)
            self._forward_input = None
            return self.subspace
```

Key points:
- Uses `setW` on the existing `.what`/`.where`/`.when` Basis slots (no new attributes).
- `self.subspace.whereEncoding.p = 0` preserves the current `forward` behavior at line 4072.
- No codebook access here — PerceptualSpace decodes the byte buffer and does all codebook work.

- [ ] **Step 4: Verify the test passes**

- [ ] **Step 5: Run targeted tests (don't wait for full suite yet; Task 3 needs PerceptualSpace update)**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py -v`

Some tests may fail here because PerceptualSpace still expects `_raw_input`. That's fine — Task 3 fixes it. If ALL lexicon_ownership tests still pass here, great; if some fail referencing `_raw_input`, note and proceed.

- [ ] **Step 6: Stage and hand back**
```bash
git add bin/Spaces.py test/test_lexicon_ownership.py
git status
```

Report: "Task 2 complete. `InputSpace.forward` now lexes and populates subspace. Task 3 must update PerceptualSpace to consume it."

---

## Task 3: Simplify `PerceptualSpace._lex_and_embed` to codebook-only

**Context:** With Task 2 landed, `upstream_vspace.subspace` arrives with `what` (indices) + `where` (offsets) populated. `_lex_and_embed` should no longer tokenize — it just reads those indices, does codebook lookup → vectors, materializes onto its own subspace.

**Files:**
- Modify: `bin/Spaces.py` — `PerceptualSpace._lex_and_embed` at ~line 4503.
- Append to: `test/test_lexicon_ownership.py`.

- [ ] **Step 1: Append a failing test that `_lex_and_embed` no longer reads raw bytes**

```python
def test_perceptualspace_lex_and_embed_reads_subspace_not_raw():
    """After lex move: PerceptualSpace._lex_and_embed reads upstream
    subspace.what (indices) populated by InputSpace, NOT upstream
    subspace._raw_input."""
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.PerceptualSpace._lex_and_embed)
    assert '_raw_input' not in src, \
        "_lex_and_embed should read upstream subspace.what, not _raw_input"
    assert 'vocab.forward' not in src, \
        "_lex_and_embed should not call vocab.forward (that does lexing); " \
        "use codebook lookup on pre-lexed indices instead."
```

- [ ] **Step 2: Verify it fails**

- [ ] **Step 3: Rewrite `_lex_and_embed`**

Current code at `bin/Spaces.py:4503-4561` (roughly — may have drifted; grep `def _lex_and_embed` to locate). Replace with:

```python
def _lex_and_embed(self, upstream_vspace):
    """Decode the upstream null-terminated byte buffer, do codebook work
    (OOV discovery + insert + index resolution), populate this subspace,
    and materialize.

    InputSpace.forward has already populated upstream_vspace with:
      - what.W:  [B, N, nWhat] null-terminated UTF-8 byte buffer
      - where.W: [B, N] byte offsets (long)
      - when.W:  [B, N] sequential positions (long)

    This method owns all codebook operations (InputSpace never touches it).
    """
    what_buf = upstream_vspace.what.getW()
    if what_buf is None:
        raise RuntimeError(
            "PerceptualSpace._lex_and_embed: upstream subspace.what is empty. "
            "InputSpace.forward must lex into subspace.what before "
            "PerceptualSpace.forward runs.")

    vocab = self.subspace.what  # Embedding -- the codebook lives here
    dev = TheDevice.get()
    batch = what_buf.shape[0]
    nObj = self.outputShape[0]

    # Decode byte buffer -> token text per slot. Empty slots (all-zero)
    # decode to empty string.
    batch_tokens = []
    for b in range(batch):
        row = []
        for n in range(what_buf.shape[1]):
            bytes_row = what_buf[b, n].tolist()
            # Truncate at first null
            end = bytes_row.index(0) if 0 in bytes_row else len(bytes_row)
            if end == 0:
                row.append("")
                continue
            text = bytes(bytes_row[:end]).decode('utf-8', errors='replace')
            row.append(text)
        batch_tokens.append(row)

    # OOV discovery + insert on our codebook
    oov_seen = set()
    oov_words = []
    for row in batch_tokens:
        for text in row:
            if (text and text not in vocab.pretrain.key_to_index
                    and text not in oov_seen):
                oov_words.append(text)
                oov_seen.add(text)
    if oov_words and not getattr(vocab, 'byte_mode', False):
        for word in oov_words:
            vocab.insert(word)
        if vocab.optimize_embedding:
            model = getattr(vocab, '_model', None)
            if model is not None:
                model.rebuild_optimizer()

    # Index resolution
    null_idx = vocab.wv.key_to_index.get("\x00", 0)
    what_indices = torch.full((batch, nObj), null_idx, dtype=torch.long, device=dev)
    for b, row in enumerate(batch_tokens):
        for n in range(min(len(row), nObj)):
            if row[n]:
                what_indices[b, n] = vocab._token_to_index(row[n])

    # where / when come straight from the upstream buffer
    where_raw = upstream_vspace.where.getW()
    when_raw = upstream_vspace.when.getW()
    where_indices = where_raw[:, :nObj].long() if (where_raw is not None and self.nWhere > 0) else None
    when_indices = when_raw[:, :nObj].long() if (when_raw is not None and self.nWhen > 0) else (
        torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1) if self.nWhen > 0 else None)

    self.subspace.whereEncoding.p = 0
    self.subspace.set_forward_content(what_indices, where_indices, when_indices)
    self.subspace.normalize("input", target="what", normalize=True)
    self._embedded_input = self.subspace.materialize()
    self._last_tokens = batch_tokens  # for plotting / debug

    return self.subspace
```

Key changes from current:
- No call to `vocab.forward(raw_input, return_meta=True)` — Embedding no longer does tokenization for this path.
- Reads upstream `what.getW()` byte buffer, decodes per-slot tokens.
- OOV discovery + insert + `_token_to_index` all happen here (moved from Embedding.forward).
- Reads upstream `where.getW()` / `when.getW()` for offsets/positions.

- [ ] **Step 4: Verify the test passes**

- [ ] **Step 5: Run targeted tests**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py test/test_basicmodel.py test/test_mm_xor.py -q --tb=short 2>&1 | tail -30`

Likely issues:
- Some test reads `inputSpace._raw_input` → delete that read; what we used to stash is no longer stashed.
- Some test reads `perceptualSpace._last_tokens` → point it to `inputSpace._forward_input['tokens']` instead.
- `_embedded_input` timing differs — PerceptualSpace now materializes in `_lex_and_embed`; check callers aren't reading it before `forward()` is called.

Fix each, or note in concerns if unclear.

- [ ] **Step 6: Run a larger subset**
`.venv/bin/python -m pytest test/test_lexicon_ownership.py test/test_basicmodel.py test/test_mm_xor.py test/test_streaming_ar_training.py test/test_xor_spaces.py -q --tb=short 2>&1 | tail -30`

- [ ] **Step 7: Stage and hand back**
```bash
git add bin/Spaces.py test/test_lexicon_ownership.py test/  # catch any test patches
git status
```

Report: "Task 3 complete. `_lex_and_embed` is codebook-only. Ready for Task 4 regression + cleanup."

---

## Task 4: Full regression + cleanup dead references

**Context:** After Tasks 1-3, no code should read `_raw_input`, call `vocab.forward(raw_input, return_meta=True)`, or depend on PerceptualSpace doing tokenization. Sweep for stragglers and run the full suite.

**Files:**
- Modify: any file with stragglers discovered by grep.
- Run: full pytest.

- [ ] **Step 1: Grep for stragglers**

```bash
grep -rn "_raw_input" bin/ test/ --include="*.py"
grep -rn "vocab.forward" bin/ test/ --include="*.py"
grep -rn "return_meta=True" bin/ test/ --include="*.py"
```

For each hit:
- `_raw_input` setters: delete — nothing stashes raw bytes anymore.
- `_raw_input` readers: delete the branch or replace with `_forward_input` meta read.
- `vocab.forward(raw_input, ...)`: the codebook-lookup path is gone from the main flow; if a test is directly exercising it, leave it (it still works for the Embedding-standalone case). If production code is reading raw bytes, that's a bug — escalate.

- [ ] **Step 2: Rename `_lex_and_embed` → `_embed` (optional but cleaner)**

Now that the method doesn't lex, the name is misleading. Rename `PerceptualSpace._lex_and_embed` → `PerceptualSpace._embed`. Update the one call site in `PerceptualSpace.forward` (around line 4568).

- [ ] **Step 3: Rename `chunking_mode='raw'` → `chunking_mode='cached'`**

In `bin/Spaces.py` `PerceptualSpace.forward`, the current mode list is `lexicon|raw|bpe`. Rename `raw` → `cached` to match the architectural intent (cached BPE-pair lookup). `bpe` and `cached` both still raise `NotImplementedError` — this is just a naming fix.

- [ ] **Step 4: Run full suite**
```bash
.venv/bin/python -m pytest test/ -q --tb=short 2>&1 | tail -15
```
Expected: ~825 passed, 3 pre-existing `TestServerQueries` HTTP 500 failures (unrelated). If any new non-server failures appear, diagnose with grep + fix.

- [ ] **Step 5: Verify final lexicon_ownership pass**
```bash
.venv/bin/python -m pytest test/test_lexicon_ownership.py -v
```
Expected: all PASS (18 from prior plan + new ones from this plan).

- [ ] **Step 6: Stage and hand back**
```bash
git add -A bin/ test/
git status
```

Report: "All tasks complete. Lexing lives in InputSpace; codebook + (future) chunking live in PerceptualSpace. Full suite green modulo 3 pre-existing server failures. Ready for final commit."

---

## Post-refactor state (informational)

**Moved from PerceptualSpace to InputSpace:**
- Tokenization (`_token_stream` — invoked through peer vocabulary reference for now; future cleanup can move the Lex onto InputSpace directly)
- Packing tokens into a null-terminated byte buffer on `subspace.what.W`

**Stayed on PerceptualSpace (codebook is its jurisdiction):**
- Codebook (`Embedding.wv._vectors` — the actual Parameter)
- OOV discovery + insertion (trigger and mechanics)
- Index resolution (`_token_to_index`)
- Embedding lookup (indices → vectors)
- Chunking (lexicon/bpe/cached; bpe and cached still stubbed)
- Attention, VQ, percept composition

**Interface:**
- `InputSpace.subspace.what.W` = `[B, N, nWhat]` null-terminated UTF-8 byte buffer
- `InputSpace.subspace.where.W` = `[B, N]` byte offsets
- `InputSpace.subspace.when.W` = `[B, N]` sequential positions
- `PerceptualSpace.forward` decodes the byte buffer and does all codebook work

**Unchanged:**
- `<lexer>` XML param (already read by InputSpace)
- `<chunking>` XML param (already read by PerceptualSpace)
- `Embedding.forward` (still works standalone for tests / `train_step` / debugging)
- AR modes (ARLM, ARUS, ARIR) — they use `arir_step` / `_ar_buffer`, unaffected
- `prepInput` (still does byte-level string→tensor for legacy batched call sites)
