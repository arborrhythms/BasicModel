# SparseLayer + Conceptual-Embedding Completion — Implementation Plan

> **STATUS: IMPLEMENTED 2026-07-02** (same session, inline execution). All tasks
> landed; execution deviations from the staged sequence are recorded in the
> session report: Tasks 3–5 landed as ONE edit (the legacy encode read the
> `_csw_*` store directly, so a storage-only intermediate would have broken the
> suite); Task 6's tests are unit-level on the leg contract plus the existing
> full-model smoke; `SparseLayer` gained a `device=` ctor arg; helpers in the
> sparse test files now stamp `_n_ps_codes`/`_n_ws_codes` (fail-loud routing
> needs the layout stamps). The todo.md binding block this plan corrects was
> removed by the user mid-implementation; the spec-path fix lives in the
> `_populate_cs_symbols` docstring instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw per-order `torch.sparse` concept transform with a dedicated
`SparseLayer` (tanh forward, transpose reverse, prunable COO store), fix the CS$\to$SS
symbol leg so gradient flows through the 0-D symbol activations, and complete the
Gardenfors-embedding mechanics: closest-link pruning, `.when` consistency, the
statement$\to$edge API on object-concepts, and Hebbian strengthening of the
word$\equiv$object tie.

**Architecture:** A concept's position in conceptual space is COMPOSED from its
definition (sparse weighted combination of its parts/wholes/sub-symbols), per
ramsified order, through two DISTINCT per-order sparse maps: a percept family
(PS+WS $\to$ CS) and a symbol family (SS $\to$ CS). Activations are bounded to
$(-1, 1)$ by tanh; the symbol IS the signed activation (0-D), and the dictionary
atom carries the direction. Codebooks stay EMA-only; gradient trains edge values.

**Tech stack:** PyTorch 2.12 (cpu tests; MPS at runtime), pytest, existing
`bin/Layers.py` / `bin/Spaces.py` / `bin/Language.py`.

---

## Ground rules (project conventions — override any defaults)

- **NO git writes by the implementer.** The user performs ALL git operations.
  Every "commit" checkpoint below means: STOP, report what changed, let the user
  commit. Never run `git add/commit/checkout/stash`.
- **Fail loud:** numerical divergence or contract violations raise; never
  `nan_to_num` or silently gate away.
- **Code comments are one-liners.** Algorithms/architecture belong in `doc/*.md`.
- **Docs use LaTeX math** (`$\to$`, `$\sigma$`, `$\le$`), never Unicode glyphs —
  `make doc` (pandoc/xelatex) fails on missing glyphs.
- **Targeted tests while iterating**; `make test` is the FINAL gate only.
- **Byte-identical gate:** all `symbolicOrder=0` configs (most of the suite) must
  be bit-for-bit unchanged. Sparse-active (`symbolicOrder>=1`, parallel) outputs
  WILL change (tanh + squash) — that is the intended fix, and the sparse tests
  are updated accordingly in Tasks 4–6.
- **Test idiom:**
  `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest <node> -q`
- **iCloud eviction:** if imports fail despite pip "satisfied", re-materialize
  the venv (`pip install --force-reinstall --no-deps torch==2.12.0`, then
  `make install`); run pip with the sandbox disabled.
- **Dataflow rule:** cross-space interaction goes ONLY through `forward()`
  arguments/returns (fields riding a returned SubSpace are the sanctioned
  carrier — precedent: `_bind_carrier`). No stashed cross-space pointers.

## Design summary (what changes and why)

1. **`SparseLayer(Layer)`** (new, `bin/Layers.py`): COO sparse linear map with
   learnable values. `forward(x) = tanh(W @ x)`; `reverse(y) = tanh(W^T @ y)`
   (transpose autoencoder pair — NO LDU inverse, NO atanh entry: inputs are
   presences/activations, not logit-domain codes). Kernel: export-safe
   scatter-add (`index_select` + `index_add_`) by default; `kernel="spmm"`
   opt-in (`torch.sparse.mm`, COO on MPS). Supports `add_edge` (idempotent),
   `remove_edges` (for pruning), tail-preserving value growth.
2. **Two families per order** on ConceptualSpace, replacing `_csw_*`:
   `_sparse_percept[k]`: `[n_ps+n_ws] -> caps[k]`; `_sparse_symbol[k]` (k>=1):
   `[sum(caps[:k])] -> caps[k]`. `add_concept_weight` keeps its signature and
   routes by source block. Combine pre-tanh: `a_k = tanh(lin_p + lin_s)`.
3. **Presence squash:** `_subspace_presence` returns `tanh(presence)` so the
   percept-family input is in `[0, 1)` (raw presences are unbounded sums;
   without the squash every presence $\ge 1$ would saturate identically).
4. **Symbol leg fix** (the grad bug): `_sparse_concept_forward` returns the
   per-order activations; `ConceptualSpace.forward` stamps them on the returned
   subspace as `_concept_activations` `[N, B]` (grad-bearing).
   `SymbolSpace.forward_concept_to_symbol` builds the leg as
   `activation x SS-row` — gradient flows through the activation; the SS
   codebook row sync stays `no_grad` (EMA-only identity). Fallback (no
   activations stamped) keeps today's detached-copy behavior.
5. **Closest-link pruning** (`prune_concept_links`): keep the maximal part and
   the minimal whole among a concept's links, using WITHIN-tower relations only
   (never cross-tower): drop `_WORD_CLASS` when a specific whole is linked;
   drop a whole that is a taxonomy ancestor of another linked whole; drop raw
   parts that are constituents of a linked raised symbol H. Runs at the tail of
   `refine_over_collected` (host-side, gated) — mint stays unconstrained.
6. **`.when` consistency check:** at the tie site, when `.when` tensors are
   available, all percepts knit into one location-concept must share `.when`;
   mismatch raises (fail-loud). No-op when unavailable.
7. **A-merge:** the per-span location concept IS the word A-symbol when the
   span's surface text is known (reuse `_word_obj_meta[key]`'s A), and the
   specific word-whole is added alongside `_WORD_CLASS` (pruning later removes
   the generic).
8. **Statement$\to$edge API:** `assert_concept_relation(cid, part=..., whole=...,
   sym_part=..., sym_whole=..., weight=...)` refines object-concepts (B): first
   concrete part removes the `_ATOM` pole, first concrete whole removes the
   `_UNIVERSE` pole; edges re-populate, so B enters the embedding exactly when
   its definition first has content.
9. **Hebbian strengthen:** re-minting a word (key hit in
   `create_word_object_meta`) strengthens C's edge values (`no_grad`,
   `eta=0.1`, clamp `[-4, 4]`). Weakening (dis-occurrence) is DEFERRED — in a
   text-only learner word and object always co-mint, so it is vacuous today.

**Explicitly deferred (documented, NOT in this plan):** NL wiring of
`assert_concept_relation` (chart/`RelativeTruthStore` $\to$ edges at high
sentence confidence), serial word$\leftrightarrow$object substitution around
compose, Hebbian weakening, symbol `.where` bounding-span stamping (the
Mereology.md 2026-06-29 REVISIT), byte-pid vs word-pid part pruning (needs PS
promotion provenance).

## Verified anchors (2026-07-02; lines shift as you edit — re-grep, trust names)

| Anchor | Location |
|---|---|
| `class Layer(nn.Module)` (`nInput/nOutput`, `self.layers`, `getParameters`) | `bin/Layers.py:686` |
| `class SigmaLayer2(Layer)` (pattern for a non-Grammar substrate layer) | `bin/Layers.py:4384` |
| `class ConceptualSpace(Space)` | `bin/Spaces.py:12230` |
| `_csw_tables/_grow_csw_values/add_concept_weight/concept_weights` | `bin/Spaces.py:13834-13910` |
| `_build_csw` / `cs_sparse_encode` (to retire) | `bin/Spaces.py:13912-13985` |
| `getParameters` / `_maybe_rebuild_optimizer_for_csw` | `bin/Spaces.py:13939-13967` |
| `source_code_activation` / `_subspace_presence` | `bin/Spaces.py:13987-14185` |
| `cs_decode` / `cs_source_layout` / `cs_forward_content` | `bin/Spaces.py:14012-14073` |
| `_concept_source_order` / `_csw_concept_row` / `_populate_concept_weights` | `bin/Spaces.py:14089-14171` |
| `_sparse_concept_forward` | `bin/Spaces.py:14187` |
| `_populate_cs_symbols` (the knit; `_WORD_CLASS` whole) | `bin/Spaces.py:13487` |
| `_autobind_cross_tower` (S2a/S2b/S2c lifecycle caller) | `bin/Spaces.py:13444` |
| `_maybe_autobind_meta` (word_texts available here) | `bin/Spaces.py:13161` |
| `create_word_object_meta` (A/B/C; key-hit re-mint path) | `bin/Spaces.py:14271` |
| `resolve_identities` / `synthesize_higher_order` / `_concept_raise_set` | `bin/Spaces.py:13642-13725` |
| `refine_over_collected` (pruning hook site at its tail) | `bin/Spaces.py:14223` |
| `ConceptualSpace.forward` (calls `_sparse_concept_forward`; returns `subspace`) | `bin/Spaces.py:15634`, call at `:15775`, return at `:15901` |
| `bind_streams` SS leg fit | `bin/Spaces.py:15522`, SS fit `:15576-15596` |
| `SymbolSpace.forward_concept_to_symbol` | `bin/Language.py:12734` |
| Model stamps `_symbolic_order/_serial/_n_ps_codes/_n_ws_codes` | `bin/Models.py:875-895` |
| Parallel body: `cs.forward` then `forward_concept_to_symbol` then `bind_streams` | `bin/Models.py:6537-6712` |
| Sparse tests to update | `test/test_cs_sparse_weights.py`, `test/test_sparse_concept_e2e.py`, `test/test_cs_to_ss_forward.py` |
| Relation/lifecycle tests to extend | `test/test_cs_symbol_table.py`, `test/test_mereology_word_binding.py` |

Note: the Phase-1/2 edge store (`add_edge`/`scatter_concept_event`/`_edge_*`)
was ALREADY retired from `bin/Spaces.py` — only the `_csw_*` store exists. The
2026-06-27 handoff doc is stale on this point (Task 13 records that).

---

### Task 0: Baseline

**Files:** none (verification only).

- [ ] **Step 0.1:** Confirm the venv imports (iCloud eviction check):

Run: `cd <repo> && BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -c "import torch, Spaces, Layers, Language, Models; print(torch.__version__)"`
Expected: `2.12.0` (or current). If ModuleNotFoundError: re-materialize per Ground rules.

- [ ] **Step 0.2:** Baseline the targeted suites this plan touches:

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py test/test_cs_to_ss_forward.py test/test_cs_symbol_table.py test/test_mereology_word_binding.py test/test_sparse_concept_e2e.py -q`
Expected: all pass. Record the counts; these are the regression baseline.

### Task 1: SparseLayer core (storage, growth, forward)

**Files:**
- Modify: `bin/Layers.py` (append after `SigmaLayer2`, near end of class section)
- Create: `test/test_sparse_layer.py`

- [ ] **Step 1.1: Write the failing tests**

Create `test/test_sparse_layer.py`:

```python
"""SparseLayer: COO sparse linear substrate (tanh forward / transpose reverse)."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
from Layers import SparseLayer  # noqa: E402


def _dense(layer):
    # Dense [nOutput, nInput] equivalent of the layer's COO store.
    W = torch.zeros(layer.nOutput, layer.nInput)
    for (r, c), i in layer._index.items():
        W[r, c] = layer.values[i]
    return W


def test_add_edge_idempotent_and_grows_values():
    ly = SparseLayer(4, 3)
    i0 = ly.add_edge(0, 1, weight=0.5)
    i1 = ly.add_edge(0, 1, weight=9.9)      # dup: same slot, weight kept
    assert i0 == i1 and ly.nnz == 1
    assert float(ly.values[i0]) == 0.5
    ly.add_edge(2, 3, weight=-1.0)
    assert ly.nnz == 2 and ly.values.shape == (2,)


def test_forward_matches_dense_tanh():
    torch.manual_seed(0)
    ly = SparseLayer(5, 3)
    for (r, c, w) in [(0, 0, 0.3), (0, 4, -0.7), (1, 2, 1.2), (2, 1, -0.2)]:
        ly.add_edge(r, c, weight=w)
    x = torch.rand(5, 2)
    got = ly.forward(x)
    want = torch.tanh(_dense(ly) @ x)
    assert torch.allclose(got, want, atol=1e-6)
    assert got.min() > -1.0 and got.max() < 1.0


def test_forward_linear_no_tanh_and_1d_squeeze():
    ly = SparseLayer(3, 2, nonlinear=False)
    ly.add_edge(1, 0, weight=2.0)
    x = torch.tensor([3.0, 0.0, 0.0])
    out = ly.forward(x)                      # [2]
    assert out.shape == (2,) and float(out[1]) == 6.0


def test_empty_layer_returns_zeros():
    ly = SparseLayer(4, 3)
    out = ly.forward(torch.rand(4, 2))
    assert out.shape == (3, 2) and torch.all(out == 0)


def test_grow_preserves_trained_values():
    ly = SparseLayer(4, 3)
    ly.add_edge(0, 0, weight=1.0)
    with torch.no_grad():
        ly.values[0] = 0.123                 # simulate training
    ly.add_edge(1, 1, weight=1.0)            # growth must keep the tail
    assert float(ly.values[0]) == 0.123 and ly.nnz == 2


def test_forward_differentiable_in_values_and_input():
    ly = SparseLayer(3, 2)
    ly.add_edge(0, 0, weight=0.5)
    ly.add_edge(1, 2, weight=-0.5)
    x = torch.rand(3, 1, requires_grad=True)
    ly.forward(x).sum().backward()
    assert ly.values.grad is not None and x.grad is not None
    assert torch.any(ly.values.grad != 0) and torch.any(x.grad != 0)
```

- [ ] **Step 1.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_sparse_layer.py -q`
Expected: FAIL — `ImportError: cannot import name 'SparseLayer'`.

- [ ] **Step 1.3: Implement SparseLayer (minimal: storage + forward)**

Append to `bin/Layers.py` (after `SigmaLayer2`'s section):

```python
class SparseLayer(Layer):
    """COO sparse linear substrate: forward tanh(W @ x), reverse tanh(W^T @ y).

    The percept/symbol -> concept map of the ramsified conceptual space
    (doc/plans/2026-07-02-sparse-layer-conceptual-embedding.md). Edges are
    appended host-side as concepts mint (add_edge, idempotent) and removed by
    pruning rounds (remove_edges); values are ONE learnable Parameter grown
    tail-preserving so trained weights survive growth. No LDU inverse: the
    reverse is the transpose (sparse autoencoder pair). No atanh entry: inputs
    are presences [0,1) / activations (-1,1), not logit-domain codes.
    Kernel "scatter" (index_select + index_add_) is export-safe (MLX /
    executorch); "spmm" (torch.sparse.mm, COO kept on MPS) is the opt-in
    fast path. Empty layer -> zeros (both directions).
    """

    def __init__(self, nInput, nOutput, nonlinear=True, kernel="scatter"):
        super().__init__(nInput, nOutput)
        if kernel not in ("scatter", "spmm"):
            raise ValueError(f"SparseLayer: unknown kernel {kernel!r}")
        self.nonlinear = nonlinear
        self.kernel = kernel
        self._rows = []            # host COO row (output) indices
        self._cols = []            # host COO col (input) indices
        self._init_vals = []       # host init weights (growth tail source)
        self._index = {}           # (row, col) -> position in values
        self.values = None         # nn.Parameter [nnz]
        self._dev_cache = None     # (device, rows_t, cols_t), dirty on edit

    @property
    def nnz(self):
        return len(self._rows)

    def _device(self):
        return self.values.device if self.values is not None \
            else torch.device("cpu")

    def _grow_values(self):
        # Tail-preserving: keep trained prefix, init the new tail.
        n = self.nnz
        if n == 0:
            return
        prev = self.values
        if prev is not None and int(prev.shape[0]) == n:
            return
        dev = self._device()
        new = torch.tensor(self._init_vals, dtype=torch.float32, device=dev)
        if prev is not None:
            keep = min(int(prev.shape[0]), n)
            if keep > 0:
                with torch.no_grad():
                    new[:keep] = prev.detach()[:keep].to(dev)
        self.values = nn.Parameter(new)
        self._dev_cache = None

    def add_edge(self, row, col, weight=1.0):
        """Append edge output[row] <- weight * input[col]; idempotent."""
        r, c = int(row), int(col)
        if not (0 <= r < self.nOutput and 0 <= c < self.nInput):
            raise IndexError(
                f"SparseLayer.add_edge out of range: ({r},{c}) for "
                f"[{self.nOutput},{self.nInput}]")
        pos = self._index.get((r, c))
        if pos is not None:
            return pos
        pos = self.nnz
        self._rows.append(r)
        self._cols.append(c)
        self._init_vals.append(float(weight))
        self._index[(r, c)] = pos
        self._grow_values()
        return pos

    def _indices(self, device):
        # Cached long index tensors per device; rebuilt when edited.
        cache = self._dev_cache
        if cache is not None and cache[0] == device:
            return cache[1], cache[2]
        rows_t = torch.tensor(self._rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(self._cols, dtype=torch.long, device=device)
        self._dev_cache = (device, rows_t, cols_t)
        return rows_t, cols_t

    def _matmul(self, x, transpose=False):
        # Sparse matmul core; x is [n_in, B]; returns [n_out, B].
        n_from = self.nInput if not transpose else self.nOutput
        n_to = self.nOutput if not transpose else self.nInput
        if int(x.shape[0]) != n_from:
            raise ValueError(
                f"SparseLayer: input rows {int(x.shape[0])} != {n_from}")
        if self.nnz == 0 or self.values is None:
            return x.new_zeros((n_to, int(x.shape[-1])))
        rows_t, cols_t = self._indices(x.device)
        take, put = (cols_t, rows_t) if not transpose else (rows_t, cols_t)
        vals = self.values.to(x.device)
        if self.kernel == "spmm":
            idx = torch.stack([put, take])
            W = torch.sparse_coo_tensor(
                idx, vals, (n_to, n_from)).coalesce()
            if x.device.type != "mps":       # MPS has no COO->CSR kernel
                W = W.to_sparse_csr()
            return torch.sparse.mm(W, x)
        contrib = x.index_select(0, take) * vals.unsqueeze(-1)
        out = x.new_zeros((n_to, int(x.shape[-1])))
        return out.index_add_(0, put, contrib)

    def _apply_shape(self, x, transpose):
        squeeze = x.dim() == 1
        out = self._matmul(x.unsqueeze(-1) if squeeze else x, transpose)
        if self.nonlinear:
            out = torch.tanh(out)
        return out.squeeze(-1) if squeeze else out

    def forward_linear(self, x):
        """W @ x with NO nonlinearity (for pre-tanh summation of families)."""
        squeeze = x.dim() == 1
        out = self._matmul(x.unsqueeze(-1) if squeeze else x, False)
        return out.squeeze(-1) if squeeze else out

    def forward(self, x):
        return self._apply_shape(x, transpose=False)

    def reverse(self, y):
        """Transpose decode: tanh(W^T @ y) (the sparse autoencoder pair)."""
        return self._apply_shape(y, transpose=True)
```

- [ ] **Step 1.4: Run to verify pass**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_sparse_layer.py -q`
Expected: 6 passed.

- [ ] **Step 1.5: CHECKPOINT — report to user (SparseLayer core) for optional commit.**

### Task 2: SparseLayer reverse, removal/compaction, kernel parity

**Files:**
- Modify: `bin/Layers.py` (SparseLayer)
- Modify: `test/test_sparse_layer.py`

- [ ] **Step 2.1: Write the failing tests** (append to `test/test_sparse_layer.py`):

```python
def test_reverse_matches_dense_transpose_tanh():
    torch.manual_seed(1)
    ly = SparseLayer(5, 3)
    for (r, c, w) in [(0, 0, 0.3), (1, 2, 1.2), (2, 1, -0.2), (2, 4, 0.9)]:
        ly.add_edge(r, c, weight=w)
    y = torch.rand(3, 2)
    got = ly.reverse(y)
    want = torch.tanh(_dense(ly).t() @ y)
    assert torch.allclose(got, want, atol=1e-6)


def test_remove_edges_compacts_and_preserves_survivors():
    ly = SparseLayer(4, 3)
    ly.add_edge(0, 0, weight=0.1)
    ly.add_edge(1, 1, weight=0.2)
    ly.add_edge(2, 2, weight=0.3)
    with torch.no_grad():
        ly.values[1] = 0.999                 # train the survivor
    ly.remove_edges([(0, 0), (2, 2)])
    assert ly.nnz == 1 and (1, 1) in ly._index
    assert float(ly.values[ly._index[(1, 1)]]) == 0.999
    ly.remove_edges([(1, 1)])
    assert ly.nnz == 0
    assert torch.all(ly.forward(torch.rand(4, 2)) == 0)


def test_spmm_kernel_parity():
    torch.manual_seed(2)
    a = SparseLayer(6, 4, kernel="scatter")
    b = SparseLayer(6, 4, kernel="spmm")
    for (r, c, w) in [(0, 5, 0.4), (3, 0, -1.1), (2, 2, 0.7)]:
        a.add_edge(r, c, weight=w)
        b.add_edge(r, c, weight=w)
    x = torch.rand(6, 3)
    assert torch.allclose(a.forward(x), b.forward(x), atol=1e-6)
    assert torch.allclose(a.reverse(x[:4] * 0.5), b.reverse(x[:4] * 0.5),
                          atol=1e-6)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `BASICMODEL_DEPTH= BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_sparse_layer.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'remove_edges'` (reverse/parity may already pass).

- [ ] **Step 2.3: Implement `remove_edges`** (add method to SparseLayer):

```python
    def remove_edges(self, pairs):
        """Drop (row, col) edges (pruning rounds); compact values, keep survivors."""
        drop = {(int(r), int(c)) for (r, c) in pairs} & set(self._index)
        if not drop:
            return 0
        keep = [i for i in range(self.nnz)
                if (self._rows[i], self._cols[i]) not in drop]
        old_vals = self.values
        self._rows = [self._rows[i] for i in keep]
        self._cols = [self._cols[i] for i in keep]
        self._init_vals = [self._init_vals[i] for i in keep]
        self._index = {(r, c): i
                       for i, (r, c) in enumerate(zip(self._rows, self._cols))}
        if not keep:
            self.values = None
        else:
            with torch.no_grad():
                kept = old_vals.detach()[torch.tensor(keep)].clone()
            self.values = nn.Parameter(kept.to(self._device()
                                               if old_vals is None
                                               else old_vals.device))
        self._dev_cache = None
        return len(drop)
```

- [ ] **Step 2.4: Run to verify pass**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_sparse_layer.py -q`
Expected: 9 passed.

- [ ] **Step 2.5: CHECKPOINT — report to user (SparseLayer complete).**

### Task 3: ConceptualSpace stores edges in SparseLayer families

**Files:**
- Modify: `bin/Spaces.py` (`_csw_tables` area, `add_concept_weight`,
  `concept_weights`, `getParameters`, `_maybe_rebuild_optimizer_for_csw`)
- Modify: `test/test_cs_sparse_weights.py`

The public API (`add_concept_weight(order, concept_local, source_global,
weight)`, `concept_weights(order, concept_local)`) is KEPT — storage and
compute move into the two per-order SparseLayer families. The `[PS|WS|SS...]`
global source indexing is preserved at the API surface; internally cols split
at `n_ps + n_ws`.

- [ ] **Step 3.1: Read the current test helper** (fixture name/shape):

Run: `sed -n '1,45p' test/test_cs_sparse_weights.py`
Expected: a `_cs(...)`-style helper constructing a bare ConceptualSpace. Reuse
it in the new tests below (adjust the helper name if it differs).

- [ ] **Step 3.2: Write the failing test** (append to `test/test_cs_sparse_weights.py`):

```python
def test_family_layers_split_percept_vs_symbol_sources():
    cs = _cs(16, symbolic_order=2)           # helper: sparse-active CS
    n_ps = int(cs._n_ps_codes)
    n_ws = int(cs._n_ws_codes)
    # PS/WS-block edge -> percept family; SS-block edge -> symbol family.
    cs.add_concept_weight(1, 0, 2)                    # a PS source
    cs.add_concept_weight(1, 0, n_ps + 1)             # a WS source
    cs.add_concept_weight(1, 0, n_ps + n_ws + 0)      # an SS_0 source
    p, s = cs._sparse_families(1)
    assert p.nnz == 2 and s.nnz == 1
    # Merged query view keeps the global [PS|WS|SS...] column indexing.
    cols = [c for (c, w) in cs.concept_weights(1, 0)]
    assert cols == [2, n_ps + 1, n_ps + n_ws + 0]
    # The learnable values surface through getParameters (optimizer pickup).
    ids = {id(t) for t in cs.getParameters()}
    assert id(p.values) in ids and id(s.values) in ids
```

- [ ] **Step 3.3: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py::test_family_layers_split_percept_vs_symbol_sources -q`
Expected: FAIL — no `_sparse_families`.

- [ ] **Step 3.4: Implement the family store.** In `bin/Spaces.py`, next to
`_csw_tables` (keep `_csw_rows` / `_csw_row_next` / `_csw_concept_row` — the
concept-row allocator is unchanged), add:

```python
    def _sparse_families(self, order):
        """The (percept, symbol) SparseLayers of ramsified ``order`` (lazy)."""
        fams = getattr(self, "_sparse_fam", None)
        if fams is None:
            fams = {}
            object.__setattr__(self, "_sparse_fam", fams)
        o = int(order)
        got = fams.get(o)
        if got is None:
            from Layers import SparseLayer
            caps = self._order_caps()
            n_ps = int(getattr(self, "_n_ps_codes", 0) or 0)
            n_ws = int(getattr(self, "_n_ws_codes", 0) or 0)
            n_ss = sum(int(c) for c in caps[:o])
            percept = SparseLayer(max(1, n_ps + n_ws), int(caps[o]))
            symbol = (SparseLayer(max(1, n_ss), int(caps[o]))
                      if o > 0 else None)
            got = (percept, symbol)
            fams[o] = got
        return got
```

Rewrite `add_concept_weight` (same signature, route by block):

```python
    def add_concept_weight(self, order, concept_local, source_global, weight=1.0):
        """Append a sparse weight edge ``concept_local <- weight * source_global``
        for ramsified ``order``; ``source_global`` indexes the stacked
        ``[PS|WS|SS_0..SS_{order-1}]`` layout. Routes to the percept family
        (PS/WS block) or the symbol family (SS block). Idempotent per edge."""
        o = int(order)
        split = (int(getattr(self, "_n_ps_codes", 0) or 0)
                 + int(getattr(self, "_n_ws_codes", 0) or 0))
        percept, symbol = self._sparse_families(o)
        s = int(source_global)
        if s < split:
            return percept.add_edge(int(concept_local), s, weight=weight)
        if symbol is None:
            raise IndexError(
                f"add_concept_weight: SS source {s} at order 0 (no symbol family)")
        return symbol.add_edge(int(concept_local), s - split, weight=weight)
```

Rewrite `concept_weights` as the merged view:

```python
    def concept_weights(self, order, concept_local):
        """The ``(source_global, weight)`` entries of ``concept_local`` at
        ``order`` in the stacked layout, deterministically ordered by source."""
        o = int(order)
        fams = getattr(self, "_sparse_fam", None) or {}
        got = fams.get(o)
        if got is None:
            return []
        percept, symbol = got
        split = (int(getattr(self, "_n_ps_codes", 0) or 0)
                 + int(getattr(self, "_n_ws_codes", 0) or 0))
        c = int(concept_local)
        out = [(col, float(percept.values[i]))
               for (row, col), i in percept._index.items() if row == c]
        if symbol is not None:
            out += [(split + col, float(symbol.values[i]))
                    for (row, col), i in symbol._index.items() if row == c]
        return sorted(out, key=lambda sw: sw[0])
```

Update `getParameters` and the optimizer-rebuild counter to read the families:

```python
    def getParameters(self):
        """Optimizable parameters: the inherited set PLUS the per-order sparse
        family values (created host-side as edges are added, so the optimizer
        trains them). With no edges this is exactly the inherited set."""
        base = list(self.params)
        for (p, s) in (getattr(self, "_sparse_fam", None) or {}).values():
            for ly in (p, s):
                if ly is not None and ly.values is not None:
                    base.append(ly.values)
        return base

    def _maybe_rebuild_optimizer_for_csw(self):
        """Rebuild the model optimizer when the sparse edge count changed, so
        fresh family values become trainable. Debounced; no-op unwired."""
        model = getattr(self, "_model", None)
        if model is None or getattr(model, "_optimizer", None) is None:
            return
        n = sum((p.nnz + (s.nnz if s is not None else 0))
                for (p, s) in (getattr(self, "_sparse_fam", None) or {}).values())
        if n != int(getattr(self, "_csw_registered_count", -1)):
            object.__setattr__(self, "_csw_registered_count", n)
            try:
                model.rebuild_optimizer()
            except Exception:
                pass
```

Delete `_csw_tables`, `_csw_device`, `_grow_csw_values` and the
`_csw_store/_csw_index/_csw_vals` state they managed. Keep `_build_csw` /
`cs_sparse_encode` untouched for now (Task 5 retires them after the forward
moves off them; they must not reference the deleted state — if `_build_csw`
breaks compilation of remaining tests before Task 5, stub it to `return None`
with a one-liner noting Task 5 removes it).

- [ ] **Step 3.5: Fix the storage-level tests in `test_cs_sparse_weights.py`.**
`test_add_concept_weight_dedup_and_query` should pass unchanged (API kept).
`test_csw_rebuild_preserves_trained_weights_on_grow` re-targets the family
Parameter — update its body to train `cs._sparse_families(o)[0].values[...]`
and assert survival after a further `add_concept_weight`, mirroring
`test_grow_preserves_trained_values` in `test/test_sparse_layer.py`.

- [ ] **Step 3.6: Run the storage tests**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py -q`
Expected: storage/population tests pass; `test_sparse_encode_*`,
`test_encoder_uses_torch_sparse`, `test_forward_content_*` may still pass via
the untouched legacy encode — all green here means proceed.

- [ ] **Step 3.7: CHECKPOINT — report to user.**

### Task 4: cs_forward_content on the families (tanh + squash)

**Files:**
- Modify: `bin/Spaces.py` (`cs_forward_content`, `_subspace_presence`)
- Modify: `test/test_cs_sparse_weights.py`

- [ ] **Step 4.1: Update the semantic tests.** In
`test/test_cs_sparse_weights.py`:

Replace `test_sparse_encode_matches_dense` with:

```python
def test_forward_content_activation_is_bounded_tanh():
    cs = _cs(16, symbolic_order=1)
    cs.add_concept_weight(0, 0, 0, weight=3.0)        # big weight saturates
    n_src = int(cs._n_ps_codes) + int(cs._n_ws_codes)
    ps = torch.full((int(cs._n_ps_codes), 2), 5.0)    # raw unbounded presence
    ws = torch.zeros((int(cs._n_ws_codes), 2))
    dictionary = torch.randn(int(cs.nVectors), 8)
    content, a_list = cs.cs_forward_content(torch.tanh(ps), torch.tanh(ws),
                                            dictionary)
    for a in a_list:
        assert a.min() >= -1.0 and a.max() <= 1.0     # tanh-bounded
    # Order-0 concept 0 fires: tanh(3 * tanh(5)) close to 1, not 15.
    assert 0.9 < float(a_list[0][0, 0]) <= 1.0
```

Replace `test_encoder_uses_torch_sparse` with:

```python
def test_forward_content_kernels_agree():
    cs = _cs(16, symbolic_order=1)
    cs.add_concept_weight(0, 0, 1, weight=0.7)
    cs.add_concept_weight(1, 2, int(cs._n_ps_codes) + 0, weight=-0.4)
    ps = torch.rand(int(cs._n_ps_codes), 2).tanh()
    ws = torch.rand(int(cs._n_ws_codes), 2).tanh()
    d = torch.randn(int(cs.nVectors), 8)
    c1, a1 = cs.cs_forward_content(ps, ws, d)
    for (p, s) in cs._sparse_fam.values():            # flip kernels
        for ly in (p, s):
            if ly is not None:
                ly.kernel = "spmm"
    c2, a2 = cs.cs_forward_content(ps, ws, d)
    assert torch.allclose(c1, c2, atol=1e-6)
```

Keep `test_forward_content_shape_and_stacking`,
`test_forward_content_differentiable`,
`test_forward_content_positive_atoms_and_signed_activation` — update their
expectations only if they assert exact unbounded values (bounded now);
`test_sparse_encode_differentiable_in_weights_and_activation` and
`test_sparse_encode_empty_order_is_zero` re-target
`cs.cs_forward_content` (empty order still yields zero activations:
`tanh(0) = 0`).

- [ ] **Step 4.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py -q`
Expected: the two new/changed tests FAIL (old linear semantics).

- [ ] **Step 4.3: Rewrite `cs_forward_content`** (replace the loop body; keep
signature and `(content, a_list)` return):

```python
    def cs_forward_content(self, ps_act, ws_act, dictionary):
        """Ramsified per-order sparse encode/decode over the family layers.

        ``ps_act`` / ``ws_act`` are squashed presences in ``[0, 1)``; order
        ``k``'s signed activation is ``a_k = tanh(P_k @ [ps|ws] + S_k @
        [a_0..a_{k-1}])`` -- percept + symbol families summed PRE-tanh, so the
        symbol (0-D, signed, bounded) composes across orders. The concept code
        is the activation scaling the strictly-positive atom (radial:
        magnitude = certainty, sign = present vs anti-present). Returns
        ``(content [B, N, ConceptDim], [a_0..a_S])``."""
        caps = self._order_caps()
        atoms = F.softplus(dictionary)
        a_list = []
        slabs = []
        source_p = torch.cat([ps_act, ws_act], dim=0)
        for k in range(len(caps)):
            percept, symbol = self._sparse_families(k)
            lin = percept.forward_linear(source_p)
            if symbol is not None and symbol.nnz > 0 and k > 0:
                lin = lin + symbol.forward_linear(torch.cat(a_list[:k], dim=0))
            a_k = torch.tanh(lin)
            a_list.append(a_k)
            slabs.append(self.cs_decode(k, a_k, atoms))
        content = torch.cat(slabs, dim=1)
        return content, a_list
```

- [ ] **Step 4.4: Squash the presences.** In `_subspace_presence`, change the
final return to:

```python
        # Squash to [0, 1): raw presences are unbounded sums and would
        # saturate the tanh fold identically for every presence >= 1.
        act = self.source_code_activation(event, W, nonneg=True)
        return None if act is None else torch.tanh(act)
```

(`source_code_activation` itself is unchanged — its unit tests keep passing.)

- [ ] **Step 4.5: Run the suite**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py test/test_sparse_concept_e2e.py -q`
Expected: test_cs_sparse_weights green. If `test_sparse_concept_e2e.py` asserts
exact legacy magnitudes, update those assertions to bounded-range checks
(activations in $(-1,1)$; gradient reaching family values) — the e2e test's
INTENT (grad + shape + non-degenerate) is preserved.

- [ ] **Step 4.6: CHECKPOINT — report to user.**

### Task 5: Retire the legacy encode (`_build_csw` / `cs_sparse_encode`)

**Files:**
- Modify: `bin/Spaces.py`

- [ ] **Step 5.1: Find remaining consumers**

Run: `grep -rn "cs_sparse_encode\|_build_csw\|_csw_store\|_csw_vals\|_csw_index" bin/ test/ --include=*.py`
Expected: only `_sparse_concept_forward`'s `_csw_store` gate check (and
possibly stale comments). Update that gate to the families:

```python
        fams = getattr(self, "_sparse_fam", None) or {}
        if not any(p.nnz or (s.nnz if s is not None else 0)
                   for (p, s) in fams.values()):
            return folded                        # no edges -> fallback
```

- [ ] **Step 5.2: Delete** `_build_csw` and `cs_sparse_encode` (their MPS
COO/CSR handling now lives in `SparseLayer._matmul`). Fix any comment
references found in Step 5.1.

- [ ] **Step 5.3: Run the touched suites + a broad smoke**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_sparse_weights.py test/test_sparse_concept_e2e.py test/test_cs_symbol_table.py test/test_mereology_word_binding.py -q`
Expected: all pass.

- [ ] **Step 5.4: CHECKPOINT — report to user.**

### Task 6: Grad-preserving 0-D symbol leg

**Files:**
- Modify: `bin/Spaces.py` (`_sparse_concept_forward`, `ConceptualSpace.forward`)
- Modify: `bin/Language.py` (`forward_concept_to_symbol`)
- Modify: `test/test_cs_to_ss_forward.py`

- [ ] **Step 6.1: Write the failing test** (append to
`test/test_cs_to_ss_forward.py`; reuse the file's existing fixtures for a
sparse-active CS + SymbolSpace — read its helpers first:
`sed -n '1,44p' test/test_cs_to_ss_forward.py`):

```python
def test_symbol_leg_is_activation_times_row_and_carries_grad():
    # Build a sparse-active CS with one populated order-0 concept, run
    # cs.forward so _concept_activations is stamped, then build the leg.
    cs, ss, sub = _sparse_active_cs_with_event()   # adapt to file fixtures
    out_sub = cs.forward(sub)
    acts = getattr(out_sub, "_concept_activations", None)
    assert acts is not None and acts.requires_grad
    leg = ss.forward_concept_to_symbol(out_sub)
    assert leg is not None
    # Gradient reaches the sparse family values THROUGH the leg.
    leg.materialize().sum().backward()
    p, _ = cs._sparse_families(0)
    assert p.values.grad is not None and torch.any(p.values.grad != 0)


def test_symbol_leg_fallback_without_activations_stays_detached():
    cs, ss, sub = _plain_cs_with_event()           # sparse-inactive path
    out_sub = cs.forward(sub)
    assert getattr(out_sub, "_concept_activations", None) is None
    leg = ss.forward_concept_to_symbol(out_sub)
    if leg is not None:                            # None contract preserved
        assert not leg.materialize().requires_grad
```

(Write `_sparse_active_cs_with_event` / `_plain_cs_with_event` as small local
helpers modeled on the file's existing smoke-test setup — same config objects,
one with `symbolicOrder>=1` + `serial=false` + one populated concept weight,
one default.)

- [ ] **Step 6.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_to_ss_forward.py -q -k symbol_leg`
Expected: FAIL — `_concept_activations` never stamped.

- [ ] **Step 6.3: Return + stamp the activations.**
In `_sparse_concept_forward`, return a pair on ALL paths: every early
`return folded` becomes `return folded, None`; the live tail becomes:

```python
        content, a_list = self.cs_forward_content(ps_act, ws_act, dict_W)
        acts = torch.cat(a_list, dim=0)          # [N, B] signed, grad-bearing
        N, D = int(folded.shape[1]), int(folded.shape[2])
        if int(content.shape[1]) != N:
            return folded, None
        Dc = int(content.shape[2])
        if Dc == D:
            return content, acts
        return (content[..., :D] if Dc > D
                else F.pad(content, (0, D - Dc))), acts
```

In `ConceptualSpace.forward` (the call site at ~`:15775`):

```python
        folded, _cs_acts = self._sparse_concept_forward(
            folded, subspace, word_subspace)
        # The 0-D symbols ride the returned subspace (dataflow rule): the
        # SS leg is built FROM these grad-bearing activations downstream.
        object.__setattr__(subspace, "_concept_activations", _cs_acts)
```

(Stamp `None` too — it clears a stale value from a prior pass on the same
subspace object.)

- [ ] **Step 6.4: Rebuild the leg from the activations.** In
`bin/Language.py` `forward_concept_to_symbol`, after the existing codebook
sync (`W[:rows, :cw] = ...`, which stays `no_grad` — EMA-only identity), replace
the leg construction:

```python
        acts = getattr(concept_sub, "_concept_activations", None)
        if acts is not None and torch.is_tensor(acts) \
                and int(acts.shape[0]) >= N and W is not None:
            # 0-D symbol: signed activation scales the row-aligned identity
            # row; grad flows through the ACTIVATION, rows stay EMA-only.
            a_t = acts[:N].t().unsqueeze(-1)         # [B, N, 1]
            rows = W[:N].detach().to(a_t.device, a_t.dtype)
            cw = min(D, int(rows.shape[-1]))
            leg = a_t.new_zeros((int(a_t.shape[0]), N, D))
            leg[..., :cw] = a_t * rows[:, :cw].unsqueeze(0)
            sym_event = leg
        # else: legacy detached-copy fallback (sym_event already built above)
        sub = SubSpace(inputShape=(N, D), outputShape=(N, D),
                       nInputDim=D, nOutputDim=D)
        sub.copy_context(concept_sub)
        sub.set_event(sym_event)
        return sub
```

(The `sym_event = event.detach()` fallback above this block is unchanged, so
`test_forward_concept_to_symbol_returns_row_aligned_leg` and the smoke tests
keep their behavior on the non-sparse path. If that existing test runs
sparse-active, update it to expect activation-scaled rows.)

- [ ] **Step 6.5: Run the file**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_to_ss_forward.py -q`
Expected: all pass (old + new).

- [ ] **Step 6.6: CHECKPOINT — report to user (the grad bug fix).**

### Task 7: `.when` consistency check at the tie site

**Files:**
- Modify: `bin/Spaces.py` (`_populate_cs_symbols`, `_autobind_cross_tower`,
  `_maybe_autobind_meta` caller threading)
- Modify: `test/test_cs_symbol_table.py`

- [ ] **Step 7.1: Find what the caller can thread**

Run: `grep -n "percept_where=" bin/Spaces.py | head -5`
Expected: the `cs.forward`-side call of `_maybe_autobind_meta` (~`:13154`)
passes `percept_where` read from PS `subspace.where`. Read that call site and
thread the matching `subspace.when` the same way as `percept_when=` (add the
kwarg to `_maybe_autobind_meta`, `_autobind_cross_tower`,
`_populate_cs_symbols`; default `None` everywhere — byte-identical when absent).

- [ ] **Step 7.2: Write the failing test** (append to
`test/test_cs_symbol_table.py`, reusing its bare-CS helper):

```python
def test_populate_cs_symbols_when_mismatch_raises():
    cs = _bare_cs()                                  # file's existing helper
    pid = torch.tensor([[1, 2]])
    where = torch.tensor([[0, 1]])
    when = torch.tensor([[7, 8]])                    # DIFFERENT .when: invalid
    spans = torch.tensor([[[0, 2]]])
    try:
        cs._populate_cs_symbols(pid, where, spans, percept_when=when)
        assert False, "mismatched .when must fail loud"
    except ValueError as e:
        assert ".when" in str(e)


def test_populate_cs_symbols_when_uniform_ok():
    cs = _bare_cs()
    pid = torch.tensor([[1, 2]])
    where = torch.tensor([[0, 1]])
    when = torch.tensor([[3, 3]])                    # same .when: fine
    spans = torch.tensor([[[0, 2]]])
    cs._populate_cs_symbols(pid, where, spans, percept_when=when)
```

- [ ] **Step 7.3: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py -q -k when`
Expected: FAIL — unexpected keyword `percept_when`.

- [ ] **Step 7.4: Implement.** In `_populate_cs_symbols`, add
`percept_when=None` to the signature; inside the span loop, before minting,
collect the `.when` of every percept whose `.where` nests, and check:

```python
                # Tie condition includes .when: one location-concept knits
                # percepts of ONE presentation moment (fail loud on mixes).
                if percept_when is not None and torch.is_tensor(percept_when):
                    w_n = int(percept_when[b, n])
                    if span_when is None:
                        span_when = w_n
                    elif span_when != w_n:
                        raise ValueError(
                            f"_populate_cs_symbols: .when mismatch in span "
                            f"({span_when} vs {w_n}) -- percepts of different "
                            f"moments knit into one location-concept")
```

(`span_when = None` initialized per span `k`.) Thread the kwarg through
`_autobind_cross_tower(pid_2d, percept_where, ws, percept_when=None)` and its
caller.

- [ ] **Step 7.5: Run**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py test/test_mereology_word_binding.py -q`
Expected: all pass.

- [ ] **Step 7.6: CHECKPOINT — report to user.**

### Task 8: Closest-link pruning rounds

**Files:**
- Modify: `bin/Spaces.py` (new `prune_concept_links`; hook in
  `refine_over_collected`)
- Modify: `test/test_cs_symbol_table.py`

Pruning drops, per concept: (a) the generic `_WORD_CLASS` whole when a
specific whole is also linked; (b) any linked whole that is a taxonomy
ANCESTOR of another linked whole (keep the minimal/tightest); (c) any linked
raw part that is a CONSTITUENT of a linked raised symbol H (keep the
maximal). All comparisons are within ONE tower — never cross-tower. Dropped
links are removed from BOTH the relation sets and the SparseLayer edges.

- [ ] **Step 8.1: Write the failing tests** (append to
`test/test_cs_symbol_table.py`):

```python
def test_prune_drops_generic_word_class_when_specific_whole_linked():
    from Spaces import _WORD_CLASS
    cs = _bare_cs()
    c = cs.new_concept()
    cs.add_part(c, 5)
    cs.add_whole(c, _WORD_CLASS)                     # generic
    cs.add_whole(c, 42)                              # specific word-whole
    dropped = cs.prune_concept_links()
    assert (c, "whole", _WORD_CLASS) in dropped
    assert set(cs.concept_wholes(c)) == {42}


def test_prune_drops_constituents_of_linked_raised_symbol():
    cs = _bare_cs()
    H = cs.synthesize_higher_order([7, 8])           # raised: Parts(H)={7,8}
    c = cs.new_concept()
    cs.add_part(c, 7)                                # constituent AND ...
    cs.add_part(c, ("sym", H))                       # ... the raised H itself
    cs.add_whole(c, 42)
    dropped = cs.prune_concept_links()
    assert (c, "part", 7) in dropped
    assert ("sym", H) in cs.concept_parts(c) and 7 not in cs.concept_parts(c)


def test_prune_removes_sparse_edges_of_dropped_links():
    from Spaces import _WORD_CLASS
    cs = _cs_sparse_active()                         # sparse-active helper
    c = cs.new_concept()
    cs.add_part(c, 1)
    cs.add_whole(c, _WORD_CLASS)
    cs.add_whole(c, 2)
    cs._populate_concept_weights(c)
    p, _ = cs._sparse_families(0)
    before = p.nnz
    cs.prune_concept_links()
    assert p.nnz == before - 1                       # the generic-whole edge
```

- [ ] **Step 8.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py -q -k prune`
Expected: FAIL — no `prune_concept_links`.

- [ ] **Step 8.3: Implement** (place after `refine_over_collected`):

```python
    def _whole_ancestors(self, whole_pos):
        """Within-tower taxonomy ancestors of a WS whole position (transitive,
        cycle-guarded); empty when the relation store is unwired."""
        store = self._relation_store()
        out = set()
        if store is None:
            return out
        seen = set()
        cur = int(whole_pos)
        while cur not in seen:
            seen.add(cur)
            parent = store.taxonomy_parent(cur)
            if parent is None:
                break
            cur = int(parent)
            out.add(cur)
        return out

    def prune_concept_links(self):
        """Closest-links pruning round (periodic; mint stays unconstrained):
        keep the MAXIMAL part and the MINIMAL whole among each concept's
        links. Within-tower comparisons only: (a) drop `_WORD_CLASS` when a
        specific whole is linked; (b) drop wholes that are taxonomy ancestors
        of another linked whole; (c) drop raw parts that are constituents of
        a linked raised symbol. Removes relation entries AND sparse edges.
        Returns the dropped ``(concept, side, code)`` triples."""
        parts, wholes = self._concept_tables()
        raised = getattr(self, "_concept_raised", None) or set()
        dropped = []
        for c in list(parts.keys()):
            ws_links = {w for w in wholes.get(c, set())
                        if not isinstance(w, tuple) and w != _UNIVERSE}
            drop_w = set()
            if len(ws_links) > 1 and _WORD_CLASS in ws_links:
                drop_w.add(_WORD_CLASS)
            for w in ws_links:
                ancestors = self._whole_ancestors(w) if w not in drop_w else ()
                drop_w |= (ws_links & set(ancestors))
            sym_parts = {x[1] for x in parts.get(c, set())
                         if isinstance(x, tuple) and len(x) == 2
                         and x[0] == "sym" and x[1] in raised}
            covered = set()
            for h in sym_parts:
                covered |= {p for p in self.concept_parts(h)
                            if not isinstance(p, tuple)}
            drop_p = {p for p in parts.get(c, set())
                      if not isinstance(p, tuple) and p in covered}
            for w in drop_w:
                wholes[c].discard(w)
                self._drop_concept_edge(c, w, side="whole")
                dropped.append((c, "whole", w))
            for p in drop_p:
                parts[c].discard(p)
                self._drop_concept_edge(c, p, side="part")
                dropped.append((c, "part", p))
        return dropped

    def _drop_concept_edge(self, concept_id, code, side):
        """Remove ``concept_id``'s sparse edge for a raw part/whole code (the
        pruning counterpart of ``_populate_concept_weights``); no-op when the
        concept has no allocated row / no edge."""
        rows = getattr(self, "_csw_rows", None) or {}
        order = self._concept_source_order(concept_id)
        c_local = rows.get((int(order), int(concept_id)))
        if c_local is None:
            return
        n_ps = int(getattr(self, "_n_ps_codes", 0) or 0)
        col = int(code) if side == "part" else n_ps + int(code)
        percept, _symbol = self._sparse_families(order)
        percept.remove_edges([(int(c_local), col)])
```

Hook it at the tail of `refine_over_collected`, just before its return:

```python
        # Periodic closest-links pruning: parts-of-parts / wholes-of-wholes
        # accumulated at mint are dropped here, not enforced at runtime.
        self.prune_concept_links()
```

- [ ] **Step 8.4: Run**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py test/test_mereology_word_binding.py test/test_mereology_raise.py -q`
Expected: all pass (existing lifecycle tests must survive the new hook — if a
lifecycle test asserts an exact whole-set that now loses `_WORD_CLASS`, that
assertion updates to the pruned expectation, which IS the intended semantics).

- [ ] **Step 8.5: CHECKPOINT — report to user.**

### Task 9: A-merge — the location concept IS the word A-symbol

**Files:**
- Modify: `bin/Spaces.py` (`_autobind_cross_tower`, `_populate_cs_symbols`,
  `_maybe_autobind_meta`)
- Modify: `test/test_mereology_word_binding.py`

- [ ] **Step 9.1: Verify span/word alignment**

Run: `grep -n "_staged_analysis_spans" bin/Spaces.py | head` and read the
staging site.
Expected: spans are per-WORD `(start, end)` in word order. If K (spans) aligns
with `word_texts[b]` indexing, the merge below is safe; if not, key only the
aligned prefix and leave the rest presentation-local (the fallback path).

- [ ] **Step 9.2: Write the failing test** (append to
`test/test_mereology_word_binding.py`, reusing its gated-model fixture):

```python
def test_word_A_symbol_is_location_concept_no_duplicate_knit():
    model = _mereology_raise_model()                 # file's existing fixture
    _run_word_presentation(model, "hello")           # existing helper pattern
    cs = model.conceptualSpaces[-1]
    A, B, C = cs._word_obj_meta["hello"]
    # The span knit reused A: no second concept holds the same parts with
    # only the generic word-class whole.
    from Spaces import _WORD_CLASS
    parts_A = set(cs.concept_parts(A))
    dupes = [s for s in cs.symbols_needing_processing()
             if s != A and set(cs.concept_parts(s)) == parts_A
             and set(cs.concept_wholes(s)) == {_WORD_CLASS}]
    assert dupes == []
    # A carries BOTH the specific whole and (pre-prune) possibly the generic.
    assert any(w != _WORD_CLASS for w in cs.concept_wholes(A))
```

(Adapt fixture/helper names to the file's existing ones — read its head first:
`sed -n '1,60p' test/test_mereology_word_binding.py`.)

- [ ] **Step 9.3: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_mereology_word_binding.py -q -k location_concept`
Expected: FAIL (duplicate loc_sym exists today).

- [ ] **Step 9.4: Implement the merge.**
- `_maybe_autobind_meta`: move the `create_word_object_meta` loop BEFORE
  `_autobind_cross_tower` (A exists when the knit runs), and pass the word
  key/whole info through:

```python
            word_bindings = self._autobind_word_wholes(
                pid_2d, vec_tensor, word_groups, tokens, ws, word_texts)
            for (w_parts, w_whole, w_key) in (word_bindings or ()):
                self.create_word_object_meta(w_parts, w_whole, key=w_key)
            self._autobind_cross_tower(pid_2d, percept_where, ws,
                                       percept_when=percept_when,
                                       word_texts=word_texts)
```

- `_autobind_cross_tower(..., word_texts=None)` forwards `word_texts` to
  `_populate_cs_symbols`.
- `_populate_cs_symbols(..., word_texts=None)`: per span `k`, resolve the
  word key (`word_texts[b][k]` when in range) and reuse A:

```python
                key = None
                if (word_texts is not None and b < len(word_texts)
                        and word_texts[b] is not None
                        and k < len(word_texts[b])):
                    key = word_texts[b][k]
                wom = getattr(self, "_word_obj_meta", None) or {}
                loc_sym = wom.get(key, (None,))[0] if key is not None else None
```

Then the existing per-percept loop uses `loc_sym` (minting fresh only when
still `None`), and the whole side becomes:

```python
                if loc_sym is not None:
                    self.add_whole(loc_sym, _WORD_CLASS)   # generic; pruned later
                    self._populate_concept_weights(loc_sym)
```

(The specific whole is already on A via `create_word_object_meta`; the merge
makes the knit and A one concept, so its weights now carry parts + BOTH
wholes until the pruning round keeps the tightest.)

- [ ] **Step 9.5: Run**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_mereology_word_binding.py test/test_cs_symbol_table.py -q`
Expected: all pass. If an existing test counts minted concepts, its expected
count DROPS by the merged duplicates — update with a one-line note.

- [ ] **Step 9.6: CHECKPOINT — report to user.**

### Task 10: `assert_concept_relation` — statements refine object-concepts

**Files:**
- Modify: `bin/Spaces.py` (new method near `create_word_object_meta`)
- Modify: `test/test_cs_symbol_table.py`

- [ ] **Step 10.1: Write the failing tests**:

```python
def test_assert_relation_replaces_poles_and_enters_embedding():
    from Spaces import _ATOM, _UNIVERSE
    cs = _cs_sparse_active()
    A, B, C = cs.create_word_object_meta([1], 2, key="cat")
    assert set(cs.concept_parts(B)) == {_ATOM}
    # "a cat has whiskers" (whiskers-object = another concept's B, say 9):
    cs.assert_concept_relation(B, sym_part=9)
    assert _ATOM not in cs.concept_parts(B)          # pole replaced
    # "cats are animals":
    cs.assert_concept_relation(B, sym_whole=11)
    assert _UNIVERSE not in cs.concept_wholes(B)
    # B's definition now has content -> it holds sparse edges (min-support).
    order = cs._concept_source_order(B)
    assert cs.concept_weights(order, cs._csw_rows[(order, B)]) != []


def test_assert_relation_raw_codes_and_weight():
    cs = _cs_sparse_active()
    c = cs.new_concept()
    cs.assert_concept_relation(c, part=3, whole=4, weight=0.5)
    assert 3 in cs.concept_parts(c) and 4 in cs.concept_wholes(c)
```

- [ ] **Step 10.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py -q -k assert_relation`
Expected: FAIL — no `assert_concept_relation`.

- [ ] **Step 10.3: Implement**:

```python
    def assert_concept_relation(self, cid, part=None, whole=None,
                                sym_part=None, sym_whole=None, weight=1.0):
        """Statement channel of the conceptual embedding: refine ``cid``'s
        definition from an asserted relation ("a body has a leg" adds a part;
        "cats are animals" adds a whole). Raw codes index PS/WS; ``sym_*``
        reference other concepts. The FIRST concrete part replaces the
        ``_ATOM`` pole, the first concrete whole replaces ``_UNIVERSE`` (the
        wide-open object narrows), then the sparse edges re-populate -- the
        concept's position moves with its definition. Host-side, at Reset."""
        parts, wholes = self._concept_tables()
        c = int(cid)
        if part is not None:
            parts.setdefault(c, set()).discard(_ATOM)
            self.add_part(c, int(part))
        if sym_part is not None:
            parts.setdefault(c, set()).discard(_ATOM)
            self.add_part(c, ("sym", int(sym_part)))
        if whole is not None:
            wholes.setdefault(c, set()).discard(_UNIVERSE)
            self.add_whole(c, int(whole))
        if sym_whole is not None:
            wholes.setdefault(c, set()).discard(_UNIVERSE)
            self.add_whole(c, ("sym", int(sym_whole)))
        self._populate_concept_weights(c)
        if weight != 1.0:
            self._scale_last_asserted(c, weight)
        return c
```

For `weight != 1.0`, `_scale_last_asserted` sets the just-added edges' values
under `no_grad` via each family's `_index` (concrete helper, ~8 lines: look up
the edge positions for the newly added cols and assign `weight`). If
`add_part`/`add_whole` reject `("sym", int)` tuples, extend them minimally —
`synthesize_higher_order`/`reify_concept` already store sym-tuples, so the
storage supports it.

- [ ] **Step 10.4: Run**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py -q`
Expected: all pass.

- [ ] **Step 10.5: CHECKPOINT — report to user.**

### Task 11: Hebbian strengthening of the word$\equiv$object tie

**Files:**
- Modify: `bin/Spaces.py` (`create_word_object_meta` key-hit path)
- Modify: `test/test_cs_symbol_table.py`

- [ ] **Step 11.1: Write the failing test**:

```python
def test_word_object_tie_strengthens_on_reoccurrence():
    cs = _cs_sparse_active()
    A, B, C = cs.create_word_object_meta([1, 2], 3, key="cat")
    order = cs._concept_source_order(C)
    row = cs._csw_rows.get((order, C))
    before = dict(cs.concept_weights(order, row))
    A2, B2, C2 = cs.create_word_object_meta([1, 2], 3, key="cat")  # re-mint
    assert (A2, B2, C2) == (A, B, C)
    after = dict(cs.concept_weights(order, row))
    assert any(after[c] > before[c] for c in before)  # Hebbian bump
    assert all(v <= 4.0 for v in after.values())      # clamped
```

- [ ] **Step 11.2: Run to verify failure**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py -q -k strengthens`
Expected: FAIL (weights unchanged on re-mint).

- [ ] **Step 11.3: Implement.** In the `key in wom` branch of
`create_word_object_meta`, after the A re-decomposition:

```python
            # Hebbian: word/object co-occurred again -> strengthen the META
            # tie's edge values (no_grad; codebooks stay EMA-only). Weakening
            # awaits a dis-occurrence signal (vacuous in a text-only learner).
            self._hebbian_strengthen(C, eta=0.1, cap=4.0)
```

And the helper:

```python
    def _hebbian_strengthen(self, cid, eta=0.1, cap=4.0):
        """Bump ``cid``'s sparse edge values by ``eta`` (clamped to ``cap``)."""
        rows = getattr(self, "_csw_rows", None) or {}
        order = self._concept_source_order(cid)
        c_local = rows.get((int(order), int(cid)))
        if c_local is None:
            return
        for fam in self._sparse_families(order):
            if fam is None or fam.values is None:
                continue
            with torch.no_grad():
                for (r, _c), i in fam._index.items():
                    if r == int(c_local):
                        fam.values[i] = min(cap, float(fam.values[i]) + eta)
```

- [ ] **Step 11.4: Run**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_symbol_table.py test/test_mereology_word_binding.py -q`
Expected: all pass.

- [ ] **Step 11.5: CHECKPOINT — report to user.**

### Task 12: Gates — compile smoke + full suite

**Files:** none (verification).

- [ ] **Step 12.1: Fullgraph/eager smoke** (the sparse path + stamp touched
`ConceptualSpace.forward`; verify no new graph break — remember: an inductor
`CppCompileError` on this repo's path is NOT a real break, use eager):

Run: `MODEL_COMPILE=eager BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_sparse_concept_e2e.py -q`
Expected: pass.

- [ ] **Step 12.2: The final gate**

Run: `make test`
Expected: green (mlx D2/D3 xfails excepted). Any failure in a
`symbolicOrder=0` config is a byte-identical violation — fix before
proceeding, do not rationalize.

- [ ] **Step 12.3: CHECKPOINT — report full results to user.**

### Task 13: Documentation updates

**Files:**
- Modify: `doc/Architecture.md` (section A "Symbolic weights"; the
  ConceptualSpace row of the Spaces table)
- Modify: `todo.md` (top block + the line-96 SigmaLayer item)
- Modify: `doc/plans/2026-06-27-sparse-coding-cs-handoff.md` (staleness note)

Use LaTeX math (`$\to$`, `$\sigma$`) — NO Unicode glyphs.

- [ ] **Step 13.1:** Rewrite `doc/Architecture.md` section A to describe
`SparseLayer` (dedicated class; tanh forward; transpose reverse; two families
per order sized by `order_capacities`; presences squashed; pruning rounds keep
closest links). Replace the "SigmaLayer needs a sparse option" sentence — that
approach is superseded (SigmaLayer's atanh-entry contract does not fit
presence inputs). Update the ConceptualSpace table row: `_csw_vals` $\to$
per-order `SparseLayer` families.

- [ ] **Step 13.2:** `todo.md`: mark the line-96 item implemented-as-SparseLayer
(one line, date). In the top binding block: correct the spec path to
`doc/old/mereological-order-raising.md`; note the S2a knit now reuses the word
A-symbol (Task 9), the `.when` check (Task 7), and that closest-links is
enforced by pruning rounds, answering the "ANY covering span" open question.

- [ ] **Step 13.3:** Add to the 2026-06-27 handoff doc header: a 3-line note
that the Phase-1/2 `_edge_*` store was retired; the surviving design is the
per-order family `SparseLayer`s of THIS plan (link it).

- [ ] **Step 13.4:** `make doc` if available quickly, else at minimum grep the
touched docs for non-ASCII: `grep -nP "[^\x00-\x7F]" doc/Architecture.md todo.md | head`
Expected: only pre-existing occurrences (do not introduce new ones).

- [ ] **Step 13.5: CHECKPOINT — report to user.**

### Task 14: Final handoff

- [ ] **Step 14.1:** Summarize for the user: what changed per task, test
counts (baseline vs final), the intentional numeric change at
`symbolicOrder>=1`, and the deferred list (NL statement wiring, serial
substitution, Hebbian weakening, symbol `.where` stamping, PS-provenance part
pruning). The user commits.

---

## Addendum (same day, after user review): poles as vectors; META reshape

Two design refinements from Alec, implemented on top of the tasks below:

1. **NOTHING / EVERYTHING replace ATOM / UNIVERSE** (better names; back-compat
   aliases kept in `bin/Layers.py`). The poles are VECTORS of the presence
   lattice, not fake WS positions: $nothing = [0,0,\ldots]$ (bottom; as a part
   it contributes $W \cdot 0 = 0$ -- realized as the ABSENCE of an edge) and
   $everything = [1,1,\ldots]$ (top; as a whole it contributes the sum of its
   weights -- realized as the trailing CONSTANT-1 "everything" column of each
   percept family, i.e. a learnable bias). This also fixes a latent collision:
   the old sentinels (101/102) collided with real WS codebook columns once
   `n_ws > 101`. Consequences: the percept family is `[PS|WS|everything]`
   (`cs_source_layout` gains an `"everything"` offset; SS offsets shift by 1);
   a freshly-minted pole-pair object B now ENTERS the embedding with exactly
   one bias edge (activation $\tanh(w)$ -- the maximally-general position);
   `assert_concept_relation` retires the bias edge when a concrete whole
   replaces the pole; `prune_concept_links` drops EVERYTHING (the loosest
   whole of all) whenever any other whole is linked. For text the attainable
   concrete bounds are LETTERS (below) and the SENTENCE (above); refinement
   moves the poles inward toward them (sentence-whole minting is not yet
   wired -- spans are word-level today).
2. **The META is a first-order COMBINATION, not a subsumption.**
   ``create_word_object_meta`` now mints C with TWO PARTS --
   $Parts(C) = \{sym(A), sym(B)\}$, no whole -- combining the symbol for the
   word and the symbol for the object. ``reify_concept`` keeps its part/whole
   semantics for ``conceptualize_chain`` (chains genuinely use the
   asymmetry). Side effect: with two parts and no whole, C no longer has the
   1-part/1-whole SHAPE, so ``resolve_identities`` leaves the META active
   instead of clearing it. The word/object pairing can also be kept
   separately in references (`bin/References.py`) for serial-mode
   substitution -- unchanged here.

## Continuation (2026-07-02, second pass): finish the situating + MM-20M items

> **STATUS (end of second pass):** C1 DONE (live-slab SBOW; pool from the
> slab's own stage dictionary). MM-20M work: THREE root-caused fixes landed --
> the SS-identity-rows clone (backward version-counter crash on the live
> path), the normalized presence readout (raw dot-sums saturated to the
> all-ones input-blind constant -- the collapse's first layer), and the radix
> reconstruction RENDER wiring + the LIVE reconstruction seed
> (``_reconstruction_seed()``; the loss was a grad-dead constant off the
> detached STM snapshot). Verified by `test_radix_recon_render.py` + extended
> sparse suites and training reproductions (sO=0 XOR now trains to ~0.000
> output loss). REMAINING (design decisions, documented in todo.md MM-20M
> STATUS notes): the sO=1 collapse's second layer (stage-0 sparse content
> replaces the butterfly slab; the joint/sentence concept over both
> word-symbols never mints -- the everything-bias makes order-1 XOR-expressible
> once it does), and reconstruction fidelity (loss magnitude, `.where`-band
> recovery aliasing, content-chunk granularity -- the reverted 2026-07-01
> ground).
>
> **THIRD PASS (Alec's review corrections, same day):** (1) presence readout
> revised from cosine to the NORMALIZED SUM -- slot-mean projection onto the
> unit code direction, $/\sqrt{D}$ (hypercube-diagonal units): the code row's
> scale is removed but the EVENT magnitude is preserved (objects in the unit
> hypercube differentiate by magnitude). (2) The JOINT/sentence concept is
> implemented as the SYMBOLIC mixing (``create_joint_concept``: first-order,
> parts = the row's word A-symbols, whole = EVERYTHING $\to$ the bias edge,
> keyed per sentence type, Hebbian re-occurrence), minted from
> ``_maybe_autobind_meta`` per batch row with $\ge 2$ words. (3) The
> SUBSYMBOLIC butterfly mixing is preserved as a truly-invertible OPTION:
> ``<sparseReplace>false</sparseReplace>`` keeps the butterfly content advance
> while the sparse activations still feed the symbolic loop. IDENTIFIED
> CONSTRAINT: replace mode requires CS ``nVectors`` == the STM slot width; a
> larger inventory needs the salience SELECTION step (Architecture.md,
> parallel-mode attention) before replace-mode XOR is testable end-to-end.

**Scope decision:** "finish implementation" = the load-bearing remainder of the
sparse design (the SBOW situating on the LIVE codes -- old Phase 3), then the
todo.md MM-20M section (XOR output at `symbolicOrder=1`; reconstruction render
at `symbolicOrder=0`). The dictionary already trains by gradient
(`similarity_codebook` params registered in `self.params`), so a separate
VQ-EMA-snap subsystem stays deferred; likewise serial substitution / NL
statement wiring / Hebbian weakening / sentence-whole minting (not needed for
the MM-20M items per todo's own diagnosis).

### Task C1: SBOW situates the live sparse codes (grad-bearing)

Two sites in `bin/Models.py`:
- Park site (`_forward_body` t=0, ~`:6594`): keep the parked t=0 slab
  GRAD-BEARING when the stage CS is sparse-active (legacy detached park
  otherwise). Same gate (`conceptualSimilarityScale > 0`), same attr.
- `conceptual_sbow_loss` (~`:3328`): when the parked slab carries grad, the
  WINDOW is the slab itself (the composed codes) and the codebook rows are the
  POOL -- `conceptual_sbow_loss_codes(slab, pool=rows)` -- so the
  substitutability gradient reaches the sparse family values and presences
  (this dissolves the documented forward-disconnect NOTE for the sparse path;
  the legacy no-grad snap-gather stays for sparse-inactive configs).

Test: `test_sparse_concept_e2e.py` -- forward under `MM_sparse_concept.xml`
with pre-populated edges parks a live slab; `conceptual_sbow_loss().backward()`
reaches `_sparse_fam` values. Byte-identical: `conceptualSimilarityScale=0`
never parks; sparse-inactive parks detached (legacy path bit-for-bit).

### Task C2: MM-20M reproduction (B0) then fixes

- B0: short runs (`bin/train.py --model data/MM_20M_xor.xml` and
  `data/MM_sparse_concept.xml`, few hundred XOR batches) to re-establish both
  symptoms POST-rework: the `'h h'` output collapse at `symbolicOrder=1` and
  the blank radix/bpe reconstruction at `symbolicOrder=0`. The tanh rework
  changed the sparse forward's numerics; re-diagnose before fixing
  (systematic-debugging discipline).
- B1 (output @ sO=1): fix informed by B0 + C1 (the todo's hypothesis: SBOW
  anti-collapse + joint mixing).
- B2 (reconstruction @ sO=0): the render path is
  `OutputSpace.reconstruct_buffer` -> `_reverse_text_vectors` ->
  `vocabulary.reconstruct_to_buffer` (`bin/Spaces.py:21321`) -- lexicon-only
  today; wire the radix/meronomy (and bpe) decode, diffing against
  `XOR_exact`'s working lexicon decode.

## Self-review notes

- Every requirement from the 2026-07-02 design discussion maps to a task:
  SparseLayer+tanh+transpose (1, 2), two families per order (3, 4), squash
  (4), store unification/retirement (5), grad-preserving 0-D symbol leg (6),
  `.when` (7), closest-links pruning (8), degenerate-whole fix via A-merge +
  pruning (8, 9), statement channel + pole replacement (10), Hebbian tie (11),
  EMA-only codebooks preserved (6, 11 — `no_grad` writes only), docs (13).
- Types/signatures cross-checked: `add_concept_weight(order, concept_local,
  source_global, weight=1.0)` unchanged; `cs_forward_content(ps_act, ws_act,
  dictionary) -> (content, a_list)` unchanged; `_sparse_concept_forward` now
  returns a PAIR (all call sites updated in Task 6);
  `forward_concept_to_symbol(concept_sub)` signature unchanged.
- Known judgment calls the implementer may hit: exact fixture names in the
  test files (read the file head first — steps say so); `test/test_sparse_concept_e2e.py`
  magnitude assertions (bounded semantics now); a lifecycle test asserting
  `_WORD_CLASS` in a whole-set after Task 8 (pruned now — intended).
