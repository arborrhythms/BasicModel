# Iterated Symbolic Loop — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. **Alec does ALL git writes** — no
> commit steps appear below; each GATE is where he commits.

**Goal:** Collapse the per-order ramsified SparseLayer families into ONE
square untyped SparseLayer with iterated (fixed-$K$ unrolled) wave forward
semantics, per the approved design
[2026-07-02-iterated-symbolic-loop.md](2026-07-02-iterated-symbolic-loop.md).

**Architecture:** The collapsed layer is a new **`AttentionLayer`**
(bottom-up attention over the concept inventory — what it IS), subclassing
the `SparseLayer` substrate (how it works); the legacy QKV class frees the
name by becoming `QKVAttentionLayer` (decision 5). Rows = the whole concept
inventory ($N$ = `nVectors`); columns = $[N \mid 1]$ (bias). Snap block =
rows $[0, n_{snap})$, $n_{snap} = \max(1, N/2)$ (the old `caps[0]`);
relation pool = rows $[n_{snap}, N)$. Forward:
$a^{i+1} = \tanh(W[a^i \mid 1] + s)$, $i = 0..K-1$, $K$ = `symbolicOrder`,
$s$ = the snap presence padded to $N$ (ADDITIVE source term — Alec
2026-07-03; snap rows have no in-edges so they read $\tanh(a^0)$ each step,
accepted). Kripke groundedness probe is a host-side `no_grad` diagnostic,
never traced.

**Tech stack:** torch COO sparse (`SparseLayer`), fullgraph compile
discipline, pytest (`make test` is THE gate).

**Approval decisions (Alec, 2026-07-03):**

1. **Source term: ADDITIVE** — the formula as written; no hard clamp.
2. **Capacity: first-come + LOUD report** — mint-order allocation; pool
   overflow warns visibly and counts; no eviction machinery now.
3. **Cycles: documented fact, no intervention** — we won't know we are
   oscillating from inside; some loops may be behaviorally advantageous.
   Loops within the machine mind are a fact we document and invite a
   solution to (for both human and machine minds). Diagnostics
   (wave-QE statistic, groundedness probe) are report-only observability;
   damping stays an unused escalation dial.
4. **Keep `symbolicOrder`** — with the doc note that we are NOT forcing
   ramsification: the value is now the MAXIMUM POSSIBLE conceptual order
   (i.e. the order reached if a novel concept were introduced at every
   iteration), read operationally as the iteration count $K$.
5. **The layer is named `AttentionLayer`** (Alec 2026-07-03): it IS
   bottom-up attention — the horizontal channel (which items, within a
   level) lifted into relation space — `SparseLayer` only describes how it
   works. Symbolic/conceptual heat is the SAME bottom-up channel at the
   retrieval site (NOT top-down); top-down attention is the vertical,
   property/goal-driven channel (Rosch Basic Level, scope handoff). Heat
   $\leftrightarrow$ wave integration is recorded future work in the design
   doc, out of scope here. The legacy QKV `AttentionLayer` is renamed
   `QKVAttentionLayer` (kept, still retired from enlistment).

**Review corrections folded in (2026-07-03 code review):**

- "Partial prefixes activate meaningfully" is imprecise: under a full snap
  the TAIL links (suffix sublists) complete first (the vine folds
  right-to-left); a partially perceived prefix yields graded sub-threshold
  evidence only. Docs must say "graded partial evidence".
- $K$-footgun: configs with `symbolicOrder=1` run ONE iteration — depth
  $\ge 2$ vines never fully activate. XOR rerun configs set $K$
  deliberately (Task 8).
- `relate(x, x)` (same sym as both part and whole) now collapses to ONE
  untyped edge (idempotent `add_edge`); this MERGE is the documented
  behavior. True self-edges (a relation among its own constituents) raise.

---

## Task 1 — `AttentionLayer`: the bottom-up-attention layer (+ legacy rename)

**Files:**
- Modify: `bin/Layers.py` (legacy QKV class 5329–5510; `SparseLayer.__init__`
  ~4480, `add_edge` ~4533; new subclass after `SparseLayer`)
- Modify: `bin/Models.py:70`, `bin/Language.py:28` (imports),
  `bin/Layers.py:8042` (comment), `bin/Layers.py:15215` (test hook)
- Modify: `test/test_basicmodel.py:374` (`TestAttentionLayer` — targets the
  QKV class)
- Test: `test/test_sparse_layer.py`

- [x] **1.1 Rename the legacy QKV class** `AttentionLayer` $\to$
  `QKVAttentionLayer` (it is retired from enlistment; kept for
  back-compat). Update ALL references:
  `grep -rn "AttentionLayer" bin/ test/` — the imports at
  `bin/Models.py:70` and `bin/Language.py:28`, the internal uses at
  `bin/Layers.py:5346/5425/5501/5506/15215`, the comment at
  `bin/Layers.py:8042`, and `test/test_basicmodel.py:374` (rename the
  TestCase to `TestQKVAttentionLayer` and its constructor calls). The
  explanatory comments at `bin/Spaces.py:9367/9751/11870/12587/12633/16822`
  SHOULD be updated to say "QKV ``QKVAttentionLayer``" where they name the
  class. Verify: `pytest test/test_basicmodel.py -k QKVAttention -q`.

- [x] **1.2 Write the failing tests** (append to `test/test_sparse_layer.py`):

```python
def test_self_edge_forbidden_when_flagged():
    ly = SparseLayer(5, 4, forbid_self_edges=True)
    with pytest.raises(ValueError, match="self-edge"):
        ly.add_edge(2, 2)
    ly.add_edge(2, 3)                      # off-diagonal still fine
    ly.add_edge(2, 4)                      # bias col (== nInput-1) fine


def test_self_edge_allowed_by_default():
    ly = SparseLayer(5, 4)
    ly.add_edge(2, 2)                      # rectangular role layers unaffected


def test_attention_layer_is_square_and_forbids_self_edges():
    ly = AttentionLayer(8)                 # N=8 concepts -> [9 x 8], no roles
    assert (ly.nInput, ly.nOutput) == (9, 8)
    with pytest.raises(ValueError, match="self-edge"):
        ly.add_edge(3, 3)


def test_attention_layer_wave_step_matches_formula():
    ly = AttentionLayer(4)
    ly.add_edge(2, 0, weight=0.5)          # relation row 2 reads concept 0
    a = torch.zeros(4, 1); a[0, 0] = 1.0
    s = a.clone()
    out = ly.wave_step(a, s)
    assert torch.allclose(out[2], torch.tanh(torch.tensor([0.5])))
    assert torch.allclose(out[0], torch.tanh(torch.tensor([1.0])))  # tanh(s)
```

- [x] **1.3 Run to verify failure:**
  `pytest test/test_sparse_layer.py -k "self_edge or attention_layer" -x -q`
  — expect `TypeError`/`NameError`.

- [x] **1.4 Implement.** In `SparseLayer.__init__`, accept and stash
  `forbid_self_edges=False`; in `add_edge`, AFTER the role-offset
  resolution and bounds check, before the idempotency lookup:

```python
if self._forbid_self_edges and int(row) == int(col):
    raise ValueError(
        f"SparseLayer: self-edge ({int(row)}, {int(col)}) forbidden -- "
        f"x = {{x}} is the Quine atom (see "
        f"doc/plans/2026-07-02-iterated-symbolic-loop.md)")
```

  Then the subclass (after `SparseLayer` in `bin/Layers.py`):

```python
class AttentionLayer(SparseLayer):
    """BOTTOM-UP ATTENTION over the concept inventory (the iterated
    symbolic loop, doc/plans/2026-07-02-iterated-symbolic-loop.md): one
    square untyped store [N x N+1] whose edges are fuzzy set-membership
    degrees; the wave a^{i+1} = tanh(W [a^i | 1] + s) propagates perceptual
    salience one membership-weighted hop per step (the horizontal channel
    lifted into relation space -- doc/Architecture.md "Parse time" sec C).
    Named for what it IS; SparseLayer is how it works. Self-edges are
    forbidden (x = {x}, the Quine atom); longer cycles are deliberate."""

    def __init__(self, n_concepts, device=None):
        super().__init__(int(n_concepts) + 1, int(n_concepts),
                         device=device, forbid_self_edges=True)

    def wave_step(self, a, s):
        """One bottom-up-attention hop: tanh(W [a | 1] + s)."""
        x = torch.cat([a, a.new_ones((1, int(a.shape[-1])))], dim=0)
        pre = (self.forward_linear(x) if self.nnz > 0
               else torch.zeros_like(a))
        return torch.tanh(pre + s)
```

  (Match `SparseLayer.__init__`'s actual signature for the device/roles
  positional order when writing the `super().__init__` call.)

- [x] **1.5 Verify:** `pytest test/test_sparse_layer.py test/test_basicmodel.py -q`
  — all pass.

---

## Task 2 — Allocator collapse: one square layer, two-block caps

**Files:**
- Modify: `bin/Layers.py` (`ConceptAllocator` 4754–4879: `layer`,
  `assign_row`, `settle`, `_layers`)
- Modify: `bin/Spaces.py` (`_concept_alloc_of` sizer closure ~12484;
  `_order_caps` 14192; `order_capacities` ~14175; `_csw_concept_row`
  14472; `_csw_rows` 14461)
- Test: `test/test_cs_symbol_table.py`, `test/test_cs_sparse_weights.py`

- [x] **2.1 Two-block caps.** Rewrite `_order_caps` (keep the cache
  pattern; drop the dyadic `order_capacities` static — delete it and its
  pin `test_order_capacities_dyadic_sums_to_total`):

```python
def _order_caps(self):
    """Cached (n_snap, n_pool) split of THIS CS's ``nVectors`` rows: the
    order-0 snap block and the single untyped relation pool (v3 -- the
    dyadic per-order split is retired with ramsification)."""
    N = int(self.nVectors)
    key = N
    cache = getattr(self, "_order_caps_cache", None)
    if cache is None or cache[0] != key:
        n_snap = max(1, N // 2)
        object.__setattr__(self, "_order_caps_cache",
                           (key, (n_snap, max(1, N - n_snap))))
    return self._order_caps_cache[1]
```

  `order_slice(0)` (snap block) and `order_slice(1)` (pool) keep working
  unmodified over the 2-tuple; `snap_settle_qe`/`cs_snap_order0` callers
  (14394, 14546) are untouched.

- [x] **2.2 Square sizer.** In `_concept_alloc_of`'s `_sizer` closure:
  return `(N + 1, N, dev, None)` where `N = nVectors` (read via the
  existing `caps_fn` host getattr: `N = sum(caps_fn())`), roles retired;
  `ConceptAllocator.layer()` constructs an **`AttentionLayer(N, device=dev)`**
  (Task 1.4 — square, untyped, self-edges forbidden by construction).
  Update `_cap` closure: keep returning `len(caps) - 1` (now 1) — it only
  feeds `order_of`'s cap, which survives as bookkeeping.

- [x] **2.3 Single layer + two row regions.** In `ConceptAllocator`:
  `_layers` collapses to one entry (key `0`); `layer(order=0)` ignores
  `order` and lazily builds THE square layer. Row allocation gets two
  counters on the layer store: order-0 keys allocate in $[0, n_{snap})$,
  everything else in $[n_{snap}, N)$. Rewrite `_csw_concept_row` to
  return the GLOBAL row:

```python
def _csw_concept_row(self, order, concept_id):
    """First-seen GLOBAL row of ``concept_id``: order 0 allocates in the
    snap block ``[0, n_snap)``, relations in the pool ``[n_snap, N)``.
    ``None`` on overflow -- LOUDLY (decision 2: first-come + report)."""
    alloc = _concept_alloc_of(self)
    n_snap, n_pool = self._order_caps()
    ly = alloc.layer(0)
    if int(order) == 0:
        row = ly.assign_row(("snap", concept_id), capacity=n_snap,
                            base=0)
    else:
        row = ly.assign_row(("pool", concept_id), capacity=n_pool,
                            base=n_snap)
    if row is None:
        n = int(getattr(self, "_csw_overflow_count", 0)) + 1
        object.__setattr__(self, "_csw_overflow_count", n)
        warnings.warn(
            f"CS relation pool exhausted ({n_pool} rows): concept "
            f"{int(concept_id)} gets records only, no weighted reading "
            f"(overflow #{n})", RuntimeWarning)
    return row
```

  `assign_row(key, capacity, base)` grows a `base` offset + per-region
  counter (`_row_next` becomes `{base: next_local}`); `row_of`/
  `pop_row_key`/`put_row_key` keep the single `_tensor_rows` map (global
  rows). Keys are namespaced (`("snap", cid)` / `("pool", cid)`) so a cid
  that is BOTH an order-0 concept and later composes relations cannot
  collide.

- [x] **2.4 `settle()` becomes bookkeeping-only.** Rows are PERMANENT in
  the single layer — no record/row migration. Body: recompute
  `self.placement[cid] = self.order_of(cid)`; delete the
  `pop_row_key`/`put_row_key` move. Callers (`bin/Spaces.py:13927`,
  `bin/Spaces.py:14890`, Layers-internal 4851–4866) stay as-is.
  `_csw_rows` view: key by `(alloc.placement.get(cid, 0), cid)` over the
  one layer's `_tensor_rows` (back-compat shape).

- [x] **2.5 Verify:**
  `pytest test/test_cs_symbol_table.py test/test_sparse_layer.py -q` —
  failures at this point should ONLY be pins of retired behavior
  (dyadic caps, per-order layers); rewrite those pins now (two-block
  caps; `alloc.layer(k)` returns the same object for all `k`).

---

## Task 3 — `_populate_concept_weights` v3: untyped edges

**Files:**
- Modify: `bin/Spaces.py` (`_populate_concept_weights` 14481–14530;
  `add_concept_edge` ~14230–14270; DELETE `cs_source_layout` 14359–14373;
  `_retire_concept_edge` ~14695; `_set_concept_edge_value` 14717;
  `_hebbian_strengthen` 14795)
- Test: `test/test_sparse_concept_e2e.py`, `test/test_cs_sparse_weights.py`

- [x] **3.1 Write the failing pin for the FIXED defect** (the chain-link
  edge that stratification dropped — motivation defect 1):

```python
def test_chain_link_edge_to_rest_link_is_populated(sparse_space):
    cs = sparse_space                     # fixture: _sparse_active() True
    a, b, c = mint_order0(cs, 3)          # helper per existing e2e fixtures
    head = cs.create_joint_concept([a, b, c])
    alloc = _concept_alloc_of(cs)
    ly = alloc.layer(0)
    head_row = ly.row_of(("pool", head))
    # the head link's part is the REST LINK (same untyped pool) -- the
    # edge the ramsified gate silently dropped must now exist:
    rest = [x for (r, x) in alloc.records(head)
            if r == "part" and x[0] == "sym"][0]
    rest_row = ly.row_of(("pool", int(x[1])))
    assert (head_row, rest_row) in ly._index
```

  (Adapt helper names to the fixture actually present in
  `test_sparse_concept_e2e.py` — mirror its existing vine test's setup.)

- [x] **3.2 Run to verify failure:**
  `pytest test/test_sparse_concept_e2e.py -k rest_link -x -q`.

- [x] **3.3 Implement `add_concept_edge` v3** — global coordinates, no
  order/role args: `add_concept_edge(row, col, weight=1.0)`; keep the
  fail-loud guard that a SNAP-region row accepts no edges
  (`row < n_snap` raises — order-0 composes nothing, unchanged
  principle). Bias column is `col == N`.

- [x] **3.4 Implement `_populate_concept_weights` v3:**

```python
def _populate_concept_weights(self, concept_id):
    """Decompose ``concept_id`` into the UNTYPED square store (v3): every
    sym constituent gets ONE edge (row = the relation's row, col = the
    constituent's row); EVERYTHING -> the bias column; raw refs stay
    reference-store-only. Direction/order live in the record store and
    the NESTING (the vine), never in typed columns. Self-edges raise
    (Quine atom); relate(x, x) merges to one edge (documented).
    MIN-SUPPORT and the singleton exemption unchanged. NO-OP unless the
    sparse transform is active -> byte-identical."""
    if not self._sparse_active():
        return
    alloc = _concept_alloc_of(self)
    order = self.cs_concept_order(concept_id)      # bookkeeping only
    if order == 0:
        self._csw_concept_row(0, concept_id)       # reserve snap row
        return
    parts = alloc.refs(concept_id, "part")
    wholes = alloc.refs(concept_id, "whole")
    def _is_sym(x):
        return isinstance(x, tuple) and len(x) == 2 and x[0] == "sym"
    sym_refs = ([("part", x) for x in parts if _is_sym(x)]
                + [("whole", x) for x in wholes if _is_sym(x)])
    n_raw = (len([x for x in parts if not _is_sym(x) and x != _NOTHING])
             + len([x for x in wholes
                    if not _is_sym(x) and x != _EVERYTHING]))
    n_poles = int(_NOTHING in parts) + int(_EVERYTHING in wholes)
    if (concept_id not in alloc.singletons
            and n_raw + len(sym_refs) + n_poles < 2):
        return
    c_row = self._csw_concept_row(order, concept_id)
    if c_row is None:
        return                                     # loud overflow already
    for (_role, x) in sym_refs:
        so = self.cs_concept_order(int(x[1]))
        s_row = self._csw_concept_row(so, int(x[1]))
        if s_row is None:
            continue
        self.add_concept_edge(c_row, s_row)        # untyped; self-edge raises
    if _EVERYTHING in wholes:
        self.add_concept_edge(c_row, int(self.nVectors))   # bias col
    self._maybe_rebuild_optimizer_for_csw()
```

  Port the EXACT v2 min-support arithmetic (compare with the current body
  before replacing — the sketch above mirrors it, verify pole handling
  1:1). `_maybe_rebuild_optimizer_for_csw` is unchanged (nnz-count
  debounce works on one layer).

- [x] **3.5 Rewrite the order/role-addressed maintenance paths** (all
  mechanical — global `(row, col)` addressing):
  - `_retire_concept_edge`: `remove_edges([(c_row, N)])` (bias col), no
    role_slice.
  - `_set_concept_edge_value`: `pos = ly._index.get((c_row, s_row))`; note
    in the docstring that for `relate(x, x)` the part- and whole-
    assertions address the SAME merged edge (decision: merge).
  - `_hebbian_strengthen`: single layer,
    `ly.hebbian_strengthen_row(global_row, ...)`.
  - `assert_concept_relation`'s bias-retirement branch (~14783):
    `remove_edges([(row_of(c), N)])` — no `old_order` layer lookup.
  - DELETE `cs_source_layout` and `_sparse_families`' symbol-family
    role machinery that nothing references anymore (grep
    `forward_linear_roles|role_slice|cs_source_layout` in `bin/` must
    come back empty outside `SparseLayer` itself; keep
    `forward_linear_roles` on `SparseLayer` only if `test_sparse_layer.py`
    pins it — otherwise delete it too).

- [x] **3.6 Verify:**
  `pytest test/test_sparse_concept_e2e.py test/test_cs_sparse_weights.py -q`
  — rewrite remaining pins of role blocks/source layout as you hit them
  (Task 7 finishes the sweep).

**GATE A (Alec commits):** Tasks 1–3 green on
`pytest test/test_sparse_layer.py test/test_cs_symbol_table.py test/test_sparse_concept_e2e.py test/test_cs_sparse_weights.py -q`.

---

## Task 4 — The wave: `cs_forward_content` v3 + `cs_symbolic_phase`

**Files:**
- Modify: `bin/Spaces.py` (`cs_forward_content` 14419–14450;
  `cs_symbolic_phase` 14553–14579; `cs_decode` 14345 — keep for order-0
  callers, forward stops using per-order slabs)
- Test: `test/test_symbolic_iteration.py` (rewrites), new
  `test/test_iterated_symbolic_wave.py` (Task 6)

- [x] **4.1 Implement the iterated forward:**

```python
def cs_forward_content(self, a_0, dictionary):
    """Iterated clamped-SOURCE wave over the single untyped layer (v3):
    a^{i+1} = tanh(W [a^i | 1] + s), i = 0..K-1, K = symbolicOrder (the
    MAXIMUM POSSIBLE conceptual order; we are NOT forcing ramsification).
    ``s`` is the order-0 snap presence padded to the inventory -- an
    ADDITIVE source term each step (Alec 2026-07-03): snap rows carry no
    in-edges, so they read tanh(a_0) per step. A depth-d vine completes
    at iteration d (tail links first -- graded partial evidence before
    that). Fixed-K unroll, no data-dependent control flow; the per-step
    wave-QE settle statistic is recorded no_grad, report-only."""
    N = int(self.nVectors)
    B = int(a_0.shape[-1])
    n_snap, _n_pool = self._order_caps()
    K = int(getattr(self, "_symbolic_order", 0) or 0)
    ly = _concept_alloc_of(self).layer(0)
    atoms = F.softplus(dictionary)                     # [N, CDim] > 0
    s = torch.cat([a_0, a_0.new_zeros((N - n_snap, B))], dim=0)
    a = s                                              # a^0 = padded snap
    qes = []
    for _i in range(K):                                # K static: unrolls
        a_next = ly.wave_step(a, s)                    # tanh(W [a|1] + s)
        with torch.no_grad():
            qes.append((a_next - a).abs().mean())
        a = a_next
    object.__setattr__(self, "_cs_wave_qe",
                       torch.stack(qes) if qes else a_0.new_zeros(0))
    content = a.t().unsqueeze(-1) * atoms.unsqueeze(0)  # [B, N, CDim]
    return content, a
```

  Notes that MUST hold: `ly.nnz` is host-static at trace time (same
  recompile-on-growth behavior as v2's `symbol.nnz > 0` branch);
  `forward_linear` exists (`bin/Layers.py:4622`, pre-tanh); the decode is
  byte-equivalent to v2's per-order `cs_decode` slabs cat'ed (same
  `a[c] * atoms[c]` math over the full inventory).

- [x] **4.2 `cs_symbolic_phase` v3:** replace the
  `content, a_list = ...; acts = torch.cat(a_list, dim=0)` pair with
  `content, acts = self.cs_forward_content(a_0, dict_W)` — `acts` is
  already the `[N, B]` signed grad-bearing stack; the downstream contract
  (`_concept_activations` stamped at the cutover, SS leg) is unchanged.
  The cutover site `bin/Models.py:6843-6862` is NOT touched.

- [x] **4.3 Byte-identity + compile check:**
  - `pytest test/ -k "byte_ident or sO0 or symbolic_order" -q` (locate the
    actual sO=0 identity pins via `grep -l "byte" test/` and run those
    files) — must be untouched-green (everything gated on
    `_sparse_active()`).
  - `MODEL_COMPILE=eager pytest test/test_symbolic_iteration.py -q` —
    fullgraph trace of the unrolled loop (an inductor `CppCompileError`
    on the iCloud path is NOT a graph break; eager tests the trace).

- [x] **4.4 Rewrite `test/test_symbolic_iteration.py` expectations** from
  stratified one-shot to wave semantics (same fixtures, new activation
  assertions — a depth-1 relation still completes at $K=1$; the ramsified
  same-order drop assertions invert into edge-exists assertions).

**GATE B (Alec commits):** wave forward green;
`pytest test/test_symbolic_iteration.py test/test_sparse_concept_e2e.py -q`
plus the byte-identity files; `MODEL_COMPILE=eager` smoke above.

---

## Task 5 — Kripke groundedness probe (host-side diagnostic)

**Files:**
- Modify: `bin/Spaces.py` (new method next to `cs_forward_content`)
- Test: `test/test_iterated_symbolic_wave.py` (Task 6 exercises it)

- [x] **5.1 Implement:**

```python
@torch.no_grad()
def cs_groundedness_probe(self, a_0, k=None, tol=1e-3):
    """KRIPKE groundedness (host-side, eager, report-only). Run 1
    iterates from a^0 = 0 WITH the source: what lights up is GROUNDED
    (support traces to perception -- the least-fixed-point reading).
    Run 2 continues from run 1's terminal state with the SOURCE RELEASED
    (s = 0): rows persisting above ``tol`` are UNGROUNDED -- sustained by
    a loop, answering to nothing outside it (rumination/addiction shape;
    see the design doc's human-problems call-out). Returns
    ``(grounded [N] bool, ungrounded [N] bool)`` or ``None`` inactive.
    Under the additive source, released snap rows decay to 0 by
    themselves (no in-edges) -- no masking needed."""
    if not self._sparse_active() or a_0 is None:
        return None
    N = int(self.nVectors)
    B = int(a_0.shape[-1])
    n_snap, _ = self._order_caps()
    K = int(k) if k is not None else 2 * max(
        1, int(getattr(self, "_symbolic_order", 0) or 0))
    ly = _concept_alloc_of(self).layer(0)
    s = torch.cat([a_0, a_0.new_zeros((N - n_snap, B))], dim=0)

    def _iterate(a, source, steps):
        for _ in range(steps):
            a = ly.wave_step(a, source)
        return a

    lit = _iterate(torch.zeros_like(s), s, K)          # ground-up, source ON
    grounded = lit.abs().amax(dim=-1) > tol
    free = _iterate(lit, torch.zeros_like(s), K)       # source RELEASED
    ungrounded = free.abs().amax(dim=-1) > tol
    return grounded, ungrounded
```

  This is an eager island (like `_chain`) — call it from tests/probe
  scripts only; it must NOT be reachable from the compiled forward.

- [x] **5.2 Quick check:**
  `pytest test/test_iterated_symbolic_wave.py -k grounded -x -q` (test
  written in Task 6 — if executing in order, defer the run to 6.3).

---

## Task 6 — New wave/cycle/groundedness tests

**Files:**
- Create: `test/test_iterated_symbolic_wave.py` (mirror the fixture style
  of `test/test_sparse_concept_e2e.py` — real space, `_sparse_active()`
  on, no `_n_ps_codes` stamps)

- [x] **6.1 Write the suite** (adapt fixture/mint helpers to the e2e
  file's actual names):

```python
def test_depth_d_vine_completes_at_iteration_d(sparse_space):
    """A depth-d vine needs d iterations; tail links complete first."""
    cs = sparse_space
    ids = mint_order0(cs, 4)                    # w1..w4, snapped one-hot
    head = cs.create_joint_concept(ids)         # 3 links, head depth 3
    a0 = one_hot_snap(cs, ids)                  # [n_snap, B=1]
    ly = _concept_alloc_of(cs).layer(0)
    rows = {d: link_row(cs, head, depth=d) for d in (1, 2, 3)}
    acts = wave_trajectory(cs, a0)              # [a^1..a^K], K >= 3
    assert acts[0][rows[1]].abs() > 0.1         # tail link at i=1
    assert acts[0][rows[3]].abs() < acts[2][rows[3]].abs()
    assert acts[2][rows[3]].abs() > 0.1         # head completes at i=3


def test_no_self_edge_via_populate(sparse_space):
    """A relation among its own constituents raises (Quine atom)."""
    cs = sparse_space
    alloc = _concept_alloc_of(cs)
    C = alloc.new_concept()
    alloc.add(C, "part", ("sym", C))            # C = {C}
    alloc.add(C, "whole", ("sym", mint_order0(cs, 1)[0]))
    with pytest.raises(ValueError, match="self-edge"):
        cs._populate_concept_weights(C)


def test_relate_x_x_merges_to_one_edge(sparse_space):
    """Same sym as part AND whole -> ONE untyped edge (documented)."""
    cs = sparse_space
    (x,) = mint_order0(cs, 1)
    sx = cs.singleton_concept(x)
    C = cs.reify_concept(sx, sx)
    ly = _concept_alloc_of(cs).layer(0)
    c_row = ly.row_of(("pool", C))
    s_row = ly.row_of(("pool", sx))
    assert sum(1 for (r, c) in ly._index if r == c_row and c == s_row) == 1


def test_cycle_flagged_by_wave_qe_not_settling(sparse_space):
    """A strong 2-cycle keeps the wave-QE statistic from decaying."""
    cs = sparse_space
    A, B_ = two_pool_concepts(cs)               # reify over distinct syms
    ly = _concept_alloc_of(cs).layer(0)
    add_cycle_edges(ly, A, B_, weight=3.0)      # A <-> B, strong
    run_wave(cs)                                # populates cs._cs_wave_qe
    qe = cs._cs_wave_qe
    assert qe[-1] > 1e-3                        # did not settle -- REPORTED,
    # never raised: cycles are a documented fact (Alec 2026-07-03), the
    # statistic is observability, not enforcement.


def test_groundedness_probe_separates_vine_from_free_loop(sparse_space):
    cs = sparse_space
    ids = mint_order0(cs, 3)
    head = cs.create_joint_concept(ids)         # grounded structure
    A, B_ = two_pool_concepts(cs)
    add_cycle_edges(_concept_alloc_of(cs).layer(0), A, B_, weight=3.0)
    a0 = one_hot_snap(cs, ids)
    grounded, ungrounded = cs.cs_groundedness_probe(a0)
    ly = _concept_alloc_of(cs).layer(0)
    assert grounded[ly.row_of(("pool", head))]        # vine: grounded
    assert ungrounded[ly.row_of(("pool", A))]         # loop: self-sustained
    assert not ungrounded[ly.row_of(("pool", head))]  # vine dies unsourced


def test_weak_loop_decays_strong_loop_persists(sparse_space):
    """Temporal membership: composed degree decides echo vs assembly."""
    cs = sparse_space
    A, B_ = two_pool_concepts(cs)
    ly = _concept_alloc_of(cs).layer(0)
    add_cycle_edges(ly, A, B_, weight=0.3)      # weak: tanh contracts
    a0 = zero_snap(cs)
    g, u = cs.cs_groundedness_probe(a0, k=12)
    assert not u[ly.row_of(("pool", A))]        # transient echo
    set_cycle_weights(ly, A, B_, weight=3.0)    # strong: self-asserting
    g, u = cs.cs_groundedness_probe(a0, k=12)
    assert u[ly.row_of(("pool", A))]            # reverberating/ungrounded
```

  Helpers (`mint_order0`, `one_hot_snap`, `wave_trajectory`,
  `two_pool_concepts`, `add_cycle_edges`, `link_row`) are small
  file-local functions built on the e2e fixture's existing patterns;
  `add_cycle_edges` writes `ly.add_edge(row_A, row_B); ly.add_edge(row_B,
  row_A)` then sets values under `no_grad` (the probe reads trained
  values). `wave_trajectory` calls `cs_forward_content` with the space's
  `_symbolic_order` forced to each $i$ (or refactor the loop body into a
  helper the test can step — prefer forcing $K$, no production change).

- [x] **6.2 Run to failure first**, then fix helpers/impl until:
  `pytest test/test_iterated_symbolic_wave.py -q` — all pass.

**GATE C (Alec commits):** Tasks 4–6 green.

---

## Task 7 — Test sweep: rewrite the ramsified pins

**Files:**
- Modify: `test/test_cs_sparse_weights.py` (44 hits — the bulk),
  `test/test_sparse_concept_e2e.py` (5), `test/test_cs_symbol_table.py`
  (1), `test/test_syntactic_order.py` (1)

- [x] **7.1 Classify every hit** of
  `cs_source_layout|forward_linear_roles|add_concept_edge|_order_caps|order_capacities|order_slice|role_slice|_sparse_families`
  in `test/`:
  - DELETE: dyadic-caps pins, source-layout offset pins, role-block
    width/slice pins, order-legality DROP pins (the defect is fixed —
    replace with edge-EXISTS pins, see 3.1).
  - REWRITE: `add_concept_edge(order, row, role, col)` calls to the v3
    `(row, col)` form; per-order `alloc.layer(k)` structure pins to
    single-layer pins; order-0-edges-raise stays but via the snap-region
    row guard.
  - KEEP: everything reading the discrete store, singleton,
    `meta_word_object`, chain idempotency, `order_slice(0)` snap reads.

- [x] **7.2 The regression floor stays green untouched:**
  `pytest test/test_mereology_word_binding.py test/test_mereology_raise.py test/test_ramsification_table.py -q`
  (ramsification-TABLE tests pin the fold recorder, not edge legality —
  verify, and only touch if they reach the retired APIs).

- [x] **7.3 Full local sweep:** `make test` on a QUIET tree (no editors
  writing `bin/*.py` — `inspect.getsource`-pinned tests). Expected: green
  (mlx D2/D3 xfails excepted).

**GATE D (Alec commits):** `make test` green.

---

## Task 8 — Docs, XSD, configs

**Files:**
- Modify: `doc/plans/2026-07-02-iterated-symbolic-loop.md` (status $\to$
  APPROVED + decisions — done at plan-writing time; verify),
  `doc/Architecture.md` (the sparse-CS section), `data/model.xsd`,
  `data/MM_sparse_concept.xml`, `data/MM_20M_xor.xml`
- Check: `make doc` (LaTeX math only — no Unicode glyphs)

- [x] **8.1 `data/model.xsd`** `symbolicOrder` comment (NO `--` inside XML
  comments): "symbolicOrder is the symbolic iteration budget K of the
  conceptual wave. We are NOT forcing ramsification: the value is the
  maximum possible conceptual order, i.e. the order reached if a novel
  concept were introduced at every iteration. 0 disables the symbolic
  phase."

- [x] **8.2 `doc/Architecture.md`:** the ATTENTION corrections landed
  2026-07-03 pre-execution (sec 2 two-channel split, "Parse time" sec C
  heat-as-bottom-up + AttentionLayer naming, the top-down penetration
  re-attribution, sec A's APPROVED pointer) — do NOT redo them. Remaining
  here: replace sec A's stratified per-order description (role-split
  columns, dyadic capacities, $a_k$ composition formula) with the landed
  wave reading; carry the design doc's cycles posture: loops are a
  DOCUMENTED FACT of un-ramsified taxonomies (and of human minds);
  diagnostics observe (wave-QE, groundedness probe), nothing intervenes;
  solutions are invited, for both human and machine minds. Use "graded
  partial evidence; tail links complete first" for partial activation
  (never "prefix recognition").

- [x] **8.3 Configs:** `data/MM_sparse_concept.xml` `symbolicOrder`
  $1 \to 3$ (sentence vines are depth $\ge 2$; $K = 3$ covers the XOR
  presentations — note the $K$-footgun in the config comment).
  `data/MM_20M_xor.xml`: set the same if it activates the symbolic phase;
  leave `sO=0` controls untouched.

- [x] **8.4 `make doc`** — pandoc/xelatex must pass (LaTeX-math rule).

---

## Task 9 — Experiments: XOR rerun + P4 probes against the wave

**Files:**
- Read: `doc/plans/2026-07-02-two-phase-loops-sparse-relation.md`
  EXECUTION NOTES 4–5 (the demux/attenuation root-causes)
- Create (scratchpad): `probe_wave_grads.py`, `probe_wave_variance.py` —
  recreate the P4 probe pattern (harnesses MUST drive `m.runEpoch`)

- [x] **9.1 XOR reruns:** the sO=0 control must still solve to 0.000
  (byte-identity); the sO=3 `MM_sparse_concept` run replaces the P4
  "undecided plateau at 0.175" data point. Record loss curves; ANY
  Inf/NaN raises (fail-loud house rule).

- [x] **9.2 Gradient probe:** re-measure the early-stage combine-gradient
  ratio under the wave (P4 baseline: $\sim$100$\times$ attenuation).
  Hypothesis to test: the $K$-step settle with the additive source gives
  the symbolic side non-degenerate gradients (richer paths through $s$
  every step). Report the number next to the P4 number.

- [x] **9.3 Snap-blindness probe:** re-measure batch-std of the slot-mean
  readout at init (P4: $\sim$5e-6). The wave does not change the snap —
  expect unchanged; record it so the deferred training-dynamics pass has
  the paired baseline.

- [x] **9.4 Append EXECUTION NOTES** to THIS file: suite counts at each
  gate, deviations, probe numbers, and any mid-execution design
  refinements (house pattern from the two-phase plan).

**GATE E (Alec commits; done):** `make test` green + experiment notes
appended.

---

## Self-review checklist (writer, done 2026-07-03)

- Spec coverage: design-doc migration sketch items 1–6 map to Tasks 2, 3,
  4, 4.3, 6/7, 9; the four approval decisions and three review
  corrections each have a landing site (decisions 1$\to$4.1, 2$\to$2.3,
  3$\to$6.1/8.2, 4$\to$8.1; corrections$\to$8.2, 8.3, 3.4).
- No placeholders: every code step carries the actual code; test helpers
  are named and specified against the e2e fixture pattern (executor
  adapts NAMES, not intent, to the fixture file).
- Type consistency: `_csw_concept_row` returns GLOBAL rows everywhere
  (2.3, 3.4, 5.1, 6.1); `add_concept_edge(row, col, weight)` is the only
  v3 signature; `layer(0)` is the single square `AttentionLayer`
  throughout (constructed in 2.2, defined in 1.4, stepped via
  `wave_step(a, s)` in 4.1 and 5.1); the legacy QKV class is
  `QKVAttentionLayer` after 1.1; wave-QE attribute is `_cs_wave_qe` in
  4.1 and 6.1.

---

## EXECUTION NOTES (2026-07-03, executed Tasks 1-9; all gates green)

Suite counts: 2927 (pre-rework baseline) $\to$ 2942 at Gate D
(`make test`, 0 failures) $\to$ 2942 at Gate E (`make test`, quiet tree),
verbatim: `2942 passed, 45 skipped, 32 xfailed, 7 xpassed, 8 warnings,
4 subtests passed in 727.78s (0:12:07)`. Gate A was FOLDED INTO
Gate B: the plan's Gate-A file list contained v2-forward-dependent tests
that could only go green when Task 4 landed the wave.
Load-bearing deviations and mid-execution refinements:

1. **`add_concept_edge` writes through `_sparse_families(0)`** -- the
   load-bearing `_sparse_fam` registration for getParameters /
   optimizer-rebuild.
2. **The plan's `_retire_concept_edge` is `_drop_concept_edge` in code**;
   the min-support check runs BEFORE row reservation (v2-faithful; the
   sketch had it after).
3. **`wave_step`'s nnz guard removed in review** -- empty-layer shape
   validation instead, fail-loud.
4. **`test_symbolic_iteration.py` needed NO rewrite** -- its configs are
   sO=0; it pins the WS codebook leg, wave-agnostic.
5. **Row keys are NAMESPACED** (`("snap", cid)` / `("pool", cid)`) so one
   cid can hold both rows; `put_row_key` is now caller-less (kept as
   store API).
6. **`relate(x, x)` merges part+whole to ONE untyped edge** (idempotent
   `add_edge`) -- documented, pinned.
7. **MID-EXECUTION DESIGN DECISION (Alec 2026-07-03).** The groundedness
   probe MASKS the EVERYTHING-bias column in BOTH runs
   (`wave_step(a, s, bias=0.0)`) -- the pole is a standing axiom, not
   perception; unmasked it lit bias-bounded rows in run 1 and sustained
   them in run 2, making the probe uninformative on production chains.
   The production wave keeps bias $= 1$.
8. **Ungroundedness analysis** (recorded in the design doc): all
   automatic minting is DAG-by-construction; ungrounded requires
   statement-channel asserted mutual containment plus loop gain past
   $1$ (Hebbian cap $4.0$ / assertion weights).
9. **Config inventory.** MM_sparse_concept is the ONLY parallel
   sO $\ge 1$ config; the other 24 sO=1 configs omit `<serial>` and
   derive serial mode (wave inactive) -- sO there is only the legacy
   mode selector.

Task 9 experiment record (CPU, `MODEL_COMPILE=eager`, `BASIC_SEED=0`;
runs via `bin/Models.py <config>`, probe harnesses drive `m.runEpoch` --
scratchpad `probe_wave_grads.py` / `probe_wave_variance.py`):

10. **XOR sO=0 control** (`data/MM_20M_xor.xml`, full 200 epochs):
    output loss 0.1750 $\to$ 0.0000 (first 0.0000 $\approx$ epoch 120;
    final 0.0000; predictions crisp 0/1, all four correct). Matches the
    P4 record -- run-level byte-identity holds. No Inf/NaN.
11. **XOR sO=3** (`data/MM_sparse_concept.xml`, full 300 epochs):
    0.1750 at epoch 1, a brief excursion (0.36/0.52, epochs 2-3), back
    on the plateau by epoch 5, flat thereafter (min $\approx$ 0.167 near
    epoch 250), FINAL 0.1752, predictions all $\approx 0.50$ --
    UNDECIDED, statistically the P4 sO=1 plateau (0.175). This REPLACES
    the P4 data point: the wave rework does not move the production XOR
    number. No Inf/NaN.
12. **The wave ran DARK in that run (root-caused).** The rectified snap
    read is $\approx 0$ on the settled field (at init: magnitudes
    $\sim$7e-6; after 5 training epochs the terminal activations are
    EXACTLY 0.0 -- the rectifier fully clamps), so the source term is
    dead, `_cs_wave_qe` $= [0, 0, 0]$ exactly, and the symbolic phase
    contributes nothing: the plateau is the subsymbolic pipeline alone.
    Confirms P4's snap-blindness root cause carries to the wave era;
    deferred to the training-dynamics pass.
13. **Capacity finding: the config comment does not reach runtime.**
    Stage CS sizing derives from the stage output width
    (`stage_space_concept = [cs_out[0], ...]`, Models.py
    $\approx$ 5756), so every stage store is $N = 8$ (snap 4 + pool 4),
    NOT the `<nVectors>32</nVectors>` (snap 16 + pool 16) the Task-8/P5
    sizing comment promises. The mint load overflows from batch 1: LOUD
    snap-block overflow fired (events #1/#2/#5/#6 surfaced -- concepts
    7, 8, 10, 11; the stage-0 counter reached 16 in a 5-epoch probe);
    the 4-row pool filled exactly and never overflowed. Trained stage-0
    store: nnz $= 4$ (two metas $\times$ two sym edges, Hebbian $1.4$);
    stages 1-2 stay empty.
14. **Gradient probe** (epoch-1 mean $|$grad$|$ on `cs.combine` params,
    sO=3 vs the sO=0 variant of the same config): sO=3 stage 0 $=$
    3.5e-9 vs stage 2 $=$ 8.6e-6 -- cross-stage attenuation
    $\sim$2400$\times$; stage-0 cross-config ratio (sO=0 / sO=3)
    $\sim$5700$\times$ at epoch 1, $\sim$140-630$\times$ at epochs 2-3.
    P4 baseline: $\sim$100$\times$. The 9.2 hypothesis (richer gradient
    paths through the additive source $s$) is NOT confirmed on this
    config: with the source dead (item 12) the symbolic side feeds no
    gradient, and the two-phase demux recursion attenuates as before or
    worse.
15. **Snap-blindness probe:** batch-std of the slot-mean readout at init
    $=$ 2.2e-6 mean / 8.9e-6 max (P4 $\sim$5e-6) -- paired, unchanged;
    the wave does not touch the snap.
16. **Gate E ran twice (environmental, not the rework).** The first run
    failed ONE test (`test_explicit_dimensions.py::
    TestXorExactCliReconstruction::test_output_mse_is_crisp`, MSE 0.1085
    vs $< 0.05$; passed in isolation): the machine-local torchinductor
    user cache (`/var/folders/.../torchinductor_arogers`) held a
    precompiled header re-touched after its `.pch` was built, so clang
    rejected it (`CppCompileError`) inside CLI-subprocess tests --
    `idempotent.xml` crashed outright in triage; `XOR_exact.xml` fell
    back off inductor and the seeded run under-converged (mushy
    $\approx 0.5$ predictions). The test file is untouched by the
    rework. Cache cleared, both nodes green in isolation, full rerun
    green (the verbatim line above).
