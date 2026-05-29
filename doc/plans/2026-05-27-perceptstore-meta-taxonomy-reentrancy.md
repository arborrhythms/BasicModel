# PerceptStore + Two-Codebook + META Taxonomy + CS Reentrancy

> **Continuation plan.** Builds on the substrate refactor at
> [`doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md`](2026-05-26-two-loop-pi-sigma-substrate.md)
> (Stages 1–6, all landed). This plan adds Stages 7–10 that:
> 1. Replace the Lexicon / MPHF / BPE lookup with a `PerceptStore` (radix trie + hash map + inverse table + codebook + byte fallback).
> 2. Split the unified codebook into independent **Perceptual** and **Symbolic** codebooks bridged by a META taxonomy.
> 3. Add a `MetaLayer` (binary `GrammarLayer`) that performs the perceptual/semantic pairing as a runtime grammar op.
> 4. Reintroduce `<conceptualOrder>` as **CS-on-CS reentrancy** (a symbolic-loop iteration internal to CS) with the iteration count = max taxonomic height reachable per forward pass.
>
> What's retired from the substrate refactor:
> - `SymbolicSpace.insert_paired_word`'s "orth row = copy of PS vector" contract (Stage 1.D+1.B's dual-storage decision).
> - PARALLEL mode's PS-on-CS iteration (Stage 1.E).
> - **Stage 1.A's "PS owns both pi and sigma" decision** — superseded by the per-space sigma/pi ownership revision below.
>
> What survives unchanged:
> - PS single-arg `forward(x_subspace)` signature (Stage 1.A). Body becomes pi-only (see revision below).
> - Butterfly mode on PiLayer and SigmaLayer (Stages 5–6) — the layers themselves still carry the cascade machinery; what changes is which Space owns them.
> - `LanguageLayer` as canonical signal-router parser (Stage 3).
> - `LiftLayer` / `LowerLayer` as binary GrammarLayer subclasses (Stage 4).
> - PiLayer / SigmaLayer inheriting from GrammarLayer (Stage 2).
> - CS as STM container (Stage 1.C); gains TWO sigmas (see revision below).

## Per-space sigma/pi ownership revision (supersedes Stage 1.A)

The Stage 1.A refactor placed both `pi` and `sigma` on `PerceptualSpace`. The 2026-05-27 clarification revises this, and *also* commits to the **per-order CS pipeline** (Ramsified) — one CS stage per `<conceptualOrder>`, each with its own owned operators.

- **`PerceptualSpace.pi`** (PiLayer) — applies pi to incoming IS. PS does **only pi**; the sigma half moves to CS. Single PS (not per-order); subsymbolic is single-pass per the locked decision.
- **`ConceptualSpace[t].sigma_in`** (SigmaLayer, **per-order**) — applies sigma to the incoming contribution at stage `t`. At stage 0 this is the IS→CS fold (the role `sigma_percept` played pre-Stage-1.C). At higher stages, it folds whatever new contribution enters at that order. Ramsified weights — each stage learns its own fold appropriate to its taxonomic depth.
- **`ConceptualSpace[t].sigma_cs`** (SigmaLayer, **per-order**) — the **residual-CS** iteration kernel at stage `t` for **PARALLEL mode** higher conceptual orders. Ramsified weights per stage, so stage `t`'s `sigma_cs` lifts from order `t` to order `t+1` with its own learnable transform.
- **`SyntacticLayer` (via `WordSubSpace.languageLayer`)** — **replaces per-stage `sigma_cs`** as the higher-order iteration kernel when **SERIAL mode** is active. The same per-order CS pipeline applies; per-stage, the signal router dispatches a compose round instead of `sigma_cs` firing. Each stage still owns its `sigma_in` (incoming fold), but `sigma_cs` stays dormant in SERIAL — the SyntacticLayer's per-stage dispatch is the substitute.

### Why per-order CS

The pre-substrate-refactor codebase already had `self.conceptualSpaces: ModuleList[ConceptualSpace]` of length `<conceptualOrder>` (the Ramsified pattern). The substrate refactor preserved that structural list but left individual stages without owned sigma operators after `sigma_percept` retired. This revision **fills in each stage's owned operators** as (`sigma_in`, `sigma_cs`) pairs:

- **Specialization**: each taxonomic depth gets its own learnable sigma. Stage 0 learns "lift from leaf percept to first-order grouping"; stage T-1 learns the deepest taxonomic abstraction. One-size-fits-all weights couldn't specialize this way.
- **Mode coherence**: the per-stage pipeline is the same in both modes. SERIAL walks per-word through the stages with SyntacticLayer dispatch per stage; PARALLEL walks once with `sigma_cs` per stage. The Φ operator in the unified equation becomes per-stage rather than monolithic.
- **Symmetry with PS-side butterfly**: PS.pi is one layer (subsymbolic is single-pass), but each CS stage gets its own pair of sigmas because the symbolic side has order. The butterfly cascade then applies independently per stage.

### Per-stage equation

The unified update applies *per stage* in the pipeline:

```
CS_t1[k] = sigma_in[k]( contribution_t1[k] ) + SS_t1[k] + Φ[k]( CS_t0[k] )
```

where `k` ∈ `[0, conceptualOrder)` indexes the stage, `Φ[k]` = `sigma_cs[k]` in PARALLEL or `SyntacticLayer` dispatch in SERIAL, and `contribution_t1[k]` is the incoming contribution at stage `k` (= `PS.pi(IS_t)` at stage 0; the prior stage's output at stages > 0).

Net result, per mode and per stage:

| Mode | PS step (single) | CS[k] incoming | CS[k] residual |
|---|---|---|---|
| **SERIAL** | `pi(IS_t)` per word | `sigma_in[k]` on the incoming contribution | `SyntacticLayer` compose round at stage k |
| **PARALLEL** | `pi(IS)` once at t=0 | `sigma_in[k]` on the incoming contribution | `sigma_cs[k]` lifts order k → k+1 |

Butterfly mode (Stage 5) is inherited transparently: each `sigma_in[k]` and `sigma_cs[k]` accepts `butterfly=True, N=<auto>` via the GrammarLayer base-class machinery. PS.pi (the single layer) also accepts butterfly.

Implementation work this implies (folded into Stage 10):
1. Drop `self.sigma` from `PerceptualSpace.__init__` (Stage 1.A's per-PS sigma retires).
2. Update `PerceptualSpace.forward(x_subspace)` body: `return self.pi(x.materialize())`.
3. Per-stage CS construction: the existing `self.conceptualSpaces: ModuleList[ConceptualSpace]` list (length `<conceptualOrder>`) becomes the canonical CS pipeline; each `ConceptualSpace` instance gains owned `self.sigma_in` and `self.sigma_cs`.
4. Update `ConceptualSpace.forward(new_idea_subspace)` to apply `self.sigma_in` to the incoming contribution before STM bookkeeping.
5. Forward loop in `bin/Models.py` walks the per-stage CS pipeline:
   - PARALLEL: stage 0 ingests `PS.pi(IS)`; stages 1..T-1 apply their `sigma_cs` to prior stage's output.
   - SERIAL: per word, walk stages 0..T-1; at each stage, sigma_in folds the incoming; SyntacticLayer dispatches one compose round; output feeds next stage.
6. Reverse symmetric: walk stages T-1..0 in reverse with each stage's `sigma_cs.reverse` (PARALLEL) or SyntacticLayer reverse-dispatch (SERIAL).

## Context

The substrate refactor delivered the architectural shape: PS owns input processing (pi + sigma + butterfly), CS holds STM, SS hosts grammar dispatch via the signal router. Two lexicon-layer details remained provisional and surfaced as bugs:

1. **Dual-storage divergence.** Stage 1.D+1.B left PS.vocabulary and SS.codebook's paired orth rows as *independent* trainable Parameters (initial copies that drift with training). The reverse decode picks up `\x01` instead of trained word vectors because: (a) PS.vocabulary trains while SS quantizes against a different vector; (b) only the first paired SS row receives gradient through the dispatch chain; (c) nearest-neighbour lookup snaps to small-norm initial vectors when the reverse output is near-init. Tying the two stores was attempted; the better resolution is **separate codebooks with structural taxonomy bridging**.
2. **Single PS-CS pass can't represent discontiguous percepts.** Forces higher-order concepts to be intensional (prototype vectors), with extensional structure living in the SS taxonomy. The mechanism for walking that structure is **CS reentrancy** — CS iterating on itself T = `<conceptualOrder>` times, each step lifting one taxonomic level.

Together, these become a coherent architecture:

- **Perceptual layer = surface forms**: stable, arbitrary, indexed via radix trie + inverse table for **exact** invertibility. Vectors are a learnable payload, not the carrier of reversibility.
- **Symbolic layer = meanings**: trained for semantic similarity, taxonomic structure, grammatical role binding. Orders > 0 emerge from CS reentrancy.
- **META taxonomy = the bridge**: signed-integer cross-codebook references that pair a percept (positive idx into PS) with a meaning (negative idx into SS). Created at runtime by the `MetaLayer` grammar op; enforced at sentence-parse boundaries.

## Architectural decisions (locked 2026-05-27)

1. **Two codebooks, independent index spaces.** `PS.codebook` and `SS.codebook` are separate `nn.Parameter` stores. Drift between them is permitted; reconciliation is structural (via META taxonomy), not vector-space (via index-tying).

2. **Surface invertibility is structural, not learned.** `PerceptStore` has a per-row `inverse_table: percept_id → canonical bytes`. Reverse-decoding a percept ID is an exact lookup. Vector content carries gradient, not reversibility.

3. **META taxonomy uses signed integers as cross-codebook references.** Positive `i` → `PS.codebook[i]`. Negative `i` → `SS.codebook[-i - 1]` (or some symmetric encoding). A META node's children list contains some of each, all stored as signed integers.

4. **`MetaLayer` is a binary `GrammarLayer` subclass.** Always available as a reduce op on the signal router. Enforced at sentence-parse boundaries (the parser must visit it for word→object slot bindings). Off-parse it fires opportunistically (when temporal-proximity + co-attention signals indicate pairing).

5. **PARALLEL mode's PS-on-CS iteration retires.** Subsymbolic is **single-pass**. One pi + one sigma fire in parallel across percepts; no iteration on the subsymbolic side.

6. **`<conceptualOrder>` returns as CS reentrancy depth.** `CS.forward` iterates on itself T = `<conceptualOrder>` times. Each iteration lifts the represented symbol's order by one (height in the SS taxonomy). Percepts are order 0. CS reverse unwinds T times symmetrically.

7. **CS reentrancy uses a learned sigma-like transform.** `ConceptualSpace.sigma_cs` (a SigmaLayer) is the iteration kernel. Taxonomy meaning emerges from training pressure (loss); the operation itself is mathematically just `tanh(W @ atanh(x) + b)` per step. Invertibility is automatic via LDU. Alternative B (discrete taxonomy parent-lookup) and C (learned-transform-anchored-to-taxonomy) noted but not the default.

8. **`<chunking>` knob remains on PS.** Defaults to `<chunking>radix</chunking>`. Legacy values `lexicon`, `bpe`, `byte` are preserved in git for revival but no longer the default. The active path goes through `PerceptStore`.

9. **Promotion threshold is exposed.** `<chunkPromotionThreshold>` (default `4`) and `<chunkPromotionMinLength>` (default `2`) on `PerceptualSpace`. A byte-fallback chunk promotes to a permanent percept when its hit count reaches the threshold and its length meets the minimum.

10. **SBOW/CBOW redirects to SS.codebook.** The user already noted this is partially happening in serial mode (masked-neighbour context). With two independent codebooks, perceptual similarity training stays on PS (optional) and semantic similarity training goes to SS (default). The `<trainEmbedding>` XML knob is repurposed to indicate which codebook is the SBOW/CBOW target.

## Design

### PerceptStore (Stage 7)

A self-contained store at `bin/PerceptStore.py` (new file) with the following components:

```python
class PerceptStore(nn.Module):
    """Authoritative store for perceptual identity + invertibility.

    Components:
      - radix_trie: RadixTrie holding canonical byte sequences. Supports
        arbitrary-length chunks, prefix sharing, longest-match lookup,
        online insertion, and exact reconstruction.
      - hash_map: dict[bytes -> int] fast cache from canonical chunk
        to percept ID. Mirrors the radix trie's contents; rebuilt on
        load.
      - inverse_table: list[bytes] indexed by percept ID. Maps ID back
        to canonical bytes. Structural invertibility.
      - codebook: nn.Parameter[V, D] learned vector payload per percept.
        V grows as inserts happen; pre-allocated cap with growth.
      - byte_fallback: BytesFallbackEncoder for unknown chunks; tracks
        promotion-candidate hit counts.
      - promotion_threshold: int (XML <chunkPromotionThreshold>, default 4)
      - promotion_min_length: int (XML <chunkPromotionMinLength>, default 2)

    Forward path (lex chunk -> vector):
      1. Lex produces canonical bytes.
      2. hash_map.get(bytes) -> percept_id if known.
      3. else radix_trie.longest_match(bytes) -> permanent percept_id +
         residual bytes (which fall through to byte_fallback).
      4. byte_fallback.encode(residual) returns a temporary vector;
         increments hit count.
      5. If hit count >= promotion_threshold AND length >= min_length,
         insert into trie + hash_map + inverse_table + codebook.

    Reverse path (percept_id -> canonical bytes):
      1. inverse_table[percept_id] -> bytes. Exact, no learning.

    Persistence:
      - state_dict carries codebook (the Parameter).
      - vocab_extras carries radix_trie + hash_map + inverse_table +
        byte_fallback state + per-chunk hit counts. Same pattern as
        the existing BPE ChunkLayer's pure-Python state.
    """
```

**RadixTrie** is the authoritative data structure. Each node represents a byte-string prefix; leaves carry percept IDs. Insertion is online; longest-match is O(L) where L is the chunk length. Persistence is via a flat `(prefix, percept_id, children)` serialization.

**Byte fallback** encodes unknown chunks by composing per-byte vectors (e.g., `sum(byte_codebook[b] for b in chunk)` with normalization), and increments a hit counter for the chunk. When the counter passes threshold, the chunk gets a permanent percept ID + learned vector (init from the byte-fallback encoding).

### Two-codebook split + META taxonomy (Stage 8)

`PerceptualSpace.vocabulary` is replaced by `PerceptualSpace.percept_store: PerceptStore` (Stage 7's class). The PS codebook lives there.

`SymbolicSpace.codebook` becomes purely symbolic — no paired-row contract with PS. Trained for semantic similarity via SBOW/CBOW + symbolic dispatch. The existing `subspace.what` machinery stays.

**Taxonomy storage** moves to `SymbolicSpace.taxonomy`, a Python dict-or-buffer holding parent/child relations as signed integers:

```python
# Convention:
#   positive i  -> PS.codebook row at index i, percept_store inverse_table[i]
#   negative i  -> SS.codebook row at index -i - 1
# A taxonomy entry maps parent_signed_idx -> list of child_signed_idx.
self.taxonomy: dict[int, list[int]] = {}
```

**META nodes** are taxonomy entries whose children list contains at least one positive (PS) and at least one negative (SS) ref. Their parent index is a negative ref (META nodes live on the SS side). A META node's SS row holds a learned "fused" vector representing the bound pair.

**Insert/lookup helpers** on `SymbolicSpace`:

```python
def insert_percept(self, canonical_bytes) -> int:        # delegates to PS.percept_store
def insert_symbol(self, init_vec=None) -> int:           # allocates SS.codebook row
def insert_meta(self, ps_idx, ss_idx, fused_vec) -> int: # allocates META node + SS row
def taxonomy_children(self, signed_idx) -> list[int]
def taxonomy_parent(self, signed_idx) -> int | None
def is_meta(self, signed_idx) -> bool
```

**Reverse decode path** (replaces `decode_reverse_meta`):

```
terminal CS state
  -> SS.codebook nearest match -> sym_idx (negative)
  -> taxonomy_children(sym_idx) -> walk to PS-side children
  -> PS.percept_store.inverse_table[ps_idx] -> canonical bytes
```

Exact at the symbol-correctness level. No nearest-neighbour against vectors at the surface step.

### MetaLayer (Stage 9)

```python
class MetaLayer(GrammarLayer):
    """Bind a perceptual idea to a semantic idea, creating a META node.

    arity = 2; rule_name = "meta"; tier = 'C'.

    forward(left, right):
        left:  CS-tier vector derived from a PS percept.
        right: CS-tier vector derived from an SS symbol.

        Internally:
          - Identify the percept_id (from left's STM-attached origin).
          - Identify the symbol_idx (from right's nearest SS match).
          - Call SymbolicSpace.insert_meta(percept_id, symbol_idx,
            fused_vec=combine(left, right)) if no META node yet exists
            for this pair; else return the existing META vector.
        Returns the META node's SS.codebook vector.

    reverse(parent):
        parent is a META vector. Find the META node via SS.codebook
        nearest match; walk taxonomy_children to recover (left, right).
        Approximate at the discrete identity level; vector-space exact
        per fused_vec storage.
    """
```

Registered with the signal router as a binary reduce op at tier 'C'. Enforced at sentence-parse boundaries (the parser must dispatch it for every word + object pairing in a sentence's parse). Off-parse it can still fire — when CS state's quantization-to-symbol pairs with the next STM slot's quantization, MetaLayer can fire opportunistically. This is the incidental-learning channel.

### CS reentrancy (Stage 10)

#### Unified update equation

The general CS update rule applies **per stage `k`** in the per-order CS pipeline (`k ∈ [0, conceptualOrder)`):

```
CS_t1[k] = sigma_in[k]( contribution_t1[k] ) + SS_t1[k] + Φ[k]( CS_t0[k] )
```

where `contribution_t1[k]` is the incoming contribution at stage `k` (= `PS.pi(IS_t)` at stage 0; the prior stage's output at stages > 0), and `Φ[k]` is the **mode-dependent** per-stage iteration kernel applied to the residual CS state at stage `k`. Per-mode simplification:

- **SERIAL** (per-word ingestion, per-stage pipeline):
  - At stage 0, `contribution_t1[0] = PS.pi(IS_t)` (nonzero — from `PerceptStore.lookup → PS.forward`). At stage k > 0, `contribution_t1[k] = output of stage k-1`.
  - `sigma_in[k]( contribution_t1[k] )` = the per-stage IS→CS fold (or stage-to-stage lift) applied by stage `k`'s owned `sigma_in`.
  - `SS_t1[k]` = symbolic contribution at stage k: **the word's own symbol vector PLUS taxonomy-driven contributions from any META parents/children of that symbol**. This is the "multiple symbolic sigmas via CS reentrancy from SS without grammar" mechanism — when a word's symbol identity has a registered META binding (from earlier MetaLayer dispatch), the META parent's other children automatically contribute their CS-tier vectors via taxonomy walk, producing the richer-than-just-symbol-identity `SS_t1[k]` contribution. Feed-forward semantic association (hearing "apple" activates "fruit", any META-bound object representations, etc.) without requiring grammar to fire. Implementation: SS-side spreading activation up the taxonomy tree from `SS.codebook[word_sym_idx]`, summed.
  - `Φ[k]` = per-stage STM shift + `SyntacticLayer` compose-round dispatch at stage k. The shift moves the stage's `STM[1..7]` down to `STM[0..6]`; the new `CS_t1[k]` lands at `STM[7]`; then the SyntacticLayer (`WordSubSpace.languageLayer`, the signal router) dispatches one compose round over the stage's new STM contents. **This is what replaces `sigma_cs[k]` in SERIAL mode** — the per-stage iteration kernel for higher conceptual orders in SERIAL is the signal router, not the learned `sigma_cs[k]`. Grammar dispatch at each stage is how SERIAL reaches higher orders.
  - "Free of interference" assumes no top-down CS expectation forces a prediction; under that assumption, `sigma_in[k](contribution)` and `SS_t1[k]` add cleanly to the stage's `STM[7]` without contention with `CS_t0[k]`. See "No-top-down-expectation premise" below.

- **PARALLEL** (CS reentrancy across the per-stage pipeline):
  - `contribution_t1[0] = PS.pi(IS)` once at t=0 (subsymbolic fires once for the whole sentence). For stages k > 0, `contribution_t1[k]` is the prior stage's output.
  - `sigma_in[k]` fires per stage, folding the contribution into that stage.
  - `SS_t1[k] = 0` per iteration (no symbolic loop dispatch in PARALLEL; reentrancy is CS-on-CS).
  - `Φ[k] = sigma_cs[k]` (each stage's own learned CS-side lifting transform).
  - Reduces to `CS_t1[k] = sigma_in[k]( contribution_t1[k] ) + sigma_cs[k]( CS_t0[k] )`. Pure CS evolution per stage; no symbolic-loop input competes with the per-stage iteration kernel.

If both modes coexist in a single forward pass (SERIAL ingestion followed by PARALLEL refinement), the order is:

1. SERIAL: for each new word, walk stages 0..T-1; at each stage k, `sigma_in[k](contribution) + SS_t1[k]` → stage-k STM[7]; SyntacticLayer dispatches at stage k; output feeds stage k+1; final stage's output lands in the global STM.
2. PARALLEL: walk stages 0..T-1; at each stage k, `sigma_cs[k]` lifts the prior stage's output by one taxonomic order.

Step 2's input is step 1's output (per stage).

#### Implementation

**Per-stage pipeline.** The existing `self.conceptualSpaces: ModuleList[ConceptualSpace]` (length `<conceptualOrder>`, structurally preserved through the substrate refactor) becomes the canonical CS pipeline. Each `ConceptualSpace` instance in this list owns **its own** pair of SigmaLayers — weights are Ramsified per stage, never shared:

```python
# Inside ConceptualSpace.__init__ — one pair per stage instance.

# 1. Incoming fold (per-stage). At stage 0 folds PS.pi(IS); at stage k>0
#    folds the prior stage's output. Fires in both SERIAL and PARALLEL.
self.sigma_in = SigmaLayer(
    percept_dim, percept_dim,
    invertible=True, monotonic=False,
    butterfly=True, N=<auto>,  # inherits butterfly per Stage 5
)

# 2. Residual-CS iteration kernel (PARALLEL mode only). Each stage's
#    sigma_cs lifts order k -> k+1 with its own learnable transform.
self.sigma_cs = SigmaLayer(
    percept_dim, percept_dim,
    invertible=True, monotonic=False,
    butterfly=False,  # CS reentrancy is per-position; no flattening needed
)
```

Per-stage `forward` applies that stage's owned `sigma_in` to the incoming contribution before STM bookkeeping:

```python
# ConceptualSpace.forward (called per stage by the outer pipeline driver).
def forward(self, new_idea_subspace):
    # Apply this stage's IS→CS fold (sigma half of the old Stage 1.A body).
    contribution = new_idea_subspace.materialize()
    folded = self.sigma_in(contribution)
    # STM bookkeeping (unchanged from Stage 1.C) — owned per stage.
    new_idea_subspace.set_event(folded)
    self._stm_shift_and_push(new_idea_subspace)
    return folded  # so the pipeline driver can feed it into stage k+1
```

**PARALLEL pipeline driver** (in `Models.py._forward_body`) walks stages 0..T-1 and applies each stage's `sigma_cs` to the residual:

```python
# Pipeline driver — outside ConceptualSpace, in _forward_body.
def _forward_parallel_pipeline(self, ps_event):
    x = ps_event  # = PS.pi(IS), fires once at t=0
    for k, cs_k in enumerate(self.conceptualSpaces):
        # Stage k folds the incoming contribution via its own sigma_in.
        folded = cs_k.sigma_in(x)
        # Residual lift: stage k's own sigma_cs (Ramsified per order).
        # CS_t1[k] = folded + sigma_cs[k]( CS_t0[k] ).
        lifted = cs_k.sigma_cs(cs_k.current_state())
        x = folded + lifted
        cs_k.update_state(x)
    return x  # terminal stage's output
```

Reverse symmetric (walks stages T-1..0):

```python
def _reverse_parallel_pipeline(self, y):
    for k in reversed(range(len(self.conceptualSpaces))):
        cs_k = self.conceptualSpaces[k]
        # Subtract residual lift, then invert the per-stage fold.
        lifted = cs_k.sigma_cs(cs_k.prev_state())
        folded = y - lifted
        y = cs_k.sigma_in.reverse(folded)
    return y  # IS-tier reconstruction
```

**SERIAL pipeline driver** (in `Models.py._forward_body_per_word`) replaces each stage's `sigma_cs[k]` with a `SyntacticLayer` compose-round dispatch:

```python
def _forward_serial_pipeline(self, ps_event_for_word_t):
    x = ps_event_for_word_t  # = PS.pi(IS_t), fires per word
    for k, cs_k in enumerate(self.conceptualSpaces):
        folded = cs_k.sigma_in(x)
        # Stage-k STM push.
        cs_k.push_to_stm(folded)
        # SyntacticLayer compose round at stage k (replaces sigma_cs[k]).
        composed = self.word_subspace.languageLayer.compose(cs_k.stm_view())
        x = composed
    return x
```

SERIAL's per-stage signal-router dispatch is the substitute for PARALLEL's `sigma_cs[k]`. The dispatch site is already wired in `Models.py._forward_body_per_word` from Stage 3; the change is to call it **per stage**, not per word only.

`<conceptualOrder>` continues to mean "number of stages in the CS pipeline":
- In PARALLEL: T stages with per-stage `sigma_cs[k]` lifting order k → k+1.
- In SERIAL: T stages with per-stage `SyntacticLayer` compose rounds.

Default `1`. MM_xor uses `3`. MM_5M uses 5–8.

**Why option (A)** (learned sigma) over (B) discrete taxonomy parent-lookup for `sigma_cs`: invertibility is automatic via LDU; gradient flows uninterrupted; taxonomy structure emerges as a training-pressure equilibrium rather than a pre-programmed lookup. Risk: the learned sigma may not perfectly align with the desired taxonomy structure unless the loss encourages it. Mitigation: SBOW/CBOW on SS provides the implicit taxonomy training signal; explicit taxonomy-walk training data can be added later if needed.

## Stages

### Stage 7: PerceptStore implementation

**Goal.** Create `bin/PerceptStore.py` with radix trie + hash map + inverse table + codebook + byte fallback + promotion. Wire it into `PerceptualSpace` as `self.percept_store` (replacing the `Lexicon`-based `self.vocabulary` for the `<chunking>radix</chunking>` config path).

**Prerequisites.** None beyond substrate refactor.

**Files modified.**

- `bin/PerceptStore.py` (NEW) — `PerceptStore`, `RadixTrie`, `BytesFallbackEncoder` classes.
- `bin/Spaces.py`:
  - `PerceptualSpace.__init__` — read `<chunking>` knob; if `"radix"` (the new default), construct `self.percept_store = PerceptStore(...)`; for legacy values, retain the existing `Lexicon`/`Embedding`/`MPHF`/`ChunkLayer` wiring under `<chunking>lexicon|bpe|byte</chunking>`.
  - `PerceptualSpace.vocabulary` property — returns `self.percept_store` when radix, else falls back to legacy lexicon. Keeps the existing API stable.
  - `PerceptualSpace.forward` — already single-arg post-Stage-1.A; the lookup goes through `self.percept_store.lookup(...)` when the radix path is active.
- `data/model.xsd` — add `<chunking>radix</chunking>` as a valid enum value; add `<chunkPromotionThreshold>` and `<chunkPromotionMinLength>` elements under `PerceptualSpace`.
- `data/MM_xor.xml` — switch `<chunking>lexicon</chunking>` → `<chunking>radix</chunking>` (or omit, letting the default fire); add `<chunkPromotionThreshold>4</...>` if non-default.

**Tests.**

- New `test/test_percept_store.py`:
  - Radix trie insertion + longest-match.
  - Hash map cache consistency.
  - Inverse table exact roundtrip: `percept_id → bytes → percept_id`.
  - Byte fallback encoding produces a vector + increments counter.
  - Promotion triggers after `promotion_threshold` hits and `>= min_length`.
  - Codebook grows on insert; existing rows preserved.
- Updated `test/test_perceptual_loopback.py`:
  - `PerceptualSpace.percept_store` exists for `<chunking>radix</chunking>` config.
  - Legacy chunking paths still work for `<chunking>lexicon|bpe|byte</chunking>`.

**Acceptance.**

- `make xor` runs end-to-end with `<chunking>radix</chunking>`; words are inserted into the PerceptStore; inverse table reproduces the surface bytes exactly.
- Targeted tests pass.
- `git status` shows uncommitted modifications. No commits, no stashes.

### Stage 8: Two-codebook split + META taxonomy

**Goal.** Decouple `PS.codebook` and `SS.codebook`. Move taxonomy storage onto `SymbolicSpace.taxonomy` with signed-integer cross-codebook references. Retire `insert_paired_word`'s "orth = copy of PS vector" contract.

**Prerequisites.** Stage 7 (PerceptStore is the PS codebook owner).

**Files modified.**

- `bin/Spaces.py`:
  - `SymbolicSpace.insert_paired_word` — retire (or make it a thin wrapper that calls `insert_meta`).
  - Add `SymbolicSpace.insert_percept(canonical_bytes) -> int` (delegates to `PS.percept_store`).
  - Add `SymbolicSpace.insert_symbol(init_vec=None) -> int` (allocates SS.codebook row; returns negative signed index).
  - Add `SymbolicSpace.insert_meta(ps_idx, ss_idx, fused_vec=None) -> int` (allocates META node + SS row; registers taxonomy children).
  - Add `SymbolicSpace.taxonomy: dict[int, list[int]]` and helpers (`taxonomy_children`, `taxonomy_parent`, `is_meta`).
- `bin/Models.py`:
  - `_decode_reconstructed_inputs` — replace vector-nearest-neighbour with `SymbolicSpace.taxonomy_children(sym_idx) → PS percept ids → percept_store.inverse_table[ps_idx]`. Drop the existing bounds-check defensive code (no longer needed; lookup is structural).
- `data/model.xsd` — `<SymbolicSpace>` gets a `<taxonomy>` element for serialization.

**Tests.**

- New `test/test_two_codebook_meta_taxonomy.py`:
  - `insert_percept` creates a PS row and inverse_table entry; returns positive idx.
  - `insert_symbol` creates an SS row; returns negative idx.
  - `insert_meta(ps_idx, ss_idx, fused_vec)` creates a META node; taxonomy_children returns both idxs.
  - Reverse decode: terminal CS state → SS quantize → walk META → PS row → bytes (exact).
- Update `test/test_unified_lexicon_codebook.py`:
  - The `insert_paired_word` contract no longer requires SS row = copy of PS row.
  - Replace with `insert_meta` + structural assertions.

**Acceptance.**

- `make xor`'s reverse decode produces the correct surface bytes (not `\x01 \x01 \x01`).
- The dual-storage divergence is structurally impossible (PS and SS don't share index space).
- Targeted tests pass.

### Stage 9: MetaLayer as binary GrammarLayer

**Goal.** Implement `MetaLayer(GrammarLayer)` and register it with the signal router as a binary reduce op at tier 'C'. Enforce at sentence-parse boundaries.

**Prerequisites.** Stages 7 + 8.

**Files modified.**

- `bin/Layers.py`:
  - New `class MetaLayer(GrammarLayer)`. `arity=2`, `rule_name="meta"`, `tier='C'`.
  - `forward(left, right)` identifies the percept_id and symbol_idx, calls `SymbolicSpace.insert_meta` (idempotent — returns existing META if one is already registered for the pair), returns the META vector.
  - `reverse(parent)` walks the taxonomy to recover left and right.
- `bin/Language.py`:
  - Register `MetaLayer` in the signal router's tier-'C' reduce op set.
  - Update the parser's reduce-op rule list to include `"meta"` as the binding op for word↔object pairings at sentence boundaries.
- `bin/Spaces.py`:
  - `SymbolicSpace.insert_meta` is idempotent on (ps_idx, ss_idx); subsequent calls return the same META idx + update the fused vec via EMA.

**Tests.**

- New `test/test_meta_layer.py`:
  - `MetaLayer(arity=2, rule_name="meta", tier='C')`.
  - `forward(left, right)` returns a META vector; the META node is registered on the SymbolicSpace taxonomy.
  - `reverse(parent)` returns a pair of vectors corresponding to the META's children.
  - Idempotent: calling `MetaLayer.forward` twice on the same pair returns the same META idx.
  - Signal-router dispatch fires `MetaLayer` at sentence-parse boundaries.

**Acceptance.**

- `make xor` produces a META node per word↔object binding inserted during training.
- META vectors are trainable; they accumulate gradient from the loss.
- Targeted tests pass.

### Stage 10: Per-stage CS pipeline + sigma migration

**Goal.** (1) Migrate `sigma` from PS to CS. (2) Each CS stage in the existing `self.conceptualSpaces` ModuleList gains its own owned `sigma_in` (incoming fold, fires in both modes) and `sigma_cs` (PARALLEL-mode residual kernel). (3) Walk the per-stage pipeline mode-dependently: PARALLEL applies `sigma_cs[k]` per stage; SERIAL replaces `sigma_cs[k]` with `SyntacticLayer` compose-round dispatch per stage. (4) PS becomes pi-only.

**Prerequisites.** Substrate refactor (Stage 1.C provides STM bookkeeping; Stage 3 provides the SyntacticLayer/signal router; Stage 1.A leaves `self.conceptualSpaces` ModuleList structurally in place). Stage 8 (two-codebook architecture) is a clean prerequisite but not strictly required.

**Files modified.**

- `bin/Spaces.py`:
  - `PerceptualSpace.__init__` — drop the SigmaLayer construction (Stage 1.A's `self.sigma`). PS owns only `self.pi`.
  - `PerceptualSpace.forward(x_subspace)` — body becomes `return self.pi(x.materialize())` (drop the `+ self.sigma(x)` term).
  - `ConceptualSpace.__init__` — construct **two SigmaLayers per stage instance**: `self.sigma_in = SigmaLayer(percept_dim, percept_dim, invertible=True, monotonic=False, butterfly=True, N=<auto>)` for the incoming fold; `self.sigma_cs = SigmaLayer(percept_dim, percept_dim, invertible=True, monotonic=False)` for residual-CS lift. Both are Ramsified per stage when ConceptualSpace is instantiated inside `self.conceptualSpaces`.
  - `ConceptualSpace.forward(new_idea_subspace)` — apply `self.sigma_in` to the incoming contribution, then do the existing STM shift + push (Stage 1.C semantics).
  - `ConceptualSpace.reverse(y_subspace)` — symmetric: `self.sigma_in.reverse` for the fold; `self.sigma_cs.reverse` when PARALLEL is unwinding.
- `bin/Models.py`:
  - `_create_per_stage` — already constructs `self.conceptualSpaces: ModuleList` of length `<conceptualOrder>`. Verify each stage gets its own `sigma_in` and `sigma_cs` instances (no shared parameters across stages).
  - `_forward_body` (PARALLEL path) — walk stages 0..T-1: stage 0 ingests `PS.pi(IS)` through its `sigma_in`; stages 1..T-1 apply their `sigma_cs` to the prior stage's output (and `sigma_in` on whatever residual contribution arrives, typically zero per the unified equation in PARALLEL).
  - `_forward_body_per_word` (SERIAL path) — for each word, walk stages 0..T-1: at each stage, `sigma_in[k]` folds the incoming; SyntacticLayer dispatches one compose round at stage k; output feeds stage k+1. After all stages, STM shift + push of the terminal-stage state.
  - `_reverse_body` — already walks stages in reverse (Stage 1.A landed this); update to use each stage's `sigma_in.reverse` and `sigma_cs.reverse` (PARALLEL) or SyntacticLayer reverse-dispatch (SERIAL).
- `data/MM_xor.xml`:
  - `<conceptualOrder>3</conceptualOrder>` — interpretation: in PARALLEL, 3-stage pipeline with per-stage `sigma_cs`; in SERIAL, 3-stage pipeline with per-stage SyntacticLayer compose rounds per word.
- `data/model.xsd`:
  - `<conceptualOrder>` semantics documented per mode (matches the existing semantic; the change is just that each stage owns more layers).

**Tests.**

- New `test/test_cs_reentrancy.py`:
  - `self.conceptualSpaces` is a `ModuleList` of length `<conceptualOrder>`.
  - Each `ConceptualSpace[k]` has its own `sigma_in` and `sigma_cs` (parameter ids differ across `k`; no shared weights).
  - PARALLEL forward walks stages 0..T-1; each stage's `sigma_cs` is called once with its own `CS_t0[k]`.
  - PARALLEL reverse walks stages T-1..0; each stage's `sigma_cs.reverse` and `sigma_in.reverse` are called; roundtrip is exact modulo LDU precision.
  - SERIAL forward per word walks stages 0..T-1; `SyntacticLayer.compose` is called once per stage (replaces `sigma_cs[k]`).
  - Identity init: at `<conceptualOrder>1</...>`, only stage 0 is active; pipeline reduces to the single-stage substrate behavior.
- Updated `test/test_cs_stm_bookkeeping.py`:
  - Each stage's STM is per-stage; the terminal stage's STM is what surfaces to the top-level model state.
- Updated `test/test_two_mode_dispatch.py`:
  - PARALLEL drives the per-stage pipeline through `sigma_cs[k]`; SERIAL drives it through per-stage `SyntacticLayer` dispatch. The `<conceptualMode>` knob selects between these per-stage drivers (no longer between "single CS with T iterations" and "per-word ingestion").

**Acceptance.**

- `make xor` with `<conceptualOrder>3</...>` constructs a 3-stage CS pipeline; each stage's `sigma_in` and `sigma_cs` are distinct `nn.Parameter` blocks.
- PARALLEL forward + reverse roundtrip is exact across the full 3-stage pipeline (modulo LDU precision).
- SERIAL per-word forward + reverse dispatches `SyntacticLayer.compose` once per stage per word.
- Targeted tests pass.

## Critical files

| File | Stage | Change type |
|---|---|---|
| `bin/PerceptStore.py` (NEW) | 7 | Created |
| `bin/Spaces.py` `PerceptualSpace` | 7 | Read `<chunking>radix</...>` knob; construct PerceptStore; route legacy values unchanged |
| `bin/Spaces.py` `SymbolicSpace` | 8 | Drop paired-row contract; add insert_percept / insert_symbol / insert_meta; taxonomy storage |
| `bin/Models.py` `_decode_reconstructed_inputs` | 8 | Structural decode via taxonomy + inverse table |
| `bin/Models.py` `_forward_body` / `_forward_body_per_word` | 10 | Per-stage pipeline drivers (PARALLEL: `sigma_cs[k]`; SERIAL: `SyntacticLayer.compose` per stage) |
| `bin/Layers.py` `MetaLayer` (NEW class) | 9 | Binary GrammarLayer; insert_meta; idempotent |
| `bin/Language.py` signal-router op registry | 9 | Add MetaLayer to tier-'C' reduce ops; sentence-parse enforcement |
| `bin/Spaces.py` `ConceptualSpace.__init__` / `.forward` / `.reverse` | 10 | Per-stage owned `sigma_in` + `sigma_cs`; Ramsified across `self.conceptualSpaces` |
| `data/model.xsd` | 7, 8, 9, 10 | New enums, elements, semantics |
| `data/MM_xor.xml` | 7, 10 | `<chunking>radix</...>`; conceptualOrder semantics |

## Cross-cutting concerns

### What's retired by this plan

- `SymbolicSpace.insert_paired_word`'s "orth row = copy of PS vector" contract (Stage 1.D+1.B's dual-storage decision).
- PARALLEL mode's PS-on-CS iteration (Stage 1.E's iteration cadence). `<conceptualMode>` may simplify or retire.
- Vector-space nearest-neighbour reverse decoding against PS.vocabulary in `_decode_reconstructed_inputs`; replaced by structural taxonomy + inverse table.
- The bounds-check defensive code I added in `Spaces.py:decode_reverse_meta` (no longer needed once decoding is structural).
- The env-gated debug surface in `Models.py:_decode_reconstructed_inputs` (no longer needed for the same reason).

### What survives unchanged

- PS single-arg `forward(x_subspace)` (Stage 1.A).
- PS.pi + PS.sigma butterfly mode (Stage 5).
- Signal router (`LanguageLayer`) as canonical parser (Stage 3).
- PiLayer / SigmaLayer inheriting from GrammarLayer (Stage 2).
- LiftLayer / LowerLayer as binary GrammarLayer subclasses (Stage 4).
- CS as STM container (Stage 1.C); just gains sigma_cs for reentrancy in Stage 10.
- The substrate refactor's flat-slab dim invariant.
- Tetralemma butterfly (Intersection, Union, Conjunction, Disjunction).

### No-top-down-expectation premise (load-bearing assumption)

The unified CS update equation `CS_t1 = PS_t1 + SS_t1 + Φ(CS_t0)` is "free of interference" between the three terms *only under the assumption that the architecture does not generate top-down CS-level predictions* — i.e., there is no expectation signal flowing from CS_t0 that competes with the new word's `PS_t1` + `SS_t1` ingestion at SERIAL time t1.

**Why this matters**: a top-down expectation would mean `CS_t0` carries not just preserved history but also a *prediction* of what `PS_t1` and `SS_t1` "should" be. Adding the actual `PS_t1` + `SS_t1` to a pre-existing prediction would create either (a) destructive interference if the prediction was wrong, or (b) constructive reinforcement if right — either way breaking the clean additive update.

**Current architecture honors this premise**: the substrate refactor does not have a CS-side prediction generator. `CS.forward` in Stage 1.C does STM bookkeeping; no predictive layer projects an expected next-state. So the assumption holds by construction.

**Future revisit**: if a top-down expectation generator is added later (e.g., a forward-anticipation layer that predicts the next word's CS-tier vector from the current STM contents), the unified update equation will need an interference term. Options:
- Predictive coding: `CS_t1 = PS_t1 + SS_t1 + Φ(CS_t0) - prediction(CS_t0)` — the prediction is subtracted out so only the residual (prediction error) gets added.
- Attention modulation: gate `PS_t1` and `SS_t1` contributions by their match against the prediction.
- Explicit expectation slot in STM that's separate from the history slots.

These are out of scope for the current plan; the assumption is flagged so the design space is open for them.

### Documentation updates

- `doc/Architecture.md` — update the Sigma / Pi ownership section to include `sigma_cs` on CS; update modes section to reflect "subsymbolic single-pass; symbolic reentrancy T-step on CS"; document META as a grammar op.
- `doc/Spaces.md` — update PS section for PerceptStore; update SS section for two-codebook split + META taxonomy; update CS section for sigma_cs.
- `doc/Language.md` — add MetaLayer to the grammar ops list.

### XML knob summary

New / changed knobs from this plan:

| Knob | Default | Meaning |
|---|---|---|
| `<PerceptualSpace><chunking>radix</...>` | `radix` (was: `lexicon` or `bpe` per-config) | New default; legacy values preserved |
| `<PerceptualSpace><chunkPromotionThreshold>` | `4` | Byte-fallback chunk promotes after this many hits |
| `<PerceptualSpace><chunkPromotionMinLength>` | `2` | Byte-fallback chunk must be at least this long to promote |
| `<architecture><conceptualOrder>` | `1` | CS reentrancy iteration count per forward pass |
| `<architecture><conceptualMode>` | unchanged or retired pending decision | If retired, single SERIAL mode |

## Verification

Whole-plan acceptance:

1. **`make xor` with `<chunking>radix</...>` and butterfly=true converges and produces correct surface reconstruction.** This is the single most important signal. The reverse path through SS → taxonomy → PS percept → inverse table → bytes should produce something like `'hello world'` (the input) after sufficient training.
2. **Targeted tests pass** for each stage's gate test file plus all the substrate-refactor tests from Stages 1–6.
3. **`grep -nE "insert_paired_word\|paired_orth_to_sem" bin/`** returns only retired-doc breadcrumbs (Stage 8 retirement complete).
4. **`grep -nE "PARALLEL.*PS.forward.*CS"` in bin/** returns only retired-doc breadcrumbs (Stage 10 retirement of PS-on-CS iteration).
5. **Reverse roundtrip** on a minimal config: `_run_pipeline_rev(forward(IS)) ≈ IS` at the byte level (exact via inverse table) for words present in the trained percept store.
6. **`<conceptualOrder>` knob** drives the CS reentrancy iteration count; reverse roundtrips at each T value tested (1, 3, 8).
7. **MetaLayer dispatch test**: a fixed pair of (word, object) presentations produces a single, idempotent META node; the META vector trains via gradient.
8. **No git commits, no stashes** in the working tree across all stages.

## Risks / open items

1. **Learned sigma_cs may not align with taxonomy structure unless the loss provides taxonomy-walk pressure.** Mitigation: SBOW/CBOW on SS contributes; explicit "lift one taxonomy level" training data can be added later.
2. **Promotion threshold sensitivity.** Default 4 is a guess. Configs with noisy data may need 10+; small-corpus configs may want 2. The knob is exposed for tuning.
3. **`<conceptualMode>` simplification.** With PARALLEL's PS-on-CS iteration retired, the SERIAL vs PARALLEL distinction at the substrate level becomes thin. May collapse to a single mode (signal router always dispatched after T-step CS reentrancy). Decided at Stage 10 implementation time.
4. **MetaLayer reverse** (vector → pair) is approximate when no META node yet exists for the parent's quantized identity. The decoding should fall back to "no META binding" rather than fabricating one.
5. **Promoted percepts on resumed runs.** The radix trie + hit counts must persist into checkpoints (`vocab_extras`) so promotion state survives checkpoint load.
6. **Tetralemma butterfly tests** are already broken from my recent element-pair refactor; deferring their rewrite is fine for now.
7. **Stage 9 enforcement at sentence parsing.** "The parser must dispatch MetaLayer for every word↔object pairing in a sentence's parse" is a soft requirement; how strict the enforcement should be depends on grammar config. Worth a brief discussion before implementation.

## Resolved decisions (2026-05-27)

0. **Unified CS update equation** (applies per stage k in the per-order pipeline): `CS_t1[k] = sigma_in[k]( contribution_t1[k] ) + SS_t1[k] + Φ[k]( CS_t0[k] )`. Mode-specific:
   - SERIAL: `sigma_in[k]( contribution_t1[k] )` + `SS_t1[k]` nonzero per word ingestion; `Φ[k]` = per-stage STM shift + `SyntacticLayer` compose-round dispatch. Free of interference under "no top-down expectation" assumption.
   - PARALLEL: `SS_t1[k] = 0`; `contribution_t1[0] = PS.pi(IS)` once; `Φ[k] = sigma_cs[k]`; per-stage CS-on-CS iteration.

00. **Per-space sigma/pi ownership + per-order CS pipeline** (supersedes Stage 1.A):
    - `PerceptualSpace.pi` — pi-only on IS input. (PS no longer owns sigma.) Single PS — subsymbolic is single-pass.
    - `self.conceptualSpaces: ModuleList[ConceptualSpace]` of length `<conceptualOrder>` is the canonical CS pipeline; each stage's weights are Ramsified.
    - `ConceptualSpace[k].sigma_in` — sigma applied to the incoming contribution at stage `k` (IS→CS fold at stage 0; stage-to-stage fold at k > 0), fires in both modes.
    - `ConceptualSpace[k].sigma_cs` — sigma for PARALLEL-mode residual-CS lift at stage `k` (order `k` → `k+1`).
    - `SyntacticLayer` (via `WordSubSpace.languageLayer`) — replaces `sigma_cs[k]` as the per-stage higher-order iteration kernel in SERIAL mode.

1. Two codebooks, independent index spaces.
2. Surface invertibility via radix trie inverse table; vectors carry gradient, not reversibility.
3. META taxonomy uses signed-integer cross-codebook references.
4. MetaLayer is a binary GrammarLayer subclass, always available, enforced at sentence parsing.
5. Subsymbolic is single-pass; no PS-on-CS iteration.
6. `<conceptualOrder>` returns as CS reentrancy depth.
7. CS reentrancy uses option (A): learned SigmaLayer on CS; taxonomy meaning emerges from training.
8. `<chunking>` defaults to `<radix>`; legacy `lexicon`/`bpe`/`byte` preserved for revival.
9. Promotion threshold exposed: `<chunkPromotionThreshold>4</...>`, `<chunkPromotionMinLength>2</...>`.
10. SBOW/CBOW redirects to SS.codebook (with optional PS-side similarity training retained).

## Implementation strategy

The four stages are mostly sequential but some independence exists:

- **Stage 7 first** (PerceptStore). Concrete, well-bounded, can be implemented and tested in isolation. Wires into PS via the chunking knob.
- **Stage 8 right after** (two-codebook + META taxonomy). Builds on Stage 7's PerceptStore. The reverse decode path becomes structural.
- **Stage 9 then** (MetaLayer). Needs Stage 8's `insert_meta` API.
- **Stage 10 in parallel** with Stage 9 if desired (orthogonal to the lexicon work). Or sequentially after Stage 9.

Each stage gets the standard subagent dispatch cycle (per `superpowers:subagent-driven-development`): implementer → spec reviewer → code quality reviewer → mark complete. All work in the existing main branch / working tree per the user's git policy (no commits, no stashes by subagents; user commits at checkpoints).

For implementation in a fresh session, this plan is self-contained — no need to re-derive the architectural decisions. Hand a fresh session this file plus the substrate-refactor master plan, and they can execute Stages 7–10 directly.
