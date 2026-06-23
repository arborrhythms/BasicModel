# Truth-Grounded Reasoning for Queries

> Plan. A small **NeuralToolUser** that uses two hard-edged truth tools —
> `isTrue(A)` and `isPart(A, B)` — inside a `reason()` orchestrator to
> evaluate a claim, **materializing verified true ideas into LTM** so they can
> serve as links in later chains, and **returning the chain**. This wires the
> built-but-uncalled reasoning engine (`reason()` / `consequents()` /
> `verify_relation()` have ZERO production callers — `todo.md` "Reasoning (D2)")
> into a live, gated path. The conceptual framing lives here (the product-doc
> home for it); code comments describe technique only.

## Status (2026-06-23)

**Tool vocabulary = the grammar `<queries>` ops.** The reasoner's hard tools ARE
the introspection ops enumerated in `data/complete.grammar` `<queries>`:
`exist(X)`→isTrue leaf; `equal(X,Y)`→isomorphic=True = fraction of shared
parts & wholes (`_idea_identity`), isomorphic=False = norm of difference;
`part(X,Y)`→parthood; `query(X[,Y])`→LTM lookup (hard retrieval now; the soft
`.where` full-truth-space read is Phase 3); `quantize(X)`→snap to nearest real
idea; `wholes(X)`→proximal containing wholes; `parts(X)`→proximal contained
parts. **An `isPart` chain is repeated `wholes()` toward the goal** — the
canonical climb primitive. `isomorphs(X)` was removed from the grammar (subsumed
by `equal`; deleted from `complete.grammar` + `test/fixtures/transitional_pos.grammar`).

> **Duplication note (`reasoning.py` vs `ConceptualSpace.reason`).** They share
> the modus-ponens hop loop and the store-row iterator (`_partof_rows` ≈
> `_iter_relation_rows`); the target-direction, beam, MIN-trust, direct
> fast-path, posture, and `materialize` are new. The canonical primitive is
> `wholes()`. **Recommended consolidation (pending, touches `Spaces.py`):** add
> `target`/`beam`/`trust_combine` to `ConceptualSpace.reason` + a `rel_type`
> filter on `_iter_relation_rows`, delete `_partof_rows`, and have both climb via
> one shared `wholes()`.

**Landed**, gated dark, full-suite green — `bin/reasoning.py`,
`test/test_truth_grounded_reasoning.py` (50 tests), gates `<queryReasoning>` +
`<answerLossWeight>` (`model.xsd` + `Models.py`); suite **2773 passed, 0
failed**, byte-identical off:

- Phase 0 — `QuerySpec` framing. ✅
- Phase 1 — hard tools `exist`/`is_true`, `is_part_direct`. ✅
- Phase 2 — beam-limited, MIN-trust, target-directed chain (`is_part`); Socrates
  syllogism passes. ✅
- **Consolidation** — `wholes`/`parts`/`_chain_to_target` are now the single
  canonical climb primitives (static) on `ConceptualSpace`; `_iter_relation_rows`
  gained a `rel_type` filter; `reason()` gained `target`/`beam`/`trust_combine`
  (byte-identical at defaults, delegates to `_chain_to_target` when targeted);
  `reasoning.py` deleted `_partof_rows` and delegates. `reason()` and `is_part`
  now climb via ONE loop. ✅
- Phase 4 — lemma write-back (`materialize`), floor-guarded. ✅
- Phase 6 — refuting edge → FALSE/BOTH; **partial-order / cycle guard**
  (`_creates_cycle`) at edge insertion. ✅
- Phase 7 — chain trace rendering. ✅
- Phase 3 (mechanism) — `where_read` (the soft `.where` read over the full
  truth-space via `GlobalAttention`, grounded/keys-detached) + the
  `InterveningIdeaGenerator` (MLP query head → `where_read` → M, recurrent).
  Unit-tested with synthetic typed spaces; **dark**. ✅
- Phase 5 (primitives) — `proof_score` (signed→[0,1] map) + `answer_loss` (NLL
  on the proof score, differentiable through the soft route only) + the
  `<answerLossWeight>` gate. ✅

**Remaining — needs a live model + a training run to VALIDATE (not just
unit-test):**

- Wire `where_read`/`InterveningIdeaGenerator` into the live forward via
  `model._addressable_spaces` (`Models.py:8095`) behind `<queryReasoning>` — the
  consumer of the parked `_global_attention_obs` + the recurrent threading.
- The `<queryReasoning>` dispatcher in `serve.py`/`infer`: parse a query into a
  `QuerySpec` (package the chart operands), run the reasoner, emit the posture.
- Actually train the policy with `answer_loss` on a QA task and confirm the
  generator learns useful intermediates.
- `verify_relation` episodic grounding (Workstream G, deferred — no source).

## 0. The core principle — soft guesswork, hard reasoning

**The reasoning is hard-edged; only the guesswork is neural.**

| | What it is | How it is implemented |
|---|---|---|
| **HARD** (exact, discrete, deterministic) | the *tools* and the *deduction*: `isTrue(A)`, `isPart(A,B)`, parthood/identity coverage, trust composition, the verdict | exact methods — `_idea_parthood`, `_idea_identity`, `ground`/`isTrue`, the `reason()` step. No gradient flows *through* a verdict. |
| **SOFT** (neural, learned, differentiable) | the *guesswork*: **which** tool to call, on **which** operands, **which** intermediate idea `M` to try when `A→B` is not direct, **which** of several candidates to keep | the existing signal-router soft-superposition policy (`LanguageLayer` `Language.py:4744`, `superposition_scale` `Language.py:6134`, soft-DP `logsumexp` scoring `Language.py:6187`) — a chooser already in the gradient path. |

The NeuralToolUser is a *policy over tool calls*, not a differentiable prover.
When it does not know the answer it **guesses** (soft) a candidate operand or
intermediate, then **checks** the guess with a hard tool. The returned chain is
a sequence of hard-verified steps; learning only improves the guessing — it
never softens the deduction. (This is the same chooser-over-exact-ops pattern
the surviving signal router uses for grammar; the legacy hard-parse
`NeuralToolUser` was retired 2026-06-22 — this is a small new tool-user scoped
to reasoning, reusing that soft-route substrate rather than inventing ML.)

## 1. Goal

Answer a query by reducing it to **at most two atomic judgments over concept
extensions** and grounding each in stored truth:

1. **All queries reduce to `isTrue(A)` or `isPart(A, B)`** for concept
   extensions `A`, `B`. Every other surface form (`isEqual`, `exist`,
   open-variable "what is part of B?") is built from these two.
2. **`isTrue(A)`** holds iff `A` exists as an **ultimate (absolute) truth**,
   OR `A` is a **single idea with positive trust**. No chaining.
3. **`isPart(A, B)`** holds iff it is **deducible directly**, OR there is a
   **chain of true intervening ideas** connecting `A` to `B`. Search is
   **candidate-limited** — the tool-user emits a small ranked set of candidate
   chains, not an exhaustive closure.

Deliverable: a `reason()` orchestrator driven by the NeuralToolUser that
produces a posture (`TRUE`/`FALSE`/`UNKNOWN`/`BOTH`), a confidence, optional
variable bindings, a **chain** (the proof trace), and a training signal so the
*tool-use policy* is learned.

## 2. The reduction, formally

| Query surface | Reduces to | Grammar layer (already exists) |
|---|---|---|
| "does A exist / is A true?" | `isTrue(A)` | `ExistLayer` `Language.py:4290` (rule `exist`) |
| "is A part of B?" | `isPart(A, B)` | `IsPartLayer`→`QueryPartLayer` `Language.py:4071,4260` |
| "is A equal to B?" | `isPart(A,B) ∧ isPart(B,A)` | `QueryEqualLayer` `Language.py:4265` (mutual parthood) |
| "what is part of B?" | `isPart(X, B)`, X open | `QueryLayer` + variable binding |
| "is A a kind of B?" (taxonomic) | `isPart(A, B)` under the **extensional** reading | see §3.1 |

The query/assertion split already exists: a rule with `query="true"` is routed
by `_dispatch_method_name_for_rule` (`Language.py:4276`) from the **assertive**
`isPart`/`isEqual` to the **interrogative** `queryPart`/`queryEqual`, which
dispatch to `QueryLayer` (`Language.py:4200`). `QueryLayer.forward` returns a
graded-parthood bivector `[pos = _parthood_geometric(left,right), neg = 0]` in
`[0,1]` (`Language.py:3975,4241`). **The parser is not the gap — the gap is a
consumer that reads those operands, calls the tools, and emits a chain.**

`isEqual` is mutual parthood (already: `QueryEqualLayer` returns
`part(A,B)·part(B,A)`, `Language.py:4271`); `exist`/`isTrue` is the leaf
predicate. Subsumption ("A is a kind of B") is `isPart` over **extensions**
(§3.1). So the whole query language collapses to `{isTrue, isPart}` plus
boolean combination.

## 3. Truth substrate — what "true" and "part of" mean here

### 3.1 Two relations, kept distinct (extensional subsumption)

Keep is-a (taxonomic subsumption) and part-of (mereology) as distinct
relations, but unify the *query* by reading subsumption **extensionally**:

```
isPart(A, B)  ≡  extension(A) ⊆ extension(B)
```

- **Order-0 / physical parthood** (`paw ⊑ cat`, `byte ⊑ word`) — the `.where`
  meronomy, geometric and transitive over spatial extents
  (`RunStructureLayer.contained_mask` `Layers.py:3114`,
  `record_cross_tower_meronomy` `Spaces.py:13241`).
- **Taxonomic subsumption** (`men ⊑ mortal`, `Socrates : man`) — a relation
  over symbols whose geometric meaning is *equivalence-class subset*. Stored as
  a `REL_PARTOF` row / WS-META edge, NOT as a `.where` extent.

Both are tested by the same hard coverage metric `_idea_parthood(C, A)`
(`Spaces.py:14213`) — "fraction of C's signed energy that A also carries",
`[0,1]`. **Pitfall:** the `men ⊑ mortal` edge is sound *only* extensionally; it
must not be composed with order-0 `.where` transitivity (the classic
part-whole transitivity fallacy). Keep the two edge kinds on distinct
`rel_type`s. ("Socrates promoted to a token of type man" = instance checking:
the most specific class whose extension contains the individual.)

### 3.2 Where truth lives — the unified store

`TernaryTruthStore` (`Layers.py:7470`) is the home and is **already wired for
writes** when `<ltmConsolidation>` is on (`Models.py:7700→7753/7758` appends
each STM end-state; `provision_ltm` `Models.py:2417` loads `<truthSet>` at
load). Each row is `(NP1, VP, NP2)` + monotonic `timestamp` + **separate**
scalar `trust ∈ [-1,1]` (NOT baked into magnitude):

| `rel_type` | row shape | meaning | use |
|---|---|---|---|
| `REL_NONE` (0) | `(NP1, ·, ·)` | an **absolute idea** | candidate for `isTrue(A)` |
| `REL_PARTOF` (1) | `(NP1, VP, NP2)` | `NP1 ⊑ NP2` | direct `isPart` edge + chain hop |
| `REL_IMPLIES` (2) | `(NP1, VP, NP2)` | `NP1 → NP2` | modus-ponens chain hop |
| `REL_OTHER` (3) | `(NP1, VP, NP2)` | any learned predicate | (not used by these two queries) |

Readers: `ideas()`, `relations(rel_type)`, `row(idx)`
(`Layers.py:7615,7605,7582`); writers: `append_idea`, `append_relation`
(`Layers.py:7570,7575`). When `<ltmConsolidation>` is off, the legacy split
(`TruthLayer` `Layers.py:5783` + `RelativeTruthStore` `Layers.py:7307`) is in
play; `_reasoning_store()` (`Spaces.py:13966`) + `_iter_relation_rows`
(`Spaces.py:13987`) already abstract both, so the tool-user targets the
abstraction, not a store. **This store is where reasoning materializes new true
ideas (§4.4).**

### 3.3 Trust: tetralemma → signed scalar (hard)

Each relation carries a **tetralemma** `(t,f,b,n)` (TRUE/FALSE/BOTH/NEITHER,
`_tetralemma_trust` `Spaces.py:13826`), **collapsed for storage** to
`trust = t − f ∈ [-1,1]` (`Spaces.py:14039`; BOTH/NEITHER are not objects of
knowing). So:

- **"single idea with positive trust"** = a `REL_NONE` row with `trust > 0`
  (read `store.row(idx)['trust']` — no un-baking).
- **"ultimate / absolute truth"** = an axiom: a provisioned `<truthSet>` row,
  or a `TruthLayer` activation whose `ground()`/`isTrue()` DoT `> 0`
  (`Models.py:1360,1378`).

These are exactly the **premise/assumption** vs **derived** node split of a
truth-maintenance system; §4.4 makes the *derived* node concrete by writing it
back to LTM.

## 4. The NeuralToolUser

A small policy that, given a `QuerySpec`, sequences hard tool calls to evaluate
the claim and returns a chain. The tools are exact; the *selection/proposal* is
the soft signal-router policy (§0).

### 4.1 Tool — `isTrue(A)` (hard, leaf, no recursion)

```
isTrue(A):                                              # returns signed DoT in [-1,1]
    r = best REL_NONE row by _idea_identity(A, r.np1)   # Spaces.py:14234
    if r and _idea_identity ≥ τ_id and r.trust > 0:     return +r.trust        # single idea, positive trust
    d = model.isTrue(A)                                 # Models.py:1378 (TruthLayer/ground coverage)
    if d > 0:                                           return d                # ultimate truth
    if (r and r.trust < 0) or d < 0:                    return negative         # refuting evidence
    return 0.0                                          # UNKNOWN — honest, not a hallucination
```

### 4.2 Tool — `isPart(A, B)` direct (hard)

```
isPart_direct(A, B):                                    # returns score in [0,1] or ⊥
    if _idea_parthood(A, B) ≥ θ:                        return _idea_parthood(A,B)   # Spaces.py:14213
    if REL_PARTOF row (A→B) with trust > 0 exists:      return trust
    if B ∈ taxonomy_parents(A) (WS-META):               return meta_trust            # Spaces.py:16953
    return ⊥
```

### 4.3 Tool — `propose_intermediate(A, B)` (soft guess) → `reason()` (hard chain)

When `isPart_direct(A,B) = ⊥`, the tool-user must find a **chain of true
intervening ideas**. This is the one place guessing happens — and the guess is
not merely *selecting* a stored next-hop, it is **creating** a candidate
intervening idea `M` (§4.3a):

- **SOFT — create candidates.** The intervening-idea generator (§4.3a) emits the
  **top-K** candidate bridges `{M₁…M_K}` (the "several candidates" beam),
  conditioned on the goal and grounded in LTM.
- **HARD — verify each guess.** For each `Mᵢ`, the exact `reason()` step
  (`ConceptualSpace.reason` `Spaces.py:14255`) accepts the hop only if
  `_idea_parthood(A, Mᵢ) ≥ θ` **and** `_idea_parthood(Mᵢ, B) ≥ θ` (or `Mᵢ`
  fires a stored relation whose antecedent covers the frontier and whose
  consequent covers `B`). The hop is real or it is not; no softness in the test.

`reason()` already does the enumerate → cover → apply → feed-back loop with
`max_steps` and per-relation single-fire. Two changes turn it into the
tool-user's hard chain executor:

1. **Target test + early-exit** when `_idea_parthood(derived, B) ≥ θ` (forward
   from `A`); optionally also expand backward from `B` and meet in the middle.
2. **Beam + MIN chain trust.** Keep only the policy's top-K frontier concepts
   per step (the candidate cap); score a chain by **MIN over its hops** (a chain
   is only as true as its weakest intervening idea). *(Today `reason()` composes
   trust by PRODUCT `t₁·t₂` `Spaces.py:14324`; product over-penalizes long
   chains. Switch to MIN here and document it inline.)*

`consequents()` (`Layers.py:7396`) and `evaluate()` (`Layers.py:7419`) are the
`REL_IMPLIES` one-step / one-shot analogues for implication chains.

### 4.3a `query()` — the `.where`-indexed read the generator conditions on

The hard chain executor can only fire when a usable `M` exists, so the soft half
must **create** an intervening idea that bridges the frontier `A` to the goal
`B`. It does not create from nothing: it conditions on the result of a
**`query()`** primitive that `.where`-indexes a single recalled idea out of the
**whole truth-space** — input, the three codebooks, STM, and LTM — and that
result **recurs** as state to the next reasoning iteration.

**`query(s)` is already assembled and sitting dark.** `_addressable_spaces`
(`Models.py:8095`) gathers exactly that truth-space as one typed, addressable
registry — `SPACE_INPUT` (the staged word brackets, with real environmental
`.where`, `Models.py:8118`), `SPACE_PART`/`SPACE_WHOLE`/`SPACE_SYMBOL` (the three
idea codebooks, `Models.py:8158,8166,8185`), `SPACE_STM` (live STM rows,
`Models.py:8129`), `SPACE_LTM` (the TernaryTruthStore rows = previous reasoning
steps, `Models.py:8139`). `GlobalAttention.forward` (`Spaces.py:520`) competes
**one** distribution across all of them and returns a **typed `.where`** (which
space + `[start,end]` bracket) plus the soft-read content `Σ αₖ·keyₖ`;
`_global_attention_step` (`Models.py:8190`) already runs it and parks the result
on `_global_attention_obs` — but never consumes it. **That parked read is
`query()`.** The only missing piece is a reasoning consumer that calls it per
step and threads the result.

So one reasoning iteration is:

1. **`query(sₜ)` — `.where`-index the truth-space.** Drive `GlobalAttention`
   with a query derived from the state `sₜ` (the goal `B`, the frontier `A`, and
   the previous result `rₜ₋₁`). It returns `rₜ = Σ αₖ·keyₖ` — **one recalled
   idea** — tagged with its typed `.where` (whether it came from current input,
   STM, a codebook prototype, or LTM). The provenance matters: an idea recalled
   from `SPACE_LTM` is a prior reasoning step; from `SPACE_INPUT` it is a given;
   from a codebook it is a known concept prototype. Because `rₜ` is a weighted
   blend of **real, detached** keys (`Spaces.py:573`), it is grounded on the
   manifold of real ideas by construction, and gradient flows only through `α`
   (and the small query head) — never through the ideas (respects §0).
2. **Generator — propose `M` conditioned on `rₜ`.** A small MLP query head reads
   `[A ; B ; rₜ]` at full content width `Dc` (the **whole** solution space, not
   a partition slice) and produces the `concept_q` that *next* `query()` will
   index with — i.e. it learns *where in the truth-space to look* for the bridge.
   The proposed `M` is `query()`'s readout; the top-K `α` is the candidate beam
   ("several candidates"), for free.
3. **Escalate to genuine synthesis only if needed.** If no *blend* of recalled
   ideas bridges, σ-fold the top-K read into ONE new code that geometrically
   contains them — `synthesize_over_set` (`Layers.py:3018`, the M→1 order-raising
   fold) — then snap it to its nearest real idea. This is the "invent a
   higher-order intermediate that was never stored" path.
4. **Verify, materialize, recur.** Hard `_idea_parthood` checks (§4.3); on
   success, write `A→M`, `M→B` (or `A→B`) back to LTM (§4.4); and **carry `rₜ`
   into `sₜ₊₁`** so the next iteration's `query()` reads a truth-space that now
   contains this step. Reasoning thus traces a `.where`-indexed trajectory whose
   own results become future `query()` targets — the recurrence the design asks
   for.

**Why a query-head-into-`query()`, not a free MLP that emits `M` directly.** A
free `M = MLP([A;B;pool(LTM)])` can land anywhere in `Dc`-space, including where
no stored idea supports it — a plausible but unverifiable bridge. Routing the
MLP's expressivity into the *query* and taking `M` as the `.where`-indexed
readout over the real truth-space keeps every proposal on the real-idea
manifold, gives the beam from top-K `α`, supplies the typed-`.where` provenance,
and reuses the wired-but-dark `<globalAttention>` path (its `consume_gate`
already trains under the answer loss). The free-MLP form survives only as step
3's synthesis, always grounded by the nearest-idea snap + the hard verify before
it can enter a chain.

### 4.4 Building true ideas in LTM (the derived node, materialized)

When a chain verifies a new fact, the tool-user **writes it back** as a true
idea so it is a reusable link next time (lemma caching / TMS derived node made
concrete):

- A derived conclusion `isPart(A,B)` with chain score `s` (MIN over hops) →
  `TernaryTruthStore.append_relation(A, partOf, B, rel_type=REL_PARTOF,
  trust=s)` (`Layers.py:7575`).
- A derived ground truth → `append_idea(C, trust=s)` (`Layers.py:7570`).
- Trust is the **hard** MIN-composed score, clamped `[-1,1]`. The store already
  rides the `state_dict`, so materialized lemmas persist and shorten future
  chains.
- `verify_relation` (`Spaces.py:14337`) can later nudge a materialized
  relation's trust toward observed order-0 support — wired once an episodic
  source exists (Workstream G, deferred).

### 4.5 What the tool-user returns

A ranked list of **candidate chains**, each
`{steps:[(tool, operands, verdict, trust)], score, posture}` — never a bare
boolean. A beam-limited, similarity-guessed search is a *candidate generator*;
keep that honesty in the API so low-margin answers read as defeasible.

## 5. Answer posture (hard)

```
support_true  = max(isTrue(A) > 0,  best isPart chain score for the claim)   # OR / MAX over candidates
support_false = max(isTrue(A) < 0,  best chain score for ¬claim)

TRUE     if support_true  ≥ τ  and support_false <  τ
FALSE    if support_false ≥ τ  and support_true  <  τ
BOTH     if support_true  ≥ τ  and support_false ≥ τ     # contradiction → tetralemma BOTH / TMS nogood
UNKNOWN  otherwise
```

`BOTH` is the defeasible *nogood* signal, not an error: route it to the
tetralemma BOTH corner and **damp** dependent chains rather than committing.

## 6. What exists vs. what is missing

| Piece | Status | Anchor |
|---|---|---|
| Query/assertion split; `queryPart`/`queryEqual`/`exist` layers | **built, live** | `Language.py:4276,4200,4290` |
| Soft signal-router policy (chooser in gradient path) | **built, live** | `Language.py:4744,6134,6187` |
| `query()` — typed `.where` read over the full truth-space (input/codebooks/STM/LTM); parked, **not consumed** | **built, wired, dark** (`<globalAttention>`) | `Spaces.py:426,520`; `Models.py:8095,8190,8213` |
| `synthesize_over_set` (M→1 order-raising fold, novel-idea synthesis) | **built, live** | `Layers.py:3018` |
| `_idea_parthood`, `_idea_identity` (hard tools' core) | **built, live** | `Spaces.py:14213,14234` |
| `ConceptualSpace.reason` (forward modus-ponens chain) | **built, TEST-ONLY** | `Spaces.py:14255` |
| `BaseModel.reason` (fwd/rev over TruthLayer) | **built, TEST-ONLY** | `Models.py:9349` |
| `consequents`/`evaluate`/`verify_relation` | **built, TEST-ONLY** | `Layers.py:7396,7419`; `Spaces.py:14337` |
| `isTrue`/`ground`/`extrapolate` | **built, live (inside `reason`)** | `Models.py:1378,1360,1387` |
| `TernaryTruthStore` rows + trust + `provision_ltm` + `append_*` | **built, wired for writes** (gated `<ltmConsolidation>`) | `Layers.py:7470,7570,7575`; `Models.py:7700,2417` |
| **The NeuralToolUser** (policy that calls the tools) | **MISSING** | — |
| **Intervening-idea generator** (MLP query head → `query()` readout, recurrent) | **MISSING** (substrate built: `query()` + `synthesize_over_set`; needs the reasoning consumer of the parked read) | — |
| **Top-K candidate beam** in the chain search | **MISSING** | — |
| **Lemma write-back** of derived conclusions to LTM | **partially** (write API exists; reasoning never calls it) | `Layers.py:7575` |
| **Policy loss** (train the tool-use steps) | **MISSING** | — |
| **Episodic source** for `verify_relation` | **MISSING** (Workstream G, deferred) | `Spaces.py:14337` |

Legacy retired (`todo.md`): the hard-parse `NeuralToolUser` executor was removed
2026-06-22; the signal-router soft route is the surviving substrate this plan
reuses. The reasoning methods are architecturally complete (~70 tests across
`test_reasoning.py`, `test_truth_ideas_routing.py`, `test_ltm_consolidation.py`,
`test_two_truth_stores.py`); the only gaps are **a policy that calls them** and
**a signal to train that policy**.

## 7. Implementation phases

Each phase is **gated dark and full-suite-gated**: off ⇒ byte-identical;
`make test` (and `make test_all RUN_SLOW=1` for slow gates) green before the
next. New gate **`<queryReasoning>`** in `<architecture>` of `data/model.xsd`
mirrors the `<ideaDecode>` recipe (`model.xsd:182` decl → `Models.py:806` read
→ guarded branch). Exemplar config: `data/MM_query_reasoning.xml` (copy a small
`MM_*.xml`, add the gate + a `<truthSet>` with syllogism axioms).

### Phase 0 — `QuerySpec` framing (ingress)
- Build `QuerySpec{predicate ∈ {isTrue, isPart}, left, right, variables}` from a
  `query="true"` rule; package the operand vectors `(predicate, left, right)`
  the chart currently discards. Normalize `isEqual → isPart∧isPart`,
  `exist → isTrue`; flag open-variable `isPart(X,B)`.
- Tests: each surface form round-trips to a `QuerySpec`; no model run.

### Phase 1 — hard tools `isTrue` / `isPart_direct` (§4.1–4.2)
- Implement the two leaf resolvers over `_reasoning_store()` + `model.isTrue`.
  Exact, no policy, no gradient.
- Tests: axiom ⇒ TRUE; positive-trust single idea ⇒ TRUE; absent ⇒ UNKNOWN;
  negative ⇒ FALSE; geometric/`REL_PARTOF`/taxonomy direct `isPart` hits.

### Phase 2 — hard chain executor (§4.3 HARD half)
- Extend `ConceptualSpace.reason`: `target=B` early-exit, top-K beam slot, MIN
  trust composition. With a trivial (uniform) candidate scorer it already
  reproduces today's behavior — this isolates the hard executor from the policy.
- Tests: **Barbara syllogism** — `man ⊑ mortal`, `Socrates : man` ⇒
  `isPart(Socrates, mortal)` via a 1-intermediate chain; cyclic store
  terminates (visited-set + `max_steps`).

### Phase 3 — `query()` + the intervening-idea generator (§4.3a, the SOFT half)
- Add a reasoning consumer of the already-parked `_global_attention_obs`
  (`Models.py:8213`): `query(s)` runs `GlobalAttention` over the **full** typed
  registry from `_addressable_spaces` (`Models.py:8095`) — `SPACE_INPUT`/`PART`/
  `WHOLE`/`SYMBOL`/`STM`/`LTM` — and returns the `.where`-typed recalled idea
  `rₜ` (+ provenance) plus the top-K `α` beam.
- Generator: an MLP query head over `[A ; B ; rₜ₋₁]` (full width) produces the
  `concept_q` that drives `query()`; the readout is the proposed `M`. Add
  `synthesize_over_set` (`Layers.py:3018`) as the step-3 synthesis fallback.
  Recur `rₜ → sₜ₊₁`.
- Reuse the wired `<globalAttention>`/`<globalAttentionConsume>` path; keys
  detached (gradient through `α`/the head only). Off ⇒ Phase-2 hard executor
  unchanged (byte-identical) — the generator only *proposes/orders* candidates;
  it never changes a hard verdict.
- Tests: `query()` returns the right typed `.where` (LTM vs input vs codebook)
  on a seeded truth-space; the generator's top-K contains the correct
  intermediate above distractors; proposals are grounded (each `M` has high
  `_idea_identity` to a real row); recurrence carries `rₜ`; beam cap honored.

### Phase 4 — lemma write-back (§4.4)
- On a verified chain, `append_relation`/`append_idea` the derived conclusion
  with MIN-composed trust; assert it is reused (a second identical query is now
  a *direct* hit, not a chain).
- Tests: derive once, then `isPart_direct` succeeds; trust persists across a
  `state_dict` round-trip.

### Phase 5 — the consumer + policy loss (gate `<queryReasoning>`)
- Wire the dispatcher: when `<queryReasoning>` on and `infer()` /
  `serve.py:/chat/completions` (`serve.py:184`) gets a query, route
  `QuerySpec` → tool-user → posture (§5) and attach the chain to the response
  (`serve.py` returns `_last_truth_assessment` at line 271). Off ⇒ generative
  path unchanged (byte-identical).
- **Policy loss** `<answerLossWeight>` (default `0.0`): supervise *whether the
  executed chain verified the claim* against gold labels, backprop through the
  soft route only (the hard tools gate which route was scored; the deduction is
  not differentiated). The `<globalAttentionConsume>` slot (`Models.py:783`) is
  the trainable feedback hook. Because the policy gets signal only on the
  candidates it actually scored, pair it with the existing per-idea
  `verify_relation`/trust supervision as an auxiliary bootstrap.
- Tests: gate off ⇒ identical generative output; gate on ⇒ posture + chain for
  the syllogism config; loss decreases on a toy QA set.

### Phase 6 — contradiction + verification
- Route `BOTH` to the tetralemma BOTH corner; damp dependent chain trust.
  Enforce partial-order axioms at edge insertion (antisymmetry, no cycles) so
  the hard chain executor terminates.
- `verify_relation` consumes order-0 episodes once a source exists — **episodic
  `.events` store = Workstream G, deferred** (no owner); this is the consumer,
  not the source.

### Phase 7 — chain rendering (explanation)
- Render the winning chain: `A is true (axiom). A ⊑ M (trust .82). M ⊑ B (trust
  .77). ⇒ isPart(A,B) TRUE, confidence .77 (= min hop).` Counter-trace on
  `FALSE`/`BOTH`.

## 8. Configuration

| Gate | Where | Default | Effect |
|---|---|---|---|
| `<queryReasoning>` (new) | `model.xsd` `<architecture>` | `false` | Route queries to the tool-user + posture instead of generative `infer()`. Off ⇒ byte-identical. |
| `<answerLossWeight>` (new) | `<training>` | `0.0` | Weight of the policy (answer) loss. |
| `<ltmConsolidation>` | `model.xsd:237` | `false` | Use the unified `TernaryTruthStore`; reasoning targets `_reasoning_store()`, which abstracts both stores. |
| `<truthSet>` | `model.xsd:248` | empty | Provisioned axioms (ultimate truths), loaded by `provision_ltm` (`Models.py:2417`). |
| `<globalAttention>` / `<globalAttentionConsume>` | `model.xsd:164,176` | `false` | Retrieval-augmented soft-read + trainable feedback (policy-loss hook). |
| `<truthCriterion>` | `model.xsd:389` | `1.0` | Learn-gate for relation acquisition; lower toward 0 to populate the store. |
| `parthood_threshold` θ / `max_steps` / `K` | `reason()` args | `0.7` / `8` / new | Hop coverage bar / proof depth / candidate beam. |

## 9. Testing

- Unit (no training): `QuerySpec` framing; `isTrue`/`isPart_direct` hard tools;
  hard chain executor + MIN + beam; lemma write-back + reuse; posture
  classifier; gate on/off byte-identity.
- `@pytest.mark.xfail` until word identity is learned (mirrors existing
  `test_reasoning.py` English-level tests): Socrates syllogism, contrapositive,
  `isEqual` semantic equivalence.
- Integration `test_query_to_answer.py`: query in via `serve.py`/`bm.py`, assert
  tool-user invoked, posture + chain returned; config `data/MM_query_reasoning.xml`.
- Full suite (`make test`, `test/test_report.py` auto-discovery) green at each
  phase; slow gates under `make test_all RUN_SLOW=1`.

## 10. Success criteria

- An axiom proves `isTrue(A)`; a positive-trust single idea proves `isTrue(A)`;
  absence returns **UNKNOWN** (no hallucination).
- A direct `REL_PARTOF` / geometric containment proves `isPart(A,B)`.
- A multi-hop chain (`Socrates : man`, `man ⊑ mortal`) proves
  `isPart(Socrates, mortal)`; the tool-user returns **several ranked candidate
  chains**, capped by the beam, and **terminates** on cyclic stores.
- The derived conclusion is **written back to LTM** and a repeat query is a
  *direct* hit.
- Negative/contradicting evidence yields **FALSE**/**BOTH** with traces.
- Gate off ⇒ byte-identical; the policy loss trains step selection on a toy QA
  set without ever making a verdict differentiable.

## 11. Risks & open questions

1. **Don't soften the deduction.** The whole design rests on §0: gradients flow
   through the *policy* (candidate scoring), never through a verdict. If a future
   change makes `isPart`/`isTrue` themselves differentiable end-to-end, the chain
   stops being hard-edged and the guarantees in §10 dissolve.
2. **Subsumption/parthood conflation.** `men ⊑ mortal` is sound only
   extensionally; never compose it with order-0 `.where` transitivity. Distinct
   `rel_type`s.
3. **Trust algebra consistency.** Use MIN-over-hops, not product *and* MIN
   (current `reason()` uses product; this plan moves to MIN). Document the
   `[-1,1] → [0,1]` map for the policy loss so negatives don't read as "unknown".
4. **Termination.** Cyclic/noisy mereonomy loops; enforce antisymmetry +
   visited-set + `max_steps`. The beam is a feature, not a degradation.
5. **Contradiction under "positive trust".** Naive `isTrue` passes both `p` and
   `¬p`; the tetralemma BOTH / nogood layer must resolve or flag it.
6. **Sparse policy gradients.** The policy is supervised only on candidates it
   actually scored (the executed beam); pair the answer loss with
   `verify_relation`/per-idea trust supervision as an auxiliary bootstrap.
7. **Lemma drift.** Materialized lemmas with mistaken trust pollute future
   chains; gate write-back on a confidence floor and let `verify_relation`
   correct trust when episodes arrive.
8. **Off-manifold proposals.** A free MLP that emits `M` directly can invent a
   bridge no stored relation supports — it *looks* like progress but never
   verifies. Mitigation is structural (§4.3a): take `M` as an attention readout
   over real LTM/codebook rows, or snap any synthesized `M` to its nearest real
   idea before the hard verify. Keep the proposer grounded; never let an
   ungrounded vector enter a chain.

## 12. References & lineage

- **In-repo lineage.** The retired hard-parse `NeuralToolUser` (removed
  2026-06-22) and the surviving signal-router soft-superposition chooser
  (`LanguageLayer` `Language.py:4744`, `superposition_scale` `Language.py:6134`,
  soft-DP `logsumexp` `Language.py:6187`) — the chooser-over-exact-ops pattern
  this tool-user reuses. The Truth/Ideas stages (`ConceptualSpace.reason`,
  `verify_relation`, `RelativeTruthStore`/`TernaryTruthStore`) provide the hard
  tools.
- **External grounding.** Description-logic subsumption as extension-subset +
  instance-checking reduction (the two-primitive reduction); Aristotelian
  *Barbara* (the canonical `isPart` chain / first unit test); truth-maintenance
  systems (premise vs derived nodes, justification traces, nogood environments —
  the `isTrue` disjunction, the materialized lemma, the BOTH posture);
  forward/backward/bidirectional chaining (the chain search direction);
  neural-guided / learned-search theorem proving and tool-use (the *soft policy
  over hard tools* split — note we deliberately keep the deduction exact, unlike
  fully-differentiable provers); defeasible/possibilistic reasoning (MIN/MAX
  trust composition, rebut/undermine defeaters).
