# Concepts — Formalization, the CS Snap Contract, and Execution Plan

> **STATUS: PLAN 2026-07-06 — the single consolidated plan for this line of
> work (framework + snap contract + execution), to be executed in a separate
> thread.** T1 (terminology rename) and T2 (WS initScale) are ready now. The
> representation calls that gated the taxonomic build are now RESOLVED — the
> sign carrier (symbol), the sparsity regularizer (rank-ordered soft-then-
> hard $L_0$, §5), and the storage domain (small init) — leaving only a
> minor read-time $\varepsilon$ and the empirical $\lambda$, so T3+ are
> design-unblocked pending Alec's final go. Framework: Alec (2026-07-05/06
> session); terminology from *The Whole Part* (Alec). Companions:
> [percept-hypercube.md](../percept-hypercube.md) (percept geometry),
> [2026-07-04-serial-derivation-reconstruction-design.md](2026-07-04-serial-derivation-reconstruction-design.md)
> (Method-1/Method-2 duality),
> [2026-07-04-union-difference-concept-ops.md](2026-07-04-union-difference-concept-ops.md)
> (the residual-bearing op pair).

## 0. Terminology — epistemic levels (*The Whole Part*)

Adopted project-wide (Alec, 2026-07-06). Everything below is a **thing**;
the levels differ in what they are *about*:

| level | store | contents | relation |
|---|---|---|---|
| **objects** | (the referent side; byte content / world) | what percepts refer to | referred-to |
| **percepts** | PS · WS · SS | **parts** · **wholes** · **symbols** — three kinds of percept | refer to objects |
| **concepts** | CS | NOT percepts | defined *over* percepts: bounded by parts and wholes; composed algebraically over symbols |

Consequences for prose and code:

- "Symbol" is never a synonym for a WS row (a **whole**) or a CS row (a
  **concept**). Symbols are percepts and live in SS.
- Epistemic level and *geometry* are orthogonal: symbols are percepts even
  while the lexicon currently rides a signed torus; whether symbols follow
  the `[0,1]` lexicon move is its own open item, untouched here.
- Known conflation sites to sweep (VERDICTS PENDING — renames touch code,
  Alec's call per site): `WholeSpace.insert_symbol` (inserts wholes);
  percept-hypercube.md's recurring "concepts/symbols" pairing; todo.md's
  "symbol/truth prototypes" description of the WS codebook; the `nObj` /
  ObjectSubSpace overload of "object" (per-word slots, not referents).

## 1. What a concept is (the ladder)

**Two axes of definition (Alec, 2026-07-06).** A concept is defined along
two orthogonal relations, at two epistemic levels:

- **Mereologically** — a set of **parts and wholes** *bounds* a concept:
  the interval between the parts it contains (from below) and the wholes
  that cover it (from above). This is the PERCEPT layer — one-sided /
  positive (a part is present or absent; there is no *negative* part) — and
  it GROUNDS the concept in percepts.
- **Taxonomically** — a set of **concepts**, by their **presence and
  absence ($\pm$)**, defines *subsequent* concepts: inclusion ($+$) and
  exclusion ($-$) of already-defined concepts compose new ones. This is the
  CONCEPT layer — two-sided / signed — and it is where the signed algebra
  (and the signed snap of §2) lives.

This is the load-bearing distinction, because it says WHERE sign belongs.
Mereology is positive (percept presence; the `[0,1]` cube of
[percept-hypercube.md](../percept-hypercube.md) §4 — absence is a
non-informative observation, not a negation). The minus sign is a *taxonomic*
operation — one concept excluding another — never a negative part. So the
signed-coefficient snap (§2) is a property of the taxonomic layer; the
mereological bounding stays on the `[0,1]` presence cube. The ladder below
is these two axes stacked: 1–3 are mereological (grounding), 4–5 taxonomic
(composition).

1. **Order 0 — cells.** A percept prototype row plus its $\sigma/\pi$-local
   generalization is a *cell*: a contiguous local region around the
   prototype (parts in PS, wholes in WS).
2. **Order $k$ — discontiguous regions.** Each fold pass can union/refine
   cells, so an order-$k$ region is possibly discontiguous and the
   effective dimensionality of what can be represented grows with order.
   The ramsification record is the bookkeeping for exactly this
   (`abstraction_order(row)` = count of non-NEITHER folds) — its LIVE
   stamping (todo: "make abstraction order canonical") is a **dependency**
   of this design, not a neighbor.
3. **A 0th-order concept is a mereological interval.** A concept is
   *bounded by parts and wholes*: the region between what it includes
   (parts, from below) and what covers it (wholes, from above) —
   $[\,\bigsqcup \text{maximal parts},\ \bigsqcap \text{minimal wholes}\,]$
   in the mereological order. (The Galois/FCA reading: extent shrinks as
   intent grows; the two frontiers are one antitone pair.) This is the
   **mereological** axis — positive, grounding.
4. **Higher concepts are taxonomic (algebraic), and SPARSELY defined over
   symbols.** A concept is defined by the inclusion ($+$) AND exclusion
   ($-$) of *other concepts*, addressed through their **symbols** — leaving
   mereology (which has no complement) for a signed algebra, computed by the
   $\sigma$ layer plus the negation ops the concept layer owns. The
   definition is a **sparse** signed weight vector over symbols
   (Alec, 2026-07-06): weights $\in [-1,+1]$ are the absence/presence of
   each symbol in the definition, **0 = don't-care**, and only a few symbols
   are non-zero. Crucially the pressure on this layer is a **growth-
   preventing regularization** — a shrinkage penalty on the concept→symbol
   weights that keeps each definition compact — NOT a force that grows it; a
   concept should depend on as few symbols as suffice. This is the
   **taxonomic** axis — signed; those $\pm$ weights are exactly the signed
   coefficients the snap of §2 recovers.
5. **Sign lives on the SYMBOLS; an idea is a sparse signed symbol vector.**
   (Alec, 2026-07-06 — resolves the "concepts vs symbols" carrier question.)
   A **symbol** is a scalar $\in [-1,+1]$ = the absence/presence of its
   corresponding **concept**: $+$ present, $-$ absent, **0 = don't-care**,
   bound to that concept by the relation table (one concept index ↔ one
   symbol index). The concept is the *region/meaning* (the positive
   mereological signature); the symbol is the *signed presence scalar* on
   it. An idea (an STM entry; the reduced root $S$) is the sparse signed
   vector of these symbol scalars — WHICH concepts, at WHAT $\pm$ presence.
   This maps directly onto the code's factored form:
   `concept_code = a · softplus(atom)` (Spaces.py:14510-14521) already
   splits the **concept** (the dense positive atom / region) from the
   **symbol** (the signed activation $a$, the presence scalar) — so "sign on
   symbols" IS the signed activation, and sparsity is 0-valued symbols. A
   symbol scalar expresses present / absent / don't-care = TRUE / FALSE /
   NEITHER; the BOTH (conflict) state stays on the 2-axis catuskoti carrier
   — the symbol is the *decided* view.
6. **Evidence accumulates on two independent axes; the symbol is the
   readout.** The symbol scalar $\in [-1,+1]$ is decided, but evidence FOR
   and AGAINST a symbol accumulates INDEPENDENTLY (Alec, 2026-07-06) — e.g.
   the luminosity / truth-set-consistency checks track positive and negative
   support separately, so BOTH (high $+$ and high $-$ = conflict) and
   NEITHER (low both) stay distinguishable rather than cancelling in a
   single running sum. The decided symbol is $\mathrm{pos} - \mathrm{neg}$,
   exactly the existing `act = pos.clamp(0,1) - neg.clamp(0,1)`
   (Spaces.py:6335) over the catuskoti's independent T/F axes. So the
   accumulator is two-sided; only the emitted symbol collapses to one
   signed scalar.

## 2. The snap contract (decoding an idea)

**Sign handling today (trace 2026-07-06 — correcting a first assumption).**
A signed axis+polarity snap is ALREADY an idiom here, so "signed snap" is
mostly *un-discarding* + *propagating*, not new invention:

- The meronomy `_snap_content` abs-snap KEEPS polarity: it selects the row
  by `cos.abs().argmax` ($x$ and $-x$ share an axis) and then recovers and
  applies the sign — `pol = sign(gather(cos, idx))`,
  `flat = pol · weight[idx]` (Spaces.py:2046-2050); the truth-store query
  mirrors it (Layers.py:7300). This branch is gated
  (`meronomy_enabled() and not monotonic`; meronomy is code-default-off but
  the shipped configs set `<meronomy>on</meronomy>`, so verify the gate per
  config).
- The concept CODE is already signed on output —
  `content = a · softplus(atom)` with signed activation $a$ riding the
  positive atom (Spaces.py:14510-14521); the representation carries sign.
- BUT the REVERSE reconstruction peel does NOT use that idiom: it ranks by
  signed cosine yet enters every row at a fixed **+1** (unit subtraction),
  and CLAMPS the sign away — `if sims[best] <= 0.0: break`
  (Language.py:2367), `factor_percept`'s `q.clamp(min=0)` /
  `sims.clamp(min=0)` (Spaces.py:16161-16162), and the order-0 CS snap
  presence `cs_snap_order0(nonneg=True)` → `clamp(min=0)` (Spaces.py:14428).
- And nowhere is there a per-row COEFFICIENT slot, so a *weighted* exclusion
  ("this concept at −0.7") is unrepresentable even where sign survives.

So the CS snap is ill-defined for a *region-valued* concept on three counts
— it discards sign on the reconstruction path, has no coefficient slot, and
has no support/region semantics. The well-defined replacement, given an
idea vector $x$:

1. **SUPPORT.** The symbols/concepts with coordinates outside an
   $\varepsilon$ dead-zone around 0 (don't-cares eliminate the rest from
   the support). $\varepsilon$: open question (fixed / per-space /
   learned).
2. **POLARITY + RELEVANCE — by PEEL, not one-shot projection.** Inner
   products against $x$ give each supported row a *signed magnitude*
   (inclusion vs exclusion, weighted by relevance) — but raw one-shot
   projections are only valid on a near-orthogonal support. Under the
   current bunched cone (the xor store's pairwise $\cos \in$
   [0.267, 0.827]) every row reads mostly the shared component. The
   crosstalk-robust form is matching pursuit: best match, **subtract**,
   repeat — `ChunkLayer.peel` (Language.py:2351) / the basis-recommender
   reverse, already in the codebase. One-shot projection is the zeroth
   iteration of the loop; run the loop. The concrete deltas on
   `ChunkLayer.peel`: (i) drop `if sims[best] <= 0.0: break`
   (Language.py:2367) so an anti-aligned row can be selected as an
   *exclusion*; (ii) keep `sims[best]` as the row's signed COEFFICIENT
   (reuse the `_snap_content` `pol = sign(q·v)` idiom, generalized from
   $\pm1$ to the real magnitude) instead of the fixed unit subtraction
   `residual - W[best]` → `residual - coeff · W[best]`; (iii) emit
   `(row, coeff)` pairs, not bare indices. This makes the peel a genuine
   signed matching pursuit / OMP step. **The `coeff` IS the symbol** (§1.5)
   — the signed presence scalar on that concept — so the peel's output is
   literally the idea's sparse symbol vector; the coefficient attaches to
   the SS symbol (via the relation table), not the CS concept row.
3. **ORDER-$k$ MEMBERSHIP — unfold before probing.** A linear probe
   against a stored prototype sees only the order-0 shadow of a
   discontiguous order-$k$ region. Membership at order $k$ applies the
   recorded $\sigma/\pi$ inversions (`invert_ramsified`, per the fold
   provenance) *before* the inner product. Without live ramsification
   stamps this stage silently degrades to order-0 matching.
4. **FRONTIER — discard the subsumed, two-sided.** Parts and wholes
   subsumed by other supported parts/wholes (or concepts) against the
   idea's boundary are discarded: keep **maximal parts** (inclusions not
   implied by others) and **minimal wholes** (tightest covers) — the two
   frontiers of the interval in §1.3. Subsumption truth: byte-interval
   containment on the percept side (`RunStructureLayer`); the relation
   table on the concept side (one concept index ↔ one symbol index per
   row). Sparse table coverage ⇒ weaker pruning ⇒ *verbose* definitions,
   not wrong ones.
5. **OUTPUT — a typed conceptual definition.** What survives is the
   uncompressed semantic form, and it is typed for the grammar:
   **head/subject = the minimal covering whole; modifiers = the surviving
   maximal parts; exclusions = the negative-polarity members.** The
   grammatical operators then COMPRESS this definition into a surface
   (NP head; `NP → ADJ + NP` peels for modifiers; negation forms for
   exclusions).

**Where it sits:** this pipeline is Method-2's front half. The forward
parse compresses surface → idea; this decode decompresses idea →
definition; the grammar re-compresses definition → surface. Method-1
(the stored-leaves replay, landed 2026-07-05, bar green) stays the exact
teacher/supervision; Method-2's free derivation is scored on surfaces
against it.

**Snap prerequisites (the contract's representation side):**

- **Signed match — mostly un-discard, not retire.** Do NOT remove the
  `_snap_content` `cos.abs().argmax`-then-`sign(q·v)` idiom: it is doing
  the right thing (axis by magnitude, polarity kept). The work is to
  *propagate* it onto the reverse peel and *drop the clamps* that discard
  sign there (Language.py:2367; Spaces.py:16161-16162; the order-0
  `nonneg=True` at 14428), plus add the coefficient slot (§2.2). Where the
  meronomy gate is off for a config, that idiom also has to be turned on.
- **Sparse signed SYMBOLS** — with sign resolved to the symbol (§1.5), both
  sign and sparsity live on the **symbol scalars** — the signed activation
  $a$ — while the **concept** atoms stay dense positive (they are the
  regions, not the presence). So the representation change is narrow: make
  the symbol vector SPARSE (most $a = 0$ = don't-care), with a genuine 0
  state, rather than the dense activation every concept gets today. The
  concept dictionary is untouched; only the presence layer sparsifies.

## 3. The fidelity program (three legs, one goal)

The end goal (Alec, 2026-07-05): not merely exact reconstruction, but
codebooks whose rows *represent* precisely, so expressing ideas in terms
of the understanding they provide is precise. Three legs:

1. **Sparsity dissolves the angular cone.** Cosine crowding is
   scale-invariant (no magnitude change touches it); sparse signed codes
   are near-orthogonal in expectation — the don't-cares are the geometry
   fix for the cone.
2. **A small-magnitude operating point keeps the folds linear** (Alec,
   2026-07-06 — adopted IN PLACE OF the unbounded dual-rays domain, which
   is NOT adopted; small-$\varepsilon$ is its linearization:
   $\tanh(\sum_k \mathrm{atanh}\, x_k) = \sum_k x_k + O((k\varepsilon)^3)$,
   and PiLayer's log-mult chart is likewise linear at 0, so one operating
   point serves $\sigma$ AND $\pi$; Boolean limits stay at their
   traditional values). Relative squash error $\approx (k\varepsilon)^2/3$
   at fold depth $k$: with `stmCapacity` 8 —
   $\varepsilon{=}0.5$ (today's WS): $\tanh(4)=0.9993$, total saturation;
   $\varepsilon{=}0.1$: 17%; $\varepsilon{=}0.02$: **0.85%** —
   effectively additive.
   Init sites (verified 2026-07-06): PS radix rows already
   `normal_(0, 0.02)` (`RadixLayer.insert`, Layers.py:10485) — the S2
   round-trip is green at that scale, the empirical proof of viability;
   WS rows `uniform_(-1, 1)` (`WholeSpace.insert_symbol`) and the generic
   `Codebook.create` prefill rescales rows to unit magnitude
   (Spaces.py:3441) — the saturating outliers. WS EMA refresh is gated
   OFF (the asymmetric hardwire, Spaces.py:3466), so a small init STICKS
   on WS; but content-derived rows (`insert_symbol(init_vec=...)` from
   the CS→WS demux) arrive at data scale — wherever an update rule or
   content write is live, the DATA scale is the attractor and the knob
   must be paired with checking those writers. No init-scale knob exists
   today; proposal: per-space `<initScale>` (WS seeds + create-prefill
   scale → ~0.02; PS already there; limits untouched).
3. **The residual is the fidelity signal.** In the signed domain,
   $\mathrm{residual} = \mathrm{idea} - \sum(\text{snapped rows})$ is
   exact and honest — literally *what the current understanding cannot
   yet say*. Persistent structured residual → MINT a row (the
   attention-to-relation promotion machinery); small never-vanishing
   residual → crowding, SPREAD the rows. Fidelity converges when
   expression through the inventory leaves nothing to carry.

**Bars (measured, not asserted):** pairwise-cosine spread of each
inventory; radial spread by composition depth (composites separate by
order instead of saturating); peel-termination margin; terminal residual
→ 0 on expressed ideas; Method-2 surface match as the end-to-end meter.

## 4. Execution plan (ordered; for the separate thread)

Each task: what · where (reuse existing code) · gate · verification.
House rules of this repo apply — targeted pytest per task
(`PYTHONPATH=test:bin BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager`), `make
test` only at a task's final gate on a quiet tree, Inf/NaN fail loud,
value-pins updated honestly.

**T1 — Terminology rename pass** (no design call; do first, it touches
names not behavior).
- `insert_symbol` → `insert_whole`: ~100 source sites — DISAMBIGUATE each
  definition first (rename only the ones inserting **wholes**, e.g.
  `WholeSpace.insert_symbol`; leave any genuine SS symbol insert). A blind
  find/replace would re-conflate, so read each def.
- The "concepts/symbols" conflation prose in `percept-hypercube.md` and the
  WS "symbol/truth prototypes" line in `todo.md` → the §0 ladder terms.
- `nObj` → `nIdeas` (Alec, 2026-07-06 — each per-word slot holds an
  **idea**, not a referent object): ~159 sites; its own commit + full-suite
  gate. `ObjectSubSpace` and the rest of the "object = slot" naming follow
  the same idea-slot rename.
- Verify: `make test` green (byte-identical behavior — pure rename).

**T2 — WS `<initScale>` + measurement** (no design call; the small-init
fidelity leg, §3.2).
- Add a per-space `<initScale>` (default WS seeds + `Codebook.create`
  prefill rescale to ~0.02; PS `RadixLayer.insert` already there; Boolean
  limits untouched). Pair with a guard that content-derived rows
  (`insert_whole(init_vec=…)` from the CS→WS demux) and any live EMA/update
  writer don't re-inflate to data scale.
- Where: `Spaces.py` (`Codebook.create` ~3441; `WholeSpace.insert_whole`),
  the XML space-config schema.
- Verify: `test_union_difference_ops` residual survival through depth-8
  folds (the contrast test); a new radial-spread probe (composites separate
  by depth, don't saturate). Bars: §3.

**T3 — Signed peel (un-discard + coefficient)** — needs ε (§5 open 1) as
the peel-stop / support threshold (minor; default a small constant).
- `ChunkLayer.peel` (Language.py:2351): drop `if sims[best] <= 0.0: break`
  (2367); keep `sims[best]` as the row's signed COEFFICIENT (= the symbol,
  §1.5) — `residual - coeff·W[best]`; emit `(symbol_row, coeff)` pairs.
  Reuse the `_snap_content` `pol = sign(q·v)` idiom (Spaces.py:2046-2050).
- Drop the reconstruction-path sign clamps: `factor_percept`
  (Spaces.py:16161-62), the order-0 `cs_snap_order0(nonneg=True)` (14428).
- Verify: a signed-peel unit test (an exclusion / negative-coefficient
  operand is recovered; multiset recovery with a negative row) + the S2
  round-trip stays green.

**T4 — Symbol sparsity + regularizer** (regularizer RESOLVED §5; only
$\lambda$ to tune).
- Add the RANK-ORDERED soft-then-hard $L_0$ penalty on definition size
  (§5): sort each concept→symbol row by $|w|$, exempt the top
  `definitionFreeSize` (=2), apply $p(n)=\lambda(n-2)$ to ranks $\ge 3$;
  hard cap at `stmCapacity` (the decode peel stops there). Concept atoms
  stay dense-positive (untouched); the ε dead-zone (§2.1) is the read-time
  support cleanup only.
- Where: the CS activation/definition path (`cs_forward_content`,
  Spaces.py:14491+); a new loss term; `definitionFreeSize` config knob.
- Verify: pairwise-cosine spread of the inventory ↑ (the cone dissolves);
  definition sizes concentrate at ≤2 with a tail to 8; terminal residual
  → 0 on expressed ideas. Sweep $\lambda$ against these bars.

**T5 — Order-$k$ membership unfold** — GATE the existing "make abstraction
order canonical" todo (ramsification live stamping).
- Apply `invert_ramsified` (the recorded σ/π inversions) BEFORE the inner
  product on the peel (§2.3), so discontiguous order-$k$ regions match.
- Verify: an order-$k$ region member (second-lobe) scores correctly where
  an order-0 probe missed it.

**T6 — Frontier + typed conceptual definition** — GATE relation-table
coverage (strength only; §2.4, the promotion plan).
- Two-sided pruning: keep maximal parts / minimal wholes; emit the typed
  definition (head = minimal covering whole; modifiers = maximal parts;
  exclusions = negative-polarity members) as the grammar's compression
  input (§2.5).
- Verify: Method-2 surface match (Task 4 of the serial-derivation plan) —
  the end-to-end meter.

**Also (independent, existing todo):** two-sided evidence accumulation
(§1.6) — wire luminosity / truth-set-consistency onto the independent ±
axes feeding `act = pos − neg` (Spaces.py:6335). Not on the critical path
of T1–T6; sequence when the truth-store work is next.

## 5. Open questions (Alec)

RESOLVED this session: the storage DOMAIN (small-magnitude init, not the
unbounded dual-rays; Boolean limits kept — §3.2); WHERE sign lives
(taxonomic, not mereological — §1); the sign CARRIER (the **symbol**, a
scalar $\in[-1,1]$ = signed presence of its concept — §1.5), collapsing the
old "signed-sparse atoms vs sparse activations" fork (sign + sparsity ride
the symbol; concepts stay dense positive); the SPARSITY PRESSURE (§1.4 —
concepts are sparsely defined over symbols, held compact by a
**growth-preventing regularization** / shrinkage penalty on the
concept→symbol weights, NOT a growth force — so this layer wants a
regularizer, answering the former "what grows it"); and the ACCUMULATOR is
two-sided (§1.6 — evidence tracks $+$/$-$ independently; only the emitted
symbol collapses to one scalar).

RESOLVED — the REGULARIZER (Alec, 2026-07-06): an $L_0$-semantics
(count-of-symbols, NOT lasso/$L_1$ — $L_1$'s magnitude shrinkage would
understate the $\pm1$ presence) **soft-then-hard schedule** on definition
size $n$:

- $n \le$ `definitionFreeSize` (=2): **free**. (The genus + differentia
  minimal definition — one superordinate concept + one distinguishing
  feature = 2 — carries no penalty.)
- `definitionFreeSize` $< n <$ `stmCapacity`: **soft**, rising —
  $p(n) = \lambda\,(n - 2)$ (or $(n-2)^2$ steeper); the pressure ramps per
  extra symbol.
- $n =$ `stmCapacity` (=8): **hard** — the structural STM ceiling; a
  definition cannot exceed what STM holds (enforced by the decode peel
  stopping at `stmCapacity`, not by penalty).

**Realization — RANK-ORDERED penalty** (avoids the $L_1$ shrinkage trap):
sort a definition's symbols by $|w|$, EXEMPT the top-2, apply the rising
penalty only to ranks $\ge 3$. Shrinkage then lands only on the marginal
symbols (the ones to drop), while the core two stay at full $\pm1$,
unpenalized — a differentiable soft-$L_0$ without hard-concrete gates
(gates remain an option for a true stochastic expected-$L_0$). Knobs:
`definitionFreeSize`, `stmCapacity` (already named), $\lambda$ (steepness —
the one empirical tune).

Still open:

1. **Dead-zone $\varepsilon$** (§2.1) — now DECOUPLED from sparsity (the
   count-penalty above does the sparsifying): $\varepsilon$ is just the
   read-time support threshold ($|a| \le \varepsilon \Rightarrow$
   don't-care). Fixed constant / per-space / learned — a minor cleanup
   knob, no longer a gating representation call.
2. **$\lambda$ steepness** — the soft-penalty weight; tuned empirically at
   T4 against the fidelity bars (§3), not decided up front.
3. **Symbols and the `[0,1]` lexicon move**: orthogonal to everything above
   — does the percept-hood of symbols eventually argue the lexicon onto the
   presence cube, or does "sign is form content" keep the torus? Note the
   tension with §1.5: a symbol's *value* is signed ($\pm$ presence), so if
   the lexicon stores symbol *identities* the sign is a separate scalar, not
   a coordinate of the stored row — worth reconciling.
