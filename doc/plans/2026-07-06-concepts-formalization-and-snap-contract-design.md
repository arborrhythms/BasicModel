# Concepts — Formalization and the CS Snap Contract (Design)

> **STATUS: DRAFT 2026-07-06 — awaiting Alec's review. NO implementation
> before approval.** Framework: Alec (2026-07-05/06 session); terminology
> from *The Whole Part* (Alec). Companions:
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
3. **A 0th-order concept is an interval.** A concept is *bounded by parts
   and wholes*: the region between what it includes (parts, from below)
   and what covers it (wholes, from above) —
   $[\,\bigsqcup \text{maximal parts},\ \bigsqcap \text{minimal wholes}\,]$
   in the mereological order. (The Galois/FCA reading: extent shrinks as
   intent grows; the two frontiers are one antitone pair.)
4. **Higher concepts are algebraic.** Concepts defined by the inclusion
   AND exclusion (negation) of other concepts leave mereology — which has
   no complement — for a signed algebra over **symbols**, computed by the
   $\sigma$ layer plus the negation ops the concept layer owns. The result
   is a (potentially) complex region in CS: a concept that includes some
   things and excludes others.
5. **An idea is a signed sparse combination.** An idea (an STM entry; the
   reduced sentence root $S$) is represented over concepts/symbols in
   $[-1,+1]$ where **0 = don't-care** (outside the support), $+$ = include,
   $-$ = exclude. A scalar coordinate expresses TRUE / FALSE / NEITHER;
   the BOTH (conflict) state stays on the 2-axis catuskoti carrier — the
   concept code is the *decided* view.

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
   signed matching pursuit / OMP step.
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
- **Sparse signed codes with real don't-cares** — the concept code is
  already signed on output, but its STORAGE is one-sided: the atom
  dictionary is dense softplus-positive
  (`concept_code = a \cdot \mathrm{softplus}(\mathrm{what})`,
  Spaces.py:14510-14521) with sign living only in the activation $a$. The
  don't-care structure this contract needs (a stored code that is 0 on
  most coordinates) is therefore a representation change — where sparsity
  lands (signed-sparse atoms vs sparse activations) is an open question.

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

## 4. Dependencies and sequencing

1. **Ramsification live stamping** (existing todo) — prerequisite for
   order-$k$ membership (§2.3).
2. **Sparse signed concept codes** — the don't-care representation
   (§2 prerequisites; where sparsity lands is Alec's call).
3. **Signed snap + $\varepsilon$ dead-zone** — the contract itself
   (§2.1–2.2): un-discard sign on the peel (drop the clamps + carry the
   coefficient), NOT retire `cos.abs`.
4. **WS `<initScale>` + measurement** — small, independent, immediately
   testable (§3.2).
5. **Relation-table coverage** (the promotion plan) — gates frontier
   *strength* only (§2.4); never a correctness blocker.

## 5. Open questions (Alec)

1. **Rename sweep verdicts** per §0: `insert_symbol` → `insert_whole`?
   the "concepts/symbols" doc phrasing; the `nObj` "object" overload.
2. **Sparsity mechanism**: signed-sparse atoms vs sparse activations;
   what training pressure grows don't-cares (an $L_0/L_1$-style prior,
   or the dead-zone itself as a straight-through gate)?
3. **Dead-zone $\varepsilon$**: fixed constant, per-space, or learned.
4. **Symbols and the `[0,1]` lexicon move**: orthogonal to the epistemic
   ladder (§0) — does the percept-hood of symbols eventually argue the
   lexicon onto the presence cube, or does "sign is form content" keep
   the torus?
