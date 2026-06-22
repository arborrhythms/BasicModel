# The idea decoder — turning a built idea back into words

*Spec authored 2026-06-20. How a C-tier idea in conceptual space is decoded into
a surface sentence. Generation is the **dual of comprehension**: run the forward
grammar operations in inverse, driven by the idea itself. This supersedes the
"decouple the driver" framing of `2026-06-19-grammar-inverses-handoff.md` — see
"Correction" below.*

> **Status: design / roadmap.** Gated behind `<ideaDecode>` (the flag already
> exists but is currently a no-op — see Stage 0). Byte-identical off. Builds on
> the declared dual grammar (`data/role_collapsed.grammar`), the σ/π folds, the
> VQ codebook + reducibility, the MetaSymbol category codebook, attention
> priming (`intent_boosts`/`ReadingAttention`/`GlobalAttention`), and the
> two-pass deterministic/stochastic learning.

## The one-sentence design

**Generation = the dual of comprehension:** run the forward grammar operations in
inverse, driven by the idea, peeling constituents and emitting words **until the
idea-residual reduces to the zero vector** (= the idea has been fully expressed).

## Correction to the grammar-inverses handoff

The handoff said goal 2 is "decouple the decode *driver* (chart `generate_rules`
→ primed symbolic space)." A verified code map (2026-06-20) found that premise is
wrong about the live wiring:

- The live `reverse()` decode is **not parse-tree-driven**. `_reverse_body`
  calls only the subsymbolic inverse chain (`ConceptualCombine.unbind` /
  `combine.reverse` on the stashed `_combine_carriers`, then `cs.reverse`
  bookkeeping) and the `PartSpace`/`InputSpace` **nearest-codebook lexicon
  decode**. Emptying `generate_rules` changes the output by **0.0** (verified).
- The grammar-reverse machinery (`WholeSpace.reverse` → `SyntacticLayer.reverse`
  → `reverseSymbols`, and the declared `<generate>` rules) is **dead code** — it
  exists but **nothing on the decode path calls it**.
- So `<ideaDecode>` (the gate on `_chart_generate_from_stm`) is **behaviorally
  inert** today — it skips rebuilding rules nothing reads.
- The forward-carrier inversion is exact **only because the forward stashed the
  carriers**. For a genuinely top-down idea (generated/recalled, never
  perceived — the real generation regime) there are no carriers, so the unbind
  falls back to the idea's own content. Order-0 still decodes (nearest
  codebook); a **composed** idea has no splitter on the path → cannot generate.

**Therefore** the build is not "swap a driver." It is: **wire the declared dual
grammar (`<generate>`) onto the reverse path**, run each op's inverse, and make
the lossy inverses real. The substrate (dual grammar, per-op reverses, category
codebook, priming, two-pass) largely exists; it is unwired and, for the lossy
ops, only stubbed.

## The decoder loop

```
R ← idea (sourced from the MUXED event: content + .where(2) + .when(2), last 4 slots)
while ‖R.content‖ > ε:                       # COMPLETENESS: residual → 0
    rule  ← reverse_grammar.next(R, category)   # WHICH inverse op to apply next
                                                # order (SVO) = learned grammar played in reverse;
                                                # allowable rules gated by the constituent's CATEGORY
    (c1, c2 | c1) ← rule.op.reverse(R)          # PEEL: the per-op inverse (see "two inverse-cases")
    w ← realize(constituent, R)                 # the codebook word for the peeled constituent:
                                                #   present in codebook → that code (named, direct)
                                                #   absent → nearest code, refined by +/−/0 (∩/∪) to fit
                                                #   at the order cued by R's GENUS (high- vs low-order noun)
    if slot is a verb: w ← inflect(w, tense=decode(R.when))   # .when → verb tense (not just "now")
    emit(w);  R ← peeled remainder              # reduce the residual
return emit                                      # already in grammar order
```

### The cues the idea carries (read off the muxed event)

| cue | source | drives |
|---|---|---|
| tense | `.when` (interval-vs-now) | verb tense inflection |
| placement | `.where` (endpoint-sum bracket) | where the percept renders / deixis |
| abstraction level | genus / order of the code | high-order vs low-order noun choice |
| named vs approximate | codebook presence (reducibility snap) | direct code vs nearest-code-refined |

## The two inverse-cases (the "peel")

1. **Exact math inverse** — run the op's reverse directly.
   *Ops:* `lift`/`lower` (Sigma/Pi natural folds, `invertible=True`), `tense`
   (phase rotation), `not`/`non`, base butterfly/invertible linear.
2. **Lossy → the +/−/0 constrained guess** — the op is non-invertible (e.g.
   `union`: `A ∪ B` does not uniquely determine `A`, `B`). Procedure:
   - guess operand-1 from semantic cues (attention priming over the codebook);
   - operand-2 = the codebook candidate such that `op(op1, cand) ≈ R`, found by
     a **three-valued feature filter** over the codebook: `+` (must have), `−`
     (must not have), `0` (don't care — obscured by the op);
   - among candidates satisfying those syntactic constraints, choose the
     **semantically motivated** one (priming).
   - *Not* a set-difference (`R − op1`): operands may overlap under the union.
   *Ops:* `union`, `intersection`, `conjunction`/`disjunction`, `bind`,
   `aspect`, `isEqual`/`isPart`, `part`, query family.

## Order, categories, and learning

- **Word order** is syntactic preference (SVO …) = the **learned grammar played
  in reverse**. `role_collapsed.grammar` declares roles only (no probabilities;
  POS is learned from participation), so the transition preferences come from
  the **learned routing**, not the file.
- **The category table gates allowable derivations.** The MetaSymbol category
  codebook (learned role participation) tells us, for a constituent's category,
  which reverse rules are legal.
- **Learn the reverse rules with the same deterministic/stochastic two-pass**
  used by the forward chooser/attention: pass A at superposition-temperature 0
  (exploit, recorded), pass B at `exploreTemperature` (explore, trimmed). The
  reverse-rule selector sits in the gradient path.

## The idea representation

The muxed event is `1 content + 2 .where + 2 .when` (`model.xml:359`; content
`nWhat = nDim − 4`). The `.where`/`.when` **are** the last 4 slots — but the STM
push stores the **content-only** slice (`CS_sub.materialize()[:, 0, :]`). So the
decoder must **source the idea from the muxed event** (which carries place/tense),
not the content-only STM slice. (Sanity-checked 2026-06-20.) For top-down ideas
with no specific grounding (e.g. an emotionally-valent recall from attention),
`.where`/`.when` default to **here/now**.

### Slots and the dimensional ladder (Alec, 2026-06-20)

The idea uses the THREE STM slots with a dimensional/order ladder:

| slot | content | dim | role |
|---|---|---|---|
| 1 | NP₁ | **3-D** | subject — spatial object (location / body / extent) |
| 2 | VP | **1-D** | the verb — temporal process |
| 3 | NP₂ | **3-D** | object / second relatum — *for relations* (transitive S-V-O) |

- **lift #1**: NP ⊕ VP → **S = 4-D** (a spatiotemporal event: 3 spatial + 1 temporal).
- **lift #2** (a second Lift): S(4-D) → **5-D**, the added dimension = **modality**
  (possibly / necessarily / …). "Lifting twice is required for modality."

So the idea is an **order ladder**: spatial(3) → spatiotemporal(4, event) →
modal(5, possible-worlds). The decoder unwinds it: `lower` from 5-D modal → 4-D S
→ identify the lift (verb) → split NP₁ / VP / NP₂.

**Grounding result (verified 2026-06-20).** BUILT and matching the design:
**lift #1 composes NP+VP → S** (folds `.what`, passes NP's `.where`, advances VP's
`.when`: "spatial from NP, temporal from VP" — `Language.py:2541-2569`;
`doc/Reasoning.md:149` even writes `S4 = lift(NP3, VP1)`); the **three-slot idea**
exists; the **verb-as-lift-operator** exists (`lexical_gate`/`verbEigEdit`) but is
**dark** (not in configs). PLANNED-ONLY / ABSENT: **NP=3-D** (`.where` is 2-D —
3-D planned, `todo.md:46`); **S=4-D as a literal widened band** (the band stays
`2 .where + 2 .when`; nWhat=1); and the **entire modality layer** — the 5th
dimension + the second lift (alethic possibly/necessarily) — is **absent / open**
(`todo.md:15`; `surface_tense.py` "modal hook noted, not built").

Two gotchas the grounding surfaced:
- **"modality" is overloaded.** In the live code "modality" means the
  `.what`/`.where`/`.when` mux (multi-*modal* channels), NOT alethic modality. The
  5th-D modal layer must be named/added distinctly to avoid colliding with it.
- **Slot order has two conventions.** The in-STM buffer is **predicate-first**
  `[predicate, idea1, idea2]` (= `[VP, NP₁, NP₂]`); but `_last_svo` and the
  persistent truth stores (`RelativeTruthStore`/`TernaryTruthStore`,
  `[cap, 3, nDim]` = slot0 NP₁, 1 VP, 2 NP₂) use **NP₁, VP, NP₂** — matching this
  design. The decoder should align to the **(NP₁, VP, NP₂)** store convention. Note
  the live "predicate" is a *relation* (part/isEqual/isPart), a generalization of
  "VP". (A `gen_diagrams.py` figure already depicts the intended 5-dim MP/VP/3·NP —
  the design is in the diagrams ahead of the code.)

## Lift / VP-application: keep NP and VP recoverable (Alec, 2026-06-20)

`lift(NP, VP)` is an NP (a 3-D spatial object) modified by a VP (a 1-D temporal
process). Read as pure function application, `VP(NP)` yields an NP at a later
time — but that **collapses the verb**: the sentence's information content
*includes* the VP, so the idea must NOT reduce to a later NP (else "the dog ate"
becomes indistinguishable from a description of the post-eating dog). Two ways to
keep the VP recoverable:

- **(A) Store NP and VP as separate components of the idea.** We have the slots
  (the relative-idea STM is depth-3: `[predicate, idea1, idea2]`). Keep NP and VP
  distinct rather than folding to one blended vector; reverse just reads them
  out. Faithful by storage, easy reverse. (The current `LiftLayer.reverse` is the
  Sigma *balanced split* → `left==right`, which does NOT separate NP from VP, so
  storage is the simple fix.)
- **(B) The verb applies but leaves significant traces of itself** *(preferred)*.
  The VP genuinely transforms the NP (so the idea is a real composition, not two
  parked operands), but the transform leaves a recoverable signature so the verb
  can be read back — i.e. NOT a pure function application. This is the
  `verbEigEdit` shape: a sparse eigenvalue edit `δ_v` applied to the NP, masked by
  the NP's own eigen-signature, complement-preserving, zero-init residual. The
  edit IS the trace; an invertible edit lets reverse recover both the NP (base)
  and the VP (the edit). Note the review's finding: `verbEigEdit` exists but is
  dark AND is currently NOT inverted in `LiftLayer.reverse` — making the edit
  invertible is the concrete work for (B).

**Decision (Alec, 2026-06-20): preserve the VP *as the Lifting itself* — NOT in
reserved dimension slots.** The verb is not stored as content (neither a `.what`
slice nor the `.when` dims 4–5); it IS the lift operator the verb selects/gates
(`lexical_gate(VP_code)` — one shared operator, a per-verb gate). Consequences:

- `lift` becomes **asymmetric**: NP is the sole *content* operand, VP is the
  *operator*. The symmetric-fold partition-blindness measured below (reverse gives
  `left==right`, exactly 0.0, because NP and VP were both content operands summed
  in atanh-domain) **dissolves** — there is no second content operand to separate.
- The idea stays the same width (no reserved slots); the verb's temporal footprint
  rides the existing `.when` shift, and its identity rides the operator.
- **Reverse = lower(result) → NP, plus identify *which* lift was applied → VP.**
  Recovering the verb is an operator-level inverse: match the transformation's
  signature against the verb-operator (gate) family — a codebook search over
  *operators*, not over content. Faithfulness therefore requires the per-verb lift
  operators to be **distinguishable by their effect** ("the verb leaves
  significant traces"): different verbs must imprint identifiable, separable
  signatures so the reverse search is well-posed.

This supersedes the (A) dim-partition / reserved-slot framing above; (A) is no
longer the plan. The remaining research question (pending the neuro-evidence
search) is whether the verb-as-time-varying-operator account is supported and how
much signature separation the operator family needs.

## Operator algebra and the dimensionality resolution (Alec, 2026-06-20)

The operators line up with DisCoCat (categorical compositional distributional
semantics: grammar → tensor contraction, nouns = vectors, modifiers = tensors),
but we deliberately pick **low-order** realizations that avoid DisCoCat's
"tensor blowup". The full operator inventory — every grammar-dispatched op with
its `forward` (compose) and `reverse` (generate) rule, annotated with the
POS↔operator↔math map and measured invertibility — is declared in
`data/full.grammar` (validated against the live parser; complements the collapsed
`role_collapsed.grammar`/`complete.grammar` by listing `part`/`assertPart`/the
`query` family that those omit).

| op | realization | order / cost |
|---|---|---|
| NP | a vector (tensor) in the `d`-space; **carries its own eigen-frame** (its activated signature) | order-1 |
| VP | sparse edit `δ_v` applied in the **NP's own eigen-frame** (gated by the NP's signature), **conditioned on the NP's class** (category codebook) | O(nnz) per verb, NP-conditional |
| ADJ | **order-1 mask / intersection on NP** (Boolean meet = an element-wise gate vector), NOT a `d×d` matrix | order-1 |
| ADV | **1-D modification of the VP's eigenvalues** (`λ_v`), NOT an order-3/4 tensor | O(1)/1-D |
| DET | order-reducer (count-noun 4-D → individual 3-D); Montague generalized quantifier | fixed |
| copula ("is" of definition) | a **relation over propositions** (identity/equality), not an operator | relation |
| and / or / not | Boolean join / meet / complement on the meaning lattice | — |

**The blowup resolution — NO global eigenbasis (corrected 2026-06-20).** Vanilla
DisCoCat makes a transitive verb `d³`, an adjective `d²`, an adverb `d⁴`. A
*single shared* eigenbasis `Q` would collapse this but is **over-restrictive**:
the basis that diagonalizes a verb for *apple* ≠ for *idea*, so one `Q` cannot
serve hundreds of thousands of distinct NPs. Instead: **the verb acts in the NP's
OWN frame, conditioned on the NP's class.** Each NP's activated eigen-signature is
the frame; the verb is a **sparse edit `δ_v` gated by that signature**
(`p_class ⊙ δ_v`, exactly what dark `verbEigEdit` does) and **conditioned on the
NP's class** via the MetaSymbol category codebook (~hundreds of classes, not 100k
instances). **What is shared is the taxonomy/class lattice, not a basis.** This
unifies (a) the old "single matrix" → the shared *class structure*, (b) the new
"separate sparse operator per verb" → sparse `δ_v`, (c) `verbEigEdit`, and (d)
"ADV modifies eigs of VP" (`δ_v ↦ m_adv ⊙ δ_v`). It is invertible in the NP's
frame (recover NP by identifying the verb's sparse signature — matching-pursuit
over the `δ` codebook = "the verb leaves significant traces"), matches the
Frobenius spider/router picture (copy the NP's active dims + pointwise edit), and
costs **O(nnz) per verb, NP-conditional, scaling on classes** — never `d²`.
Expressivity rung: a diagonal/additive edit scales/gates but cannot rotate/mix;
if needed, go **sparse low-rank** `δ_v = u vᵀ` (class-conditioned, O(d·r)).

**The "1000-dim NP is untenable" half.** Identity is an **index** (the VQ/EMA
codebook stores rows; an NP = a codebook index + low-dim typed fields `.where`/
`.when` + a few eigen-coeffs), never a dense `d`-vector per token; and NP is
**sparse in the eigenbasis**, so its effective dim is small even though the ambient
`d` is large. The `d`-space is shared (`Q`), not paid per symbol.

(Caveat: the order-1 mask for ADJ is exact for **intersective** adjectives;
subsective/privative ones — "former president", "fake gun" — are not pure masks
and would need the eigen-edit form, like a verb.)

### Pedigree & compression menu (literature, verified 2026-06-20)

The operator algebra is **Lambek/pregroup + Montague + Keenan-Faltz Boolean +
DisCoCat**, and the type→tensor-order map is a *theorem* (Coecke-Sadrzadeh-Clark
2010). The `d³` verb cube is the field's CENTRAL problem (Kartsaklis 2014:
`d=300` → transitive verb **27M params**, ditransitive **8.1B**). Established,
composable cures (all confirmed), ranked by fit to "sparse operator per verb":

| cure | cost | source |
|---|---|---|
| **CP / PARAFAC low-rank** (verb = Σ of R rank-1 tensors) — *recommended store* | `O(3·R·d)`, ~2 orders of magnitude, "little accuracy loss" | Fried-Polajnar-Clark 2015 (ACL) |
| **Tucker / factorized** (small **shared core** × per-verb factors) — *the shared-vs-per-verb answer* | `O(k·d)`, k≪d | Factorized Transitive Verbs 2016 |
| **Frobenius copy/spider** (Δ = vector→**diagonal**; copy-all → elementwise mult) | `d²`→diagonal `O(d)` | Sadrzadeh-Clark-Coecke 2013; Kartsaklis-Sadrzadeh 2013 |
| **Lexical-function** (one `d×d` matrix per word; transitive = 2 matrices summed, extra arg → diagonal identity) | `O(d²)` per word | Baroni-Zamparelli 2010; Paperno-Pham-Baroni 2014 |
| **Diagonal / element-wise** (pure eigenvalue action) + matrix-vector nets | `O(d)` | Socher 2012; Mitchell-Lapata 2008 |
| **Tensor-train / MPS** embedding compression — *for the "1000-dim NP untenable" worry* | orders of magnitude | Khrulkov-Hrinchuk-Oseledets 2019; TensorGPT 2023 |

**Resolution of the shared-vs-per-verb tension:** the literature default is "shared
structure + per-verb factors," and **Tucker** is exactly that — a small *shared
core* times *per-verb factor matrices* — the principled middle between a single
global basis (over-restrictive, rejected) and a free `d²` per verb (untenable). A
**per-verb low-rank (CP) matrix is NP-conditional** (it acts on the NP vector →
output depends on the NP) at `O(d·R)`, answering the 100k-NP objection. And the
honest empirical finding: diagonal/matrix/low-rank models perform **comparably to
or better than full cubes** (the cube is redundant) — so the compression is safe.

**Three flags from the lit:**
- **copula = identity** ("is of definition") is the *standard* treatment (η-map /
  Frobenius identity, Kartsaklis §5.1) — principled, not a hack. ✓
- **DET = generalized quantifier** (Barwise-Cooper; Keenan-Stavi; categorically
  Hedges-Sadrzadeh) — but the specific "**DET reduces dim 4→3**" is *your* geometric
  reading, NOT how the field models it. Keep as design, not citation-backed.
- **ADV = eigenvalue-edit** is the **one genuinely novel piece** — diagonal-adverb
  operators exist (Maillard-Clark-Grefenstette 2014) but the eigen-edit framing is
  yours. **modality + tense-as-a-dimension is the OPEN GAP**: the field models tense
  with temporal operators/intervals (Montague), not a dimension, and has no tensor
  realization of modality — consistent with "no generative DisCoCat LLM exists."

### VP parameterization — the N-vector operator (settled 2026-06-20)

Answering "an N-sized vector representation of an N×N matrix guaranteeing symmetry
or invertibility, justified for VP": two independent searches converge on an
**eigendecomposition-structured operator** `VP = Q · Λ · Qᵀ` (Q orthogonal so
`Q⁻¹ = Qᵀ` is exact and free). The O(N) vs O(N²) limit is real (full free
mix+scale is > O(N) unless the basis is shared/amortized or NP-derived), so pick
the spectrum by what the verb needs:

- **Symmetry option** — `Λ = diag(eᵂ)` (real, positive). `VP = Q diag(eᵂ) Qᵀ` is
  **symmetric, always invertible** (`eᵂ>0`), inverse `Q diag(e⁻ᵂ) Qᵀ`. Pure
  stretch, no rotation. The N-vector `w` = the log-eigenvalues.
- **Invertibility + rotation option** — `Λ = blockdiag(γᵢ·R(θᵢ))` (2×2 scale-
  rotation blocks; complex eigenpairs `γᵢe^{±iθᵢ}`). This is the **non-normal RNN**
  form (Kerg et al., NeurIPS 2019, `V = P(Λ+T)Pᵀ`, learnable radii γᵢ): rotation
  AND scaling, eigenvalues allowed **off** the unit circle, **invertible** for
  `γᵢ≠0` (use `γ=eᵍ`), still **O(N)**. You do NOT need to jump to butterfly for
  rotation.

In both, **ADV = eigenmodifier** edits the spectrum directly (`w`, or radii `γ`
/ angles `θ`) — exactly "ADV modifies eigs of VP."

**Sparsity / non-destructiveness (Alec, 2026-06-20).** A verb is a **sparse,
non-destructive edit** of the NP's state — "a person running" changes the
motion/over-time dims and **leaves the rest of the person identical**. So the
operator must be `I + (edit on a sparse support)`, NOT a dense diagonal: a full
`diag(eᵂ)` (all `w≠0`) has **overly general support** — it can rewrite every
dimension, which is a state *rewrite*, not a verb. The fix is to keep the spectrum
at **identity (`eᵂ=1`, i.e. `w=0`) except on a sparse set** of the verb's
characteristic directions:

`VP = I + U_S (Λ_S − I) U_Sᵀ`  with support `|S| = r ≪ N` (complement preserved).

- **Non-destructive + invertible**: identity off `S`; on `S`, `eᵂ>0` so the inverse
  is `I + U_S(Λ_S⁻¹ − I)U_Sᵀ`. The complement is byte-preserved.
- **The shared structure is a DICTIONARY of edit-directions** (verbal/state-change
  primitives), and each verb is a **sparse selection** from it (+ magnitudes) — NOT
  a global eigenbasis over the 100k NPs. This sidesteps both problems at once: no
  over-general diagonal, no over-restrictive shared NP-basis. ADV scales the
  selection; the support may be NP-class-masked.
- This is the `verbEigEdit` mechanism (soft-thresholded sparse edit,
  complement-preserving, zero-init residual). **IMPLEMENTED 2026-06-20:** the
  eig-based *verb* edit was **removed** — the verb is the **lift operator itself**
  (`lift.forward` is now a plain sigma fold), and the eigen-edit machinery was
  rewritten as the **adverb eigenmodifier** `LiftLayer.apply_adverb` (gated
  `<adverbEigEdit>`, masked by the VP's own eigen-signature). The verb's own
  sparse-operator form (the `eᵂ`/block-inverse spectrum) remains the open Stage-1
  build.

Enforce sparsity as the inductive bias (soft-threshold / L0-L1 on the edit) — it is
what makes the operator a *verb* (local edit) rather than a general rewrite.

**The eigenbasis `Q` (NP-conditional):** a short product of K Householder
reflectors (Mhammedi et al. 2017, O(KN), `Q⁻¹=Qᵀ` by construction) whose
reflection vectors are **emitted from the NP by a small hypernetwork `g(NP)`** —
so the frame is the NP's own (no global basis), answering the open sub-question.
For richer mixing, `Q` = a **butterfly/Monarch** factor (O(N log N); Dao et al.
2020/2022). Both are invertible-by-construction — your LDU requirement.

**Maps onto the codebase:** `Λ` = the `InvertibleLinearLayer`'s `D` (generalized
to 2×2 blocks, `exp`-parameterized); `Q` = the existing **butterfly** (or a
Householder product). The verb becomes the diagonal/spectrum `w_v` (+ optional
block angles) of a shared-structured invertible layer — invertible by
construction, O(N)–O(N log N), eigenvalue-editable for ADV, reusing primitives
already tested for round-trip exactness.

## Compositionality of nouns and verbs (Alec, 2026-06-20)

Three desiderata, all supported, and they collapse the algebra to one structure.

**1. Nouns are functions on ⊤ ("everything"); nouns intersect like ADJ.**
`person(⊤)=person`, `green(⊤)=green thing`, `green ∩ person = green person`. This
is the Montague/Keenan-Faltz view: a common noun **is a predicate** (type `e→t`),
the *same type* as an intersective adjective, composed by **Boolean meet**. So
**NP and ADJ share one mechanism** — an order-1 mask/meet on the universal `⊤`
(the grammar's `U` "everything" start symbol; the PartSpace ATOM↔UNIVERSE poles).
The noun-vector and the noun-function are the same object (a lattice element;
applying it = meeting it). *Caveat:* intersective only — noun-noun compounds
("toy gun", "stone lion") are **non-intersective**, like privative adjectives, and
need the verb-style eigen-edit, not a pure mask.

**2. Verbs compose from verbs: `running = fast(walking)`.** Derived verbs are
compositions of sparse edits over the shared primitive dictionary: `walking`
selects the motion primitives at magnitude `m`; `fast` is an ADV eigenmodifier
scaling them. Key: **verb composition is ADDITION in the sparse log-eigenvalue
space** — `w_running = w_walking + w_fast` (eigenvalues multiply). So
verb-from-verb and ADV-modification are the *same* operation (add in `w`), and
`running = fast(walking)` falls out for free. *Caveat:* clean for manner/degree
derivations; suppletive/idiosyncratic verbs don't decompose.

**3. VPs recurse: `VP(VP(NP))` — "she said John gave her the book".** Verbs apply
to **idea-states (propositions)**, not just atomic NPs: ideas and NPs are the same
kind of object (C-tier states), so a verb-edit applies to a composed proposition
the same way it applies to an NP. `said(she, [gave(John, her, book)])` — the inner
idea is an argument to the outer verb. The recursion **is the symbolic-order
relational pump** (ideas-of-ideas), and the decoder unwinds it by **repeated
`lower`**. *Caveat:* the binding ("her" = she) inside the recursion is the
`ContextualBindLayer` — a known stub and the genuinely hard, parse-context case.

**The unification.** Content words are one compositional algebra:
- nouns & adjectives = **meets** (restrictions on ⊤) → compose by intersection;
- verbs = **sparse non-destructive edits** → compose by **additive eigen-composition** (= ADV) and apply **recursively** to ideas (symbolic order);
- one algebra: meet for nominal restriction, eigen-addition for verbal edits,
  recursion for sentential embedding. (Montague/Boolean predicates + the
  eigen-edit verb algebra, unified.)

### Proper vs abstract nouns: points vs regions (Alec, 2026-06-20)

The noun-as-function-on-⊤ produces a **point** or a **region** by kind — which is
the Montague type distinction, geometrically:

- **Proper noun** — `Alec(⊤)` = a **POINT** (a specific individual; `.where` extent
  ≈ 0, the degenerate bracket `[x,x]`). Montague type **`e`** (individual
  constant). This is the **reducibility base case** — it snaps to a single
  codebook code (the `Felix`/`Alec` case the decoder started from); decoding fires
  the terminal test immediately and emits one word.
- **Abstract / common noun** — `person(⊤)` = a **REGION** (nonzero `.where`
  extent `[start,end]`) defined by **`π` composition** (intersection of defining
  properties = **intension**) or **`σ` composition** (union of instances =
  **extension**). Montague type **`e→t`** (a predicate / set). The two definitions
  are dual (extension of the intension; the ramsification / Galois connection) and
  the project already carries both — the subsymbolic system learns *extension*,
  the taxonomy / `RelativeTruthStore` learns *intension*.

This finally grounds **DET geometrically**: a determiner **collapses a region to a
point** — "the person" = a *point* selected from the person-*region*. That is the
literal meaning of "DET reduces a count noun 4→3": drop the extent/count
dimension (region → individual). **ADJ** stays within the region (intersect →
a *smaller* region). And `.where`'s endpoint-sum bracket already encodes the
distinction for free: **extent 0 = point (proper / determined), extent > 0 =
region (kind)**.

Decoder consequence: a proper/determined noun decodes as a point-code (terminal,
one word); an undetermined common noun decodes as the region (the kind word), and
a DET on it collapses the region to an instance point before realization.

## rewrite(): surface realization at the object→word boundary (Alec, 2026-06-20)

The lexical leaf — the **object→word boundary** where the codebook reverse
(`decode_reverse_meta`) yields a **lemma** — ends in a surface realizer
`rewrite()` that inflects the lemma to its surface form (`run` → ran / will run /
running) from the tense/aspect features (read off `.when`, set by `TenseLayer`).
It is a **dual-route ("words and rules") realizer**:

- a per-word, **feature-keyed LOOKUP** over the word's stored surface forms (its
  paradigm cells) — handles **irregulars / memorized** forms ("ran"); only
  *unpredictable* cells need storing (sparse, lives in the lexicon/codebook entry);
- a shared, **general TRANSFORM** applying the regular rule (`-ed`/`-s`/`-ing`)
  from the features — the **productive** route for regulars and **novel/unseen**
  words. NB: this must be a **char-level seq2seq** (LSTM-with-attention or a tiny
  transformer), **not a fixed-vector MLP** — inflection is variable-length string
  transduction, not classification (SIGMORPHON evidence).

Lookup wins when a cell is stored; otherwise the transform fires. The dual route
is what avoids over-regularization ("runned") while still generalizing to novel
words. It **replaces the dead `aspect` slot** and **subsumes `morphology`'s
generation role** (feature-driven, not token-driven, so it works at decode where
no surface token exists).

**BIDIRECTIONAL — rewrite() runs on BOTH sides of the lexical interface (Alec,
2026-06-20):**
- **forward (comprehension / analysis):** surface form → (lemma, features).
  `ran` / `running` / `runs` / `run` all map to the **same lexeme `run`**, with the
  inflection extracted as features (→ `.when` tense). So **many surface structures
  collapse to one lexeme.**
- **reverse (generation / realization):** (lemma, features) → surface form
  (`run` + PAST → `ran`).

**Payoff:** the codebook / lexicon is over **lexemes, not surface forms** — the
whole inflectional paradigm collapses to one entry per lexeme (vocabulary economy
+ generalization: "run" is one concept regardless of inflection), and surface
variation lives in the **features**, not in separate codebook rows. This is
standard bidirectional (FST-style) morphology — analysis + generation — with the
dual route (paradigm lookup + char-seq2seq) on **each** side. Feasible (morphological reinflection is small/solved);
standard for compositional / morphology-aware generators (mainstream LLMs only
skip it by folding inflection into subword BPE). Needs a `RewriteLayer` (or a
decode post-process); planned. (Grammar: `data/full.grammar`.)

## Build order

- **Stage 0 — Felix (order-0 proper noun, `S → NP`).** Already decodes via
  nearest-codebook. Use it to verify the loop plumbing, the muxed-idea sourcing,
  and the residual→0 completeness test. Repurpose the (currently inert)
  `idea_decode` flag. No structural learning here.
- **Stage 1 — exact compositional (`lift`: `S → NP VP`).** Exercises
  `Sigma.reverse` + grammar-ordered (SVO) emission + residual reduction. No
  lossy guessing — the first real generative split.
- **Stage 2 — lossy compositional (`intersection`/`union`: "black cat").** The
  first exercise of the +/−/0 constrained inverse.

Throughout: learn the reverse-rule selection via the two-pass; gate the category
table over allowable derivations; keep `<ideaDecode>` off → byte-identical.

## Per-op forward/reverse review (completed 2026-06-20)

A per-method review (read the live bodies + the nearby-LLM tiers) produced the
following CONFIRMED/CORRECTED tiers. Three corrections from the nearby-LLM list:
the relational/query family are bare stubs (not "lossy-acceptable"); `ExistLayer`
and `AspectLayer` are exact identities (not lossy); and `LiftingLayer`/
`LoweringLayer` (Layers.py) are NOT the grammar lift/lower (name collision — they
have no reverse). Next step is to walk each op and EMPIRICALLY round-trip it
("see if they are faithful").

**Tier A — strong / exact (run reverse directly):** `SigmaLayer`, `PiLayer`
(unary reverse exact via LDU; binary `generate` = balanced split left==right,
exact on the *sum* only); `LiftLayer`/`LowerLayer` (Language.py — the REAL
grammar lift/lower; reverse = the balanced split; `.where` duplicated to both
children, `.when` shifted; `verbEigEdit` NOT inverted); `NotLayer`/`NonLayer`
(exact involutions); `TenseLayer` (exact ±δ rotation, BUT reverse reads
out-of-band `_op` — tense must be re-decided from `.when` vs the clock);
`ExistLayer`/`AspectLayer` (exact identities — CORRECTED from lossy); base
`InvertibleLinearLayer`/NonNeg/Contractive (exact LDU; ergodic is a
forward-carrier → keep off for top-down generation).

**Tier B — lossy, real codebook recommender exists (extend into +/−/0):**
`UnionLayer`/`IntersectionLayer`, `ConjunctionLayer`/`DisjunctionLayer`. Reverse
has TWO modes: with a populated `basis` → `Ops.disjunctionReverse`/
`conjunctionReverse` (a genuine codebook size/overlap search); without → silently
`(parent,parent)`. CORRECTION: the recommender is a size/overlap search, NOT a
literal +/−/0 ternary filter — the `+` maps to the hard `S≥parent`/`S≤parent`
feasibility mask; there is no explicit `−`/`0` channel. The +/−/0 design is the
TARGET to build on top of `Ops._binary_op_recommend` (Layers.py:12576); step 1 is
guaranteeing a populated basis so it never degrades to `(parent,parent)`.

**Tier C — genuine stubs (reverse = bare `(parent,parent)`, build from scratch):**
`IsEqualLayer`, `IsPartLayer`, `PartLayer`/`AssertPartLayer`, `QueryLayer`/
`QueryPartLayer`/`QueryEqualLayer`, `ContextualBindLayer`. Note `IsPart`/`Part`
FORWARD returns `right` (the part operand is DESTROYED, not entangled); the
parthood geometry lives in `QueryLayer.forward`, not `Part`. **DEFERRED to the
reasoning layer** — these are query operators reasoning consumes, not decoder
ops; their reverses are captured in `2026-06-20-reasoning.md`, NOT built here.

**Tier D — math may be exact but depends on a carrier absent at generation:**
`MorphologyLayer` (reverse re-analyzes the surface token = the output we're
generating → read tense from `.when` vs now + add a real re-inflection step);
`SymbolizeLayer` (exact iff the META is in the persisted taxonomy, else balanced
split); `PrepositionLayer` (phrase content reverse EXACT via ±0.6 `.where`
rotation; only the marker is lossy + `theta=0.6` is shared across prepositions).

**Legacy (Layers.py, dormant — not in `role_collapsed.grammar`):** `EqualLayer`,
`True`/`False` (destroy a pole), `Swap`/`Copy` (discard an operand),
`Area`/`Luminosity`/`IsaPart` (collapse to a scalar — undecodable). `IsaPartLayer`
≠ the grammar's `isPart`. Ignore unless revived.

### Measured faithfulness (empirical round-trips, 2026-06-20)

Ran the round-trips (`test/_faithfulness_lift.py`, `test/_faithfulness_ops.py`):

| op (POS) | result |
|---|---|
| `lift` (VP) / `lower` (DET) | exact on the sum (`fwd∘rev` Δ≈6e-8) but **balanced split `l==r` exactly 0.0** → partition-blind as a symmetric content-fold (resolved by treating VP/DET as the *operator*, not a content operand) |
| `not` (negation) | **exact involution** `not(not(x))==x` (Δ=0.0); `not(x)` genuinely flips. No work needed. |
| `intersection` (ADJ) | no basis → `(parent,parent)` stub; **with codebook basis → reconstruction Δ=0.0, operands are codebook rows** ✅ |
| `conjunction` (and) / `disjunction` (or) | same: stub without basis; **exact reconstruction (Δ=0.0) with basis** ✅ |
| `union` (ADV) | stub without basis; **with basis the recovery is greedy/approximate — reconstruction Δ≈0.3 observed** (largest-part + residual-cover); the one recommender that is genuinely lossy and needs the +/−/0 hardening |

Takeaways: the four set/logical reverses are faithful **only if a populated codebook
basis is threaded in** (the live dispatch does pass `basis=subspace.what`,
Language.py:5401); without it they silently return `(parent,parent)`. Min-family
(AND/intersection) reconstructs exactly; **OR/union is the approximate one**.

**Remaining ops measured (`test/_faithfulness_rest.py`) — all match the review:**

| op | measured | verdict |
|---|---|---|
| `non` | `non(non(x))==x` Δ=6e-8 | **exact involution** ✓ |
| `tense` | `reverse(forward(x))==x` Δ=1.5e-8 (`.when` shifted) | **exact rotation given the tense** (decoder must pick tense from `.when` vs now) |
| `aspect`, `exist` | fwd/rev Δ=0.0 | **exact identities** ✓ |
| `sigma`,`pi` (unary) | `reverse(forward(x))==x` Δ≈0 | **exact LDU inverse** ✓ |
| `isEqual` | fwd=max, rev=`(parent,parent)` | **stub** |
| `isPart`,`part` | **fwd=`right` (drops left)**, rev=`(parent,parent)` | **stub** (part destroyed) |
| `query`/`queryPart`/`queryEqual` | fwd=lossy truth, rev=`(parent,parent)` | **stub** (stage-4 QA) |
| `morphology` | cold fwd/rev Δ=0.0 (passthrough) | **carrier-dep**: needs tense from `.when`+re-inflection |
| `preposition` | phrase Δ(r,phrase)=6e-8; marker slot l==r=0.0 | **content EXACT** (`.where` un-rotated); marker lost |
| `symbolize` | cold fwd=(a+b)/2, rev=balanced (parent/2) | **carrier-dep**: exact iff META in taxonomy, else balanced split |
| `contextualBind` | rev=`(parent,parent)` | **stub** (binding needs parse context) |
| legacy `equal`/`true`/`swap`/`copy` | `(parent,parent)` / identity-stub / drops an operand | **dormant** (not in grammar) |

**Complete verdict.** Faithful as-is (exact): `lift`,`lower` (on the sum),
`not`,`non`,`tense`,`aspect`,`exist`,`sigma`,`pi`, and `intersection`/`conjunction`/
`disjunction` *with a basis*. Faithful with care: `union` (greedy ~0.3),
`preposition` (content exact, marker needs binding), `symbolize` (needs persisted
taxonomy). Stubs to build: the truth/query family + `contextualBind` (stage-4 QA,
downstream). Carrier-dep: `morphology`. **Stage 1 rests only on the exact tier —
all green.**
