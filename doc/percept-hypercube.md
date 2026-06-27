# Percepts on the positive unit hypercube `[0,1]^D`

> **Design.** A percept is a vector of independent **presence** features. Each
> coordinate is a membership in `[0,1]` — `0` = *nothing* (absent), `1` =
> *everything* (fully present) — so a percept lives on the **positive unit
> hypercube** `[0,1]^D`. Percepts are **one-sided**: they have no negative
> half. The percept "opposite" is therefore the **complement** `1-x`, not the
> negation `-x`. This doc states the percept geometry once; the code points
> here. Concepts and symbols are a *different* geometry — the signed sphere —
> covered at the end and in
> [plans/2026-06-23-conceptual-similarity-space.md](plans/2026-06-23-conceptual-similarity-space.md).

This is the partner of the **concept** similarity space. Percepts = the cube
(presence); concepts = the sphere (meaning ⊕ certainty). Keeping them distinct
is what resolves a long-standing confusion where a single `[-1,1]` encoding was
carried everywhere even though percepts never use the negative half.

---

## 1. Presence, not signal

A percept feature answers *how present is this?* — a one-sided measurement:

| value | meaning |
|---|---|
| `0` | nothing / absent (the empty corner) |
| `1` | everything / fully present |
| `x ∈ (0,1)` | partial presence |

There is no "negative presence." A coordinate going to `0` is a definite
observation of *absence*, not a signed quantity. The model already encodes this
on the percept path:

- **`Layers.PartSpace.factor_percept`** ([bin/Spaces.py](../bin/Spaces.py)):
  *"only its POSITIVE content participates (percepts are one-sided, spec §2;
  negative input coordinates are structurally invisible)"* — evidence
  `a ∈ [0, +1]`, and *"a zero percept yields `a = 0`"*.
- **The σ/π membership folds** (`SigmaLayer2` / `PiLayer2`,
  [bin/Layers.py](../bin/Layers.py)) run on memberships `m ∈ [0,1]` through
  `log`/`exp`, with `m = 0` (the bottom element "nothing") floored to
  `EPS_LOG` before the log.

So the *semantics* are `[0,1]` everywhere on the percept path. What diverged was
the *representation*: input was normalized to `[-1,1]` and the lexicon codebook
is a signed `[-1,1]` torus — carrying a negative half that the percept
semantics never read. Moving percepts to `[0,1]` makes the representation match
the semantics.

---

## 2. The antipode is the complement `1-x` (part/copart — not the catuskoti)

A percept's "opposite" is the **complement**: where it is present, the opposite
is absent, and vice versa.

```
antipode_percept(x) = 1 - x
```

This is **mereological**: the opposite of a *part* is its **copart** — the rest
of the whole. With the whole normalized to `1`, `copart = 1 - part`. A percept
carrier of the form `[part, copart] = [x, 1-x]` is what historically put
percepts on a flat torus; but in a strict mereology `part + copart = 1`, so the
**copart axis is redundant** (`= 1-x`) — derivable, not independent. Dropping it
leaves the single presence coordinate `x ∈ [0,1]`, with `1-x` recovered as an
*operation* (the antipode), not a stored dimension.

This is **not** the catuskoti/tetralemma bivector. That carrier —
`TRUE=[1,0], FALSE=[0,1], BOTH=[1,1], NEITHER=[0,0]` — has **two independent
axes** (it must, to express *both* and *neither*), and belongs to the
**concept/truth** layer. The two carriers look alike (both 2-component) and were
conflated; they are not the same:

| carrier | axes | redundant? | layer |
|---|---|---|---|
| **part / copart** `[x, 1-x]` | dependent (`copart = 1-part`) | **yes** → collapse to `[0,1]` | **percept** |
| **catuskoti** `[TRUE, FALSE]` | independent (BOTH / NEITHER) | no → keep 2-D / signed | **concept / truth** |

### Complement and negation are the same reflection

On `[0,1]` the complement `1-x` and on the signed sphere the negation `-x` are
the **same operation** — reflection through the *uncertain center* — in shifted
coordinates. Let `y = x - ½`:

```
1 - x  =  ½ - y   =  -(x - ½)  =  -y
```

So the complement in `[0,1]` *is* negation once you center at `½`. This is why a
single SBOW routine serves both spaces with one parameter (§6).

---

## 3. σ/π: the membership lattice (zero, identity, absorber)

The mereological fold/split operators run on `[0,1]` memberships and form a
bounded lattice with `0 = nothing (⊥)` and `1 = everything (⊤)`:

| operator | role | identity (no-op) | absorber |
|---|---|---|---|
| **σ** (sigma) | synthesis / union | `0` — `x ∪ ∅ = x` | `1` — `x ∪ ⊤ = ⊤` |
| **π** (pi) | analysis / intersection | `1` — `x ∩ ⊤ = x` | `0` — `x ∩ ∅ = ∅` |

The log-space fold floors `0` (`EPS_LOG`) because in π (the product/AND form)
nothing *absorbs*: `log(0) = -∞`. "Zero" (the membership value) and "identity"
(the lattice no-op element) are different roles — σ's identity is `0`, π's is
`1`.

---

## 4. Percepts vs concepts: absence-informativeness and the straight map

The percept origin and the concept origin mean **opposite** things:

- **percept `0`** = absence — a *certain* observation;
- **concept `0`** (sphere center) = *uncertainty* — no information.

Whether you may map one onto the other depends on a single modeling question —
**is absence informative?** — and the resolution is **per level**:

- **Percept level — absence is non-informative.** You learn only from what is
  present, so `0 = nothing = no evidence`. The **straight map**
  `[0,1] presence → positive orthant` of the concept space is then justified:
  the percept origin lands on the concept origin (uncertain), which is *correct*
  because percept-absence carries nothing to contradict it.
- **Concept level — absence is informative.** "Certainly not a cat" is a real,
  signed fact — but that negative structure is **grown by concept operations**
  (negation, the catuskoti, `NotLayer`), **not** produced by the percept map.
  The percept never emits a negative coordinate; the concept space develops its
  own negative half.

Consequence for the architecture: the **percept→concept seam is presence-
preserving and one-sided** — percept presences `a ∈ [0,+1]` (`factor_percept`)
enter non-negative, and there is **no re-center at that seam**. NB: this is NOT
a vector *injection* of percepts into the concept space. `PerceptDim` and
`ConceptDim` are **decoupled** (a concept is a distinct-dimensional atom, never
a sum/map of percept vectors). Concretely (the ramsified sparse CS transform,
`cs_forward_content`), the non-negative PS/WS **presences** are the *input
activations* to a per-order signed **sparse weight matrix** that produces the
concept activation; the concept's negative half is **grown by the signed
weights** (a negative weight = a feature's presence anti-correlates with the
concept) and signed activations, exactly the "negativity is a concept
operation, not the percept map" principle above. The sanctioned signed↔membership
bridge — `χ(a) = (1+a)/2` and `χ⁻¹(m) = 2m-1` (`Ops.eval_chart` /
`eval_chart_inv`, [bin/Layers.py](../bin/Layers.py)) — stays where it belongs:
the **truth/catuskoti** boundary, not the percept seam.

---

## 5. The geometry split

| | **percepts** | **concepts / symbols** |
|---|---|---|
| space | positive unit hypercube `[0,1]^D` | signed unit sphere / ball |
| coordinate | per-axis **presence** | one **direction** (meaning) + **radius** (certainty) |
| "opposite" | complement `1-x` | negation `-x` |
| sides | one-sided (`negative half unused`) | two-sided (sign is content / truth-polarity) |
| metric | per-axis membership; cosine after centering at `½` | cosine on the sphere |
| magnitude | *is* the presence, per axis (set by observation) | a separate scalar = certainty |
| antipode is | a complement *operation* | a stored direction |

> **Realization note (ramsified sparse CS).** In code the concept "direction"
> is a **strictly-positive** `ConceptDim` atom (`softplus(atom)` — a positive
> mereological-feature signature, not a signed sphere vector), and the
> **sign + magnitude** ("signed sphere / radius = certainty") live in the
> **scalar activation** that scales it: `concept_code = signed_activation ×
> softplus(atom)`. So the "signed sphere" describes the activation's role; the
> stored atom itself is positive. Sign comes from the signed sparse weights.

---

## 6. SBOW situates concepts, **not** percepts

`embed.conceptual_sbow_loss_codes(window, pool, ...)`
([bin/embed.py](../bin/embed.py)) situates **concept** codes by an in-group
attraction and an out-group repulsion (word2vec **CBOW-NS**): in-group rotated
toward the gaussian-weighted neighborhood centroid `pode_dir`; out-group
**repelled from random negative codes** (pairwise SGNS). The out-group is *not*
pulled toward an antipode `-pode_dir` — because concepts carry negation, `-pode`
*is* a code's negation, and attracting the out-group to it collapsed every code
onto the present/absent (`±pode`) axis. The gradient is **tangential** (operates
on unit directions), so a code's magnitude/radius is untouched — only its angle
rotates.

**SBOW applies to concepts only.** Percepts are deliberately **not**
SBOW-situated. A percept's position is *anchored* by two things that
distributional situating would fight:

1. **The perceptual metric** — percepts are *grounded*; their location is the
   sensory similarity of the actual signal (the wrapped-MSE / `[0,1]`-presence
   metric), and percept **identity is decode-by-nearest-neighbor** in that
   metric — so moving a percept vector changes what it decodes to (breaks the
   grounding).
2. **The mereological encoding** — the σ/π part/whole composition and the
   `.where`/`.when` locality are carried in the percept tower. (Whether the
   part↔whole *algebra* is guaranteed by construction — radix + invertible
   folds + tower constraints — vs. only anchored is analyzed in §10; either
   way, the **grounding/identity** is position-dependent and SBOW would scramble
   it.)

The conceptual similarity layer is exactly where codes are free to float by
substitutability; the percept layer is not. The "antipode = `1-x` complement"
geometry of §2 still describes the percept cube — it is just an *operation*
(used by `factor_percept`, the `.where`/`.when` brackets), not a force that
moves percept vectors around.

---

## 7. Encodings that saturate at the bounds

The percept cube `[0,1]` and the concept sphere `[-1,1]` are **bounded**, so a
fixed-point encoding that spends all its bits on the bounded range is more
bit-efficient than a float that wastes most of its exponent on unused dynamic
range:

- **UNORM** (unsigned normalized int) → `[0,1]`, hardware-clamped = the
  **percept** cube.
- **SNORM** (signed normalized int) → `[-1,1]`, hardware-clamped = the
  **concept/symbol** sphere.
- **Q-format fractional** (`_Sat _Fract`, Q15/Q31) with saturating arithmetic
  is the CPU/DSP equivalent.

These are **storage/quantization** formats. Training stays in `float32`/`bf16`,
where there is no auto-`±1` dtype (IEEE overflows to `±∞`); the practical
saturators are functions — **`tanh`** (soft; already the percept fold's
`atanh → tanh`) or `clamp` (hard). Quantizing a codebook to SNORM/UNORM is
standard QAT: keep a float master, quantize in the forward, and backprop with a
**straight-through estimator** (the VQ codebooks already use STE) or stochastic
rounding for the coarse 8-bit case.

A note on precision: over a bounded unit range, **SNORM16 ≥ bf16** almost
everywhere (uniform `~3e-5` grid vs bf16's `~0.4%` relative, which is coarser
than SNORM16 for `|x| > ~2⁻⁷`); SNORM8 only matches bf16 near the boundary and
is a real loss off it.

---

## 8. Migration status

Moving percepts to `[0,1]` is **percept-scoped** and **not byte-identical**
(the default percept path reads sign as form content — see §9). Staged:

**Done (this pass):**
- **Input normalization → `[0,1]` for presence data (provenance-branched).**
  `TheData.normalize`/`denormalize` (`which="input"`) map measured features
  (tensor input; `input_presence=True`) to the `[0,1]` presence range — the
  `_compute_ranges` docstring's long-declared target — retiring the `* 2 - 1`
  signed map. **Signed text embeddings** (list input; `input_presence=False`)
  **keep `[-1,1]`**: they are concept-ish signed vectors, and the *half-done*
  move (input `[0,1]` while the lexicon codebook is still the signed torus)
  breaks the invertible embedding **reconstruction** chain. The test-first
  probe caught exactly this as a **consistent (not flaky)** `XOR_exact`
  reconstruction regression (`'hello world' → ''`, `'loving world' → 'hello'`;
  classification still correct), so the embedding case waits for the lexicon
  move (pending below). Output normalization is **unchanged** (`[-1,1]`).
  Verified: exact round-trip on both branches; the one test pinning the literal
  `[-1,1]` input target updated; full suite green.
- **SBOW out-group → word2vec SGNS (§6).** `conceptual_sbow_loss_codes`'s
  out-group is now pairwise repulsion from random negatives, replacing the
  `-pode` antipode (which collapsed concept codes onto the present/absent axis —
  a real bug, since concepts carry negation). The earlier percept-SBOW `center`
  param was **reverted**: percepts are *anchored*, not SBOW-situated (§6), so
  there is no percept SBOW to center. (The percept antipode `1-x` of §2 remains —
  it is a geometric *operation*, not an SBOW force.)

**Done (2026-06-25 — the `[0,1]` percept-cube lit up; note meronomy ships ON, see §9 correction):**
1. **Torus codebook → `[0,1]`.** The radix percept Codebook reads `[0,1]` via a
   UNORM straight-through clamp in `Codebook.getW` (`embed._unorm_ste`), gated on
   `meronomy_enabled() and is_percept_store` — a percept-store marker set ONLY on
   the radix Codebook (`PartSpace._build_what_basis` radix branch), never the
   lexicon Embedding nor the concept/symbol sphere. The σ/π membership folds
   already run on `[0,1]` under meronomy; the decode (`_snap_content` /
   `_nearest_idx` / `codebookDistance`) uses `_presence_mse_score` (un-wrapped,
   complement-aware) on the percept store. Seed writes route to the master
   parameter (`RadixLayer.insert` → `_basis.W`), not the clamped read view.
2. **No seam adapter needed.** The single `getW` chokepoint makes BOTH the
   forward gather and the reverse decode read `[0,1]`, so the round-trip is
   coherent without a seam (verified: MM_mereology `'hello world' → 'hello world'`,
   reconstruction loss 0). `factor_crossing` is a no-op on no-CS-codebook configs
   and is not on the critical path.
3. **STM — no edit.** Spied STM pushes read `[0,1]`/non-negative under
   meronomy-on (consistent with the migration; NOT signed as the audit assumed) —
   no breaking mix for percept-only configs (round-trip + full suite green).
   Mixing `[0,1]` percepts with *signed* concept rows in one store is a **part-B**
   concern, deferred until the concept taxonomy exists.

Also: `percept_store` relocated off the bare PartSpace onto `self.subspace` (the
dataflow container — Spaces are operators, state rides the subspace).

**Open:** the **`[0,1]` lexicon move** — the default-config signed lexicon
Embedding (synthesis=lexicon) stays signed for now (§9: sign is form content).
Whether to move it to `[0,1]` (completing the half-done input-`[0,1]`) is an open
decision, separate from the radix percept cube.

(No percept SBOW: percepts are anchored, not distributionally situated — §6.)

---

## 9. The live path vs. the meronomy chart (audit `wwgslspjl`)

> **Correction (2026-06-25).** This audit predates the decision that **meronomy
> ships ON** (`data/model.xml` `<meronomy>on</meronomy>`). So the `[0,1]`
> one-sided percept chart is **not** "dark by default" — it is the **live
> default**, and lighting up the codebook decode + the `[0,1]` row re-home (§8)
> *completed* the migration rather than flipping a dark flag. The "off by
> default" phrasing below reflects the audit's then-current default and is
> superseded. The **signed range staying load-bearing for concepts/symbols**
> (last paragraph) is unchanged and correct — that is the part-A / part-B
> boundary: percepts on `[0,1]`, concepts/symbols on the signed sphere.

A precise read-only audit established the key fact that makes this a real change
rather than a config flip: **the `[0,1]` one-sided percept design is the
*meronomy* chart, dark by default; the live/default percept path is the signed
`[-1,1]` torus**, where the sign is **form content** (`word` and `-word` are
distinct tokens). Three sign-readers fire on the default path:

- the percept fold `SigmaLayer.forward` — `atanh()`/`tanh()` (odd functions);
- the percept **VQ** nearest-row (ranks signed vectors);
- the torus decode `_nearest_idx` / `_snap_content` — full-vector wrapped-MSE
  where `word ≠ -word` ("sign is form content").

The one-sided `[0,1]` pieces (`factor_percept`'s `clamp(min=0)`, the
`SigmaLayer2`/`PiLayer2` membership folds, the `cos.abs()` snap) are **all
`meronomy_enabled()`-gated, off by default**. So "percepts are one-sided / on
`[0,1]`" is the **opt-in** chart, and lighting it up changes live behavior
(expect XOR / reconstruction to move) — hence the percept-scoped staging in §8
and the mandatory seam adapter. The **signed range stays load-bearing** for
concepts/symbols (the `use_dot_product` sphere codebook, the `cos.abs`
NEG-quotient snap, `_pole_aligned_score`, the catuskoti `pos-neg` readers,
`eval_chart`) and must not be collapsed.

---

## 10. Mereological guarantees: by-construction vs. metric-anchored

Why can't percepts be SBOW-situated? Not because composition would break — it
wouldn't. The encoding is **hybrid** (audit `wupf3e962`):

**Guaranteed by construction (independent of vector positions):**
- The mereological **structure** is **byte-positional**: the ordered split
  (`RadixLayer.spell_out` longest-prefix-match), part order (slot index `N`),
  the exact pid↔bytes table, the `.where`/`.when` brackets (encoded from byte
  offsets), and "`A` isa `B`" / run-containment (pure byte-interval containment,
  `RunStructureLayer`). None read a percept vector.
- The σ/π fold **invertibility**: `compose(generate(y)) == y` for *arbitrary*
  weights/inputs — a property of the `L·D·U` / butterfly **form** — **but only
  when the fold is built `invertible=True`/butterfly**; the default bare σ/π is a
  plain `LinearLayer` with no exact inverse (so this is structural *when
  configured*, not universally live).

**Anchored by the perceptual metric (position-dependent):**
- Every vector→token identity/decode is **nearest-neighbor in the torus
  wrapped-MSE metric** (`_nearest_idx`, `_snap_content`, `decode_reverse_meta`,
  `RadixLayer.reverse`, `factor_percept`). **A percept's vector position *is* its
  identity.**
- The composition **value** (*which* whole a part-set folds to) is the **learned**
  `SigmaLayer` over the (movable) part vectors. Note: "radix" is a **trie**
  (longest-match), **not place-value arithmetic** — it supplies ordered
  references; the fold supplies the value.

**So SBOW would break the *grounding*, not the algebra.** Relocate percept
vectors and the structure + invertibility survive intact, but every
nearest-codebook decode re-routes: the model would compose/decompose perfectly
while **decoding to the wrong tokens** — a grounding failure masked by intact
algebra. That is the precise reason percepts stay anchored (vector-position =
identity) and only **concepts** (mediated identity) are SBOW-situated.

*Caveats:* whether live `synthesis=radix` configs build `invertible=True` is
unverified; much of the meronomy/order-raising spine is `<mereologyRaise>`-gated
and may be dark by default (this analysis is of the code as written).

## References

- Concept partner: [plans/2026-06-23-conceptual-similarity-space.md](plans/2026-06-23-conceptual-similarity-space.md)
- Mereology / σ-π folds: [Mereology.md](Mereology.md)
- Lexicon / codebook geometry: [Lexicon.md](Lexicon.md)
- Truth / catuskoti / `eval_chart`: [Logic.md](Logic.md)
- Code: `TheData.normalize` ([bin/data.py](../bin/data.py)),
  `conceptual_sbow_loss_codes` ([bin/embed.py](../bin/embed.py)),
  `factor_percept` / σ-π folds ([bin/Spaces.py](../bin/Spaces.py),
  [bin/Layers.py](../bin/Layers.py)).
