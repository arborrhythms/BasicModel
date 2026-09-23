# Item 11a brief

Preserved from `todo.md` at the start of this landing, including the review
prerequisites. The normalization-lifetime question remains for Alec; the
implementation choice and measurements are in [the landing record](README.md).

- **11a. WholeSpace analysis as intersections of properties** (Alec,
   2026-09-23; new). WholeSpace's analysis is neither computationally
   complete nor the dual of PartSpace's. Today it is region means of the
   unity snapped to VQ prototypes, a fixed lookup table of four property rows
   (LETTER, DIGIT, CAPITAL, punctuation) and a monotone pi membership slot;
   none of these can form a learned "a or b". The design, by tower:
   **OR belongs to WholeSpace** — a property is a union of primitive
   elements, the letter class `a ∨ b ∨ c …` over the byte atoms, a word the
   maximal run over which a property holds, and a narrower type the
   intersection of properties (CAPITAL ∧ LETTER); **AND belongs to
   PartSpace** — a whole is the co-presence of its parts, the synthesis
   fold; and their **combination belongs to the concept**: XOR is the kind
   over WholeSpace's property `A ∨ B` and not PartSpace's whole `A ∧ B`,
   the negation through the whole's `c⁻`. `A` and `B` must be positionally
   grounded across the two independently coded towers — the same `.where`
   in both — for the concept's combination to name one occurrence.
   This means one occurrence/extent with positions or roles inside it;
   A and B need not occupy the same raw byte position. An observed zero
   supplies counterevidence; a missing observation remains unknown.
   Completeness is a property of that combined architecture, not of either
   tower (Codex, 2026-09-23). The negation is evaluated per position — the
   whole's `c⁻` at the positions where `A ∨ B` holds — since read over the
   whole field every concept co-present with anything is "both" and the
   negative channel would discriminate nothing (two-truths §1.1). Words, spaces and digits
   are a priori today only because we hand them to the mind; they are to be
   defined in terms of primitive elements, as learned properties over the
   byte atoms, with the complement side of each bifurcation available so the
   algebra is complete. **Before the redesign, from the September 23
   landing review** (Claude; decisions Alec): (i) the snap reads in the
   field's own chart — cosine times slot norm for unit-ball codes, the `√D`
   divisor only for cube-valued slots — and `conceptEvidenceFloor` is
   recalibrated in those units (measured: slot norms .16–.97 at D = 1024,
   presence ceiling .03, union of eight slots ≤ .22, so the .5 use floor
   was unreachable; two-truths §1.1); (ii) the field's occurrence axis
   becomes the subject's extent, a run of positions from the towers'
   `.where`, with pairs kept per position inside it — co-presence is
   collected and negation scoped per extent, not per tile; (iii)
   co-presence discovery writes positive parts only; a negated part is
   learned or sealed, never witnessed; (iv) Codex confirms no SymbolSpace
   reader maps a leg row to a codebook row by identity in the parallel path
   (the leg now has two rows per concept) and records the rule `row // 2`
   in the shared-index invariant; (v) open for Alec: max-normalisation
   after each optimizer step applies to every assigned row for life while
   seal-minted rows are never normalised; and (vi) the union-over-admitted-slots
   read of §1.1 is confirmed (Alec, 2026-09-23) with the floor kept as a
   calibrated parameter.
   Codex's item, design and build (Alec, 2026-09-23), from this
   brief: fix the property algebra, its folds and reverses, and what stays a
   priori, in the Architecture doc before code; build it behind the existing
   analysis/synthesis knobs and delete the lookup-table path it replaces —
   no two permanent modes; Claude reviews the landing. **Exit:** the four a-priori property
   rows reproduced as learned properties over bytes; segmentation by
   constant-type runs unchanged on the ladder corpus; and the **end-to-end
   XOR test point** — the parallel model on XOR input, WholeSpace's property
   `A ∨ B` and PartSpace's whole `A ∧ B` from primitive elements, grounded
   at one `.where`, snapped to order-0 atoms, the concept `(A ∨ B) ∧ ¬(A ∧
   B)` over them through the whole's `c⁻`, and the readout — passing unseeded,
   in place of item 11's `xfail` pyramid gate. Interacts with item 10 (the
   membership folds of the towers): measure there before adopting a fold
   here.
