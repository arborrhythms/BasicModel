# Serial-Mode Reconstruction via Grammatical Derivation — Design (DRAFT)

> **STATUS: DRAFT 2026-07-04 — awaiting Alec's review. NO implementation
> before approval.** Resolves the Task-6 finding of
> [2026-07-03-reconstruction-fidelity-execution.md](2026-07-03-reconstruction-fidelity-execution.md)
> (grammar's serial exact round-trip structurally unreachable through the
> tensor reverse). Framework: Alec, 2026-07-04.

## The framework (Alec, 2026-07-04)

Serial-mode reconstruction is a GRAMMAR DERIVATION, not a tensor regroup.
Bottom-up parses collapse word-concepts into object-concepts, then apply
grammatical transformations until 1, 2, or 3 concepts remain:

- **1 concept** — NP: a single idea (`exist_O1` start state);
- **2 concepts** — NP + VP: an idea plus a transform (`verb`/`lift`);
- **3 concepts** — NP R NP: two ideas related — the "is of DEFINITION"
  (copula, `isEqual query="false"`) vs the "is of PREDICATION"
  (`isPart query="false"`), the assertive forms feeding the relation
  table (refinement deferred; couples to
  [2026-07-04-attention-to-relation-promotion.md](2026-07-04-attention-to-relation-promotion.md)).

Reconstruction from the reduced state proceeds one of two ways:

- **Method 1 — undo the stored syntax tree**: replay the derivation
  recorded on the way up. Exact by construction.
- **Method 2 — optimal reverse under the given BU/TD attention
  activations**: infer a derivation from the reduced state alone (the
  forward derivation erased). Top-down productions (NP $\to$ ADJ + NP
  etc.) generate multi-word sentences from a single idea; 2- and 3-slot
  ideas have their own start states (see `data/complete.grammar`).

**Teacher/student relation:** Method 1 is the natural supervision signal;
Method 2 is the trainable target. The serial round-trip bar measures how
well Method 2's free derivation reproduces Method 1's exact one.

## Grounding — the machinery already exists (survey 2026-07-04)

Per Alec: the grammar is already present in the generate and compose
subsections. Confirmed, with the routing gap identified:

| Piece | Site | Status |
|---|---|---|
| Grammar start states + productions | `data/complete.grammar:54-179` (`U`, `isEqual_O1`, `isPart_O1`, `exist_O1`; NP ops conjunction/…; VP ops lift/lower/verb/adverb; preposition/bind; tense/morphology) | present |
| Stored parse tree (Method 1 substrate) | `generate_rules` — populated by the forward (`Language.py:10692`), replayed by `_reverse_from_S` (`Models.py:9301`) | present; ALREADY drives the D3 objective (multi-slot reverse `[B,128,D]`) |
| Free inference (Method 2 substrate) | `reconstruct_from_idea` + `clear_grammar_cache` + `_chart_generate_from_stm` (`Models.py:8299`; `Language.py:11200`) | present; infrastructure pinned by `test_reconstruct_from_idea.py` |
| The collapse | `_stm_reducer` (`Models.py:7027`, `BinaryStructuredReductionLayer` over the grammar's arity-2 ops); depth capped by `syntacticOrder` | present |
| **The gap 1 (routing)** | `recon_bench`'s decode → `perceptualSpace.reconstruct_data` → the TENSOR reverse; `_reconstruction_seed` serial arm seeds `snap[:, :1, :]` (`Models.py:3536`) — the grammar generation path is never entered | THE Task-6 blocker |
| **The gap 2 (surface sharpness)** | per-op reverses in `SyntacticLayer.reverse` are identity stubs for non-invertible rules (`Language.py:7546`) | constrains Method-2 sharpness; Method 1 unaffected for the D3-style replay |

## Requirement: the NULL-word pathway (Alec, 2026-07-04)

The static per-word loop (`for p in range(N_words)`, Models.py:8016 — the
recompile-churn fix: fixed slab width, not the dynamic per-sentence
count) is the right choice for compile, BUT it mandates a NULL-word
pathway through BOTH forward() and reverse() for the trailing empty
positions. Today this is a FORWARD-ONLY SHORTCUT: `word_at(p)`
(Spaces.py:8980) returns the ZERO-PADDED slab slice past the sentence,
and the `word_active` gate masks its contribution to zero — bit-identical
ONLY because zero is the identity element for the current compose op, and
it gives reverse() NO signal. (The existing `null_percept_idx` is on the
percept/IR-mask axis, NOT the word axis.)

The principled form — the ∅/NOTHING pole on the WORD axis:
- **forward()**: a padding position carries a genuine NULL-word encoding
  to a NULL idea-contribution that composes as the EXPLICIT identity
  (robust by construction, not by the zero-is-identity accident).
- **reverse()/derivation**: blind Method-2 derivation generates a
  variable word count from one idea and MUST emit NULL-words past the
  sentence end AND recognize them on decode — this is how sentence
  LENGTH round-trips without the scaffold supplying it. The mereological
  form of EOS: NOTHING on the temporal-word axis, symmetric with the
  clock owning exact time and the record store owning exact extents
  (see the encoding design's `.when` note).
- **Round-trip bar consequence**: exact reconstruction = right words AND
  right count (NULL-words in the right trailing positions), recovered
  blind, not received.

**Padding-cost fork (Alec, 2026-07-04) — coupled to the NULL-word.** The
static loop's bound trades three ways: (i) ACTUAL word count = no
padding but UNBOUNDED graphs (the churn, rejected); (ii) FIXED MAX
`N_words` (current churn-fix) = ONE graph but MAX padding (every
sentence padded to the slab); (iii) NEXT-POWER-OF-TWO = the sentence's
loop rounds up to the next $2^k \ge$ its word count, giving log2(max)
distinct lengths (a bounded handful, fits `cache_size_limit`) with
$\le 2\times$ worst-case padding. The choice is CONDITIONAL on the
NULL-word's cost: IF NULL processing is made FAST (padding positions
skip the heavy forward/reverse, cost ~0) → keep (ii), one graph, cheap.
IF NULL stays full-cost (today) → adopt (iii), pow2-bucketed loop
lengths, so NULL positions are FEW rather than MAX. Both still require
the CORRECT NULL-word (compose-identity forward, emit/recognize
reverse); they differ only in making NULL cheap-per-position (A) vs
few-in-number (B). On the current dev data (word-counts 1..7): (iii) =
buckets {1,2,4,8} = 4 graphs vs (ii)'s 1-graph-pad-to-8.

## Design

1. **Route the serial decode through Method 1** (smallest correct step):
   in serial mode, `recon_bench`'s decode (and the round-trip metrics)
   consume `_reverse_from_S`'s replayed multi-slot surface event instead
   of the single-slot tensor reverse. The scaffold/blind distinction from
   the parallel path maps onto serial as stored-tree/free-derivation.
   Bar: `test_mm20m_grammar_derivation_roundtrip` — Method-1 replay
   round-trips the input exactly (this SHOULD be reachable: exact by
   construction once routed; any residual is a real defect of the replay
   machinery and gets fixed, not accepted).
2. **Method 2 as the trained bar** (the serial analogue of Gate 2b's
   blind recovery): `reconstruct_from_idea=True` — derivation inferred
   from the reduced STM + attention state, scored against the input
   (teacher-forced variants against Method 1's tree available as
   diagnostics). Expected to be TRAINING-dependent; measured, pinned by
   the standing convergence formula if reachable, reported precisely if
   not (the per-op reverse stubs are the known ceiling — surfacing which
   rules' inverses matter is itself a deliverable).
3. **Per-op reverse stubs $\to$ ERRORS (Alec, 2026-07-04):** the identity
   stubs for non-invertible rules in `SyntacticLayer.reverse`
   (`Language.py:7546`) are converted to loud errors. Rationale: the
   resulting error inventory is the decision procedure — each firing
   names a rule that either GETS a real `reverse()` written or is
   REMOVED from the grammar. No silent pass-through survives (the fifth
   application of the fail-loud pattern). Method-1 replay runs first
   with the errors in place; the rules it actually exercises define the
   initial inverse-coverage worklist.
4. **Relation-table coupling (deferred, recorded):** assertive
   `isEqual`/`isPart` firings during the parse feed the relation table;
   the attention-to-relation-promotion plan (2026-07-04) governs
   promotion. Method 2's decompositions should eventually prefer
   promoted wholes. Out of scope here; interface noted.

## Open questions — resolutions (Alec, 2026-07-04)

1. **RESOLVED — free-run only.** No teacher-forced per-production loss.
   Density comes from the SCAFFOLD-MASKING curriculum (see the encoding
   design's header note): masking only a fraction of the scaffold gives
   dense per-masked-tile feedback within a free-run derivation.
2. **DEFERRED (Alec: "ask again if necessary")** — whether Method-1
   replay takes grammar's round-trip slot immediately with Method 2 as
   the staged trained bar. RE-ASK when writing this design's execution
   plan (it becomes load-bearing at the bar-definition task).
3. **RESOLVED by a coupling principle, not a schedule:** "forward() and
   reverse() do the concept encoding and decoding; they are not
   separable, since the idea produced varies somewhat across time for
   deterministic inputs." Consequences: (a) SEQUENCE the encoding pass
   FIRST — the `.when` the ideas carry is defined there, and the
   derivation decodes ideas that include that temporal context; (b)
   Method-2 scoring compares SURFACES (derived sentence vs input), never
   idea tensors — cross-run idea equality is NOT expected for
   deterministic inputs (the clock advances); (c) determinism bars stay
   within-run (consecutive no_grad forwards), never cross-run.
4. ~~The identity-stub per-op reverses~~ — RESOLVED (Alec 2026-07-04):
   converted to errors; the error inventory drives write-the-reverse vs
   remove-the-rule. See design point 3.
