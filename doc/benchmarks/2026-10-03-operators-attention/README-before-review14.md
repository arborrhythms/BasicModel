# Decoder, operators and 6.8 — §13 measured candidate, held for Claude

2026-10-04. Working HEAD remains `802abb1acc95e1bddc8cb237b13230a336681c49`.
One working tree; nothing committed. The settled §13 design is implemented and measured once on frozen source. §12 remains measured, without
retry: class **0/10**, reconstruction **7/10**, joint **0/10**, MM_xor **10/10**,
sum **10/10**. Its [complete receipt](README-before-review13.md) is preserved.
The accepted 6.9 baseline remains class MSE **.1147481948**, reconstruction
**0/4**, zero ownership conflicts; the prior class **9/10** and reconstruction
**5/10** remain historical comparisons.

## Settled implementation

Native perceptual content is the live basis of order-zero conceptual codes.
The serial dictionary has no free word row or extra learned location table:
its cache is a buffer overwritten from native PS prototypes and signed 11b
feature evidence. The native part-group max fold and PS cube read are reused;
reconstruction owns these existing prototypes and the evidence weights.

The conceptual code embeds the native PS event width, with the type code's
WHERE/WHEN coordinates zero. Its remaining complement reads the recency-weighted
mean of existing occurrence roots, detached. Form is not attenuated by how many
occurrences a word has. Native row references and leaf postings supply the two-way
membership; new context rows are not minted. Both paths use the same context
snapshot before either trains. Existing rows conduct priming from words to their
occurrences and back to constituents.

Read-back uses scale-free absolute cosine on the native perceptual content,
with the existing priming weight. The activation's sign is separate from surface
identity. The antipode objective is retired: no evaluation, graph, comparison
credit or training writer. Its published reporting key remains an untrained,
detached zero, and its old helper remains a diagnostic API only.

The probabilistic-sum disjunction and other composition operators are unchanged.
The future form-band fold and connective split remain with the operators update.
Distinctness is measured, not guaranteed: codes have a shared basis, and product
composition can erase differences outside its operand's support. The audit records
start/end code and root geometry, each word's exact nonzero support and minimum
absolute coordinate, decoder margins and gradients, and read-back winners decided
by code, priming, or exact tie.

**Explicit limitation, per Alec's reply:** property/situation bootstrap learning
is deferred to the operators update. A context complement initialized at zero
cannot seed itself from occurrence means. No new co-activation objective is
implemented. The toy gate configurations have an empty complement; the wider
mechanism fixture checks isolated, detached context reads. A future split of form
and connective operators needs a nonempty complement in those gate configurations.

**Declared capacity and dimension changes:** XOR_grammar ConceptualSpace inventory
6 → 262, MM_grammar 8 → 264, reserving 256 native percept-concept addresses.
XOR_grammar InputSpace, PartSpace, ConceptualSpace and WholeSpace nDim 10 → 14,
as specified by the settled plan. Native PS content is therefore six coordinates,
plus eight event coordinates; the 14-D toy conceptual code has no remaining
context complement. Production's native PS content remains 128 coordinates.

The [settled plan record](review13-plan-settled-amendment.json) and
[diff](review13-plan-settled-amendment.diff) preserve Claude/Alec's amendments;
Codex has not edited the plan. Earlier implementations and every failing probe
remain in the [design-hold receipt](README-review13-design-hold.md) and append-only
archives. No gate training occurred during those design changes.

## One frozen measurement

The control was read first: **sum 10/10**, under its unchanged
criterion (checkerboard contrast at most 1e-4 and no class-bar pass).
Only then were XOR and MM_xor run. Each XOR run trained once, and both unchanged
bars consumed that same model. There were **30 trainings, zero retries**, with
no post-result source repair. [Provenance verification](review13-verification.json)
confirms the frozen production, test and data hashes and the measurement harness.

| Gate | §12 comparison | §13 |
|---|---:|---:|
| XOR class | 0/10 | 0/10 |
| XOR reconstruction | 7/10 | 0/10 |
| XOR joint | 0/10 | 0/10 |
| MM_xor | 10/10 | 10/10 |
| Sum control | 10/10 | 10/10 |

Bands use §20.5: at 0 is MSE < .05; at ¼ is within .02 of .25;
remaining values below .25 are between, and remaining values above are above ¼.
Counts are **0 at 0, 9 at ¼,
1 between, 0 above ¼**.

Below the comparison: **reconstruction_pass 0/10 versus 7/10**. These measured failures stand; nothing was retried.

## Per-run XOR results

The final-operator column follows each row's saved input order; full rule names,
IDs and complete greedy derivations are in [summary.json](review13-measurements/summary.json).
C = conjunction, D = disjunction, N = not; a sequence lists every final compose
operation. Read-back counts refer to emitted words: code / priming / unresolved
tie. An absent word contributes no read-back decision and still fails the
unchanged reconstruction bar.

| Run | Class MSE | Band | Class | Reconstructed | Joint | Final compose operators | Read-back code / priming / tie |
|---|---:|---|---|---:|---|---|---|
| 1 | 0.249993681 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 2 / 6 / 0 |
| 2 | 0.247814599 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 3 / 5 / 0 |
| 3 | 0.249071177 | at 1/4 | fail | 2/4 | fail | D / D / D / D | 3 / 5 / 0 |
| 4 | 0.249609258 | at 1/4 | fail | 1/4 | fail | C / C / C / C | 1 / 4 / 0 |
| 5 | 0.228082223 | between | fail | 2/4 | fail | C / C / C / C | 2 / 6 / 0 |
| 6 | 0.249827128 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 2 / 6 / 0 |
| 7 | 0.249829796 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 5 / 3 / 0 |
| 8 | 0.241484388 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 2 / 6 / 0 |
| 9 | 0.249224587 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 3 / 5 / 0 |
| 10 | 0.249988865 | at 1/4 | fail | 2/4 | fail | C / C / C / C | 2 / 6 / 0 |

Saved input order: `hello world`, `hello there`, `loving world`, `loving there`.
Final evaluation totals: **25 code, 52 priming, 0 unresolved ties**, across 77 emitted-word decisions. Priming means it changed the code-only winner or resolved a code tie; it is not a separate decode.

MM_xor best-error values (runs 1–10; unchanged < .20 convergence bar): 0.147866, 0.181406, 0.19028, 0.190208, 0.188796, 0.175365, 0.186514, 0.186708, 0.192103, 0.171143.

Sum checkerboard contrasts (runs 1–10): 0, 0, -2.98023e-08, -5.96046e-08, 0, 0, 0, 0, 0, 5.96046e-08.

## Tenth XOR training: ownership, walks and decoder margins

**0 ownership conflicts**, 23
active and 62 inactive parameter entries. Native PS
prototypes and feature evidence are reconstruction-owned; durable occurrence
roots have no gradient. Inactivity remains visible in the [ownership audit](review13-measurements/xor-10/ownership/ownership.json).
There are 1200 recorded optimizer steps, of which
800 contain decoder observations, and
1600 first-step greedy/explore logit records.

The raw STOP-minus-undo margins, gradients reaching those logits, and actual
fixed-parent margin changes are retained per step in
[events.jsonl](review13-measurements/xor-10/ownership/events.jsonl),
with [numerical summaries](review13-measurements/audit-summary.json) and
[a margin plot](review13-measurements/decoder-margin.png).
Eligibility must be read alongside the raw margin: a masked STOP cannot win
and receives no choice gradient. A rejected path can have no gradient. First-step
eligibility counts: {'compound': 6400}.

| Undo action | Observed gradients | Nonzero STOP−undo gradient | Nonzero fixed-parent margin change | First-epoch mean margin | Last-epoch mean margin |
|---|---:|---:|---:|---:|---:|
| conjunction (1) | 6400 | 0 | 0 | 2.00149 | 2.00378 |
| disjunction (2) | 6400 | 0 | 0 | 1.99877 | 1.99676 |

| Walk | Explorable / walks | Explore kept | Strict-rule violations | Kept-path stability |
|---|---:|---:|---:|---:|
| attention.input | 1600/1600 | 0 | 0 | 1589/1596 (0.995614) |
| generate.decoder | 0/3200 | 0 | 0 | 0/3196 (0) |
| compose | 1600/1600 | 0 | 0 | 1596/1596 (1) |

All departures are sampled among eligible alternatives at the chosen round;
both paths are costed before either trains; only strictly lower owner cost is
kept, with ties to greedy. The final audit records **6392
activated-candidate opportunities** and **0
activated candidates outranking own words**, across its ranking observations.
These are repeated observations, not counts of unique words. Real occurrence
conduction is also checked by the isolated, cross-batch-safe mechanism fixture.

Training read-back classifications: {'code': 2854, 'priming': 3546}.
Evaluation read-back classifications: {'code': 2, 'priming': 6}.

## Start/end geometry and support

Full code and root pairwise cosine matrices and centered singular values are
saved in [start geometry](review13-measurements/xor-10/ownership/geometry-start.json)
and [end geometry](review13-measurements/xor-10/ownership/geometry-end.json),
with tensor code matrices beside them. Root values below are the same four
sentence roots observed in this training, not another model or run.

| Phase | Root pairwise cosine range, excluding self | Root centered singular values |
|---|---|---|
| start | 0.745419 to 0.90452 | 0.00298547, 0.00223446, 0.00100208, 1.76822e-10 |
| end | 0.835667 to 1 | 0.0304115, 4.5615e-05, 1.95315e-08, 2.67875e-09 |

Support uses exact nonzeros in each word's six native perceptual content coordinates; the minimum includes zeros. Reserved PS event positions are reported separately, not counted as perceptual content.

| Phase | Word | Nonzero fraction | Minimum absolute coordinate | Minimum nonzero absolute coordinate |
|---|---|---:|---:|---:|
| start | hello | 6/6 (1) | 0.00891257 | 0.00891257 |
| start | loving | 6/6 (1) | 0.0121095 | 0.0121095 |
| start | there | 5/6 (0.833333) | 0 | 0.0257498 |
| start | world | 5/6 (0.833333) | 0 | 0.00967864 |
| end | hello | 6/6 (1) | 0.0206372 | 0.0206372 |
| end | loving | 6/6 (1) | 0.0542828 | 0.0542828 |
| end | there | 6/6 (1) | 0.0459213 | 0.0459213 |
| end | world | 6/6 (1) | 0.0458169 | 0.0458169 |

## Reading the saved failure

Because reconstruction fell below its comparison, the following is a
[read-only analysis of the tenth run's saved tensors](review13-measurements/saved-geometry-reading.json).
It constructs no model and adds no inference, training or changed gate.

Removing the free word row did **not** prevent directional collapse in this
audited run. `world`, `there` and `loving` finish nearly proportional on their
perceptual coordinates: their pairwise cosines, recomputed in float64 from
the saved float32 values, are 0.999999999970, 0.999999999925 and
0.999999999990. Their nonzero supports are all 6/6 at the end. The third
centered root singular value falls from 0.00100208 to 1.95315e-8. The
coordinates retain small differences; this is near-collapse, not a claim
of exact vector equality or a proof about the other nine runs.

Every one of the 6,400 observed live first decoder steps has exactly one
legal action; STOP is ineligible in all of them. Both binary margin-gradient
differences and all fixed-parent margin changes are exactly zero. The
generate policy's weight and bias have zero observed displacement across
all 1,200 steps. This audit records an absence of policy motion; it does not
support an explanation that more learning rate or budget would solve the
first-step choice. No rate, budget, eligibility mask or guard was changed.

The kept-path stability metric compares successive recorded decoder walks,
including both outer compose trials. Its zero value is not a separate
measurement of final-greedy-only persistence. The per-input greedy and kept
path distributions and named compose stability are saved alongside it.
The candidate is held for review; the measured reconstruction loss is not
repaired or retried in this round.

## Verification, failing probes and provenance

Final mechanism verification: **97 passed, 1 existing skip** in
[the guarded run](probes/review13-final-native-mechanisms/run.log), plus
[the settled audit probe](probes/review13-observer-settled/run.log).
The latter uses one ordinary training batch and evaluation to verify wiring;
it is not an extra gate training. No new seed call was introduced.

Each probe archives its exact pre-run source, log and guarded process report;
failures were saved before repair. The earlier design's probes remain in
[the design-hold receipt](README-review13-design-hold.md).

| Native-subspace probe | Saved result |
|---|---|
| [subspaces-before](probes/review13-subspaces-before/run.log) | 6 failed, 1 warning in 2.18s |
| [subspaces-first](probes/review13-subspaces-first/run.log) | 5 failed, 1 passed, 1 warning in 3.30s |
| [subspaces-second](probes/review13-subspaces-second/run.log) | 7 failed, 16 passed, 2 warnings in 9.03s |
| [subspaces-ported](probes/review13-subspaces-ported/run.log) | 1 failed, 39 passed, 2 warnings in 23.35s |
| [observer-native](probes/review13-observer-native/run.log) | 1 passed, 4 warnings in 0.93s |
| [settled-mechanisms](probes/review13-settled-mechanisms/run.log) | 3 failed, 128 passed, 1 skipped, 6 warnings in 117.53s (0:01:57) |
| [boundary-ports](probes/review13-boundary-ports/run.log) | 6 passed, 4 warnings in 9.71s |
| [observer-settled](probes/review13-observer-settled/run.log) | 1 passed, 4 warnings in 0.96s |
| [final-native-mechanisms](probes/review13-final-native-mechanisms/run.log) | 97 passed, 1 skipped, 6 warnings in 104.81s (0:01:44) |

All 30 measurement workers exited normally within the unchanged 8-GiB and
1,800-second guards. Assertion failures are the saved gate outcomes. Peak
worker footprint was **755,074,656 bytes**; campaign wall time was
**959.54 seconds**. Three workers at most, one CPU
thread each, unchanged 24-GiB aggregate guard.

The [contract check](review13-contracts-frozen.json) confirms unchanged published
gates, guards, assertions and seed calls, and only the declared XML capacity/width
changes in this round. Fixture ports explicitly supply an absolute sentence
reading for the answer boundary and decoder ownership; production support
eligibility is unchanged. The field-dictionary parameter fixture is separate
from the derived serial cache.

The [frozen source archive](review13-source/source.zip),
[full old/new test ports](review13-source/test-ports.json),
[seed-call audit](review13-source/seed-port-audit.json),
[measurement helpers](review13-source/reporting-source.zip),
[final delivery archive](review13-delivery/source.zip) and
[measurement verification](review13-verification.json) preserve the candidate.
§11 and §12 remain measured as saved, without retry. No attribution training,
full sweep, native run or fresh BasicModel NanoChat scoring was added. Its
[existing acceptance probe](nanochat_acceptance_probe.py) and
[three-item frozen-evaluator mechanism check](README-before-review11.md) remain;
the trained gate waits for item 4's checkpoint.

**Held for Claude's review. Nothing committed.** Bootstrap learning and the
form-fold/connective split remain explicit work for the operators update.
