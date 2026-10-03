# Item 6.9 — plan §14

One working tree; no commit. The source snapshot before this round is `before.json` / `before.zip`. Prior receipts remain historical. No seeds, guards or thresholds change except the explicitly decided sum-control criterion.

## 1. Fixed journal bound

Before repair, `query-fullgraph-before.json` retains the failing one-graph assertion from the closing sweep. `journal-recompiles-before.json` shows the separate CSLang capture when the sentence-local width changed.

The first literal 256-word-capacity allocation used 784 frames and exceeded the existing native slow ceiling: 24.8842 GiB sampled peak, stopped by the 24 GiB guard after 701.73 s. `native-peak/` preserves that unsuccessful prototype; no training result is substituted. A requested stop was never sent because the guard had already killed the process.

The final implementation selects a sentence bound from the loop's existing static word buckets (8, 16, … up to the configured ceiling; smaller configured ceilings remain unchanged). The native short sentences use 40 frames, compared with 25 under arbitrary-length sizing and 4,864 before the item-7 repair. Values remain live, with the same global-to-local address mapping. This changes storage capacity, not the admissible sentence length or a model parameter.

`journal-buckets-after/`: eight cases passed, including the unchanged fullgraph forward/backward one-capture assertion. That case also exposed an independent retained `intra` Error-registry entry; `query-guard-diagnostic/`, `query-next-guard/` and the successful `query_start_probe.py` preserve its diagnosis. The owner now resets it at the eager forward boundary.

`journal-buckets-captures-after.log` / `journal-recompiles.json`: one first capture of each CSSub/CSSym/CSLang stage (29.49 s), then no captures at the other length or either revisit (.0799–.1111 s). A second independent guard was the previously missing `_compose_exploit_actions` attribute becoming None; the sentence context initializes its action constraints before compiling. No compiler limit changed.

`native-bucket-peak/`: production batch 28 completed, with source hashes matched, in 695.43 s including the cold eager-backend graph capture (678.14 s training). Sampled peak was 23,069,531,296 bytes (21.4852 GiB), within the existing 24 GiB native slow ceiling but above the unchanged 8 GiB sweep guard. This is one first training batch, with no evaluation or seed; it is not interchangeable with the earlier full-training stage-1 peak.

## 2. Reading-mode migration

The default and explicit old mode selectors now use meronomy. `mode-dispositions.md` lists every configuration; `mode-configuration-delta.json` verifies that the parsed changes are limited to authorized reading selectors. The two mode-only configurations are retained as receipt copies and removed from data. The old mode dispatch and callers are deleted; the live RadixLayer store remains the underlying meronomy dictionary. No checkpoint has BPE extras (`checkpoint-mode-audit.json`).

`configuration-first-batches.md` preserves the initial 51 attempts (47 migrated embedding configurations plus four numeric configurations affected by the inherited default): 29 completed, 21 raised, and one reached the unchanged 8 GiB guard. Those attempts exposed the unit-staging and long-derivation repairs below. The final verification is recorded separately in `mode-final-batches/` and `final-configuration-first-batches.md`; initial timings are not substituted for final ones.

Mode-only tests and the removed helpers are enumerated in `mode-test-dispositions.json` and `retired-mode-methods.json`. Mode-independent assertions are ported to the retained code. The complete before source and all changed test bodies are preserved in `test-ports-old-new.zip`, with 69 changed files and 172 complete definition pairs indexed by `test-port-index.md`.

## 3. Remaining sweep failures

Diagnosis before repair: `compose_intersection_probe.py` deterministically replays the last legal binary choice over the original fixture without seeding. Three forced `not` operations negate its all-positive older operand; the younger operand is nonnegative. Signed intersection has no occupied pole in common, so its result is exactly zero. The deadline still reduces depth two to one. The unforced probe selected union and returned a nonzero vector. The failure is a fixture assumption about legal algebra, not a failed stack commit. The mixed-sign fixture port now passes; all original final-state assertions remain.

`data/BasicModel.ckpt` contains twelve `reverse_chooser` parameter keys, so the one-way loader migration is still required. `checkpoint-audit.json` records the inventory of both data checkpoints and both test fixture checkpoints. The other three contain no reverse-student state.

## 4. Sum control

The receipt-local control keeps the same sum grammar and now grades absolute checkerboard contrast ≤1e-4 plus failure to meet the existing class bar. All ten unseeded runs pass: the maximum absolute contrast is 2.9802322387695312e-8 and none meets the class bar. Distance from one half remains diagnostic only. The full control patch and every answer/read-back/contrast are saved under `closing/`.

## 5. Closing measurements

The ten class-gate runs finish at 8/10, the ten reconstruction-gate runs at 3/10, and the ten sum controls at 10/10. The named table finishes at 30/34, including the single passing exact round trip. Its four failures are the predeclared first class and reconstruction runs and the two MM convergence tests. All ten MM_grammar runs complete their 900 updates: eight finish near zero and two at .125, with median ending training MSE 4.356559557550099e-11 (accepted item 7: .1066). The direct-forward MM measurement remains distinct from sentence-trial joint-cost training.

The measurement campaign completes in 91.769 minutes with matching source hashes and no resource stops. The source-matched full sweep has attempted all 5,060 collected cases: 5,057 completed, with 4,672 passed, 314 skipped and 71 reported failures; three first attempts stopped without a result. Wall time is 214.66 minutes against 5,219 cases / 122 minutes previously: 159 fewer cases and 92.66 more minutes. The schedules differ (two workers here, up to ten in the reference), and this wall time includes the incident below. Complete observations, all 25 profiled cases' final times and process records are retained in the [closing receipt](closing/README.md). The measurement campaign has no retries; the sweep continuation incident is separate.

All six original repair regressions pass in this final-source sweep: the two compiled sentence/query cases, unaligned relation selection without word identities, non-aliasing definition identities, and both stored-truth ingestion cases. `closing/original-six-repairs.json` retains their outcomes and times. The subsequent sweep failures are separately preserved and classified in `closing/failure-triage.json`; the candidate remains frozen for review.

**Receipt continuation incident:** after the reasoning worker crossed 8 GiB, the continuation omitted the aborted peer's durable reports and scheduled already-attempted cases again. During the stop and repair, eight completed cases were repeated and two interrupted cases were restarted: ten case IDs and seventeen extra attempts in total. The corrected receipt retains every case's first attempt, including interrupted failures, and records all duplicates in `closing/full-sweep/duplicate-attempts.json`. `closing/peer-accounting-before/` preserves the failing probe and original scripts; `closing/peer-accounting-verification.json` verifies the correction. The receipt-only continuation completed the 1,400 never-started cases with no further duplicates, unchanged source and unchanged guards. Wall time includes the interruption, duplicate work and bookkeeping repair. The earlier claim that the sweep had no retries was incorrect. Memory/time metadata are unavailable for the two force-stopped duplicate workers; their case reports are retained.

Alec clarified during this receipt that demonstrating learning is 6.9's primary goal and lack of consistency is acceptable. The 8/10 class result demonstrates learning; ten-of-ten consistency is not treated as an additional acceptance requirement. All test assertions and measured bars remain unchanged. Reconstruction's 3/10 result remains a review concern. Alec suspects gradient conflict: the earlier stage-1 measurements established opposition before the new projection, but these gate results alone do not establish its cause under the current projected-gradient/optimizer path. No new training or source change was made in response to that hypothesis.

Alec subsequently deferred new gradient-design work to [FutureWork](../../../FutureWork.md#separate-gradient-ownership), with structural separation of gradient ownership as the leading direction. The earlier joint optimizer proposal is retained there as a secondary reference. This adds no implementation to 6.9; the candidate and closing measurements remain unchanged.

## Weekly triage

Completed. See `weekly-triage.md`; no weekly failure was repaired for this campaign.

## Migration repair details

The initial configuration measurements exposed three independent recursive traversals on a long LM_5M derivation. `deep_derivation_probe.py`, `deep_meaning_probe.py` and `deep_clause_probe.py` preserve the failures and exact-order/gradient checks. The production traversals now use explicit stacks; no recursion limit or sentence capacity changed. The original failed first-batch attempts remain under `mode-first-batches/`, `mode-repair-batches/` and `mode-repair-batches-2/`.

The default-mode migration also exposed a unit mismatch in the mixing stem: six byte chunks were being labeled with two word surfaces. `separator_bank_probe.py` and `mixing_unit_contract_probe.py` preserve it. Serial mixing now uses the native byte-constituent word-unit stem, includes gaps omitted by the analytic property tiling as perceived separator units, and preserves the existing distinction between OBJECT grammar references and unexposed WORD identities. The byte-address registry is constructed after resolving the reading mode; its mixing input range derives from the configured word slots and residual-byte capacity. No XML capacity or memory ceiling was raised. `mixing-unit-contract-after-init-order.log` verifies exact word/space bytes, two grammar leaves and no published word identities.

`migration-contracts-after/` retains the focused verification. All four separator variants, all six supplied-answer ownership cases, the generation-lesson gradient and both packed-trial cases pass. The ported historical cache quality case reaches its unchanged .8 overlap bar; its strict xfail therefore reports XPASS. A subsequent unseeded attempt returned .5, so the original expected-failure status is retained; neither result is hidden. Complete before/after bodies are retained by the port archive.

The first weekly run has finished on its frozen review-13 source. `weekly-triage.md` and `weekly-triage.json` classify every failure by immediate cause, distinguishing static evidence of historical API drift, recent changes, resource stops and unresolved onset. All 328 cases were attempted in 7.464 hours. Unique outcomes are 198 passed, 98 failed, 15 skipped, five xfailed and 12 stopped without a case result. Fourteen process-stop events include two cases that already reported an outcome; those events are not extra cases. No weekly failure was fixed or rerun for this triage.

`final-migration-ports/`: 24 passed and one quality failure (the unseeded .5 top-k result described above). The unchanged fullgraph cross-length query test captures one graph and passes forward/backward. Mixing reconstruction, UTF-8 witnesses and all journal tests pass. The real aligned mini model and provenance round-trip pass together in 14.0 s / 4.95 GiB after using the existing eager loop fixture; the mini alone previously passed in 563.1 s / 5.32 GiB with capture. No signal was sent: it had finished before the proposed diagnostic stop. `aligned-mini-timing.json` keeps the process records.

`final-configuration-delta.json` verifies the final parsed XML changes: only analysis/synthesis mode selectors, the two retired mode-only files, and removal of retired detachedReverse selectors. All other values, including capacities, learning rates and epochs, are unchanged.

The final first-batch LM_5M attempt completes (3.430 s numerical training, batch 128). The earlier 435-second recursion failures arose while reading individual byte chunks as if they were grammar units; the explicit-stack fixes still preserve deep valid derivations, and the corrected unit stem avoids that accidental grammar depth.

## Final preparation and verification

`final-configuration-first-batches.md` records all 51 final timing attempts: 32 completed, 18 exceptions, one memory stop, 8.811 minutes wall time with two workers. The exceptions and their traces are explicit. MM_add_verb sampled 8.507 GiB before the unchanged 8 GiB guard stopped it. MM_sequence_predict completed its batch of four in 393.914 s numerical training (6.704 GiB worker peak). XOR_grammar completed its configured first batch in .524 s numerical training, with reconstruction through the understanding and no unavailable surface sentences. These are first-batch results, not gate results.

After those timings, `retired-helper-import-before.log` exposed a retained test importing only a path constant from the retired `space_equiv` helper. Its port uses pathlib and explicitly adds bin to the subprocess path; assertions are unchanged. The saved `final-helper-port/` failure then exposed numeric carriers incorrectly constructing a text dictionary. The store is now created only for embedding models; `numeric-default-after/` passes the unchanged numeric cursor test. This numeric-only repair does not alter the measured embedding branch. The four numeric configuration attempts had already failed in MNIST data loading, before construction.

`final-mechanical-ports.json` lists the remaining whitespace-only cleanup (verified AST-identical), and the final old/new archive now covers 69 test files and 172 complete definition pairs. `eleven-failure-map.md` maps every preceding sweep failure to its port, source repair or expressly retired path. `final-document-links/` passes all 248 documentation cases checked at that point (one case per Markdown file). The closing measurements and full sweep use the final source snapshot in `closing/measurements/manifest.json`.

## Final sweep triage and review stop

`closing/failure-triage.json` lists every reported failure, its saved evidence and the next diagnostic or port. The 71 reported failures comprise:

| Category | Cases |
|---|---:|
| Retained fixtures expect the retired lexicon interface | 41 |
| Live span fixtures still select retired word analysis | 6 |
| Other retired-mode dispatch expectations | 4 |
| Input occurrence exceeds the configured address capacity | 7 |
| Incomplete configuration port, cursor mismatch, separator-free promotion fixtures, retired warning, retired-name guard, and eager-loop fixture inside capture | 8 |
| Category assignment, inverse reference-side invariant, and repeated-understanding stability: causes unresolved | 3 |
| Inductor reconstruction while-loop lowering (`int` lacks `meta`) | 1 |
| Strict XPASS: the unchanged .8 reconstruction-overlap assertion passes | 1 |

The three stopped first attempts are the reasoning training case at 8.001 GiB, its interrupted relation-capture peer, and the expectation-provisioning case interrupted during the receipt repair. They remain failures; later duplicate outcomes never replace them. No candidate repair was made after the closing source freeze. A stale fixture interface does not retire the behavior it was meant to test: its ownership, storage, reconstruction and target-isolation assertions still need a proper mapping.

The longest completed call took 908.51 seconds in LTM provisioning; the real Inductor aligned-loop case passed in 743.95 seconds. All 45 LTM consolidation cases passed but consumed 76.05 minutes of call time. The former 30-minute interleave timeout case passed in 1.49 seconds. `closing/performance-final.json` and the closing receipt keep the complete before/profile/final comparison, including prior failure outcomes and unavailable calls.

The first weekly run is complete and still red; its failures were triaged, not repaired in this round. The full sweep reports that warning. The production batch-28 benchmark remains above the ordinary 8 GiB target at 21.4852 GiB, within the existing 24 GiB slow ceiling.

**Stopped for review.** Nothing is committed; 6.9 is not marked accepted or closed, and the venv rebuild remains held. The proposed gradient redesign stays in FutureWork for now. `closing/source-final.zip` preserves the measured candidate; `before.zip` is the pre-§14 source, not this final candidate.
