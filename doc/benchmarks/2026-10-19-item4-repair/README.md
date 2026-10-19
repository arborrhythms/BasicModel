# Item 4 repair candidate — 2026-10-19

BasicModel baseline `9207f5e03`; WikiOracle baseline `929e4a8`. Both checkouts are on `codex/item4-repair`. This round preserves the prior working-tree spec, todo, NP and documentation edits. No commit on main.

**Item 4 is not complete.** Claude reviewed candidate `f7e74af04` and did not approve the landing sweep; the full report is [claude-review.md](claude-review.md). The follow-up repairs below address the code findings, but B3 requires Alec's decision on the unscaled selection prime and `φ=.5`. All four attenuation gains are zero; those two other effects are not. The missing copular-identity artifact prevents one requested certificate from executing. The carried initialization-distribution failure remains. Global derivation optimality, per-level learning/collapse, learned repeated-cat recognition and all-level XOR are not established. The original failed learning run is unchanged; Item 0 and Item 5.5 remain open. No full sweep or push has run in this round.

Claude's [re-review of `7285b38`](claude-followup-review.md) confirms those
repairs apart from B3, and identifies N1/N2 below. They are repaired under
the existing raw-observation and R+A decisions; B3 remains undecided.
The [short check of `282fcf918`](claude-n1-n2-review.md) confirms N1/N2 in
source and their native regressions, with no new code defect. No further
source review of those two repairs is requested. Sweep approval is still
withheld pending B3 and the disclosed gates.

## §9.1 — Does zero gain preserve the previous landing?

| Consumer check | Result | Limits |
|---|---|---|
| Native Item 5 vs repaired tree, all four gains zero | Exact equality for IDs, keys, word values, terminal STM/depth and stored role views; maximum difference 0 | Two presentations: single words and two-word fields. Expectation disabled in both to match zero-expectation consumer views. |
| Parameter match | 141 named parameters copied from Item 5 | New head parameters and expanded candidate-attention input are listed in `zero-repaired.json`; the new context columns feed a zero-initialized output at this point. |
| Standing XOR with all four gains on | Not established | The existing standing learning gates are not replaced by carrier probes. |

Diagnostics: `zero_baseline_probe.py`, `zero-baseline.json`, `zero-repaired.json`. This is a matched consumer check, not parity between the retired predictor and the new head.
It does not establish parity with expectation enabled in a multi-sentence
training departure. Claude B3 requires that measurement after the default
decision; the existing disabled-expectation result is not promoted to it.

## §9.1 / J — Do the requested math-chain certificates complete?

| Certificate | Completed result | Disposition |
|---|---|---|
| `forced_c_e_closings_exercise_empty_search_mint_and_question_storage` | Passed, 13.3 s process | `math-chain-1/worker-000.json`; forced fixture consumes all four `select_logits` results. |
| `ordinary_initial_binding_distribution_includes_every_retained_candidate` | Failed, 225.6 s process | Relative-to-uniform range 0.8779656–1.1124873, required 0.9–1.1. Same carried class documented in todo Item 0; no threshold or initialization changed. `math-chain-2/worker-000.json`. |
| `forced_ordinary_answer_fills_committed_question_without_its_own_episode` | Skipped; prerequisite absent | `copular_identity` is null in the phase registry. No replacement training or unrelated checkpoint. Existing artifact and producing receipt requested from Alec. |

A skip is not a completed certificate. The four failures in the original `item4-final-native` row are now explicit in the plan: 120 passed, 1 skipped, 4 failed and 9 unfinished out of 134 selected.

## §9.2 — What did the fixed repaired-tree run measure?

The predeclared `native-repaired-protocol.md` fixes two seeds, four arms, six training batches each, and a single held-out before/after pair. All four gains are measured at 0.5; shipped defaults remain zero. No continuation or added updates.
This measurement precedes Claude's follow-up fixes and belongs to the first
repair candidate, `f7e74af04`. It has not been rerun or regraded to rescue a
learning claim on the follow-up tree.

| Seed / order | P normalized residual before → after | S activation MSE before → after | D final MSE | Raw final role norms |
|---|---:|---:|---:|---|
| 19 graded | 1.000000 → 0.291095 | 0.213922 → 0.190287 | 0.183776 | 7.392246, 9.089376 |
| 19 shuffled | 1.000000 → 0.317584 | 0.213922 → 0.196278 | 0.185321 | 7.398427, 6.249060 |
| 23 graded | 1.000000 → 0.302525 | 0.262758 → 0.254263 | 0.248350 | 1.451101, 4.222039 |
| 23 shuffled | 1.000000 → 0.319737 | 0.262758 → 0.254406 | 0.253487 | 4.307245, 1.458676 |

The first D read is cold, so there is no matched D before value. R is not independently measured here. Nonzero raw role norms do not establish a flat collapse diagnostic. The full per-level learning gate remains false. The content-coordinate assertion applies to the two nonnegative evidence lanes; signed identity keys retain their original coordinates.

## §9.2 / B — Which departure owners received exposure?

Seed 23’s recorded-source cause is in [seed23-cause.md](seed23-cause.md). All three recorded compose departures tied; their corrections were 51, 51 and 60. Unequal paired costs belonged to other owners. The original report combined mean costs across owners with compose-only parameter change.

| Seed / order | Owner | Departures | Non-tied exposures | Explore wins |
|---|---|---:|---:|---:|
| 19 graded | narrowing | 3 | 0 | 0 |
| 19 graded | compose | 7 | 4 | 4 |
| 19 graded | candidate | 2 | 1 | 1 |
| 19 shuffled | narrowing | 7 | 0 | 0 |
| 19 shuffled | compose | 4 | 2 | 2 |
| 19 shuffled | candidate | 1 | 1 | 0 |
| 23 graded | narrowing | 5 | 1 | 1 |
| 23 graded | compose | 4 | 1 | 0 |
| 23 graded | candidate | 3 | 3 | 2 |
| 23 shuffled | narrowing | 3 | 0 | 0 |
| 23 shuffled | compose | 6 | 1 | 0 |
| 23 shuffled | candidate | 3 | 3 | 1 |

Compose and narrowing share reducer parameters; candidate attention has its own parameters. Per-batch scales, masks, policy losses, optimizer gradients and parameter changes are in `native-repaired/measurement.json`. No non-tied exposure means unexposed, not verified learning. Zero narrowing wins occur in three arms while S residual stays above 0.19; this flags insufficient exposure, not a successful stall gate. R/D thought departures are not exercised by this workload.

## §9.3 — Can the fully expected cat still be answered and stored?

| Row gain | Document gain | Content / raw floor | Native answer | Raw stored observation |
|---:|---:|---:|---|---|
| 0 | 0 | 1 | Exact | Exact |
| 0.5 | 0 | 0.5 | Exact | Exact |
| 1 | 0 | 0 | Exact | Exact |
| 0 | 0.5 | 0.5 | Exact | Exact |
| 0 | 1 | 0 | Exact | Exact |
| 0.5 | 0.5 | 0.25 | Exact | Exact |

`test_item4_native.py` uses a registered identity, a controlled confident shared head, a causal pre-arrival row read, the production sentence commit, native answer materialization and `TernaryTruthStore`. Seven carrier cases plus prime-lifetime coverage pass. The follow-up adds three-role native relation closings at `(g_R,g_D)=(1,0),(0,1),(.5,.5)`, all passing through storage. κ now follows each bank identity, so every occupied role receives its own confidence. These are mechanism checks, not the learned repeated-sentence gate.

The separate cold surface-word probe (`native_gain_probe.json`) is not counted
as the fully expected cat. Claude H1 identified the missing native-handle
route: activated codes can differ from their raw bank keys. The follow-up
native `(0.6,0.2)` and `(0,0.8)` cases identify through STM addresses and
preserve both actual lanes. Only composed values without a known handle use
the exact-distance fallback; an unmatched value passes whole.

## §9.4 — Does omission enter content?

| Native close / retained-pair check | Result |
|---|---|
| Expected absent identity, for-lane omission | 1 |
| Excess for that identity | 0 |
| Content | Only the observed cat, scaled by the declared floor; no absent symbol added |
| Answer and retained raw observation | Exact |

The omission case exercises both the closing image and the retained pair’s residual/content reader. `native_closing_gates.json` and the seven formal native certificates record the mechanism.
Retention uses the soft union `max(o,ê)` as support, so confident omissions
retain their residual. Both native writers use the same withheld-aware owner;
empty bank padding has zero weight. This corrects Claude B2/M4 without
changing Item 5's priority for surprising omissions.

## §9.5 — Does percept attenuation preserve identification?

| P gain | Attended region-gain range | Raw identification |
|---:|---:|---|
| 0 | 1.000000–1.000000 | Unchanged |
| 0.01 | 0.992363–0.992449 | Unchanged |
| 0.1 | 0.923626–0.924487 | Unchanged |
| 0.25 | 0.809064–0.811218 | Unchanged |
| 0.5 | 0.618128–0.622437 | Unchanged |
| 0.75 | 0.427193–0.433655 | Unchanged |
| 0.9 | 0.312631–0.320386 | Unchanged |
| 0.99 | 0.243894–0.252425 | Unchanged |
| 1 | 0.236257–0.244874 | Unchanged |

Native codebook lookup consumes raw coordinates before region gain, so identification is independent of the gain by construction. These nine sampled settings confirm the native handoff. Moment count remains 1024 during evaluation. S exact matching separately rejects unknown composed values; its raw norm and leaf coefficient survive the gained compose view.

## §9.6 — Does graded order beat shuffled on global derivation optimality?

| Seed | Initial R | Graded final R | Shuffled final R | Global derivation-optimality rate |
|---:|---:|---:|---:|---|
| 19 | 1.2667166 | 1.2473471 | 1.2473476 | Not established |
| 23 | 1.2457234 | 1.2112565 | 1.1584089 | Not established |

**Gate false.** Seed 23’s shuffled control has lower R. The original corpus supplies no exhaustive native derivation oracle; R is not renamed an optimality rate. Execution and parameter changes cannot certify the toy’s global measure. No extra training was used to turn this result into a pass.

## K — Have the compiled timeouts been resolved?

| Native compiled check | Result | Process time / peak memory |
|---|---|---|
| Unpacked explicit sentence boundary | Passed | 41.8 s / 1.34 GiB |
| Fullgraph output walk vs eager | Passed | 11.1 s / 0.69 GiB |
| Packed training boundary, compiled=True | Passed | See `native-final-1/result.json` |
| Unpacked explicit boundary, all four gains 0.5 | Passed | 47.0 s / 1.33 GiB; `review-compiled-8/result.json` |
| Packed document stream after review repairs | Passed | 23.7 s / 1.25 GiB; `review-compiled-8/result.json` |

`Generative.inverse_menu` previously traced all R operator dispatches for every one of R fixed menu entries: 17 operators produced 289 branches. Each entry now supplies its single operator. Word/STM choice kernels compile separately from the eager journal driver, CPU uses its native allocation default, and inference padding makes one host read before the word loop. No backend or timeout was relaxed. The failed/time-limited diagnostics remain failures; `compiled-8/result.json` is the successful rerun.

`review-compiled-6` was interrupted when Codex changed the schema during the
run. Its four completed cases passed, but the whole run is incomplete.
The two outstanding cases were rerun with fixed source in `review-compiled-8`;
the interruption is not counted as a pass.

## H3 — What does retaining the prediction cost?

The bound uses 1,024 bank symbols, 1,032 coordinates, three observed keys and
one representative key, with LTM capacity 1,024. All IDs, orders and logits
remain in both formats; the observed target retains only nonzero pairs.

| Retained representation | Keys | Bytes per row | Serialized bytes | Projected bytes at capacity | Packing time |
|---|---:|---:|---:|---:|---:|
| Full key bank | 1,024 | 34,074,906 | 5,308,353 | 34,892,703,744 | 0.599 s |
| Observation support and representative | 4 | 349,642 | 49,233 | 358,033,408 | 0.024 s |

`retention_bound.json` records allocation and timing details. This is a
metadata bound, not total model memory. A read materializes one transient
dense bank view (4 MiB here); an unpaired future prior may retain all keys
until its observation is attached. Missing keys are explicit and cannot
match zero padding. The sparse-key checkpoint regression preserves the exact
prior logits, target, omission and surprise.

## Review follow-up — Which findings are addressed?

| Claude finding | Repair / remaining decision |
|---|---|
| B1 | Per-symbol κ replaces envelope-role confidence; native three-role closings cover R and D. |
| B2 / M4 | One withheld-aware retention formula over soft union support; confident omissions count. |
| B3 | Awaiting Alec: prime weight default 0 and φ=1, or accepted changed defaults with enabled-expectation parity. No sweep approval. |
| H1 / H2 | Native STM handles identify activated codes; the compose read uses actual for/against lanes. Unknown unhandled values require distance ≤1e-5. |
| H3 | Sparse retained keys with full IDs/logits; measured bound above and exact checkpoint replay. |
| M1 / M2 | Unused arguments, scope knob and dead guards removed; migrated unknown IDs/orders are −1. Retired query-reward forecast/end stubs removed. |
| M3 | Free proposal view has zero prime and z=1; structured forecast need slot removed. Native checkpoint migration and free-view invariance checked. |
| M5 | lxml installed from the existing requirements; both canonical configs validate and retired scope values are rejected. |
| L1–L4 | CDF edge handling, configured omission threshold, dead state and stale configuration/docs repaired. |
| L5 | Word and S reads intentionally train the shared head from different causal contexts and targets; documented in ExpectationRetention. Their content gain is not applied twice. |
| N1 (re-review) | The incoming S observation now reads the raw poles through the declared `SymbolField.read` consumer. P dims only the content payload. Fixed S priors at P=0 and P=.75 exercise the same S gain. |
| N2 (re-review) | Thought keep, credit and audit all use R+A. Work remains bounded and reported without a work-price term in the judge. Native conflict cases exercise both directions where R and A disagree. |

The native causality diagnostic distinguishes the forecast of a fixed key
from the representative selected out of a changing vocabulary. The repaired
test holds the vocabulary fixed and compares all bank logits. Closing-only
optimizer tests isolate that loss from the independently weighted word/S
losses. Toggling the sentence head preserves the configured word switch.

## Scoped regressions and publication

| Question | Result |
|---|---|
| Broad affected selection | 715/715 completed: 652 passed, 33 failed, 30 skipped; `regression-2/result.json` |
| Repair of those 33 failures plus optimizer coverage | 55/55 passed; `regression-repairs/result.json` |
| Native closings, retained NP checkpoints and packed boundaries | 13/13 passed; `native-final-1/result.json` |
| Focused native/gain/attention/device coverage | 92 passed, 1 skipped; `native-and-focus-final.log` |
| Claude review of `f7e74af04` | Completed; not approved to sweep. `claude-review.md` and `claude-review-status.json`. |
| Claude re-review of `7285b38` | Completed; prior code findings resolved, B3 pending, N1/N2 found. `claude-followup-review.md`. |
| Claude short check of `282fcf918` | N1/N2 confirmed; no new code defect. `claude-n1-n2-review.md` and `claude-n1-n2-status.json`. |
| Review-specific/native/schema/NP restoration checks | 64 passed, four slow cases skipped and separately exercised; `review-targeted-5/result.json`. |
| Broader affected checks after review | 248 completed; three failures found; `review-regression-7/result.json`. Follow-up repairs recorded separately. |
| Repair of those three regressions plus review/native/thought coverage | 90/90 passed with slow cases enabled; `review-repairs-9/result.json`. |
| N1/N2 repairs, controller coverage and nonzero compiled boundary | 82/82 passed with slow cases enabled; `review-n1-n2-11/result.json`. The first attempt found an undeclared raw-pole consumer; `SymbolField.read` is now in the consumer census. |
| One full landing sweep | Not run; follows review |
| Push / WikiOracle bump / Item 4 completion | Not done |

The NP restoration fixture now preserves the native loader’s migrated sidecar fingerprints while retaining exact learned tensors. It previously restored old fingerprint bytes over migrated metadata and corrupted its next checkpoint. The production migration remains one routine at load.

Generated runner source manifests and the transient verbose compiler trace are not publication artifacts. The trace excerpt and result logs retain the diagnostic findings. Historical benchmark directories are unchanged.

Concurrent edits to the row-layout specs, Philosophy, the row diagram and
todo are preserved with the candidate. Their future row/membership changes
are not implemented as part of Item 4's expectation repair.

N2's review asks Alec to confirm whether the prior thought work price should
remain. The repair applies the user's existing instruction, “R + A for both
keeping and credit,” literally to thought too: work is metered and reported
but is not a third judge term. This interpretation is explicit in the
thinking spec; it is not recorded as an additional Alec decision. The stale
R-only / R+E+A live contracts in Architecture and GradientFlow are corrected.
