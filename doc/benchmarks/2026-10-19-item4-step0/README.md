# Item 4 outstanding gates before Item 3.5

Baseline: BasicModel `3ad4056e4`, WikiOracle `9d81ec0`. Both mains now
name these published Item 4 heads. Work is on `codex/item4-step0`.

Human-readable log/HTML whitespace is normalized. Where a log changed,
its original bytes are retained beside it as `.log.raw.gz`; JSON results
retain their captured output unchanged.

**Candidate status:** the paired reconstruction and initial-distribution
repairs pass their scoped checks. All 16 standing XOR cases pass in both
configurations. Gate 1 remains **false**: the one-successor decomposition
certificate fails in both configurations after its API and memory repairs.
This is a blocker in this round, not a new deferral to Item 0 or 3.5. Required
Claude review and the single full landing sweep remain pending; no landing
or Item 3.5 implementation is claimed.

## Gate 2 — Is the paired reconstruction invariant repaired?

| Question | Evidence / status |
|---|---|
| Known failing initialization | Archived seed 1, third training batch; `reconstruction.content = .0517327189`, with zero byte and forward-discard charges. Reproduced exactly in `paired-before.json.gz`; compact costs in `paired-costs.json`. |
| Cause | Both trials use the identical primed bank. Free inverse selection chose an against-only candidate for the correct left word. `recover_lanes` read its empty for lane as its code, losing the left meaning despite correct byte recovery. This is not a bootstrap-state leak. |
| Repair | Identify the shared code support from both candidate lanes with their coordinatewise maximum, then solve each parent lane independently. The parent's against evidence never enters its for target. |
| Known counterexample after repair | `paired-after.json.gz`: both trials' R are exactly zero in all three original batches; same seed 1 and budget, no continued training. |
| Regression | Six polarized-candidate inverse cases pass; the original unseeded bootstrap certificate passes. Broader inverse selection: 143 existing tetralemma cases pass in `invariant-checks`. Final landing-sweep coverage pending. |

Equal initial snapshots alone do not imply equal loss for arbitrary different
derivations. The repaired invariant is this fixture's exact reconstruction in
both trials; no general cost-equality rule or cost override was introduced.

## Gate 3 — Is initial conditional binding within [.9, 1.1] of uniform?

| Question | Evidence / status |
|---|---|
| Existing failure | .872515–1.063954. Threshold unchanged; no passing seed selected. |
| Initialization repair | Candidate-value columns and the reference-choice projection start at zero. Other context/operator weights retain their initialization. Both neutral paths have nonzero ordinary gradients; checkpoint keys and the RNG draw count are unchanged. |
| Remaining cause after that repair | `distribution-before/contribution-summary.json`: 3,998 conditional menus, base logit span exactly zero in every menu; prime span up to .250968. Probability ratios .845358–1.188600. The prime is added after the chooser score. |
| Prime repair | Keep the prime on. In binding menus only, rescale its fixed logit contribution to span at most log(1.1), preserving sign and ordering. Learned logits are not bounded. At neutral initialization, any conditional subset has probability/uniform in [1/1.1, 1.1], independently of seed or candidate count. |
| Coverage repair | A later ordinary run constructed zero retained candidates (`section14-repaired`), failing before the band assertion. Add six controlled native retained-bank contexts (orders 0/2; 1/4/16 stored candidates), with unforced selection and no training. Retain the original 32-document/8-question audit and assertions. Also verify that the actual compose chooser, not only the thought chooser, remains untrained during provisioning. |
| Controlled coverage | 336 menus, 3,864 retained alternatives; .991817–1.009936 of uniform. Order-one binding remains the close-time resolver's responsibility. Ordinary-document counts are reported separately. |
| Full §14 completion on repaired setup | `invariants-final`: 25 passed / 1 prerequisite skip across all §14 cases, 12 new regressions and the original paired certificate. Distribution .930257–1.077537 across 5,182 menus and 11,576 retained alternatives; `distribution-after/distribution.json` and compressed menus preserve the observations. |

## Gate 1 / spec §9.1 — Do standing fixtures hold with every level enabled?

| Question / standing fixture | Zero-gain arm | All-level arm |
|---|---|---|
| Declared configuration | P=S=R=D=0; prediction enabled, prime on, φ=.5 | P=S=R=D=.5; prediction enabled, prime on, φ=.5; shipped defaults unchanged |
| Matching and budgets | One fresh initialization per fixture/shared group; existing budgets and assertions | Same initialization; 28/28 common-constructor parameter comparisons exactly equal |
| Grounded XOR | 6/6 pass; outputs [0,1,1,0], MSE 0 | 6/6 pass; outputs [0,1,1,0], MSE 0 |
| Native XOR curriculum | 3/3 pass; outputs [0,1,1,0], reconstruction 4/4 | 3/3 pass; outputs [0,1,1,0], reconstruction 4/4 |
| Exact CLI XOR | 2/2 pass; reconstruction 4/4; output MSE <.05 | 2/2 pass; reconstruction 4/4; output MSE <.05 |
| MM_xor / MM_grammar | 3/3 pass at the unchanged .20/.26/.20 loss bars | 3/3 pass at the unchanged .20/.26/.20 loss bars |
| XOR_grammar class accuracy | 4/4; reported outputs [.0006,.9994,.9994,.0006] | 4/4; reported outputs [.0009,.9982,.9986,-.0033] |
| Grammatical reconstruction | 4/4 word multisets; pass | 4/4 word multisets; pass |
| Standing math-chain selection | 51 pass, 2 API-fixture failures, 1 copular-artifact skip | 51 pass, same 2 API-fixture failures, same skip |
| Complete initial comparison | 70/70 completed: 67 pass, 2 fail, 1 skip | 70/70 completed: 67 pass, 2 fail, 1 skip |
| Two decomposition certificates after wrapper/memory repair | 2/2 complete: one-successor fails; two-successor passes | 2/2 complete: same failure and pass; initial parameters match the recorded zero arm |
| Observed P/S/R/D gain minima | P/R/D = 1; S consumer gain path bypassed at g=0 | .708694 / .503297 / .925982 / .945317 |
| Joint all-level coverage | Matched MM_grammar baseline | MM_grammar convergence run exercises P, S, R and D attenuation in one model |
| Exclusions | No graded/shuffled regrade, derivation-optimality or collapse campaign | No extra 5.5 training or rescue training |

The [declared protocol](all-levels-protocol.json) records fresh entropy before
either arm. Each fixture (or shared-training group) gets one seed derived from
that entropy, reused only for its paired arm. Initial parameter tensors are
compared exactly at the common per-stage constructor, including native CLI
subprocesses. Temporary initialization tensors under ignored `output/` are
not trained lesson artifacts. Configuration/coverage observations live in
`all-levels-metrics`; compiled graphs omit Python counters, so coverage counts
describe observed eager reads. `run_all_levels.py` refuses to reuse a protocol.

Results and all fixture failures are in `all-levels-{zero,enabled}/result.json`;
the compact [comparison](all-levels-summary.json) retains the full grouping,
parameter comparisons and coverage counts. Numerical outputs are not identical:
XOR_grammar's MSE calculated from the rounded printed predictions rises from
approximately 3.6e-7 to 4.225e-6; both pass the unchanged .05 bar. The MM cases
and the exact CLI MSE case do not emit their exact final predictions/losses;
their existing assertions certify the reported bounds only. No extra training
was run to fill that diagnostic gap or to obtain a better number.

The two decomposition failures occur before their first reading completes:
their forced-grammar observer still unpacked three chooser results after the
API added the proposal correction. Its repair forwards the original result
tuple intact while retaining the same forced actions. The separate replay
must complete both configurations; the initial failures remain in the record.

The subsequent completed replays are `decomposition-bounded-{zero,enabled}`.
They retain the original initialization, two-document fixtures, 512/768 work
budgets, optimizer and assertions. The behavioral fixtures use the same
eager loop helper as the other §14 cases. Native compiled boundary checks
remain separate. The one-successor failure is explained, with exact R/A
components, in [decomposition-cause.md](decomposition-cause.md). Its live
alternative wins under the required R+A rule; the committed readings do
not supply the intended proof. Changing that rule, suppressing the live
alternative, trying another initialization or weakening `assert proved`
would not repair this certificate under its current contract.

| Resource / execution question | Evidence / outcome |
|---|---|
| Original compiled decomposition replay | Manually interrupted after approximately 355 seconds, with only the first premise batch completed. `decomposition-zero/interruption.json` and the sampled stack record it as incomplete, never a pass. |
| Eager replay before memory repair | `decomposition-eager-*` and `decomposition-final-*`: the first worker reaches the unchanged 8 GiB guard; companion workers are aborted. No completed certificate. |
| Memory cause | `allocation-stack-probe.log`: retained memory grows inside `narrow_words` while the MLP scores every reserved bracket slot on every round. Budget-sized padding, not the episode audit, causes the observed stop. |
| Memory repair | Score only the prefix that word brackets can occupy, then restore the full masked action menu. An assertion rejects any legal action outside that bound. Work, sampling, action addresses and gradients retain their contract. |
| Completed resource results | Both decomposition cases in both configurations complete below 2.2 GiB. The functional one-successor failure remains visible. |
| Scoped attention regressions | `bracket-memory-checks`: 82 completed, 73 passed, 9 skipped. Full versus bounded scoring agrees in action, probability, RNG state and parameter gradients for both chooser types. |
| Explicit compiled narrowing | `bracket-compiled-check`: 1/1 passed, 7.3 s, .58 GiB. The default batch's skip is not counted as this result. |
| Explicit compiled expectation boundaries | `compiled-boundaries`: 2/2 passed at g=0 and g=.5; 46.8/47.7 s, approximately 1.3 GiB each. |
| Separate audit defect | Sparse state digests now hash coalesced occupied storage without densifying logical capacity; the regression passes. The first sparse-only replay still hit the memory guard, so this repair is not credited with resolving that stop. |
| Diagnostic preflight | `allocation-probe-process.json`: the first standalone allocation probe omitted the CPU selection, failed exact initialization comparison before presentation, and was corrected to the suite's CPU device. All subsequent native probes use the recorded initialization. |

The all-level table is the original complete comparison, followed by the
separately reported scoped repairs. It is not a second full comparison or
a claim that a failed case passed. No stopped learning experiment was resumed.

This zero-gain arm keeps the prime on and φ=.5. It is distinct from the sole
B3 parity fixture (prime off and φ=1); neither result is promoted to the other.

## Gate 4 — Can the copular certificate run from its legitimate lesson?

| Question | Evidence / disposition |
|---|---|
| Registered artifact | `copular_identity` is null; no producing artifact has been identified. |
| Mechanism contract | The column-copula certificate remains unconditional. |
| Deferred prerequisite | Produce the copular-identity lesson in the agreed Item 3.5 sequence, after stage 4e; register the artifact and producing receipt, then run the ordinary-answer certificate unchanged. No substitute checkpoint or extra rescue training in this round. |
| Earlier reader proposal | The October 18 decision in the Item 5.5 plan withdrew the proposed one-equality-hop reader patch in favor of phase gating. It is not silently reinstated. |

## Gate 5 — Does the landing sweep cover the repaired production tree?

| Question | Status |
|---|---|
| One full landing sweep | Pending resolution of the one-successor certificate and the required candidate review. Must include common-constructor PerceptMoments and the paired reconstruction case. |
| Review candidate | [Review request](review-request.md); the unresolved certificate is a current-round blocker, with its original assertion and failing initialization retained. |
| Documentation checks | `doc-links`: 412/412 passed; `git diff --check` clean. Generated source manifests are removed from this receipt under the publish rule. |
| Deferred learning tracks | Original failed Item 4 learning run and separate Item 0 / 5.5 gates remain unchanged and explicitly open. |

The earlier seed-23 chooser-credit finding is separate from seed 1's inverse
defect. Its [recorded-source cause](../2026-10-19-item4-repair/seed23-cause.md)
stands: compose departures tied, while unequal paired costs belonged to other
owners. This round does not regrade or extend that failed learning run.
