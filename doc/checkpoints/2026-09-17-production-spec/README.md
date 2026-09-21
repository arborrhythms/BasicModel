# September 17 production-spec checkpoint

**Unfinished work, preserved for the OS update.** Alec requested a pushed
checkpoint and made completing this session and its integrated spec the next
task. This checkpoint does not claim full implementation or full-suite success.

Canonical specification:
[2026-09-15-next-sentence-as-the-production-objective.md](../../plans/2026-09-15-next-sentence-as-the-production-objective.md).
Resume its §10 order and completion gates. First read the September 14 ownership
plan in full; re-read its §2 standing invariants after every context compaction.
[Top-priority work list](../../../todo.md).

## Published state

The last product-code commit is `639034cc36e0c1cc7630a88783539cd0fe696be9`.
The bounded test workflow checkpoint is `88425276f8dfdb31bc034bded213c4d5df236d10`.
No candidate below has been installed into the main runtime by this checkpoint.

**September 21 follow-up:** the remaining generation candidate is rebased by
countdown item 11 under the superseding shared-operator contract. Its copied
weights and global gradient-budget assumptions are not installed. The archive
and patch below remain unchanged historical evidence; see the
[current implementation, probe dispositions and receipts](../../benchmarks/2026-09-21-item11/README.md)
before considering any archived helper or assertion.

- Current runner/device tests: **29 passed in 138.27 seconds**, peak 0.56 GiB.
- Last full default attempt: **exit 124**, 2,907/4,458 completed, 5,136 seconds.
  It timed out; no replacement full run was started.
- Actual MPS sequence-training check passed. Two batches took 877.81 seconds
  inside `runEpoch`; construction took 1.24 seconds. Peak footprint: 2.54 GiB.
  This is not a current supervised-answer throughput result or learned-utility
  gate. GPU speed tuning remains open.

[Validation and limitations](../../Testing.md#validation) and
[durable receipts](../../benchmarks/2026-09-17-bounded-test-data/checkpoint-summary.json).

## Uninstalled candidates

The patches are review copies. `unfinished-work.tar.gz` preserves complete owned
source files, their exact base contents, candidate manifests, probes, draft
documentation, scripts and earlier evidence. The candidate manifest identifies
files covered by each candidate's validation hash. Earlier passing selections
are evidence for those exact candidate versions, not for an integrated result.

| Dependency order | Candidate | Latest known status / next gate |
| --- | --- | --- |
| 1 | [Thought history](thought.patch) | 33 reviewer cases; isolated affected selection 145 passed, 1 skipped. Install, review docs, run affected files and full default gate. |
| 2 | [Query phases](phase.patch) | 23 reviewer cases; latest isolated selection 76 passed. Includes fullgraph probes. Install after history and validate primary. |
| 3 | [Nested retention](nested.patch) | 23 focused cases passed after the final reference fix; earlier 257-case affected result predates it. Re-run affected files and primary gate. |
| 4 | [Shared query work](query-work.patch) | 13 reviewer cases; isolated affected selection 154 passed, 1 skipped. Normal controller integration remains open. |
| Following prerequisites | [Selected meaning](meaning.patch) | Partial Language change only. Three pure-dispatch probes were red; green rerun is pending. Eleven selected-relation probes are prepared but have not run. Obtain their red before implementing missing behavior. |
| After normal meaning/controller work | [Generation catalog](generation.patch) | Isolated migration/ownership probes passed. Based on older primary runtime, so rebase after the earlier stages, then rerun affected files and the full gate. |

These are partial foundations. They do not complete the normal meaning/controller,
anticipation, residual policy credit, output integration or held-out utility
requirements of the canonical spec. The selected-meaning outline in the archive
lists the missing paths and checkpoint-compatibility questions.

## Recovery and review

1. Verify the archive SHA-256 against `archive-manifest.json`; it also records
   the hash and size of every regular member. Extract into a new scratch folder.
   The archive contains no executable symlinks and no protected user documents.
2. `candidates/<stage>/` contains the owned candidate files and manifests;
   `bases/<stage>/` contains their exact pre-change contents. Review the adjacent
   patch and verify hashes before applying a stage to an isolated checkout.
   Unchanged dependencies come from the recorded primary base or prior stages.
   `candidate-manifest.json` records the original dependency symlink targets.
3. `scratch/` preserves the session's prefixed probes, draft docs, install/finalize
   helpers and evidence. Helpers contain absolute paths and historical base/head
   assertions. They are recovery material, **not a script queue to execute**.
   Rebase and review them against the resumed tree, preserve user-owned changes,
   and validate each item before publishing it. Old success receipts cannot be
   reused as validation for changed sources.
4. The main resume helpers are `full-spec-install-thought-history.py`,
   `full-spec-install-query-phase.py`, `full-spec-install-nested-retention.py`
   and `full-spec-install-query-work.py`, with their matching `-docs.py` helpers.
   Their required base hashes and protected-file hashes will need an explicit
   review after this checkpoint. `full-spec-bounded-stage-config.json` lists
   stage evidence; `full-spec-meaning-implementation-outline.md` describes the
   incomplete next implementation; `full-spec-arithmetic-isolation-probes.py`
   preserves unrun arbitrary-symbol isolation probes.
5. Significant training must use MPS here, with finite memory and time limits.
   Finish the legacy slow-marker audit: 19 manually gated functions and the
   GPU-placement class lacked the central marker; seven measured training
   functions also merit slow classification. These proposed marker additions
   were not applied at the checkpoint. Default reruns should follow measurement,
   not simply hide failed tests behind slow markers.
6. Keep the original per-item red → fix → affected files → full default green →
   commit/push → parent gitlink commit/push workflow. The OS-checkpoint exception
   does not waive that workflow for future implementation. Trailer on both:
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

Do not remove unused reasoning methods without Alec's review. The held legacy
retirement was not installed; only its hold note is retained here. Do not
implement the September 16 two-truths/forgetting specs in this resumed session;
Alec reserved them for work after this session. Preserve the protected uncommitted
README and user documents. `todo.md` was explicitly authorized for this checkpoint.
