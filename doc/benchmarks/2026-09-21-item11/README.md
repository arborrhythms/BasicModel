# Item 11: generation ownership and end-to-end output

Baseline: `3f92150` after the reviewed item 12 landing (`afdcdfa`). The preserved
September 17 archive was checked against its SHA-256 and every regular member's
hash and size before inspection. No archived installer was executed.

## Rebased contract

The separate generation catalogue exists with `outputInLoop` either off or on;
only the walk chooser is conditional. Catalogue entries are non-owning views
of the shared numerical hosts. Alias reordering adds no parameter, checkpoint
key or RNG draw. Ordinary output dispatch enters a model-local scope, resolves
declared interfaces and restores the previous scope on exceptions and nested
calls. Reconstruction uses its recorded compose dispatch. Existing natural-fold
interfaces retain their host's inverse, including a two-pass adapter.
Only exact, live per-space natural-fold bindings enter this catalogue;
inactive-space declarations do not create walk actions or borrow another
space's geometry.

No independent numerical copies or extra objective were introduced. Existing
rule-meaning chooser migration preserves weights and Adam moments through
reordering, removed actions, new actions and marker-free old checkpoints.
Shared operator weights retain their host names and moments through strict
reload and a further optimizer step; a missing host parameter remains an error.
The conclusion and named context still detach before the trainable output
conditioner. [Gradient contract](../../GradientFlow.md).

## Preserved-probe disposition

| Archived probe family | Current disposition |
| --- | --- |
| Catalogue in both modes; detached output versus live input | Catalogue restored in both modes. Independent-weight assertions are inverted to shared identity and single optimizer ownership. Actual conclusion-cut and shared-inverse credit checks remain in `test_prepared_answer_boundary.py`. |
| Unscoped/scoped binary inverse; error restoration | Scoped declaration lookup uses the same host; input replay stays outside that scope. Error restoration and rejection of an undeclared output operator are checked. |
| Copy saved comprehension weights into a new generation owner | Superseded: there is no new numerical owner or new numerical checkpoint key. Strict shared-host/Adam round trips cover both output modes. |
| Independent weights/moments, partial checkpoint, flat natural folds | Weight/moment identity is shared. Missing host state is rejected. Existing natural-fold declarations resolve to their exact shared hosts. |
| Unary output inverse | Declared catalogue lookup is exercised; the existing host adapter is retained. The candidate's bypass of the host adapter is discarded with its copied weights. |
| Reordered aliases | Shared host identity and empty non-loop state remain stable without consuming RNG. Existing chooser migration separately covers rule-row changes. |
| Detached/live global gradient budget | Superseded by §8.4's ordinary weighted sum and local state cuts. Existing gradient-factorization and joint-objective tests cover the current contract. |
| Supplied/unlabelled Adam gating | Actual output steps train the active conditioner and the walk's shared maps. Only dedicated answer parameters must stop on absent labels; shared maps may still train from reconstruction. Warmed-Adam checks cover present, past, future and missing-label inputs in both output modes. |

No main-tree reasoning method is removed. The archived raw candidates and
earlier receipts are unchanged; none is reused as current validation.

## Development evidence

The first four catalogue probes fail on the baseline: the non-loop catalogue
is empty and no declared ordinary-output resolver/scope exists. After the
runtime change, all 52 catalogue, compose-meaning and peer-pipeline cases pass.
[Initial failure log](initial-red.txt) and
[all development attempts and dispositions](development-attempts.json)
retain the raw receipts, including a corrected nonexistent test selector that
collected no cases.

A broader integration probe initially demanded shared-map credit in both
native modes. The non-loop fixture executes dedicated synthesis and a
parameter-free reverse; requiring a shared-map update there was an invalid
probe assumption, also noted in the preserved candidate. The corrected probe
requires a live conditioner update and zero conclusion gradient in both modes,
and a shared-map update when the walk executes those maps. The absent non-loop
shared-map gradient is recorded as absent, not as a learning result.

Explicitly running the existing slow absent-label tests exposed stale
`what_step_policy` assertions and a retired chooser parameter enumeration.
They now enumerate the production `synthesis_parameters()` owner and warm the
live generate chooser's Adam state before checking both output modes.

The first full attempt, `20260921-075518-982b22`, exposed a real construction
regression in the new catalogue: a parallel aligned model retained a natural
fold declaration for an inactive subsymbolic space. Its existing dispatcher
had no host for that interface, but the catalogue tried to resolve it. The
attempt was stopped at **3,837/4,693** completed cases, exit 130, after the
failure; it is not an acceptance receipt. [Raw result](first-full-result.json.gz),
[source manifest](first-full-source-manifest.json), [run log](first-full-run.txt),
[failure](default-interface-failure.txt).

The catalogue now snapshots exact live bindings on this default path and
rejects cross-space natural-fold lookup. Both the original construction case
and a new wrong-space alias probe fail before the correction
([two-case red](default-red-result.json.gz)); all 25 catalogue and row-local
optimizer cases pass afterward ([targeted green](default-green-result.json.gz)).
Ordinary structural aliases still share their declared numerical host.

The second full attempt, `20260921-082056-a33381`, stopped at **1,797/4,694**
completed cases, exit 130, after the unseeded MentalModel compatibility test
produced a non-finite routing probability. This was reproduced at seed 3 on
both baseline `3f92150` and the current runtime: inputs, candidate expansions
and scores are **bit-identical**, including the overflowing candidate. Seeds
0–2 complete normally on both trees. This is an existing untrained-model
numerical limit, not evidence of robust generation or a catalogue regression.
[Failed receipt](second-full-result.json.gz), [source manifest](second-full-source-manifest.json),
[run log](second-full-run.txt), [failure](mentalmodel-failure.txt),
[baseline probe](mentalmodel-baseline.txt), [current probe](mentalmodel-current.txt),
[tensor comparison](mentalmodel-comparison.json), [probe script](mentalmodel_probe.py).

The compatibility fixture now restores its RNG state and uses seed 0. A
separate seed-3 case explicitly checks that the existing non-finite guard
raises; the runtime guard and numerical operators are unchanged. The complete
compatibility file passes **11/11** in `20260921-083207-bb1df0`, exit 0,
9.85 seconds ([receipt](hierarchical-result.json.gz),
[source manifest](hierarchical-source-manifest.json), [run log](hierarchical-run.txt)).
This makes the two intended assertions reproducible rather than relying on
ambient worker initialization. It does not claim the seed-3 model can forward
successfully or hide its overflow.

The third full attempt, `20260921-083404-415408`, completed **4,695/4,695**
cases: **4,361 passed, 330 skipped, 3 failed, 1 expected failure**, exit 1.
Only the three typed-result adapter fixtures failed: they replace the entire
language space and stub every numerical synthesis operation, but did not
provide its new `generation_scope` interface. Their stub now supplies a null
context; the behavior assertions and production code are unchanged.
[Raw result](third-full-result.json.gz), [source manifest](third-full-source-manifest.json),
[run log](third-full-run.txt), [failure](adapter-fixture-failure.txt).

## Affected and supervised integration receipts

The broad affected run `20260921-081431-7fdbd2` completes **230/230** cases:
**212 passed, 18 skipped**, exit **0**, in **234.66 seconds**, without a cache
retry. [Receipt](affected-result.json.gz), [run log](affected-run.txt),
[source manifest](affected-source-manifest.json), [summary](affected-summary.json).
It covers catalogue ownership, output and reconstruction boundaries, compiled
walks, meaning/peer contracts, grammar dispatch, shared gradients and objectives.

The explicit slow run `20260921-081850-49fd99` passes **4/4**, exit **0**, in
**110.37 seconds** (3.19 GiB peak). It exercises both native conditioner modes
and both warmed-Adam missing-label modes with `RUN_SLOW=1`; the bounded runner
places these cases on MPS. [Receipt](integration-result.json.gz),
[source manifest](integration-source-manifest.json), [run log](integration-run.txt).

These two receipts match **630 source files**, fingerprint
`d8722750142b320a8816c576286962bdf1106ea609d1a58d6220892cb95b305d`
(SHA-256 of the sorted compact JSON `validated_source` map).
Only the subsequent deterministic MentalModel fixture and adapter stub differ
from that snapshot; the runtime and these supervised integration probes are
unchanged.

After both fixture corrections, `20260921-085306-14909c` completes **76/76**
cases across the adapter, catalogue, prepared-answer, output-walk and supervised
output files: **60 passed, 16 skipped**, exit 0, in **76.37 seconds**, without
a cache retry. [Receipt](final-affected-result.json.gz),
[source manifest](final-affected-source-manifest.json), [run log](final-affected-run.txt),
[summary](final-affected-summary.json). This snapshot is the one used for the
replacement full receipt below.

The real native supplied-answer probe uses initialization seed 0 and sampling
seed 11, two input rows and one ordinary `runBatch`/Adam step on CPU. No action
is forced, and no answer seed is injected. Both modes update their active
conditioner. [Captured gradient evidence on the corrected source](supervised-gradients.json):

| `outputInLoop` | Supplied output loss | Conclusion gradient | Shared groups updated by the step and reached by output loss |
| --- | ---: | ---: | --- |
| false | 0.04460762 | 0 | None in this fixture's parameter-free reverse |
| true | 0.04076102 | 0 | `operator.CS.lift`, `operator.CS.lower` |

The step includes the normal weighted objectives. Output's own derivative is
measured before backward; a later parameter change is not misattributed solely
to output. A forced shared-inverse boundary test separately isolates that path.
One-step supervised evidence does not establish useful learned language,
expectation or reasoning. Countdown items 9 and 8 retain those learning gates.

## Full receipt

The replacement full run `20260921-085748-b59239` completes **4,695/4,695**
cases: **4,364 passed, 330 skipped, 1 expected failure**, exit **0**, in
**1,059.34 seconds**, with **no waived failure and no cache retry**.
[Full receipt](full-result.json.gz), [source manifest](full-source-manifest.json),
[run log](full-run.txt), [receipt metadata](receipt-info.json).

The final affected and full receipts match all **630 source files**, fingerprint
`215308c7bed6922c72e6599319f6fefd62e124bb0032b75d10e183d77f3c1cd0`.
The current tree was checked against every hash after completion. Counts above
are unique cases; the existing flag test emits five passing phase reports for
one case, preserved separately in the metadata. The expected failure is the
same cleared-cache word-recovery case as the reviewed baseline. No new skip or
expected-failure marker was added.

The command uses `--batch-size 8 --max-files 1`, ten bounded workers and the
unchanged 8 GiB per-worker / 28 GiB aggregate caps. Peak aggregate memory is
**14.73 GiB**. All earlier failures have explicit dispositions above; none of
those failed full attempts is an acceptance receipt.

Final documentation-link run `20260921-091700-f39470` passes **69/69** cases,
exit 0, with the same validated source. Its raw receipt, manifest and run log
are identified in the receipt metadata above.
