# Item 6.9 — continuation review repairs, October 1

Stopped for review under plan §10 and Claude's October 1 hand-off. Nothing
committed. The separator and policy repairs and all seventeen ports pass their
focused checks and the final sweep. The XOR reconstruction gate remains red.
Both native objective-conflict arms stopped at the unchanged 8 GiB guard before
their first cost/gradient snapshot, so that measurement remains incomplete.
The final sweep completed every case except one retained 30-minute timeout;
no assertion failed.

The accepted implementation of steps 5a, 6 and 7 is retained. The preceding
continuation receipt and its red results remain unchanged historical evidence.
The ordered separator, policy ownership, fixture ports, objective measurements
and final verification are recorded below. Suite trimming and item 6.85 await
review.

The candidate starts at published HEAD `d679df2b` with the prior uncommitted
6.9 changes. The venv is unchanged; the 8 GiB worker and 24 GiB aggregate guards
are retained. No seed is pinned, and no measurement fixture, epoch budget or class/reconstruction
threshold is changed. The additive control changes only its receipt-local
copy of the grammar, as explicitly requested.

## Separator repair

`separator-red/` preserves both failing cases before repair: an ordinary word
and a word whose object row is deliberately unavailable. Both pushed the space.
`separator/` retains the old source, complete changed bodies and repair diff.

Whitespace is classified at the eager boundary from the staged unit's surface
or byte span, independently of object admission. Only the mixing grammar's
push mask changes. The perception mask and full reconstruction target remain.
The reverse traversal and owned derivation use the same leaf membership, while
sentence closing still uses the original unit positions (including a trailing
separator).

The first follow-up check (`separator-green/`) caught the integer occurrence
prepass still counting a separator. Its 15 cases completed: 13 passed and the
two new probes failed on the unavailable flag. `separator-metadata/` saves that
pre-repair source and the complete old/new bodies. The corrected follow-up
(`separator-metadata-green/`) passes all ten cases, including trailing spaces,
unknown-word retention, punctuation retention, Unicode/ASCII whitespace, actual
read-back and compiled capture. No failed assertion was removed or relaxed.

`separator-thirty/` records ten fresh unseeded runs each of the sum-only control,
class gate, and reconstruction gate, with actual answers/read-backs and the
specified checkerboard contrast. All trials are retained. The raw distance from
one half is reported; the `half_at_float32_precision` field is an additional
numerical-equality diagnostic, not a new acceptance threshold. Finite-budget
training may finish near, rather than bit-exactly at, one half.

The thirty runs completed in 774.28 seconds with matched source and no resource
stops. The sum-only answers have a maximum distance from one half of .0056681,
MSE .2500001–.2500148, and contrast at most 5.96e−8. None solves XOR; 0/10 is equal to one half under the saved default
`torch.allclose` diagnostic. The raw deviations are retained, not relabelled as
ten exact-half passes.
The class campaign gets all four right in 10/10 and meets the .05 bar in 8/10.
The dedicated reconstruction campaign passes 1/10; every read-back now has two
words and every unavailable flag is false. All raw values and read-backs are in
[the thirty-run table](separator-thirty/results.md); misses remain red.

## Walk policy ownership

`policy-red/` saves the two failing probes: actual exploit and explore answer
generation each published `_output_policy_cost`. `policy/` holds the old source,
complete old/new bodies and diff. The repair keeps the realization graph live
and leaves policy credit to the call after the sentence trials. The first focused
checks and four unchanged historical policy cases are preserved in `policy-green/`.

That first follow-up completed with two focused passes and four historical
failures: suppressing the attribute alone leaves zero-valued policy gradients
through the compiled loop's lifted inputs. A zero gradient can still advance
Adam momentum. `policy-gradient-red/` reproduces that distinction in both
compiled trials; the eager variants pass. `policy-gradient/` keeps complete
old/new bodies. Trial generation now captures detached policy parameter views
before the loop and does not accumulate policy credit. Its numerical inverse
and answer paths remain live. `policy-gradient-green/` passes all four focused
eager/compiled cases, with an absent policy gradient and a nonzero idea gradient.
The original four policy cases are byte-for-byte equal to published HEAD
(`policy/unchanged-historical-tests.json`). Their final recheck passes in
`policy-final-and-scope-green/`, including the no-answer phase. All 14 selected
cases pass in 1,123.86 seconds, with peak aggregate memory 18.95 GiB and no resource stops.

## Packed-sentence scope follow-up

Code inspection identified a scope error introduced with the leaf mask: it
was copied before restricting the reverse traversal to a single sentence.
`separator-scope-red/` saves two failures, with and without whitespace. The
second sentence incorrectly popped the first sentence's leaf and reported
underflow. `separator-scope/` records the complete old/new bodies. Leaf
membership is now intersected with the selected sentence before either reverse
pass. This does not change the single-sentence campaigns above. The new probes,
separator probes and native packed-versus-streaming parity are included in the
same final check as the historical policy cases.

## Seventeen ports

`ports-red/` reproduces all seventeen failures. `ports/` preserves the old
files, the patch, and complete old/new bodies for all seventeen cases, including
the shared helper for the fifteen unchanged case bodies. The shared helper and
direct builder now include last-unary IDs and inverse-exclusion masks. The
inverse case requires a hard valid bounded candidate attaining the least
compose residual, while preserving its availability and known-right assertions.
`ports-green/` passes all seventeen cases, as does the final sweep.
[Complete old/new case bodies](ports/case-bodies.json) include the shared
helper changes alongside all fifteen unchanged consuming case bodies.

## Objective-conflicts stage 1

The candidate is frozen in `final-source.json` and `final-inputs.json` for these
measurements and final verification. Only receipt-local observation and the
specified cut arm are installed in memory. No configuration or production source
is changed. Hook-only validations constructed no model. The first runtime
attempt (`objective-conflicts/failed-attempt-01-XOR_grammar-step5a/`) found a
logging error when the optional grammar-lesson dictionary was `None`, before
any optimizer step. Its source, log and failure are retained; the observer now
handles that absent term. This is an observer repair, not a statistical rerun.
The four requested arms have been attempted once. Both XOR arms complete in
about 69 seconds; step 5a ends at MSE .0281231 (4/4 right) and the cut at .194819
(2/4). Neither reconstructs all four. In 1,600 active comparisons, step 5a keeps
the worse expectation trial 297 times (18.56%), mean worsening .00317355 and
maximum .137341. Trial reconstruction is inactive in this configuration; a
zero reconstruction conflict count is not evidence of protection.

Both native benchmark arms hit the unchanged 8 GiB guard before their first
cost or gradient snapshot, after about 7.5 seconds (sampled peak 8.93 GiB).
There is no native reach, selection, magnitude or endpoint result. Its effective
weights and codebook buffer ownership are saved. This portion of stage 1 remains
incomplete; no smaller batch, configuration change or guard exception was used.
See [all measurements](objective-conflicts/README.md) and
[every weight and its purpose](objective-conflicts/weights-purpose.md).

The XOR parameter-group labels were corrected from saved sufficient statistics:
InputSpace's registered OutputSpace back-reference had included the reading
head in perception and the shared vocabulary in the reading map. The original
observations remain; `regroup_xor.py` checks exact nonzero support before
recovering each norm/cosine. No training or forward pass was repeated. The two
trial parameter-version digests match, gradient reads preserve parameter and
optimizer-gradient versions, and all sampled RNG states are unchanged.

## Final verification

The [single-run named XOR table](final-measurements/candidate/table.md) is
**34/35**. XOR_grammar's class gate passes (4/4, MSE .01267379); its reconstruction
gate remains red. The requested **one MM_20M_xor exact round trip passes (1/1)**.
Slow proof selectors and their actual observations remain in the table and raw
logs. These are candidate-only measurements, with no HEAD rerun.

MM_grammar's ten fresh unseeded runs have median ending training MSE
**0.000818674336316**, with three runs ending at .125. All ten
rows are retained in [the MM measurements](final-mm-summary.json). This is the
same raw-forward 900-update measurement used in the accepted receipt; it is a
descriptive measurement, not a sentence-trial grammar-learning proof.

The [source-matched full sweep](full-sweep/summary.md) selected **5,219 cases**:
**4,895 passed, 322 skipped, one expected failure and one timeout**. The four
extra passed subtest reports are recorded separately. All six previously
accepted repairs, seventeen ports and four unchanged policy cases pass (27/27),
as do the thirteen new separator/scope/policy probes. There are no assertion
failures and no unexecuted cases; the timeout is the sole case without a completed
outcome.

The retained timeout is
`test/test_item9b_schedule.py::test_native_interleave_supplies_context_then_reads_the_same_sentences[True]`:
1,801.036 seconds at the unchanged 1,800-second worker limit, sampled peak
5.147 GiB. The initial runner stopped dispatch and interrupted three sibling
attempts. The same collected list then continued without rerunning any completed
case or the timed-out case. Those three interrupted cases completed and passed;
all original attempts are preserved in the
[combined coverage ledger](full-sweep/combined-coverage.json). This is one
continued sweep, not a second full sweep. Its original three-hour deadline,
8 GiB worker guard and 24 GiB aggregate guard remained in force.

Total wall time was **122.077 minutes**, including
3.486 minutes of continuation preparation;
runner time was 118.591 minutes. Against round 5's 5,122 cases
and 89.6 minutes, this is **+97 cases and +32.477 minutes**.
Peak aggregate memory was 17.380 GiB. No compile-cache retry
occurred. The sweep's overall result remains `resource_stops` because of the
retained timeout.

[Final verification](final-verification.json) confirms all 723 tested source
files and supporting inputs match the frozen snapshot, the environment is
unchanged (both `pip freeze` files are saved), and `git diff --check` passes.
HEAD is still `d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`. Receipt summaries were
closed after all workers stopped; their local links were then checked separately.
The original measurement version of `finalize.py` is preserved as
`finalize-before-continuation.py.txt`; `resume_full.py` records the continuation.

The accepted item-7 baseline remains historical: **44/49 XOR table, 12/15 exact
round trips, MM_grammar median .1066**. Earlier red depth-three evidence and prior
6.9 receipts are retained. No seed, threshold, guard, production configuration,
environment or assertion was changed to improve these results. No suite trim or
item-6.85 design is implemented. The reconstruction misses, native measurement
stops and full-sweep timeout remain explicit for review.
