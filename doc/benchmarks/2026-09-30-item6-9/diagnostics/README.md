# Why this receipt takes time

Read-only diagnosis requested during the step-5 campaign. No run, seed,
threshold, source file, configuration, thread setting or guard was changed.

The native path still captures graphs with `MODEL_COMPILE=eager`.
`BasicModel._sentence_path_cost` calls `_compiled_reconstruct`, which maps
the eager backend to `aot_eager` and uses `torch.compile(..., fullgraph=True)`
for its nested reconstruction loops. `SentenceCompose.compile_word_brick`
also uses `torch.compile` whenever the backend is not `none`. “Eager” here
does not disable Dynamo or AOT graph capture.

The current native run's four initial evaluation batches took
235.5027, 70.8779, .5248 and .4660 seconds. A one-second, 10 ms interval
[live stack sample](native-live.sample.txt) of PID 58597 during the next
training warm-up has the main thread's 79 samples beneath
`dynamo__custom_eval_frame` / `dynamo_call_callback`. The sample confirms
active graph capture at that instant; it is not a whole-run profiler.
Sampling briefly perturbs the cold measurement and is disclosed here.

The existing component timers independently locate most of the earlier
warm-up cost in sentence scoring: 66.6% immediately before step 5, versus
3.8% of the five measured warm training batches. Before this item the
corresponding fractions were 72.0% and 6.2%. Those records include all
batches and late captures, without dropping outliers. See
[before-step-5 timing](../native-prepair/batch-timing.json) and
[before-item timing](../native-before/batch-timing.json).

After the final run completed, its
[batch timer](../native-after/batch-timing.json) measured 1108.7173 seconds
for the first training batch: 790.0417 in sentence scoring, 147.2504 in
greedy composition and 163.1762 in explore composition. The next training
batch took 2.4305 seconds. The first two evaluation batches plus the first
training batch account for about 97% of the 1457.98-second process time.
This locates the dominant cold-start cost; it does not equate every second
of those batches with compiler execution. The five measured warm batches
took 1.9949, 2.3234, 1.9542, 11.6625 and 2.3569 seconds. The 11.6625-second
batch contains 10.0579 seconds of explore composition and stays in the
reported mean. No outlier was removed to improve throughput.

The exact-round-trip tests have a separate, repeated training cost:
`test_mm20m_xor_exact_roundtrip` trains a fresh model for 160 epochs.
Across the completed tables each attempt's process median was about
108–111 seconds and peak memory about 4.7 GiB. A current attempt spent
111.27 seconds inside the test and 116.48 seconds in the worker, so worker
startup is a small fraction. Thirty attempts per HEAD/candidate table,
with memory reservations around 6.3 GiB each under the 24 GiB aggregate
guard, limit their concurrency to three. Lighter work fills other slots.

The completed table campaigns took 1463.0, 1230.3, 1220.1, 1193.3 and
1201.1 seconds (before changes and after steps 1–4), totaling 105.1 minutes.
These tables must be repeated in order under the hand-off. Affected
compiled/native checks add substantial cold work: the native unlabelled
chooser test took 1257.61 seconds; the final compiled packed-sentence test
took 564.22 seconds. The required MM measurements each perform 900 updates.

At diagnosis, the 14-core, 36 GiB host had roughly ten model workers
using 98–100% of one core each. The aggregate CPU snapshot was 73.52% user,
10.45% system, 16.02% idle. Swap-in/out counters were zero, though about
4.6 GiB was compressed. This supports CPU/capture and guarded scheduling
as the observed bottlenecks, rather than a stalled worker or swapping.
The runner deliberately limits numerical-library threads to one per
worker to avoid oversubscription.

The most useful future performance investigation is reconstruction graph
capture and specialization reuse. It is distinct from reducing the
declared training budget or weakening a gate. No such optimization was
inserted into this source-matched learning campaign.
