# Validation runtime correction, September 30

Alec rejected the 3–4 hour workflow. The previous full sweep took **155 minutes**
before the separate affected-file and learning measurements were included.
Its runner used three workers and 644 fresh batches. The saved
[timing audit and failing scheduling probes](scheduling-before.json) show:

| Measurement | Recorded value |
|---|---:|
| Full-sweep wall time | 9,282 seconds |
| Summed worker time | 27,366 seconds |
| Summed test execution time | 24,165 seconds |
| Cases taking more than two minutes | 54 of 5,100 |
| Execution time in those 54 cases | 22,984 seconds, 95.1% |
| Longest case | 1,500 seconds |
| Initial round-5 thought/reasoning selection | 315 pass, 5 skip; 2,509 seconds |
| Final item-7 selection | 202 pass; 308 seconds |

The expensive cases are native training, backward and compilation integrations;
this campaign has not started a million-sentence training run. The audit names
every case above two minutes and keeps its recorded time and worker peak.
Increasing concurrency alone cannot make those individual cases fast. Ten
heavy workers together would require about 42.6 GiB at their prior peaks,
so blindly using the normal ten-worker setting would break the 24 GiB budget.

The old coordinator was stopped before it launched further affected files or
the sweep. Its ownership record is preserved in
[the scheduling-hold snapshot](active-final-validation-before-scheduling-hold.json).
The follow-up was paused during the correction. Existing healthy XOR/MM jobs
and all completed attempts are retained.

The [replacement scheduler](resource_schedule.py) uses the existing bounded
runner. It admits up to ten one-thread workers, reserves memory from prior
peaks with headroom, and batches lightweight cases to reduce import overhead.
Unknown cases reserve the entire 8 GiB worker allowance. Estimates govern
admission only: the original kernel/sampled limits, failure handling,
resource recycling and exact coverage checks still enforce the run.
No pytest skip, test body, seed, model configuration, assertion, threshold or
compiled/eager setting changed. Per-worker memory remains 8 GiB, the full pool
24 GiB, and each worker keeps the 1,800-second deadline. The affected-file pool
retains its 8 GiB aggregate allowance while the two existing XOR/MM workers run.

Five [scheduling checks](scheduling-probes/result.json) pass, including affordable
concurrency, memory backpressure, retained/recycled work and exact case/device
coverage. A [real bounded-worker integration](scheduling-integration/run/result.json)
also passes all five cases, retaining the 8 GiB limits.

The [historical replay](scheduling-projection.json), using all 5,100 prior cases,
projects **86 minutes** with 304 batches instead of 155 minutes with 644.
This is a scheduling estimate, not a measured speedup or a passing receipt.
The long native fixtures still need performance work before the complete suite
can become a short development loop. Item 7's model repair remains frozen;
this correction changes its validation harness only.

The [continuation](resume_scheduled_validation.py) adopts the already-running
measurements, then runs the affected files, graph gate and exactly one full
sweep. The [new harness manifest](scheduling-harness-source.json) records the
change while retaining the original source/harness manifests and all attempts.
Current progress is in [active-final-validation.json](active-final-validation.json).
The candidate remains unaccepted and uncommitted until Claude reviews it.

At 19:53 UTC, the scheduled thought/reasoning selection had completed in
**1,840 seconds (30.7 minutes)**, with 315 passes and the same five skips.
The earlier serial selection took 2,509 seconds (41.8 minutes) on the initial
AK source; this comparison is not a controlled performance benchmark of the
same source. Graph release also passed, in 22.5 seconds at 7.631 GiB.
All final XOR and MM measurements are complete. The one full sweep ran from
19:51 to 21:20 UTC with 5,122 cases. Its measured wall time is **5,378.6 seconds
(89.6 minutes)**, versus 9,282.2 seconds (154.7 minutes) previously: an observed
42.1% reduction. The estimate was 86 minutes. This comparison includes the AK
source change, ten new mechanism cases and twelve additional documentation-link
cases; it is not a controlled same-source benchmark.

The completed run used 365 worker batches, peaked at 5.263 GiB per worker and
19.169 GiB in aggregate, and had no resource stops or diagnostic reruns. It
finished every selected case once: **4,793 passed, 6 failed, 322 skipped and
one expected failure** ([summary](full-sweep/summary.json)). All 322 skips are
unchanged from the previous run. The six newly failing tests retain assumptions
about inventory-backed or independently allocated predicates; their causes
and unchanged assertions are recorded in [the findings](full-sweep/findings.md).
The receipt is red. The reduced full-sweep duration remains too long for a
short development loop; this scheduling change does not resolve the cost of
the long native training fixtures.

The later [accepted fixture-port landing](landing/README.md) runs only the six
cases, their six files, all item-7 cases and documentation links, as §25
requires. It preserves this sweep and its timings; no second full sweep or
learning campaign is launched.
