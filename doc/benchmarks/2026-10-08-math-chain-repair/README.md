# MM_math_chain repair — October 8, 2026

Status: **incomplete repair candidate, awaiting Claude’s review; no commit**.
All thirty declared math attempts ran once. All failed during epoch one;
none completed an epoch or reached held-out or beyond-range evaluation.
This is not a learning result. The first receipt remains intact.

The delivered source passed the regression checks:

| Check | Result |
| --- | --- |
| Full sweep | 5,604/5,604 unique cases completed: 5,319 passed, 284 skipped, one XPASS; exit 0 |
| Thinking gate | 57/57 passed, unseeded guard active |
| Standing thirty | Sum 10/10; XOR class and reconstruction each 10/10 on the same ten trainings; MM 10/10 |
| Thought entry in the standing thirty | Zero calls, open episodes and bound calls |

`standing-summary.json` retains every bar and the shared-training check.
`unique-case-outcomes.json` counts unique selected nodes. The raw sweep has
four call reports for the one `test_flags_match_expected` node; the raw
outcome count in `summary.json` includes those duplicate reports.

The frozen protocol declared fifty epochs for each of ten fresh starts,
replayed across answer-and-expectation, expectation-only and zero-budget
conditions, with four held-out and four beyond-range pairs. Every condition
has ten retained failed attempts and zero completed trainings. The failures
are 27 compose legality assertions, two out-of-range action indices after
the replay override, and one sentence-reader occupancy exception.
`failure-analysis.md` separates the observed traces from the suspected
menu/replay cause. `failed-batches.json` reconstructs the next scheduled
batch from saved data; no failed learner was replayed to obtain it.

Held-out binding accuracy, beyond-range accuracy, learned chain success,
and final chooser displacement are **unavailable**, not scored as zero.
The frozen driver saves final chooser displacement only after training and
evaluation finish. The learned-chaining bar cannot be assessed from these runs.

Partial first-epoch question observations, across the ten attempts in each
condition:

| Condition | Questions observed | Questions still open | Open references remaining | Episodes opened | Correct committed bindings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answer and expectation | 59 | 5 | 7 | 6 | 0 |
| Expectation only | 61 | 15 | 15 | 18 | 0 |
| Zero attention budget | 0 | 0 | 0 | 0 | 0 |

A complete first epoch would contain 1,170 questions per condition across
ten runs. None is complete here. The zero-budget attempts failed before
an observed question. The frozen observer records open references remaining
**after** an episode, not the binder’s initial open-reference count.
Ordinary corpus questions did open referent slots and episodes in both
nonzero-budget conditions, including role 2 in paired start 1. This does
not establish the requested answer-filling certificate or learning.

All 24 observed question episodes spent the full budget of 64. One inference
row was observed on questions with answer credit and three with expectation
credit alone. All four were produced by `query`, with no bound numeral and
no witnessing references; none passed the chain verifier. These are partial
training observations, not held-out chain measurements.

Across all observed closings, the raw thought-credit observer recorded
45 nonzero chooser gradients with answer credit, 43 with expectation credit
alone, and zero with budget zero. It recorded 144, 181 and 20 ties respectively.
Equality trials with a nonzero verb-shift gradient numbered 0, 1 and 2.
These gradients do not establish final chooser displacement or a learned
chain. `math-results.json` contains the per-condition and per-run reductions;
`summary.json` and each run’s `observer.json` retain the underlying reports.

The candidate adds backward `interpret.bind` choices in the shared operation
softmax, pending-reference storage and address-preserving filling, shared
LTM retrieval and batch-sized priming reads, explicit capacity precedence,
and operand filtering. Pending-name postings are derived from the shared
LTM; no per-stream registry or other document-owned state was added.
`capacity-audit.json` records the 1,048,576-row configuration budget from
50 presentations of 1,449 sentences and 7,621 word/punctuation units, worked
steps, DEF rows and margin. A full store still raises. `ltmCapacity` sizes
the durable store; `truthMaxEntries` sizes the truth activation view.

The green suites do **not** certify every requested repair. The ordinary
`the answer is five.` filling case, fully bound copula case, and the specified
pending two-sentence case in both orders are not certified. Their constructed
mechanism tests are not substitutes. The isPart filter validates occupied
operands but still permits grammar-open forms, which is narrower than the
requested both-operands condition. See `certificate-coverage.md`.

The source is frozen in `measured-source/`: 733 files, canonical source-map
SHA-256 `f944bef3dc2aef40f59d0e2dda35f4b2d14f9f2609ce1f6632abd5d2d35cf36e`.
The full sweep, thinking gate and standing thirty match that source. All
192 recorded measurement dependencies and seven frozen contract hashes
match. All 2,393 files in the first receipt match, with no added or removed
receipt files. `integrity-after-measurement.json` records these checks.
The corpus, driver, verifier, observer and protocol were preserved.

All ten preparations were made before any declared training. Their fresh
RNG archives are distinct; each start’s three saved initial chooser tensors
are bit-identical. See `paired-preparation-audit.json` and
`paired-initial-chooser-check.json`. No seed was supplied, and no failed
attempt was retried or replaced.

Development failures are retained: the two full development sweeps had
40 and 2 failures; the subsequent delivered-source sweep is the green one
reported above. `ordinary-construction.json` records one separate unseeded,
evaluation-only read of eight corpus documents, with no optimizer; all eight
question rows stayed closed in that construction. It is not a learning
trial or a successful ordinary-path filling certificate. The starting and
measured source archives preserve both sides of the repair.

No acceptance or learned-chaining claim is recorded. The earlier 6.5 learning
gates remain pending the million-sentence checkpoint. Work stops here for
Claude’s review before any commit.
