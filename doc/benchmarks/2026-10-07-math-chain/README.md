# MM_math_chain — October 7, 2026

Status: **learning gate not met; stopped for Claude's review, uncommitted**.
All thirty declared math attempts ran once. All failed before completing
their first epoch; none reached held-out or beyond-range evaluation. The
green mechanism checks do not establish learned chaining.

The accepted 6.2 mechanism was committed and pushed at
`e43638a747e373b3b10343c642f76cd1d0757ec1`; WikiOracle's submodule bump was
committed and pushed at `1970001e6ad61f720784fc6192dfd453276a164a`.
The [acceptance record](../2026-10-07-item6-2-repair/acceptance.json) states
that MM_query_reasoning's 300-epoch completion is not learned chaining and
that 6.5's learning gates remain pending the million-sentence checkpoint.

The [protocol](protocol.json) fixed thirty unseeded learning attempts and
four held-out pairs before measurement. The source and original measurement
helpers stayed frozen. [Summary](review-summary.json),
[per-run results](summary.json), and [failure analysis](failure-analysis.md)
record the result without replacing any training.

The live [thinking spec §10](../../specs/2026-10-07-thinking.md) supplies the
learning step; its [starting revision](spec-at-start.txt) is retained.

The new [configuration](../../../data/MM_math_chain.xml) uses opaque number
words and a one-slot batching sentinel, with text labels held outside the
chooser. [The corpus](../../../bin/math_chain_corpus.py) provides forty counting
documents and 117 worked problems per epoch: 1,449 sentences with answers.
Every successor from zero through nineteen is stated in both forms. The
four held-out pairs are (two, seven), (four, eight), (seven, three) and (nine,
six). Beyond-range evaluation uses (eleven, two), (twelve, three), (two,
eleven) and (three, twelve). Evaluation documents end at the question.

[The driver](../../../bin/MathChainTraining.py) groups equal-length documents
without dropping a tail or splitting a document between streams. All model
updates use the existing sentence owner and shared chooser. No arithmetic
executor or recursion definition is supplied. The former September numeric
fixtures remain historical regression fixtures; this configuration does not
load them or use their answer vectors.

The assertion loss reads only equality operands in the selected numerical
journal. The answer cost and verifier read the completed binding by lexical
identity; nearest-numeral classification and a generated answer string are
not success criteria. The verifier also checks committed inference rows and
their licensing fact and predecessor references. Observation hooks run after
the corresponding model result is fixed.

The development full sweeps are retained as `sweep-01` through `sweep-04`.
The fourth completed all 5,586 cases successfully; the final sweep also
covers the subsequent zero-budget control. Removing
the shared helper's fixed seed exposed fixture assumptions: independent
counterfactual initializations, a replacement memory too small for a legal
episode, legacy parity records mixed with ordinary thought history, and an
unforced surface question added to a fixture that already injected its own
two questions. The repairs preserve the existing assertions and isolate
their declared inputs; they do not choose an initialization by outcome.
The wider batched smoke exposed two runtime boundaries: relational rows
cannot enter the numerical NP reference bank, and loss-side text encoding
must not resize or clear the live discourse, attention or thought streams.
The proposal menu now validates the executor's existing reference contracts.
Target encoding stops before attention and source binding and disconnects
the stem's source-owner sizing callback for the duration of the read.
The successful driver smoke consumed all 51 sentences and observed eight
questions. None of those eight opened an episode or supplied a correct
binding; its three episodes occurred elsewhere. Non-tied cost observations
are plumbing evidence only, not a learning result.

The zero-budget control keeps ordinary VP composition active and consumes
no attention work. This is tested through model loading and an optimizer
pass, not by substituting an answer or arithmetic executor.
All development receipts remain beside the final measurement.

## Final checks

| Check | Result |
| --- | --- |
| Default sweep | Green: all 5,587 selected cases completed; 5,302 passed, 284 skipped, one XPASS, no failures or compile retries |
| Explicit unseeded thinking gate | 57/57 passed, including the per-row answer certificate |
| Standing sum control and floor | 10/10; MSE 0.2499999851–0.25 |
| Standing XOR classification and reconstruction | 10/10 on both, from the same ten trainings; MSE 8.88e-16–5.27e-10 |
| Standing MM | 10/10; best MSE 0.14295247–0.19777325 |
| Thought in the standing thirty | Zero calls and zero open episodes |
| Final document checks | 335/335 passed; live tracked changes pass `git diff --check` |

The [standing summary](standing-summary.json) preserves individual bars.
The [source freeze](measured-source/freeze.json) covers 732 files at digest
`cc0444cc4d920904c9b3454b4c235e332352f12aacd71d405e7ae029634245e3`.
[Final integrity checks](integrity.json) match the measured source, all 189
original helpers and the supplemental launcher helpers. The protected
standing test/configuration files and both earlier 6.2 receipts are unchanged.
The live thinking spec still matches the revision read at the start.
The [delivery record](delivery.json) records the final checks and confirms
that the learning-step changes remain uncommitted.

The standing launcher initially omitted two inherited audit modules. Both
startup failures occurred before model construction or training. Their logs
and entropy states remain in `standing-launch-failure` and
`standing-launch-failure-02`. The [supplement](measurement-supplement.json)
supplies the identical inherited modules. The continuation restores each
original sum run's saved entropy state; it does not draw a replacement start.
[All ten state comparisons](standing-entry-check.json) match both the second
startup-only launcher and the first actual training. This is launcher
continuation, not a repeated model training.

## Declared math measurement

Ten fresh entropy starts were each replayed across the three conditions,
with the same saved fifty shuffled presentations. No seed was chosen and
no failed training was retried. [Initial chooser comparisons](paired-start-check.json)
match across all three conditions for each of the ten starts. The configured
budget was 64, and the VP control used zero. Each run required fifty epochs.

| Condition | Attempts | Completed trainings / epochs | Held-out rows evaluated | Held-out and beyond-range accuracy |
| --- | --- | --- | --- | --- |
| Answer and expectation | 10 | 0 / 0 | 0 | Unavailable: evaluation not reached |
| Expectation only | 10 | 0 / 0 | 0 | Unavailable: evaluation not reached |
| Zero attention budget | 10 | 0 / 0 | 0 | Unavailable: evaluation not reached |

The ≥9/10 held-out binding bar is **not met**. The completed-chain count after
training and the VP/expectation-only accuracy comparisons are unmeasured.
In particular, a failed run is not assigned zero-percent accuracy.

| Partial training observation | Answer and expectation | Expectation only | Zero budget |
| --- | ---: | ---: | ---: |
| Sentences completed | 832 | 760 | 2,131 |
| Explicit math questions observed | 72 | 72 | 189 |
| Those questions opening an episode | 0 | 0 | 3 |
| Correct committed bindings / complete chains | 0 / 0 | 0 / 0 | 0 / 0 |
| Thought credit comparisons, all ties | 9 | 9 | 78 |
| Nonzero raw thought-chooser gradients | 0 | 0 | 0 |
| Equality trials with a verb-change gradient | 0 | 0 | 10 |

These are first-epoch observations, not evaluation results. The three
zero-budget question episodes spend no work and produce no inference step.
Other sentences open 12, 12 and 153 episodes respectively; the attention-enabled
conditions each record 226 `not` and 13 `query` acts overall. All recorded
thought credit comparisons tie exactly. The ten equality-gradient observations
occur in the fourth zero-budget start and do not establish learned successor
accuracy. The mechanism's equality certificate only proves its constructed
gradient path.

**Final chooser movement is unavailable for all thirty failed attempts.**
The driver saves it after completed training/evaluation, which none reached.
Initial chooser snapshots and zero partial thought-credit gradients do not
establish that the shared chooser's other parameter updates were zero.

| Failure | Attempts | Where |
| --- | ---: | --- |
| Stale eight-row priming surface at a two- or three-row batch | 18 | Nine starts in each attention-enabled condition |
| Incompatible taxonomy operand reference | 2 | Start nine in both attention-enabled conditions |
| Effective LTM capacity of 1,024 exhausted | 8 | Zero-budget starts 1–5 and 7–9 |
| Unresolved relational row rejected at ordinary sentence commit | 2 | Zero-budget starts six and ten |

`truthMaxEntries=65536` does not override the inherited `ltmCapacity=1024`.
The [failure analysis](failure-analysis.md) links the other failures to their
source boundaries and states the limits of the saved traces. All raw attempts,
including their partial question/credit observations and exceptions, remain
under `math-trainings/`. The source was not repaired during this measurement.

This candidate does not establish MM_math_chain or isolate thinking's
contribution to generalization. The 6.5 learning gates remain pending the
million-sentence checkpoint. Stop here for Claude's review before any
learning-step commit.
