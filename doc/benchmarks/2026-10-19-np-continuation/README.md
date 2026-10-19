# NP continuation — preparation and no-learning regrade

**Continuation pending Item 5 landing.** Item 5 is an uncommitted candidate
at the time of this receipt. Per Alec's sequencing instruction, no adjective
lesson, definite-adjective lesson, final evaluation or default sweep has run
in this continuation. This is a preparation receipt, not a completed landing
or a new learning result. The [Item 5 record](../../plans/2026-10-19-item5-start.md)
records its remaining review/publication work.

The [parts-of-speech specification](../../specs/2026-10-18-parts-of-speech.md)
§8 defines textual grounding. Empty intersections and products on fresh
object-concepts are expected. Norms below and in the runner's reports have
no pass threshold.

## Regrade: archived rows only

`output/np-description-flow-20261018/results.json` was re-summarized with
`summarize_stage` and its `summarize` dependency read directly from committed
source at **180288c32fd1276540e4a82d20a61f1dc8d9c63e**. No model was
constructed or restored. There were zero new training presentations, updates
or held-out readings. The four archived rows are unchanged.

| Measurement | Original archived verdict | §5a regrade |
|---|---|---|
| Exact brackets (≥.9) | 4/4, pass | 4/4, pass |
| Singular determiner-NP occurrences (≥.95) | Not applicable | Not applicable: bare descriptions |
| Duplicate writes / collisions | 0 / 0 recorded | 0 / 0 recorded; no determiner NP denominator |
| Exact answers (=1) | 4/4 recorded; absent from original gate map | 4/4, pass |
| Unlowered control | 0/4, failed gate | 0/4, report |
| Original intersection program requirement | 0/4, failed gate | 0/4, report; kept `lower` ×4 |
| Unexpected occurrences | 4 recorded | 4, report |
| Overall | **Failed** | **Passed** |

The `regraded` record retains the original gates verbatim, the original
`passed` flag and protocol gate declaration, the grader commit, and the
unchanged checkpoint hash. The effective `results.json` protocol declaration
now names §5a only; its original `protocol.json` is untouched. Supplementary
operator, stamp and norm reports are copied from the archived rows. The
[original producer receipt](../2026-10-18-occupation-flow/README.md) remains
historical evidence of the original failed protocol.

All four held-out readings kept `lower`, with description/result orders
0→1 and closed meaning norm **0.20000003278255463**. The three archived
training intersections had operand norms **0.20000003278255463** each and
output norms **0**, **0.11547006666660309**, **0**. No product was kept in
that archived lesson. None of these norms is a gate.

## Prepared source and certificates

`flow_mask` now returns 3 for `product` and `sum`, and the POS table declares
product's combined input carrier. One added certificate checks that both
words in a kept product occupy its output and the enclosing lower's head.
The existing catalogue certificate also checks both masks. The existing
exact artifact restoration certificate is parameterized over `np_definite`
and the newly registered `np_description`; its weight and occupation-state
assertions are retained.

Protocol gates are exactly `exact_brackets=.9`,
`one_order1_occurrence=.95`, `duplicates=0`, `collisions=0`, and
`exact_answers=1`. Unexpected occurrences, operation observations, lowering
stamps, kept operators and meaning norms are reports. The NP occurrence
verdict retains the occurrence, duplicate and collision requirements; the
shared stage-4e unexpected-occurrence assertion is retained separately.
Boundary checks require 18/20 brackets, 19/20 occurrences and 20/20 answers,
and reject any duplicate or collision. No seed was selected for assertions.

The runner rejects a dirty working tree before creating a run directory or
constructing a model. All three future invocations must use committed source
after Item 5 lands. This pass changes no stage-3, clock, chain, surface,
chooser or kernel implementation, and performs no identity run.

Syntax checks, two pure grader checks, and product/sum mask checks passed
without model calls. The native kept-product and artifact-restoration
certificates have **not been executed** on the pending source. The single
default sweep is **not started**; it remains reserved for the final committed
source. The eventual enclosing commit is the source archive.

## Lesson results and budget

Fixed protocol: seed **20261010**, LR **.003**, B4, two passes. Each lesson
restores its predecessor's training artifact before held-out evaluation.
Stop on the first failed gate, with no rerun or extension. Final evaluation
is conditional on both lessons passing and has no updates.

| Lesson | Training / paired updates used | Brackets | Determiner-NP occurrences | Answers | Reports / disposition |
|---|---:|---|---|---|---|
| Archived description regrade | 0 / 0 new | 4/4 | Not applicable | 4/4 | `lower` ×4; 4 extras; 0→1 stamps; pass |
| Adjective from description | 0 / 0 of 24 / 12 | Pending | Pending | Pending | Not started; Item 5 prerequisite |
| Definite adjective from adjective | 0 / 0 of 24 / 12 | Pending | Pending | Pending | Not started |
| `--stage final` | 0 / 0 | Pending: 14 held-out | Pending: 10 determiner NPs | Pending: 12 questions | Not started; no updates authorized |

Previously used: **60 presentations / 32 paired updates**, comprising the
two bare-NP lessons (12/8), the discarded adjective branch (24/12), and the
archived description lesson (24/12). This continuation has used **0/0** and
zero new held-out readings. The authorized cumulative ceiling remains
**108/56**, leaving **48/24**. Conditional evaluation allowance is 4 adjective
readings, 4 definite-adjective readings, then the final 14. No continuation
output directory or trained artifact has been created.

## §6 POS tables

The tables below are the **archived description lesson's before/after**,
read from its saved `before.json` and `after.json`. They are unchanged by
regrading. They do not stand in for continuation before/after measurements;
those will be added here only after the lessons run. Counts are actual kept
argument occupations; an unobserved default argmax is not a learned class.

| Word | Before argmax | Before nonzero counts | After argmax | After nonzero counts |
|---|---|---|---|---|
| a | `lower_argin1` | definite_argin1=2, lower_argin1=4 | `lower_argin1` | definite_argin1=2, lower_argin1=4 |
| bird | `definite_argin2` | definite_argin2=1, definite_argout1=1, lower_argin2=1, lower_argout1=1 | `lower_argin2` | definite_argin2=1, definite_argout1=1, intersection_argin2=1, intersection_argout1=1, lower_argin2=6, lower_argout1=6 |
| black | `adverb_argin1` | ∅ | `lower_argin1` | definite_argin1=1, lower_argin1=5 |
| cat | `definite_argin2` | definite_argin2=2, definite_argout1=2, lower_argin1=1, lower_argin2=1, lower_argout1=1 | `lower_argin2` | definite_argin2=3, definite_argout1=3, intersection_argin2=1, intersection_argout1=1, lower_argin1=2, lower_argin2=4, lower_argout1=4 |
| chases | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| dog | `lower_argin2` | definite_argin2=1, definite_argout1=1, lower_argin1=1, lower_argin2=2, lower_argout1=2 | `lower_argin2` | definite_argin2=1, definite_argout1=1, lower_argin1=4, lower_argin2=5, lower_argout1=5 |
| horse | `definite_argin2` | definite_argin2=1, definite_argout1=1, lower_argin2=1, lower_argout1=1 | `lower_argin2` | definite_argin2=2, definite_argout1=2, intersection_argin2=1, intersection_argout1=1, lower_argin2=5, lower_argout1=5 |
| is | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| large | `adverb_argin1` | ∅ | `lower_argin1` | intersection_argin1=1, intersection_argout1=1, lower_argin1=4, lower_argin2=1, lower_argout1=1 |
| near | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| on | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| runs | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| sees | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| sleeps | `adverb_argin1` | ∅ | `adverb_argin1` | ∅ |
| small | `adverb_argin1` | ∅ | `intersection_argin1` | intersection_argin1=2, intersection_argout1=2, lower_argin1=2, lower_argin2=2, lower_argout1=2 |
| the | `definite_argin1` | definite_argin1=3, lower_argin1=1, lower_argin2=2, lower_argout1=2 | `definite_argin1` | definite_argin1=3, lower_argin1=1, lower_argin2=2, lower_argout1=2 |
| white | `adverb_argin1` | ∅ | `lower_argin1` | definite_argin1=1, lower_argin1=4, lower_argin2=1, lower_argout1=1 |

| Table | Argument 1 | Argument 2 | Occupant cosine >.9 | Relation |
|---|---|---|---:|---|
| Before | `definite_argin2` | `definite_argout1` | 1.000000000 | head flow |
| Before | `lower_argin2` | `lower_argout1` | 1.000000000 | head flow |
| After | `definite_argin2` | `definite_argout1` | 1.000000000 | head flow |
| After | `lower_argin2` | `lower_argout1` | 1.000000000 | head flow |

These are the only pairs above .9 among the 4,186 archived pairs at each
endpoint. The full pair tables remain in the archive's `before.json` and
`after.json`. The two high cosines follow declared head flow, not a class
merge. The recorded intersection output counts equal the combined input
counts; product was unobserved.

All 17 reported words have **zero individual columns and noun_prior=false**
both before and after. Noun-prior records are identical across the two
endpoints. `a` and `the` retain their earlier labels and counts; `bird`,
`cat` and `horse` change from `definite_argin2` to `lower_argin2`; `dog`
retains `lower_argin2`. Unused words remain unobserved. This is a report of
the archive, not a gate or a new learning claim.

## Artifacts

| Role | Path | SHA-256 |
|---|---|---|
| Preceding definite lesson | `output/np-definite-restored-20261018/trained.ckpt` | `b52392d9fc05734fd6a546ef9547d642e4be9d33f77a350eda1c13335340cafa` |
| Regraded description; continuation prerequisite | `output/np-description-flow-20261018/trained.ckpt` | `33b2d3e87b9b16844a42c69881cc2a195a2a9cd8aa731e81e58bd3e3c21cf3e0` |

`np_description` is registered in `test/fixtures/phase-artifacts.json` with
this receipt. Its checkpoint hash was checked before and after regrading;
the checkpoint was never written. Adjective and definite-adjective
checkpoints and the final evaluation are pending, so no hashes or results
are claimed for them.
