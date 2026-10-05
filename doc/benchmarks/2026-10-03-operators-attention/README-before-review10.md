# Decoder exploration, operators and 6.8 — uncommitted review candidate

This work continues the one existing tree from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`. The sequence was decoder exploration
under 6.9 §26.3, the decided operators update, then 6.8-1 including §§7–9.
There is no conference freeze. **Nothing is committed or pushed. Claude reviews
before any commit. The candidate is not accepted: the XOR_grammar no-regression
condition remains unmet.**

The accepted 6.9 record remains **class MSE .1147481948, reconstruction 0/4,
zero ownership conflicts**. Class **9/10** and reconstruction **5/10** remain
prior measurements, not replaced by this work. MM_xor stayed red through the
operators stage; the new 6.8 measurement passes its unchanged bar.

**Alec's scope correction, 2026-10-03:** stop fresh BasicModel scoring, retain
the acceptance probe, and check the evaluator once on a small configuration over
a few manifest items. The NanoChat gate waits for item 4's trained checkpoint.
Claude owns the corresponding plan/item-4 clarification. No further fresh
BasicModel gate was run after this correction.

## Review disposition and validation

| Check | Recorded result |
|---|---|
| Final default suite | **4,829 passed, 285 skipped, 1 existing xfailed; all 5,115 selected cases completed; exit 0** |
| MM_xor, unchanged loss < .20 within 200 epochs | **Pass: .1826079786**, 49 observed loss evaluations |
| XOR_exact, both original 64-epoch CLI tests | **MSE 0, reconstruction 4/4**, twice |
| Grounded XOR and word-boundary variants | **All 12 pass**, exact-zero errors; no unrelated percept events |
| Final XOR_grammar, one shared 400-epoch training for both original gates | **MSE .3207185981, 3/4 class labels, reconstruction 0/4** |
| Final XOR ownership | **0 conflicts**, 29 active / 58 inactive parameters, 1,200 backwards |
| Retained reading/global capabilities | **46 tests pass**, including their slow mechanisms |
| Four shipped reference configurations | **All pass** finite forward and learned-reader gradient checks |
| Word evaluator mechanism | **Pass**, first 3 unchanged items, all 16 choices, both prefix controls, `MM_grammar_wording.xml` |
| Full trained NanoChat gate | **Deferred to item 4**, not scored here |

[Final suite summary](final-validation.json) and
[full bounded result](attention-default-final/result.json) record 123.65 seconds,
14.03 GiB aggregate peak and 3.74 GiB maximum worker peak. Limits remain 8 GiB /
1,800 seconds per worker, 28 GiB aggregate, up to ten workers, with the existing
CPU-headroom policy. The earlier weekly slow record remains failed; this default
suite and the focused slow checks do not replace it. Original skip/xfail gates
remain in force.

[Controls](controls-final/run.log) retain all 15 original slow-test results and
[their raw observations](controls-final/observations.jsonl). They are single
measurements, not a new campaign. The final class result is worse than the
accepted baseline and is not replaced by a favorable rerun. The reconstruction
bar remains red. This prevents claiming a completed acceptance of 6.8.

## Decoder and walk exploration

The decoder now evaluates greedy plus one legal departure. Exploration replays
the greedy prefix, excludes its choice at one eligible round, then follows the
best available suffix. Both complete walks are costed under identical parameters
before training. Only strictly lower reconstruction cost keeps exploration;
ties and unavailable departures keep greedy. The objective remains free-byte
error / log(256) plus antipode / log(2), with the existing common scale. Only the
selected reconstruction graph supplies its gradient. Evaluation remains greedy;
no compose journal or teacher derivation enters operation inference.

6.8 extends the same rule to input narrowing, ordinary thought, anticipatory
thought and output generation. Thought restores only its declared effect owners
between trials, then retains the winning history. Anticipation holds both
prior-only forecasts until the next observation supplies the return; previews
publish no history or policy credit. The larger prior spend is reserved against
the shared attention budget. Output selection uses answer error without giving
answer gradients ownership of decoder parameters. Tests cover strict ties,
missing alternatives, parameter versions, selected graphs, state restoration,
shared meters and compiled decoder replay.

The final training's [walk audit](xor-final/ownership/walk-audit.json) reports:

| Active walk | Explore kept | Consecutive kept-path stability |
|---|---:|---:|
| Input attention | 0 / 1,600 | 1,584 / 1,596 = .992481 |
| Compose | 687 / 1,600 = .429375 | 1,367 / 1,596 = .856516 |
| Reconstruction decoder | 1,148 / 3,200 = .358750 | 1,636 / 3,196 = .511890 |

There are zero strict-selection violations. Thought and output-generation
comparisons are not active in this XOR configuration; their contract tests are
not presented as measured learning fractions. The greedy decoder still chooses
STOP and emits one word per sentence. Exploration works mechanically, but it
has not established the expected agreement with compose's inferred operations.

## Operators

The implementation declares operand kinds, reads/writes, head, polarity, order,
relation and inverse properties. Alias dispatch uses implementation properties;
face permissions only restrict writes, and the effect ledger rejects duplicate
writes to a subsystem in one round. Field Booleans operate within brackets;
order-dependent grammar consumes identified symbols across brackets.

`sum` is a mean with its witness inverse; `chunk` stays additive. Intersection
uses the decided signed minimum with zero silence. The old all-ones universe
interpretation in the interim checkpoint is not retained as a competing
identity. `non` withdraws the expressed pole without affirming its opposite and
has no faithful inverse; `not` exchanges poles. Compound composition selects
the existing head's sigma cases with the modifier's observed field, preserves
both poles and refolds them. Inverse search uses the unchanged bounded primed
operand bank. Verb keeps the I1 head, and adverb applies repeatable multiplicative
gain with its witness inverse.

`exist`, `true`, `lookup` and unused binary `symbolize` retire. `what` supplies
unified content retrieval; `quantize` and `arma` stay thought-only. Equality's
question has a fixed grammatical identity without allocating an equality VP
row; closing still writes two canonical part rows. Deferred projections, ICA,
dimensional lift/lower, modality and morphology stay in their recorded future
scope. The [catalogue implementation section](../../specs/2026-09-29-operator-catalogue.md#12-implementation-candidate-2026-10-03-uncommitted)
and regenerated operator diagram describe the candidate.

## One attention and retained capabilities

A bounded typed bracket table carries intervals, validity, space, level,
completion and spent work. The first open read takes no gradient step. Native
paired evidence sets divide/descent/gloss eligibility; a Boolean rewrite changes
the next reading, and children read their own parts. The stop is pinned at word
extents, repeated unknown surfaces share one descent witness, and the selected
bracket drives the mereological scope handoff. The input walk is compared before
native admission using an immutable reconstruction target, so omission cannot
make its target disappear.

One registry covers input, STM, LTM, part, whole and symbol spaces. The chooser
uses the priming prior `max_v cos(key,row_v) * boost_v`; recall shares the meter.
The answer-owned learned reader consumes detached keys through a gate initialized
at zero. These replace the retired ReadingAttention/GlobalAttention modules and
flags. Word expectation produces a discrete candidate distribution with observed
history in training, self-history in inference and the shared negative-image
column. Sentence expectation retains its existing structured cycle. Byte/row
levels are declared off. Old mode/order/budget switches are rejected; binding
and inventory dimensions have separate names.

The fixed `word` whole uses the reader's maximal ASCII letter runs. Whitespace,
punctuation and digits separate words; digit and punctuation percepts remain
available to the numeric controls. Nine WholeSpace inventories were explicitly
expanded to nine rows to hold eight existing learned properties plus the fixed
whole: [exact old/new capacities](fixed-word-capacity.json). This is a declared
capacity change, not a claim that every dimension stayed unchanged. No other
capacity, seed, quality bar or resource guard was changed.

The four native reference checks use their actual shipped dimensions:
[MM_reading](native-reading-review/measurement.json),
[MM_global](native-global-review/measurement.json),
[MM_qa](native-qa-review/measurement.json), and
[MM_20M_grammar_reading](native-grammar-reading-review/measurement.json).
They use the same eager autograd loop body and explicitly set the consume gate
to .5 to test scorer gradients. This is a mechanism fixture, not a learned gate
claim. MM_qa's invalid TruthSet form is repaired. The focused capability receipt
is [46 passing checks](probes/attention-capability-slow-after/run.log).

## All XOR_grammar measurements

Each completed row is one unseeded, source-matched 400-epoch training, consumed
by both unchanged gates. These stages are not paired random initializations and
cannot establish a causal comparison. No completed statistical failure was
repeated to select a better outcome.

| Source stage | Class MSE | Class labels | Reconstruction | Decoder explore kept |
|---|---:|---:|---:|---:|
| Accepted 6.9 | .1147481948 | 4/4 | 0/4 | absent |
| [Decoder exploration](decoder-xor/summary.json) | .0312503717 | 4/4 | 0/4 | 1,790/3,200 |
| [Interim operators checkpoint](checkpoint-xor/summary.json) | .0013091945 | 4/4 | 0/4 | 1,094/3,200 |
| [Completed operators stage](operators-xor/summary.json) | .8749836335 | 3/4 | 0/4 | 1,627/3,200 |
| [Early 6.8 stage](attention-xor-02/summary.json) | .0399673232 | 4/4 | 0/4 | 477/3,200 |
| [Final learning measurement](xor-final/summary.json) | .3207185981 | 3/4 | 0/4 | 1,148/3,200 |

All five completed trainings have zero ownership conflicts and zero decoder
selection violations. The earlier incomplete `attention-xor-01` attempt failed
in the observer on the removed `schedule_context` argument; its source, error
and repaired observer remain saved. The final failed class measurement shows
four distinct roots (centered rank three); dictionary mean squared off-diagonal
cosine changes .054415 → .069890. There is no demonstrated root collapse or
ownership conflict. The final greedy compose reading differs from the last
kept training reading. That is an observation, not an established causal repair
or a waiver of no-regression. Geometry remains diagnostic, with all arrays in
the ownership receipts.

## Descriptive measurements and evaluator scope

The [profile](attention-profile-review/measurement.json) records whole-forward
mean .141470 seconds against .054590 in the
[before measurement](open_read_baseline.json), about 2.59×. Input staging averages
.041390 seconds; the field-open operation .00006265. The old `_sentence_prelude`
and new input-stage timing boundaries differ, so their open-only times are not
a like-for-like comparison. These are short CPU measurements, not throughput
acceptance bars.

On the fixed probes, without optimizer updates and after one native admission
exposure, the XOR set has both-rate 0/4 and categorical discrimination 0 at word
and sentence levels. FineWeb has both-rate 3/68 at both levels, with discrimination
.194962 at word and .060127 at sentence. The sentence pole measure pools the
native word readings; its code metric uses the sentence carrier. Byte and row
are disabled and unmeasured. No input was unread.

The requested evaluator mechanism check uses `MM_grammar_wording.xml` unchanged,
including its existing seed 931. The first three frozen manifest items retain all
16 choices and both controls. Scores are finite and ranks lie in 1–16; this is
not a learning claim. [Probe](probes/nanochat-small-mechanism/run.log),
[raw measurement](nanochat-small-mechanism.json): 4.65 seconds, .71 GiB peak.
The full gate is deferred to item 4's trained checkpoint.

Earlier fresh BasicModel attempts are retained, not scored as partial gates:
`nanochat-final`, `-02`, `-03` exceeded memory before the padding repairs;
`-04` stopped after eight items at a divide no-op; `-05` stopped after 136 at
definition-store capacity. The acceptance trace and
[divide failure](probes/divide-edge-before/run.log) precede the repair, and the
[54-case affected check](probes/divide-edge-after/run.log) passes afterwards.
A both-pole first word now splits at its right edge instead of repeatedly
requesting its left edge; it remains ineligible for gloss/descent. The normal
pure neighbors stay reachable. Native/word padding is omitted only during eager
computation, with complete carrier shapes and matching results checked. The
frozen-learning flag still permits native definitions to accumulate; that
observed limitation is retained for the future trained evaluator, with no
capacity increase or speculative reset mechanism introduced here.

## Sources, failures and complete test ports

[Final source and whole old/new test ports](final-review/manifest.json) include
every changed/new test path, including the non-Python grammar fixture. The old
contents come directly from published HEAD, and the new contents include
imports, helpers, decorators and assertions. Earlier snapshot helpers omitted
non-Python test fixtures; the final archive explicitly supplements them. The
[tracked patch](final-review/changes.patch) is accompanied by the
[source archive](final-review/source.zip), so new files are not lost.

[Contract audit](contracts-audit.json) records zero changed Python seed calls or
XML seeds. The original XOR/MM gate files, reconstruction round-trip tests,
bounded runner, Makefile, pytest configuration and frozen manifest are byte
unchanged. [Measurement source bridge](final-source-bridge.json) lists the exact
files changed after each measurement; later padding, native-evidence and split
repairs are not misrepresented as retrainings of the saved XOR model.

All saved failing probes remain in [probes/](probes/). Earlier full default
runs and their source snapshots remain beside the final green suite. The exact
interim report is [README-operators-checkpoint.md](README-operators-checkpoint.md);
its incomplete status and provisional intersection discussion are historical.
The final candidate is ready for inspection, with the class no-regression failure
and unproven greedy operation inference explicitly unresolved. Stop here for
Claude's review before committing.
