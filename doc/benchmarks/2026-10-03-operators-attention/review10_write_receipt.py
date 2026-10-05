"""Render the one review receipt from the completed, saved measurements."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'review10-measurements'
result=json.loads((OUT/'summary.json').read_text())
audit=json.loads((OUT/'audit-summary.json').read_text())
counts=result['counts']


def yes(value):return 'pass' if value else 'fail'


text=f'''# Decoder exploration, operators and 6.8 — held for Claude's review

This is the one receipt for the work from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`, in the existing working tree.
The review is [6.8 plan §10](../../plans/2026-09-27-item-6-8-one-attention.md).
**Nothing is committed or pushed. Work stops for Claude's review.**

The three requested changes are implemented and the one declared measurement
is complete: XOR_grammar **class {counts['class_pass']}/10, reconstruction
{counts['reconstruction_pass']}/10, joint {counts['joint']}/10**; MM_xor
**{counts['mm_pass']}/10**; sum control **{counts['sum_pass']}/10**.
The class, reconstruction and sum counts are below §22's **9/10, 5/10 and
10/10**. No failed run was retried, omitted or repaired after seeing its outcome.

The accepted 6.9 baseline remains **class MSE .1147481948, reconstruction 0/4,
zero ownership conflicts**. MM_xor remained red through 6.9 §17 and the operators
stage; the current ten-run measurement is under 6.8. The previous single-run
.321/.115 class comparison was **not a regression finding**: the initial round's
single runs span .0013 to .875. §10 held the candidate for this measurement,
not as a rejection. The [original receipt text](README-before-review10.md) is
preserved as history; this receipt corrects that interpretation.

## Three changes and focused checks

1. Input narrowing, compose, ordinary thought, anticipation and the shared
   decoder now sample the departure action from the eligible policy
   probabilities after excluding greedy's action, at unit softmax scale.
   The eligible round is still sampled uniformly, the prefix replayed and
   suffix greedy. The original policy supplies training credit. Both walks
   are costed before updates at the same parameters; only a strictly lower
   owner cost keeps explore, and ties keep greedy.
2. The eager audit records first-step decoder logits, STOP-over-each-binary-undo
   margins, the gradients reaching those logits on real training backwards,
   and margins before/after the actual optimizer update with the same parent
   held fixed. Diagnostic gradient queries are excluded. It adds no RNG draw,
   backward, optimizer update, learning-rate or budget change.
3. Frozen interpretation resolves known definitions without reserving words,
   minting objects, adding decompositions, finishing pending definitions or
   refreshing testimony. No capacity changed. The small evaluator check reads
   the first three unchanged manifest items with all 16 candidates and both
   controls on `MM_grammar_wording.xml`, preserving the complete definition
   inventory, reservations, identity placement and capacity. This is a
   mechanism check. The trained NanoChat gate waits for item 4's checkpoint.
   The [acceptance probe](nanochat_acceptance_probe.py) is retained; fresh
   BasicModel scoring remains stopped.

The [focused check](probes/review10-focused/run.log) passed **129 tests, with
one existing skip**, in 18.88 s. It includes eager and compiled decoder replay,
strict ties, all shared selectors, frozen known/unknown reads, the three-item
evaluator and a numerical gradient/update check for the audit. The
[one-batch observer check](probes/review10-observer/run.log) also passed: four
decoder first-step records over the same three actual backwards, zero owner
conflicts. The production measurement uses its original 400-epoch budget.

## One measurement on the frozen source

[Plan](review10-measurements/plan.json), [complete record](review10-measurements/complete.json),
[raw summary](review10-measurements/summary.json), and
[source archive](review10-source/manifest.json). Exactly 30 trainings: ten
XOR_grammar, ten MM_xor convergence runs, ten sum controls. Each XOR training
supplies **both original pytest bars**; the observer verifies the same model
identity for their two consumers. The tenth XOR training supplies the audit.
No additional training was used for it.

Class still requires all four labels and MSE < .05. Reconstruction still
requires all four word multisets with no unavailable inverse; the historical
pytest selector containing “50_pct” is unchanged. Bands follow §20.5 and the
prior receipt convention: “at 0” is MSE < .05, “at ¼” is within .02 of .25,
then remaining values below .25 are “between” and those above are “above ¼”.
These bands do not replace either gate.

| Run | Final answer MSE | Band | Correct labels | Recovered inputs | Class | Reconstruction | Joint |
|---|---:|---|---:|---:|---|---|---|
'''
for row in result['xor']:
    text+=f"| {row['run']} | {row['mse']:.9f} | {row['band']} | {row['correct']}/4 | {row['recovered']}/4 | {yes(row['class_pass'])} | {yes(row['reconstruction_pass'])} | {yes(row['joint'])} |\n"
text+=f'''
Bands: **{counts['xor_bands'].get('at 0',0)} at 0,
{counts['xor_bands'].get('at 1/4',0)} at ¼,
{counts['xor_bands'].get('between',0)} between,
{counts['xor_bands'].get('above 1/4',0)} above ¼**.

MM_xor uses its unchanged convergence test: best loss < .20 in at most 200
epochs, with the original early stop. The sum control changes only the
grammar to `sum.forward(S, S)` and retains the 400-epoch runner and §22 criterion:
absolute checkerboard contrast ≤ 1e-4 and the class bar not met. `sum` is the
decided mean. [The entire control diff](review10-measurements/sum-control.patch)
and all endpoint answers are saved.

| Run | MM_xor best MSE | Loss calls | MM bar | Sum final MSE | Sum band | Sum contrast | Sum bar |
|---|---:|---:|---|---:|---|---:|---|
'''
for mm,s in zip(result['mm'],result['sum']):
    text+=f"| {mm['run']} | {mm['best']:.9f} | {mm['calls']} | {yes(mm['passed'])} | {s['mse']:.9f} | {s['band']} | {s['contrast']:.9f} | {yes(s['sum_bar'])} |\n"
text+='''
The sum control fails its retained additive criterion; these outcomes are not
relabelled as passes. After the below-comparison counts, a source inspection
identified a possible explanation: the current answer path includes
`PrimedSymbolReader`'s content-dependent scorer and learned consume gate.
A sum-only grammar therefore does not constrain the complete answer path to
remain affine after learning. The audited XOR run records the reader's gate
and scorer as answer-owned writers. This is a source-level observation, not
measured attribution of the sum outcomes to that module. No reader was disabled,
no threshold changed and no attribution training arms, full sweep or native
run were added to this round. The failed counts and this distinction go to
Claude for review before any further change.

## Tenth XOR run: ownership, margin and walk audit

[Raw events](review10-measurements/xor-10/ownership/events.jsonl),
[audit summary](review10-measurements/audit-summary.json),
[ownership](review10-measurements/xor-10/ownership/ownership.json), and
[walk audit](review10-measurements/xor-10/ownership/walk-audit.json).
**Zero ownership conflicts over 1,200 backwards; 29 active and 58 inactive
parameters.** Both outer trials are costed at the same parameter versions.
The same training produces the final class/reconstruction results in row 10.

There are **1,600 batched first-step logit records** (both decoder paths inside
both compose trials). The audit reaches those logits on **800 optimizer
steps**, with the **400 answer-only steps explicitly recording no decoder
walks**. It records gradients for both selected and discarded decoder graphs;
the latter have zero gradient. There are 3,200 nonzero STOP-minus-undo gradient
differences out of 6,400 row/path observations for each binary action, and
6,400 nonzero fixed-parent margin changes. Duplicate greedy/explore parents
are retained so every graph can be traced, not counted as independent samples.

| First-step STOP margin | Mean at epoch 1 | Mean at epoch 400 | Range at epoch 400 | Mean fixed-parent change per update |
|---|---:|---:|---|---:|
'''
for binary in (0,1):
    first=next(row for row in audit['epochs'] if row['epoch']==0 and row['binary']==binary)
    last=next(row for row in audit['epochs'] if row['epoch']==399 and row['binary']==binary)
    row=audit['binary'][str(binary)]
    text+=f"| STOP − binary undo {binary} | {first['margin']['mean']:.6f} | {last['margin']['mean']:.6f} | {last['margin']['min']:.6f}–{last['margin']['max']:.6f} | {row['fixed_parent_change']['mean']:.9f} |\n"
text+='''
The gradient **does move the margin** in this audited run. The margin first
rises, then falls; it stays positive for every observed first-step parent.
Greedy therefore chooses STOP first in all 3,200 training walks and ends with
one-word readbacks. This distinguishes the result from an immobile margin;
it does not establish that a larger budget would solve it. The rate, optimizer
and budget remain unchanged. The
[plot](review10-measurements/decoder-margin.png) ([SVG](review10-measurements/decoder-margin.svg))
is rendered only from saved observations.

| Walk active in XOR | Explore kept | Strict-selection violations | Consecutive kept-path stability |
|---|---:|---:|---:|
'''
for name,row in audit['walks'].items():
    text+=f"| {name} | {row['explore_wins']}/{row['walks']} | {row['strict_violations']} | {row['stable_pairs']}/{row['stability_pairs']} = {row['derivation_stability']:.6f} |\n"
text+='''
The decoder's .610763 stability is descriptive alongside the initial round's
.511890, not a paired attribution to sampling. Thought and anticipation are
not activated by this XOR corpus; their mechanisms are covered by the focused
tests, with no invented training counts. Per-row modal paths, every epoch's
margin/gradient distributions, code geometry and coordinate displacement
arrays remain in the audit.

## Preserved initial-round evidence

The original decoder → operators → 6.8 sequence and every intermediate outcome
are retained in [README-before-review10.md](README-before-review10.md) and their
original folders. In particular:

| Initial-round check | Saved outcome (before the §10 repairs) |
|---|---|
| Default suite | 4,829 passed, 285 skipped, 1 existing xfailed; all 5,115 selected cases completed; 123.65 s |
| XOR_exact | Both original CLI checks: MSE 0, reconstruction 4/4 |
| Grounded and word-boundary XOR | Twelve checks at exact zero |
| Reading/global capabilities | 46 passed |
| Four shipped reference configurations | Finite forward and learned-reader gradient checks passed |
| Open-read whole forward | .14147 s versus .05459 s (2.59×); already assigned to todo item 1 |
| Historical single XOR runs | MSE .031250, .001309, .874984, .039967, .320719; each reconstruction 0/4 |

These prior results are not represented as a new full-suite/native validation
of the frozen §10 source. The failed weekly slow receipt remains failed and
all earlier fresh-model NanoChat failures, including store exhaustion at item
136, remain saved. There is no conference freeze.

## Sources, complete ports and failures before repair

[Before-repair source](review10-before/manifest.json),
[measurement source](review10-source/manifest.json),
[complete old/new test ports](review10-source/test-ports.json), and
[contracts audit](review10-contracts.json). There are **161 complete old/new
test ports** from published HEAD, including all prior ports, the non-Python
fixture and both new review helpers. The original gate files, reconstruction
round-trip tests, bounded runner, Makefile, pytest configuration and frozen
manifest are byte-identical to published HEAD. Existing assertions in the
two additionally changed test files are unchanged. The prefix/suffix decoder
test now declares one eligible alternative, preserving all its assertions;
the new controlled-draw probes test several eligible actions. No seed calls,
XML seeds, production capacities or guards changed in this review round.
The initial round's nine fixed-word-whole row changes remain separately
disclosed in the original receipt.

| Saved probe | Outcome and disposition |
|---|---|
| [review10-before](probes/review10-before/run.log) | Six failures reproduce deterministic departure selection and frozen admission before repair |
| [review10-repaired](probes/review10-repaired/run.log) | 84 pass, one failure exposes the old deterministic fixture assumption; saved before the fixture port |
| [review10-focused](probes/review10-focused/run.log) | 129 pass, one existing skip after the port; compiled decoder included |
| [review10-observer](probes/review10-observer/run.log) | One real-batch audit wiring check passes |

All thirty measurements retain their original process exit, log, final answer
arrays and assertions. The measurement source is checked throughout the run.
Each worker keeps the **8 GiB / 1,800-second** guard; at most three one-thread
workers reserve **24 GiB**, within the unchanged **28 GiB** aggregate ceiling
and existing CPU headroom. There are no additional working trees (old prunable
Git registrations are retained), no seed selection, no retries and no commits.
'''
(HERE/'README.md').write_text(text)
