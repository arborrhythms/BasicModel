# Item 6.9 — §23 corrections (2026-10-03)

Uncommitted candidate in the existing working tree, starting at published
`d679df2b5a2665d72a99ca4b6dfd47c1ba048e99` with the previous rounds' edits retained.
The completed [§22 receipt](../2026-10-03-item6-9-free-readback/README.md) remains
the measurement baseline. Acceptance measurements are deferred to the next
check-in by [§23.7](../../plans/2026-09-29-item-6-9-xor-grammar.md#237-decided-alec-2026-10-03).
No HEAD measurement, ten-run gates, attribution, full sweep, MM_grammar runs,
native run, configuration change, new seed, threshold change or commit.
The per-process guard remains 8 GiB and the timeout 1,800 seconds.

## Corrections

- The byte loss uses log-sum-exp of the emitting candidates and the null
  candidate's uniform byte mass, minus the log normalizer over all candidates
  and the null. No target-probability floor remains. The wrong-winner probe
  passes with live gradients to the recovered leaf, true code and competing
  code. Temperature stays .1; the relative-error baseline stays log 256.
- The saved compiled/eager discrepancy begins in float32 dot, norm and cosine
  reductions. [Intermediate values](numeric-before.json) and the
  [generated kernel](numeric-before-kernel-0.py) expose different reduction
  orders. The maximum logit difference is .001953125 on these large-activation
  inputs, producing a probability difference .00006058067. The original
  complete byte test differed by .00004605204. Byte scoring now accumulates
  in float64 where supported through its log sums, returning the input dtype.
  The original cost and gradient tolerances pass unchanged. MPS retains its
  supported input precision; no MPS measurement was requested this round.
- The integer derivation prepass retains unary ancestry on binary operands.
  It applies each operand's chain to candidate codes before pair composition,
  then the inverse traversal undoes that chain on its selected operand.
  No numerical witness is used. Candidate selection still uses activation ×
  cosine × priming; bank order breaks equal-residual pair ties within the
  shortlisted set. All four saved §22 roots recover their original leaves
  and recompose to their roots, in eager and fullgraph Inductor execution.
- Free input search is explicit. Answer generation's sum/chunk inverse keeps
  its balanced split. Both previously failing generation cases pass unchanged.
- Falsity takes the catalogue `Ops.union` rather than grammar mean disjunction.
  All six truth-loss cases pass, including agreeing and unknown propositions.
- The one-batch reconstruction chooser test passes unchanged, including its
  requirement that an MLP parameter move.

## Clause-closing journal

`ClauseJournal.finish_clause` reads the selected operation's actual input
values (two for a binary, one for a unary), its output, and the selected
reference addresses/relation flags. These build clause-node points, native
references and relation meanings from the operation that actually ran,
including scoped reference resolution; closing does not replay the operator
against a later state. The numerical values retain their gradient edges.
The unused second unary input is zero. Rule sequence and operand positions
remain the inverse's only operation record; neither input reconstruction nor
the answer reads these temporary numerical frames, and no witness offsets
were reintroduced.

A path without sentence transactions cannot call the open-sentence program
capture and has no closing reader. It now carries only empty placeholders
for these fields: no per-operation writes, gathers or scatters. The two
unconsumed writer helpers were removed. The saved failing probe demonstrates
real operations with populated, unread frames before the change. It now
passes, as do compact-journal gradient preservation and packed-row isolation.

## Ports and probes

[ports.json](ports.json) preserves complete old and new bodies for all six
ports. The reconstruction feature-growth and checkpoint observers read
`momentum_buffer`, retaining their state, identity, shape, enlistment and
frozen-row assertions. The direct byte fixture uses the sentence's real
primed bank. The synthetic chunk fixtures supply the leaves and intermediate
compound required by their free searches. The reverse-chooser fixture
constructs the grammar's mean parent. Its exact-pair assertions are unchanged.

The sparse-field probe found four admitted rows with native feature
memberships and exactly eight nonzero evidence poles. The physical and field
slices both happened to be `[0,8]`; changing the address observer alone did
not repair it. [The saved fields](sparse-failure-fields.json) show admission,
not a code update, supplies those memberships. The fixture now admits and
freezes its input, establishes zero memberships, then runs the same repeated
cutover reads and unchanged zero-evidence/code-stability assertions. It also
requires the admitted feature table to be nonempty.

[before.zip](before.zip), [source-final.zip](source-final.zip) and the matching
JSON manifests preserve the source. Each focused probe has its source hash,
command, complete log, process status, peak memory and duration under `probes/`.
The previous fourteen failing case records are copied under `probes/prior/`,
with classifications in [prior-failure-notes.json](prior-failure-notes.json).
Probe-development setup errors and the unsuccessful first field-slice port
are retained rather than hidden. The original numerical inverse body is also
replayed from the saved archive against the completed saved-root fixture.

## Minimal measurements

All three requested measurements completed once on the frozen source. No
retry or follow-on campaign was run.

| Measurement | Outcome | Wall time | Peak process tree memory |
|---|---|---:|---:|
| One shared XOR_grammar training (400 epochs) | Class fails; reconstruction 2/4, fails; both fail | 103.26 s | 0.631 GiB |
| MM_ladder serial supplied-answer learning, 40 epochs | Best checkpoint accuracy 0.953125 < 1.0; fails | 67.94 s | 1.559 GiB |
| MM_ladder free round trip, 3 epochs | Exact read-back 0/64; fails | 237.81 s | 4.333 GiB |

The serial supervised result is retained without a retry. It reaches the
learning assertion, with no setup, capture or optimizer exception. The
three-epoch free round trip also reaches its unchanged exact-match assertion;
the subsequent where-recovery assertion is not reached. Its saved decoded
strings and inverse tensors remain under `free-roundtrip/observations/`.
These single outcomes do not estimate reliability or establish a regression
against another unseeded run.

The XOR bars consume the **same model identity**, recorded twice in
[xor/observations.jsonl](xor/observations.jsonl). The answers and read-backs are:

| Input | Target | Answer | Grammar read-back |
|---|---:|---:|---|
| hello world | 0 | 0.4999422431 | hello world |
| hello there | 1 | 0.4999524951 | hello there |
| loving world | 1 | 0.4999217987 | world world |
| loving there | 0 | 0.5000132918 | there there |

Class accuracy is 1/4; final MSE is **0.2500203133**, in the **at ¼** band.
The checkerboard contrast is **0.00008124113**. All four inverses are
available. Neither bar is changed or waived. Full values are in
[one-training-summary.json](one-training-summary.json).

## The one-training audit

The [full audit](audit-summary.json) and raw files under `xor-ownership/`
include single-state objective gradients, every recorded optimizer
anchor/code gradient and displacement coordinate, start/end dictionary
cosine matrices and XOR-root singular values, VQ cluster sizes, trial
selection, costs and per-sentence derivation stability. The receipt-local
observer makes no RNG draws or optimizer changes and performs no extra
training. Wrong-read-back gradients are additional `autograd.grad` reads
of the existing trial graph with `retain_graph=True`; parameter `.grad`
buffers and optimizer steps are untouched.

Across **1,682 wrong word read-backs** in 885 recorded training calls, all
1,682 have nonzero recovered-leaf gradients and all 1,682 have nonzero bank
code gradients. These are raw byte-loss gradients before log-256
normalization:

| Gradient | Minimum norm | Median norm | Maximum norm |
|---|---:|---:|---:|
| Recovered leaf | 0.0003164775 | 0.3003770 | 91.01747 |
| Candidate codes | 18.66415 | 69.62440 | 163.29147 |

The audit has **zero ownership conflicts**, 18 active and 55 inactive
parameters, across 1,200 backwards. Of 800 reconstruction updates:

| Parameter | Nonzero gradient steps | Nonzero displacement steps | Largest gradient norm | Largest displacement norm |
|---|---:|---:|---:|---:|
| Active concept dictionary | 800 | 800 | 0.644612 | 0.0500914 |
| Chooser stop anchor | 768 | 40 | 2.29149e-7 | 6.15241e-9 |
| Chooser reduce anchor | 768 | 97 | 9.55973e-8 | 1.16144e-9 |
| Chooser apply anchor | 768 | 47 | 2.32283e-7 | 6.45239e-9 |

Momentum can continue a previous direction when the current gradient
reverses; the signed displacement/gradient cosines and coordinate arrays
are preserved, including positive cosines. Tiny anchor updates still
frequently round to zero. This is distinct from the previous receipt's
zero displacement for every code and anchor.

The active dictionary moved by Frobenius norm **1.56566**. Its mean squared
off-diagonal cosine rose from **0.100388** to **0.260874**, and its code norms
ended between approximately **1 and 1.32404**. The `world`/`there` cosine
rose from **0.363732** to **1.0** at float32 reporting precision. The four
centered root singular values changed from
`[1.1908001, .7976577, .5429389, 4.60581e-8]` to
`[2.0302548, .0001844342, .0000736664, 5.23906e-8]`.
The saved values show nearly parallel word codes and nearly coincident
root pairs in this failed run. Both stages' `vq.cluster_size` stay
`[1,1,1,1,1,1]`; EMA refresh remains disabled.

| Sentence | Epochs on modal derivation | Distinct derivations |
|---|---:|---:|
| hello world | 385/400 (96.25%) | 2 |
| hello there | 381/400 (95.25%) | 2 |
| loving world | 389/400 (97.25%) | 2 |
| loving there | 400/400 (100%) | 1 |

Full rule/arity/operand-position sequences are in
[xor-ownership/derivation-stability.json](xor-ownership/derivation-stability.json).
The final greedy training relative reconstruction error is **0.06028973**,
weighted **0.00602897** at the unchanged reconstructionScale .1. Nonzero
reconstruction gradients and a lower byte cost have not established either
acceptance bar in this one run.

## Verification and disposition

- **Eight previous sweep failures:** all pass in the focused correction
  probes, with the six original production/chooser assertions unchanged and
  the two authorized optimizer-state observers ported to momentum.
- **Six previous moved failures:** four fixture ports pass; the two learning
  cases above remain red. Assertions, epochs and bars are unchanged.
- **25 distinct focused regression/port cases pass** across the incremental
  probes, including the eager/Inductor saved-root checks, byte loss and
  gradient parity, real chooser movement, truth penalties and journal use.
  Setup mistakes, the initial skipped slow fixtures and superseded failed
  probes remain saved. The slow fixtures were subsequently executed on CPU
  with RUN_SLOW=1; their skips are not counted as passes.
- **Collection:** 5,009 cases collected, exit 0; this is not a sweep.
- **Documentation links:** 270 checks pass.
- Each of the three measurements has `complete.json` confirming the frozen
  source match. [corrections.patch](corrections.patch) records this round's
  production/test changes against the saved starting candidate. Data and
  grammar configurations match the starting snapshot exactly.

The dead byte gradient, unary omission, reviewed production regressions and
ports are addressed. The two unchanged XOR bars and two MM_ladder learning
cases are still red in these requested single measurements. No acceptance
campaign or attribution was started. **Stopped for review; nothing committed.**
