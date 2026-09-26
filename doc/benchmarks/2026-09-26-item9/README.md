# Item 9: expectation learning, three seeds

This receipt executes the remaining learning comparisons from
[item 9](../../../todo.md), on the
[corrected 9b runtime](../2026-09-26-item9b-corrections/README.md).
The [protocol](PROTOCOL.md) declares seeds 0/1/2, four independent training
conditions and 64 native optimizer updates per condition. A null remains a
null; Alec decides whether to change the experiment after this declared run.

All **12 native arms completed their 64 updates**, with the same 16 held-out
batches before and after. **The learning gates did not pass.** Ordered
prediction does not consistently beat both trained controls, and its FineWeb
discrimination is worse than reconstruction-only in all three seeds. Native
thought work could not be measured on these meanings. The separate frozen
text study has a failed wording prerequisite and no reasoning advantage.
Item 9 therefore remains open for Alec's decision; these measurements do not
authorize a larger run or establish learned utility.

## What is measured

Each native arm reads the same complete short sentences from 256 cached
FineWeb documents, batch size 2, with 16 held-out batches before training and
16 after. The ordinary native serial cells run through the CPU eager loop
dispatcher. This is a bounded learning comparison, not compiled throughput
or production-width training. No saved input trace supplies reconstruction.

Ordered prediction is compared with separately trained shuffled-context and
context-free predictors, plus a reconstruction-only arm. The shuffled arm
draws only from earlier contexts, excludes the current context, and preserves
choices through loss replay. Features and presence masks are both removed in
the context-free arm. The reconstruction-only arm disables prediction and
residual-policy losses. Initialization, source sentences, reconstruction
objective and optimizer count are shared.

Both feature MSE and role-presence BCE are reported. Current encodings show
the result of joint training; the fixed pretraining encodings help distinguish
predictor learning from changes in the representation. Target and predicted
variances are retained. Reconstruction byte cost and the fixed XOR/FineWeb
discrimination scores compare ordered training with reconstruction-only.
Here discrimination is measured on serial sealed-root codes, not a separate
parallel field or a new training objective.

The native thought comparison restores the same complete model and memory
checkpoint before gains 0 and 1. It asks about held-out observed meanings with
64 work units per question. Unsupported meanings, answer pairs, Brier error
against the presented positive assertion, steps and work are all retained.
Lower work at equally unsuccessful answers is not a useful-reasoning result.

A separate study repeats the earlier parsed-text grammar curriculum, then
freezes its encodings and trains the three prediction controls. Related and
unrelated continuations and the controlled thought experiment remain separate
from the native joint-learning results.

## Native comparison: gates not met

Smaller is better for both prediction losses. “Fixed” uses the same held-out
encodings captured before training, rather than each arm's changed encoder.
The [full reduction](native-summary.json) also retains initial losses,
prediction/target variances, reconstruction costs and every gate result.

| Seed | Training condition | Feature MSE | Presence BCE | Fixed MSE | Fixed BCE |
|---:|---|---:|---:|---:|---:|
| 0 | Ordered | .010828905 | .377924263 | .004733504 | .378191590 |
| 0 | Shuffled | .003012080 | .367002934 | .003129206 | .352928132 |
| 0 | Context-free | .002024853 | .570651770 | .002948680 | .570651770 |
| 0 | Reconstruction-only | .009637067 | .693044841 | .009771246 | .695668995 |
| 1 | Ordered | .009387901 | .392979264 | .318975359 | .140589759 |
| 1 | Shuffled | .008471242 | .375934094 | .337827593 | .126474231 |
| 1 | Context-free | .010801545 | .575426757 | .408293813 | .569974184 |
| 1 | Reconstruction-only | .046608318 | .678066492 | .421632230 | .625954211 |
| 2 | Ordered | .007340525 | .365777940 | .034593880 | .324858248 |
| 2 | Shuffled | .020444214 | .336288065 | .032407057 | .304440439 |
| 2 | Context-free | .003001086 | .588692307 | .029591579 | .588692307 |
| 2 | Reconstruction-only | .005279092 | .704194427 | .040050671 | .709089756 |

In seed 1, ordered prediction wins fixed-encoding feature MSE against both
controls. It still loses presence BCE to shuffled context. No seed passes
both losses on both encoding views. Each shuffled training arm makes 558
new context choices: 557 use an earlier, different context and one uses the
empty-bank start. Replay counts are 86, 62 and 93 respectively. Thus a
silently empty shuffle bank does not explain the result.

Byte cost must be no higher than reconstruction-only; discrimination CP
must be no lower. The tolerance is the declared 1e-6.

| Seed | Ordered byte cost | Reconstruction-only | Ordered XOR CP | Reconstruction-only | Ordered FineWeb CP | Reconstruction-only |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.227446672 | 1.238677055 | .110129297 | −.118326247 | −.041409969 | −.025600046 |
| 1 | 1.437000535 | 1.280316874 | .072465181 | −.050583243 | −.041296124 | .027357578 |
| 2 | 1.492445894 | 1.627180204 | −.121439874 | −.086365044 | .022368848 | .045290589 |

Reconstruction passes this comparison in seeds 0 and 2, but not seed 1.
FineWeb discrimination fails in every seed, and XOR fails in seed 2.
Neither a single favorable loss nor a mean across seeds satisfies the gate.

No sentence hits the 512-candidate basis limit; the largest basis is 34.
Incomplete reverse rows are nevertheless present and remain in the scores:

| Seed | Ordered | Shuffled | Context-free | Reconstruction-only |
|---:|---:|---:|---:|---:|
| 0 | 0 | 5 | 1 | 0 |
| 1 | 80 | 60 | 56 | 140 |
| 2 | 124 | 144 | 73 | 30 |

These are totals over the before/training/after phases. The input programs
encounter unavailable inverses for `lookup`, `quantize`, `what` and `arma`,
depending on seed and condition; the raw per-batch records retain each name
and count. Output-generation diagnostics use a separate operator catalogue.
These failures limit the reconstruction quality; they are not missing banks
or candidate-capacity exhaustion and were not removed from the comparison.

Each native ordered model supplies 25 question pairs. All 50 attempts per
seed are unsupported at both gains: “thought meaning has no grammatical VP.”
The controller therefore supplies no answer error or work measurement for
this comparison. This is **unmeasured reasoning utility**, not zero work or
a successful answer. The frozen text diagnostic below reaches the controller
but also shows no work advantage.

## Provenance and reproduction

Every final arm and the frozen-text study match the same 663-file source
snapshot, SHA-256
`86c9aa27bed5345a57e253a3f85668066789499555ba145fca0af960173f17d6`.
The [9b source archive and validation](../2026-09-26-item9b-corrections/README.md)
identify the exact runtime, configurations and tests. The
[provenance record](provenance.json) records commands, harness hashes,
process limits, timings and artifact hashes. The 12 native processes total
1,050.57 process-seconds and peak at 0.911 GiB; each had an 8 GiB cap.

Raw configurations, per-batch results and compressed logs are under
[seed 0](native/seed-0/manifest.json),
[seed 1](native/seed-1/manifest.json) and
[seed 2](native/seed-2/manifest.json). Temporary thought-restore checkpoints
remain in the local run directories; their hashes are recorded, but their
binary contents are not duplicated in this receipt. The
[diagnostic index](diagnostics/index.json) preserves unsuccessful and
superseded attempts. No seed, step budget, control or threshold was changed
after examining quality results.

From the repository root, run each seed with
`.venv/bin/python doc/benchmarks/2026-09-26-item9/run_native.py --out OUTPUT --seeds SEED --workers 1`.
Use a new output directory for each run. The source and harness are frozen
by the supervisor; simultaneous 8 GiB reservations must not exceed 24 GiB.
`summarize.py` takes the three run directories in seed order and `--out`.
The separate `text_probe.py` deliberately exits 1 after preserving the
failed wording prerequisite and completing its diagnostic comparisons.

## Harness diagnostics retained

The first supervisor attempted to call its process guard from worker threads;
the signal handler requires the main thread. That attempt produced no native
learning results. A subsequent partial ordered seed-0 run was interrupted for
roughly 48 seconds of graph-dispatch overhead per training batch. The eager
dispatcher uses the same condition/body tensors, as the prior native erosion
measurement did. Another attempt stopped because the output compiler backend
had not yet been set to `none`; it produced no complete run.

The first shuffled and context-free seed-0 attempts stopped at a runtime
`truncated` flag. The harness incorrectly described this as a candidate-limit
failure. The flag also includes unsupported inverse operations and malformed
reverse programs. Complete runs retain these rows, while reporting actual
per-sentence candidate-limit use separately. The later diagnostic identifies
the input compose catalogue separately from the output walk's operator list.
Repeating the complete seed-0 comparison after that logging correction gave
identical losses and prediction scores in all four arms; no seed, update
budget, control or threshold changed.

The source-matched serial baseline and packed/single parity measurements
precede interpretation of these results. Old packed-answer quality numbers
from before the saved-root fix remain invalid, as documented in Testing.

## Frozen text diagnostic: prerequisite failed, reasoning null

The old grammar fixture exhausted its 64-row percept reserve before learning.
It now declares 4,096 physical rows at construction; the old fixture relied
on runtime growth. Its seed, 1,000 generation updates, 8,000 composition
updates and assertions are unchanged. On this source the wording gate fails
on **5 of 56** held-out strings: four “also match” examples select *also* as
an operand, and “locomotives also are part of pistons” selects a constituent
in place of the intended object. Both the meanings and their generated
wordings are wrong. This is a red learning gate, not an expected failure.

After recording that unchanged assertion failure and verifying all 9,000
updates occurred, the measurement continues on the frozen, imperfect encoder.
It deliberately exits nonzero at the end, even though the diagnostic
comparisons finish. There is no passing-seed selection or replacement of the
failed gate. The [raw text results](semantic/text.json) retain the failure.

There are 1,400 training and 40 held-out paraphrase pairs. All 80 related and
unrelated continuations produce a selected meaning, which does not certify
that the selection is correct. The three predictors each take 1,000 updates.

| Seed | Ordered feature MSE | Shuffled | Context-free | Related remainder | Unrelated remainder |
|---:|---:|---:|---:|---:|---:|
| 0 | .0008396140 | .0134836389 | .0122223478 | 1.283323 | 1.734825 |
| 1 | .0008872874 | .0141136516 | .0122889960 | 1.290137 | 1.739119 |
| 2 | .0008369086 | .0144024631 | .0123950997 | 1.289953 | 1.737879 |

The related/unrelated signal persists in this diagnostic. Each thought
condition, however, answers all 40 questions with `(0, 0)` in every seed:
neither true nor false, Brier error **1.0**, mean work **17** at both gains.
There is no reasoning advantage. These results do not establish native joint
learning or restore the failed wording gate.
