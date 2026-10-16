# Item 6: positional operands and paired singleton warm-up

The accepted corrected-warm-up measurement is pushed on BasicModel main at
`d5d300aed`; WikiOracle main is bumped and pushed at `7bf96de`. No learning
claim accompanies that landing. This next candidate is on
`codex/item6-orders-paired-warmup`; its measured production/driver revision is
`6a14ba151`. Commits are the archive.

The [driver](../../../test/paired_identity_experiment.py) starts this fresh
run from the declared seed. Earlier development runs stopped to repair
migration defects; their checkpoints and exposures are not reused
([run notes](diagnostics/run-notes.json)).

The pass follows [Orders.md](../../Orders.md) and plan §§12–13. Operand
positions are `argin1`, `argin2`, `argout1`. Reference requests and compose
results use the carried stamp and the operation's `order_delta`; the
name-to-order table is gone. Candidate references are matched at that order.
Mint/bind/generic behavior remains on operations, including field and
higher-order operands; generic scopes do not request individual admission.
A columns still collect order-1 individuals under the existing selected-mint
and recurrence gates. No order-2 dictionary, taxonomy design or POS lesson
is added, and no identity rule is retired.

## Are the order and mechanism certificates preserved?

| Check | Result |
|---|---|
| Relation over kinds, stamped order 3, with the configured storage rung at 1 | Pass: lexical read keeps order 3; the same test checks order 17 propagation, rejects an order-1 distractor, and prevents lowering from retaining its higher-order source identity. |
| Exact NOT/OR reconstruction | Unchanged exact certificate passes: `R == [0.]*4`. |
| Migration regressions | 96 targeted checks pass; synthetic readings now supply their operand stamps, and asserted field links retain their sparse edges. Another 51 checks pass after the source-order guard. Numerical assertions are unchanged. |
| Existing open/mint/bind behavior | Four affected math certificates pass after removing the remaining order-1-only binding guard. c reaches its carried `five` structure failure. |
| Unseeded paired-training zero-cost assertion | Passed in the full sweep, failed in a targeted run. Matched accepted-main/candidate probes are identical: seeds 20261010 and 0 have zero cost; seed 1 has `reconstruction.content = .0517327189` in one third-epoch trial on both revisions. Assertion unchanged; this is a pre-existing counterexample. |

The one full sweep at `e42505ae2` completed **5,796/5,796** cases in
808.00 s: **5,498 passed, 287 skipped, one XFAIL, ten failed**. Nine failing
test names are new versus the accepted gate-off sweep; c's test also failed
at an earlier, new missing-open-slot assertion. Targeted repairs restore
all ten to passing checks except c's previously reported terminal `five`
failure. There is no second full sweep and no claim of a clean full sweep
on the repaired commit. Cleared-cache overlap remains XFAIL, adverse to
accepted 6.1's XPASS. The two probabilistic batch-diversity tests are retired
from the default suite as authorized.

Diagnostics: [sweep](diagnostics/sweep.json),
[fixture repairs](diagnostics/fixture-repairs.txt),
[math repairs](diagnostics/math-repairs.txt),
[paired certificate on accepted main](diagnostics/paired-certificate-baseline.json)
and [candidate](diagnostics/paired-certificate-candidate.json), using the
[matched probe](diagnostics/paired_certificate_probe.py) (CPU, eager;
`d5d300aed` versus `86579057d`). The later source-order guard does not enter
that Boolean grammar's reference-free trial path. [Its targeted checks](diagnostics/source-order-guard.txt) pass.

## Does the trained artifact cover all eligible departure rounds?

**80/80 eligible compose rounds covered, four sentences out of four.**
The artifact has completed the paired warm-up (38 `optimizer.step` calls). The certificate
replays the native reservoir and walk selector over its frozen greedy menus:
**1,693 selection draws per sentence**, with no additional training exposure.
With E eligible decisions per sentence, W available walks, and N decisions
across the sentences, the bound is
`ceil(max(W*E) * log(N/1e-6))`; probability of missing any declared decision
is at most **1e-6**. This is coverage at fixed learned parameters, not while
optimizer updates change the menus. The sampler is unchanged.

| Sentence | Eligible rounds | Departed at least once | Missing |
|---|---:|---:|---:|
| `the black cat runs .` | 10 | 10 | 0 |
| `the white dog sleeps .` | 10 | 10 | 0 |
| `a cat sees a dog .` | 29 | 29 | 0 |
| `a dog sees a cat .` | 31 | 31 | 0 |

[Native round numbers and bound](diagnostics/coverage.json). The final
artifact check passes in 18.84 s. The local checkpoint is
`output/item6-orders-paired-accepted-code-20261015/warmup.ckpt`;
set `BASICMODEL_EXPLORATION_ARTIFACT` to that directory to run the
[`artifact_eval` certificate](../../../test/test_exploration_coverage.py).
The committed driver recreates it from seed.

## Do paired stages 1–2 retain the singleton atoms?

**20/24 carry their atoms; gate passes (20 required).** The accepted
observation-only warm-up carried 8/24. The final run uses seed **20261010**,
**76 presentations**, learning rate **.003**, and **38 `optimizer.step` calls**
through native explore/keep trials. It completes in **472.99 s**. The
original bare-word fields and order are unchanged; stages 3–4 are not
repeated. No A columns are minted in warm-up.

Cosine is signed, against each word's final meaning vector (threshold .9).
Zero rows have undefined cosine. Programs come from each word's final
original presentation, not a later replay. The four zero interruption words
were presented once; the budget is unchanged.

| Singleton | Cosine to word | Closed meaning norm | Kept program |
|---|---:|---:|---|
| `luma` | 1.000000 | 0.200000 | leaf → STOP |
| `nemi` | 1.000000 | 0.200000 | leaf → STOP |
| `tavo` | 1.000000 | 0.200000 | leaf → STOP |
| `suri` | 1.000000 | 0.200000 | leaf → STOP |
| `pako` | undefined | 0.000000 | generic × 19 |
| `feni` | undefined | 0.000000 | tense × 19 |
| `balo` | undefined | 0.000000 | generic × 19 |
| `runi` | undefined | 0.000000 | generic × 19 |
| `dax` | 1.000000 | 0.200000 | leaf → STOP |
| `wug` | 1.000000 | 0.200000 | leaf → STOP |
| `nep` | 1.000000 | 0.200000 | leaf → STOP |
| `zot` | 1.000000 | 0.200000 | leaf → STOP |
| `cat` | 1.000000 | 0.200000 | leaf → STOP |
| `dog` | 1.000000 | 0.200000 | leaf → STOP |
| `bird` | 1.000000 | 0.200000 | leaf → STOP |
| `horse` | 1.000000 | 0.200000 | leaf → STOP |
| `black` | 1.000000 | 0.200000 | leaf → STOP |
| `white` | 1.000000 | 0.200000 | leaf → STOP |
| `small` | 1.000000 | 0.200000 | leaf → STOP |
| `large` | 1.000000 | 0.200000 | leaf → STOP |
| `runs` | 1.000000 | 0.200000 | leaf → STOP |
| `sleeps` | 1.000000 | 0.200000 | leaf → STOP |
| `waits` | 1.000000 | 0.200000 | leaf → STOP |
| `moves` | 1.000000 | 0.200000 | leaf → STOP |

Full precision and row addresses: [singletons](diagnostics/singletons.json).
Original kept journals: [presentations](diagnostics/singleton-presentations.json).
This establishes better retained singleton content in this pass, not learned
identity or an isolated causal estimate for the protocol change: the order
migration also precedes the run.

## What does balanced order-1 identity learn?

**The identity lesson fails: zero columns minted, 0/146 bindings resolved,
and 0/20 row pairs resolved.** All unconditional correctness scores are zero;
conditional accuracy is undefined because nothing resolves. There is no
learning claim and no identity rule is retired.

The run completes **136 training documents** (16 vocabulary, 32 balanced
pronouns, 88 determiner/same-kind) and **166 held-out documents**, with no
native errors and process exit 0. It uses the accepted revision-3 corpus,
gate on, the fresh paired warm-up above, and the existing .003 optimizer
reset at the identity branch. Evaluation freezes online learning. Only the
balanced gate-on branch runs; the reversed-recency evaluation is not a
separate biased-training control.

**Columns and witnesses:** zero before identity, zero after training, hence
no minted columns or column witnesses. Across the 940 committed training
and evaluation clauses, zero select an individual and zero select mint
([native column list and admission counts](diagnostics/identity-results.json)).
The recurrence gate remains in place; this is not its retirement lesson.

**Kept programs:** 43/384 training clauses have nonzero closed meaning.
All **556/556 held-out clauses close at meaning norm zero**. Their programs
are exclusively `equal`: 138 clauses use it twice, 228 three times, 56 four
times and 134 five times. Every training and evaluation program, with its
clause address, closed norm and per-step norms, is in the
[kept-program diagnostics](diagnostics/identity-kept-programs.json);
[program counts](diagnostics/identity-audit.json) retain the full distribution.
The better singleton retention has not produced usable NP readings.

**Observer:** all **690/690 annotated references** have the named missing
reason `no_individual_request`; none is a sentinel. Candidate coverage is
zero in every binding group. No annotated leaf is reported
`leaf_not_admitted` in this run; the earlier 43-leaf finding was on a
different artifact, which is not retested here.
[Full reference provenance](diagnostics/identity-observations.json) and
[all group scores](diagnostics/identity-results.json) remain in files.

Ranks below are zero-based. Introduction role and first-mention position
remain confounded in the revision-3 SVO introductions; refresh mentions
cross recency and role. The neither-candidate control is unresolved, so
this run does not establish chance binding on that control.

| Group | Cases | Resolved | Correct |
|---|---:|---:|---:|
| All bindings | 146 | 0 | 0 |
| Family: document context | 8 | 0 | 0 |
| Family: neither candidate | 32 | 0 | 0 |
| Family: pronoun | 32 | 0 | 0 |
| Family: pronoun one | 4 | 0 | 0 |
| Family: pronoun three | 54 | 0 | 0 |
| Family: reversed recency | 16 | 0 | 0 |
| Candidate count: 1 | 4 | 0 | 0 |
| Candidate count: 2 | 88 | 0 | 0 |
| Candidate count: 3 | 54 | 0 | 0 |
| Target recency rank: 0 | 58 | 0 | 0 |
| Target recency rank: 1 | 70 | 0 | 0 |
| Target recency rank: 2 | 18 | 0 | 0 |
| Target introduction role: object | 58 | 0 | 0 |
| Target introduction role: subject | 88 | 0 | 0 |
| Target first-mention rank: 0 | 66 | 0 | 0 |
| Target first-mention rank: 1 | 62 | 0 | 0 |
| Target first-mention rank: 2 | 18 | 0 | 0 |
| Candidates / recency / role: 1/0/subject | 4 | 0 | 0 |
| Candidates / recency / role: 2/0/object | 16 | 0 | 0 |
| Candidates / recency / role: 2/0/subject | 20 | 0 | 0 |
| Candidates / recency / role: 2/1/object | 24 | 0 | 0 |
| Candidates / recency / role: 2/1/subject | 28 | 0 | 0 |
| Candidates / recency / role: 3/0/object | 6 | 0 | 0 |
| Candidates / recency / role: 3/0/subject | 12 | 0 | 0 |
| Candidates / recency / role: 3/1/object | 6 | 0 | 0 |
| Candidates / recency / role: 3/1/subject | 12 | 0 | 0 |
| Candidates / recency / role: 3/2/object | 6 | 0 | 0 |
| Candidates / recency / role: 3/2/subject | 12 | 0 | 0 |
| All row pairs | 20 | 0 | 0 |
| Row-pair family / expected rows: determiner/1 | 4 | 0 | 0 |
| Row-pair family / expected rows: determiner/2 | 4 | 0 | 0 |
| Row-pair family / expected rows: determiner conflict/2 | 4 | 0 | 0 |
| Row-pair family / expected rows: same kind/1 | 4 | 0 | 0 |
| Row-pair family / expected rows: same kind/2 | 4 | 0 | 0 |

The original-artifact relation trace 2/4, both order CE gates, free readback
0/4, MM grammar recovery, compiled retention, and c's marker-headed answer
remain open and are not remeasured here. The parts-of-speech pass still
waits for readings with NP structure.
