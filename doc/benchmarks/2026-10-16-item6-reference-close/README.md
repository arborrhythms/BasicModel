# Item 6: reference at the close

The controlled closing regressions pass, including the four same-kind cases.
The native lesson remains unresolved: **0/146 bindings, 0/20 row pairs,
zero A columns and witnesses**, unchanged from the accepted balanced run.
There is **no learning claim**. Every held-out kept program uses only
`equal`; no kept reading supplies a reference request. Part 2 remains open.

The accepted positional-operands and paired-warm-up round is landed and
pushed on BasicModel main at `51f069c30`, with WikiOracle main bumped and
pushed at `f4d3139`. This next candidate follows plan §14/14.1 and
[Orders.md](../../Orders.md). Its measured source is `a3ec07fd0` on
`codex/item6-reference-close`. Commits are the archive.

The [driver](../../../test/reference_close_experiment.py) restores the
accepted paired warm-up, including the exact saved tensors, and runs the
same revision-3 balanced order-1 corpus with seed **20261010**, **136 training
documents**, **166 held-out documents**, and learning rate **.003**. The
warm-up's singleton gate was **20/24**. Gold labels enter only the grader.
Two development runs were interrupted for correctness repairs; neither
their state nor their exposures enters this measurement
([run notes](diagnostics/run-notes.json)).

## What changed, and what do the controlled certificates establish?

Reference is deferred through the kept derivation to its close. Mint and
bind retain fixed, separate semantics; the chooser selects their word
mapping. Candidates are accessible order-1 individuals from STM and cued
frames. The closing uses content explanation, predicate evidence and
trainable salience/form cues. Copula is excluded from the absorbed predicate.
It writes first-mention individuals immediately and revises accessible
occurrence references when later evidence supports a split or merge.
Occurrence content remains immutable. Order-2 kinds and their deferred
recurrence lesson are not implemented here.

| Certificate | Result and limit |
|---|---|
| Four same-kind cases after the fourth close | **4/4** parameterized cases pass: two mints stay distinct, repeated binding stays one individual, and ambiguous binding splits under later exclusive evidence. These use supplied operation requests and controlled evidence; they do not establish a learned word→operator mapping. |
| Fixed operations and bounded access | Mint always allocates; bind/pronoun never mint and remain unresolved without an accessible individual. Higher-order rows are excluded. Enclosing order-1 operations preserve the deferred request. |
| Revision | Later shared evidence merges accessible occurrences, including transitive groups and cued earlier documents. Inaccessible occurrences and stored content are unchanged. |
| Closing credit | Reference evidence trains the chooser at the close. Exact reconstruction still has zero R; ordinary expectation does not choose the kept trial. |
| Exact NOT/OR certificate | Unchanged `R == [0.]*4` assertion passes. |
| Native writes and checkpoint | First mention allocates only after a successful kept write; typed references and column slots round-trip. Empty closes preserve the original clause DAG and produce no reference gradient or metadata. |

The final source passes **12 closing regressions plus the unchanged math
c/e certificate (13 checks)**. The preceding targeted run passes all 12
closing cases, the unchanged expectation/keep separation check and the
exact NOT/OR certificate (14 checks). Logs:
[final closing checks](diagnostics/closing-regressions.txt),
[exact reconstruction and credit](diagnostics/exact-reconstruction-and-credit.txt).

The one full sweep, at initial source `48093d421`, completed **5,808/5,808**
cases in **785.64 s**: **5,494 passed, 288 skipped, one XFAIL, 25 failed**.
These are unique case counts. One node produced four passing reports;
the runner's 5,497 passing reports include those three extra executions.
Eleven new correctness failures were repaired and pass targeted checks:
nine shared a stride-zero gradient from an artificial empty reference cost,
one omitted the closing parameters from the base parameter registry, and
one cloned a shared clause DAG on an empty close, changing the stored query
structure. The unchanged c/e certificate passes on both accepted main and
the repaired source. The later enclosing-operation guard also passes its
targeted checks. There is no second sweep and no claim of a clean full
sweep on the repaired source.

**Twelve new failures remain in tests asserting retired contracts**:
nearest-left/context-stash binding (four), fixtures calling that API
(three), selected-mint admission (three), content-only access through a
synthetic owner without a native store (one), and gradients through the old
live binding slab (one). Their assertions and suite membership remain
unchanged pending clarification. **Two failures are carried**: c's
marker-headed `five` answer, and the paired zero-cost counterexample whose
`.0517327189` content charge was reproduced on accepted main in the prior
receipt. Neither numerical assertion is relaxed. Cleared-cache overlap
remains XFAIL, adverse to accepted 6.1's XPASS.

Full names, errors and classifications are in [the sweep](diagnostics/sweep.json).
Targeted evidence: [gradient repairs](diagnostics/gradient-repairs.txt),
[parameter and interleave checks](diagnostics/parameter-and-gradient-regressions.txt),
[math certificate on accepted main](diagnostics/math-accepted-main.txt),
[shared-structure repair](diagnostics/math-shared-structure-repair.txt).
The parameter/interleave log also records an import error in a new test;
the corrected test passes in the final closing log.

## What does the native balanced identity run establish?

The run completes **136 training documents** and **166 held-out documents**, with **0 native errors**, in **39215.33 s**. The process exits 0 and saves its final checkpoint. Evaluation freezes online learning; episodic first mentions may still allocate individuals. The reversed-recency evaluation uses this same balanced training branch.

**Bindings:** 0/146 resolved, 0/146 correct. The accepted run resolved 0/146 and got 0/146 correct. The full conditional scores and candidate coverage remain in [the results](diagnostics/identity-results.json).

| Binding group | Cases | Resolved | Correct |
|---|---:|---:|---:|
| all | 146 | 0 | 0 |
| family=document context | 8 | 0 | 0 |
| family=neither candidate | 32 | 0 | 0 |
| family=pronoun | 32 | 0 | 0 |
| family=pronoun one | 4 | 0 | 0 |
| family=pronoun three | 54 | 0 | 0 |
| family=reversed recency | 16 | 0 | 0 |
| candidates=1 | 4 | 0 | 0 |
| candidates=2 | 88 | 0 | 0 |
| candidates=3 | 54 | 0 | 0 |
| recency=0 | 58 | 0 | 0 |
| recency=1 | 70 | 0 | 0 |
| recency=2 | 18 | 0 | 0 |
| role=object | 58 | 0 | 0 |
| role=subject | 88 | 0 | 0 |
| first mention=0 | 66 | 0 | 0 |
| first mention=1 | 62 | 0 | 0 |
| first mention=2 | 18 | 0 | 0 |
| cell=1/0/subject | 4 | 0 | 0 |
| cell=2/0/object | 16 | 0 | 0 |
| cell=2/0/subject | 20 | 0 | 0 |
| cell=2/1/object | 24 | 0 | 0 |
| cell=2/1/subject | 28 | 0 | 0 |
| cell=3/0/object | 6 | 0 | 0 |
| cell=3/0/subject | 12 | 0 | 0 |
| cell=3/1/object | 6 | 0 | 0 |
| cell=3/1/subject | 12 | 0 | 0 |
| cell=3/2/object | 6 | 0 | 0 |
| cell=3/2/subject | 12 | 0 | 0 |

Conditional binding accuracy is undefined because no case resolves. The unresolved neither-candidate control does not establish chance binding.

## Do row pairs resolve after the document closes?

**0/20 resolved; 0/20 correct.** The accepted run resolved 0/20 and got 0/20 correct. The observer reads occurrence references after all sentences of the document, including revisions by later closes.

| Row-pair group / expected rows | Cases | Resolved | Correct |
|---|---:|---:|---:|
| determiner/1 | 4 | 0 | 0 |
| determiner/2 | 4 | 0 | 0 |
| determiner conflict/2 | 4 | 0 | 0 |
| same kind/1 | 4 | 0 | 0 |
| same kind/2 | 4 | 0 | 0 |

## Which columns were minted, with which witnesses?

The warm-up starts with **zero A columns**. Training ends with **0**; evaluation ends with **0**. There are **0 new columns** and **0 final column-witness links**. The accepted run had zero columns and witnesses. All column records and their training snapshot are in [the results](diagnostics/identity-results.json).

| Column | Witness row addresses after evaluation |
|---|---|
| None | None |

## Which programs are kept?

| Phase | Kept program | Clauses | Nonzero recorded closed meaning |
|---|---|---:|---:|
| training | equal × 4 | 175 | 0 |
| training | equal × 3 | 113 | 0 |
| training | equal × 2 | 20 | 0 |
| training | equal × 5 | 16 | 0 |
| training | verb × 5 | 13 | 10 |
| training | verb × 3 | 13 | 12 |
| training | verb × 2 | 12 | 12 |
| training | generic × 3 → conjunction → generic × 2 → whole → generic × 2 → equal → generic × 2 → lower → generic × 18 | 3 | 0 |
| training | generic × 3 → equal → generic × 2 → equal → generic × 2 → equal → generic × 2 → lower → generic × 18 | 2 | 1 |
| training | generic × 7 → tense × 3 → equal × 2 → lower → generic × 2 → part → generic × 15 | 2 | 0 |
| training | generic × 3 → lower → generic × 2 → verb → generic × 2 → verb → generic × 2 → verb → generic × 18 | 2 | 2 |
| training | generic × 10 → product → generic × 2 → whole → lower → equal → generic × 15 | 1 | 0 |
| training | generic × 3 → intersection → tense × 2 → verb → tense × 2 → verb → tense × 2 → surface → tense × 18 | 1 | 0 |
| training | generic × 3 → intersection → generic × 2 → verb → generic × 2 → disjunction → generic × 2 → part → generic × 18 | 1 | 0 |
| training | generic × 3 → conjunction → generic × 2 → verb → generic × 2 → verb → generic × 2 → intersection → generic × 18 | 1 | 0 |
| training | generic × 3 → part → generic × 2 → equal → generic × 2 → equal → generic × 5 → lower → generic × 15 | 1 | 1 |
| training | generic × 3 → part → generic × 2 → verb → generic × 2 → verb → generic × 2 → conjunction → generic × 18 | 1 | 0 |
| training | generic × 6 → verb → generic × 2 → verb → generic × 2 → verb × 2 → generic × 17 | 1 | 0 |
| training | generic × 3 → verb → generic × 2 → verb → generic × 2 → verb → generic × 2 → verb → generic × 2 → verb → generic × 18 | 1 | 0 |
| training | generic × 3 → verb → generic × 2 → verb → generic × 2 → verb → generic × 18 | 1 | 1 |
| training | verb → intersection → verb | 1 | 1 |
| training | verb × 4 → equal | 1 | 1 |
| training | verb × 3 → equal → verb | 1 | 1 |
| training | equal × 3 → lower | 1 | 1 |
| evaluation | equal × 3 | 228 | 0 |
| evaluation | equal × 2 | 138 | 0 |
| evaluation | equal × 5 | 134 | 0 |
| evaluation | equal × 4 | 56 | 0 |

Training: **43/384** clauses have nonzero recorded closed meaning; **0** reference requests in kept readings.
Evaluation: **0/556** clauses have nonzero recorded closed meaning; **0** reference requests in kept readings.

The preceding balanced identity run had 43/384 nonzero training clauses and 0/556 nonzero held-out clauses, with zero requests. These are the existing observer’s recorded meaning-band norms; all 940 recorded norms in this run are finite. Full programs and per-step norms remain in [the kept-program file](diagnostics/identity-kept-programs.json); the kept readings’ requests are in [the closing records](diagnostics/identity-reference-readings.json). This audit does not count requests in unkept exploratory trials.

Annotated reference provenance: **690** `no_individual_request`.

[Full observations](diagnostics/identity-observations.json) preserve every reason and final reference. First-mention role and position remain confounded in the revision-3 SVO introductions; the refresh mentions cross role and recency.


The original-artifact relation trace 2/4, earlier 43 unadmitted leaves,
both order CE gates, free readback 0/4, MM grammar recovery, compiled
retention and c's marker-headed answer remain open. The parts-of-speech
pass still waits for readings with NP structure.
