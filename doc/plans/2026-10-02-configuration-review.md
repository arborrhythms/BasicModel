# Configuration review (October 2)

Alec, 2026-10-02: "Some configurations (model files) may be redundant or
obsolete. Can you please review? That's costly testing time, especially for
large models."

Claude reviewed the 72 files in `data/*.xml` and `data/matrix/*.xml`. Nothing
in `data/`, `test/` or `bin/` was changed. Evidence:

- the review-13 closing sweep (5,090 cases, `doc/benchmarks/2026-10-01-stage1-and-suite-trim/closing/full-sweep/`);
- the first weekly slow run (328 cases on the review-13 snapshot, `tmp/slow-tests/latest.json`);
- Codex's §14 first batches on the current candidate (`review14/final-configuration-first-batches.md`).

Each test was mapped to the configuration files it reaches with a static call
graph. The graph follows helpers, imported test modules, fixtures and
module-level constants. A case's time is split evenly over the configurations
it reaches. The `model.xml` defaults template is counted only when a test
names no other configuration.

## 1. Where the time goes

| | Closing sweep | Weekly slow run |
|---|---:|---:|
| Case time | 9,083 s | 21,526 s (7.46 h wall) |
| `MM_ladder` | 5,928 s (238 cases) | 10,289 s (52 cases) |
| `BasicModel` | 2,403 s (48 cases) | 3,628 s (10 cases) |
| `MM_nanochat_grammar_gate` | 340 s (2 cases) | 33 s |
| `MM_ladder_idiom` | 0 s | 2,824 s (2 cases, one 30-minute timeout) |
| `BasicModel_answers_tied_benchmark` | 0 s | 2,403 s (the two production objective arms) |
| The other 67 configurations, plus unattributed cases | 412 s | about 2,350 s |

Deleting obsolete configurations saves almost no sweep time. Twenty-six
sweep cases over 60 s take **8,301 of 9,083 s (91%)**. All of them build
`MM_ladder`, `BasicModel` or the historical nanochat gate. The other 5,064
cases together take 782 s.

The obsolete configurations cost time in three other ways:

- **Migration.** Every architectural change must carry all 72 files. §14
  migrated 51 of them, and 18 now raise on their first batch.
- **Weekly failures and memory stops.** The 1024-wide June–July family (§2.4)
  runs 121 weekly cases, 58 of which failed. It also caused at least eight of
  the eleven 8 GiB stops.
- **Build-all tests.** `test_modality_configs.py` builds every file in one
  process.

## 2. Dispositions

### 2.1 Delete: no test exercises them, they no longer build, or they are declared historical (21)

| Configuration | Evidence |
|---|---|
| `MM_init_scale` | No file names it. `test_init_scale.py` now tests the codebook directly. |
| `MM_xor_step3` | Named only in a docstring; no case loads it. |
| `MM_xor_step4` | Its header says it "now matches Step 3". It is named only in comments and the build-all test's exclusion list. |
| `HeadEmission` | Its test file no longer exists. It does not build ("unresolved generate rule: S -> C"). |
| `XOR_recon` | Does not build on the candidate (WholeSpace width 10 ≠ concept width 8). Its one weekly case is an expected failure. Not in the XOR table. |
| `XOR_spaces` | Does not build (where-space capacity). Its case skips. Not in the XOR table. |
| `LM_5M_IR`, `MM_5M_AR`, `MM_5M_IR` | Do not build (flat-slab invariant). They carry million-entry percept dictionaries and two-million-entry concept dictionaries. Named only in comments and two tools. |
| `LM_5M` | Builds. Its one test (`test_svo_end_to_end.py`) skips until `data/LM_5M.ckpt` exists, and nothing trains it. |
| `MM_400M` | No case builds it; it appears only in comments and the build-all exclusion list. 7.3 GiB first batch. BasicModel replaced it as the FineWeb model. |
| `BasicModel_answers_benchmark`, `BasicModel_expectation_benchmark`, `BasicModel_long_tied_benchmark`, `BasicModel_output_tied_benchmark` | Inputs of closed benchmark rounds; their receipts can be reproduced from git. Only a parse-level list in `test_training_diagnostic_contracts.py` names them. `BasicModel_answers_tied_benchmark` stays: the production objective-conflicts slow test uses it. |
| `MM_ladder_text`, `MM_ladder_textpacked` | Named only in the same parse-level list. |
| `MM_nanochat_grammar_gate`, `MM_nanochat_grammar_pilot` | Declared historical in their own headers: "keep this file only to reproduce the original gate". The gate's trace case costs 340 s in every sweep. Keep `bin/eval_nanochat_grammar.py` only if it runs on `BasicModel.xml`; otherwise it goes too. |
| `MM_symbolic_iter` | Fixture of the retired CS→SS symbolic-iteration leg. All seven cases in `test_symbolic_iteration.py` fail on retired internals or are expected failures. |
| `MM_boolean` | No signal. In the weekly run, one expected failure runs 581 s and the other case times out at 30 minutes. XOR_exact and the concept-output tests cover the field's conjunctions. |

These tests go with the deleted files:

- `test_svo_end_to_end.py`, `test_symbolic_iteration.py` and `test_mm_boolean.py`.
- `test_modality_configs.py`. It builds every configuration in one process,
  and its 0/2 temporal-coordinate assertion predates the four-coordinate
  ladder. The weekly run stopped it at 8 GiB. The §14 per-configuration
  first-batch audit replaces it.
- The two nanochat cases that load the historical files.
- The deleted names in `test_training_diagnostic_contracts.py`'s list.
- References in `test/tools/probe_perf_compare.py` and
  `test/tools/bench_codebook_lookup.py`.

### 2.2 Defaults and one merge

- **`MM_20M_fineweb`.** `make train` (`MODEL ?=`) and `bin/train.py --model`
  still default to it. It is the 118M-parameter predecessor that `BasicModel.xml`
  replaced. Change both defaults and the Training/Installation docs to
  `BasicModel.xml`. Move its seven fast tests (FineWeb preflight, compile
  target) to `BasicModel.xml`, then delete it.
- **`MM_ltm_consolidation_stateful_fixture`** differs from
  `MM_ltm_consolidation_fixture` only in `<stateless>false</stateless>`. Its
  six cases set the flag on the base file instead.

### 2.3 For Alec: the numeric demos (5)

`ergodic`, `ergodic-only`, `simple`, `mnist` and `tomatoes` do not build on
the candidate:

- The four MNIST configurations fail while reading `data/mnist_*.csv`, which
  loads as object arrays.
- `tomatoes` fails because the Hugging Face id `rotten_tomatoes` no longer
  resolves.

No test builds them. The README quick start (`make simple`, `make ergodic`,
`make compare`) and the Makefile defaults `XML1`/`XML2` use them, so the
documented quick start fails today.

**Decided (Alec, 2026-10-02):** "Mnist is tempting as a non-text test,
let's keep that. Ergodic: I assume that's not the mathematical text test
corpus, in which case we can remove it also, and tomatoes should also go."

**Amended (Alec, 2026-10-02):** "Keep Ergodic, it's better in principle than
random weight init; if it works, we'd prefer everything is Ergodic (but that
can be future work). Deleting simple and tomatoes is fine."

- **Ergodic stays.** It is not a corpus: `ergodic.xml` and
  `ergodic-only.xml` are MNIST runs with `ErgodicLayer`'s exploration
  switched on (learned weights mixed with sampled noise; `doc/Ergodic.md`).
  The math corpus is the `math` dataset. Both files stay. FutureWork records
  "Ergodic exploration everywhere".
- **Delete `simple.xml`.** It is `ergodic-only.xml` with ergodic exploration
  off, so the default `make compare` was the ergodic A/B. That A/B moves into
  the MNIST test, which runs `ergodic` both on and off.
- **Delete `tomatoes.xml`,** with `make tomatoes` and the `tomatoes` dataset
  branch and loader.
- **Keep `mnist.xml`. Its failure is not code.** `data/mnist_train.csv` is a
  134-byte Git LFS pointer, and git-lfs is not installed on this machine, so
  the loader reads the pointer text. Install git-lfs and pull that file. The
  loader should name this cause when it meets a pointer. Add one small MNIST
  test, a short run on a subset, as the non-text case. The Makefile defaults
  and the README quick start move to `mnist.xml` and `XOR_exact.xml`.

### 2.4 For Alec: the 1024-wide June–July family (15), and `MM_add_verb`

**What the family has in common:**

- Every file uses the MM_20M geometry (1024 wide, 65,536-entry dictionaries)
  on the XOR data.
- Each differs from a sibling by one to eight settings. For example,
  `MM_global` differs from `MM_reading` only in `globalAttention`, and
  `MM_overlap_tiling` differs from `MM_mereology` only in
  `overlapWhereTiling`.
- First batches peak at 4.1–7.8 GiB.
- Every case is slow-guarded, so none runs in the sweep.

**In the weekly run:**

- 121 cases ran, and 58 failed.
- The family caused at least eight of the eleven 8 GiB stops.
- Forty-five failures come from the one 6.9 defect in Codex's weekly triage:
  mixing-leaf staging calls `len` on a row with no `word_texts`.
- About fifteen failures call internals that no longer exist (for example
  `analysis_store`, `stage_symbolic_virgin_rows`, `_ws_pos_to_row`).

The family splits into two groups.

**(a) Nine files exist for opt-in features that the training model
(`BasicModel`, `MM_ladder`) and the XOR gate never enable:**

- reading attention: `MM_reading`, `matrix/MM_20M_grammar_reading`
- global attention and its QA consumer: `MM_global`, `MM_qa`
- the D3 idea decoder: `MM_decode`, `MM_phrase_decode`, `MM_meronomy_smoke`
- overlapped where tiling: `MM_overlap_tiling`
- the word store with radial STM reduction: `matrix/MM_20M_grammar_wordstore`

*Recommendation:* retire each feature together with its code, configurations
and tests, per the no-legacy rule. Item 6.8 makes attention one bracket
mechanism, and generation is the grammar's reverse walk, so neither the
attention producers nor the D3 decoder has a place in the plan.
*Alternative:* name any feature to keep; its tests then move to a small base
configuration.

What each feature is (Alec asked for detail, 2026-10-02):

| Feature | What it does | What replaces it |
|---|---|---|
| Reading attention (`readingAttention`; June spec "(A) Reading attention", whose file is gone) | `ReadingAttention` (about 250 lines in `Spaces.py`). At each later pass of the parallel loop it scores the staged word spans, using the previous pass's concept and the STM as the query. It writes the next reading scope (`.where`), which the mereology-raising handoff consumes. A next-word cross-entropy trains it, with the true next span written during training and its own prediction at inference. | Item 6.8: one bracket over the input, narrowed; the serial order is the bracket schedule. The priming-led reading scope (`_primed_reading_step`, gated by `<relevance>`) is a separate mechanism and is not affected. |
| Global attention and its consumer (`globalAttention`, `globalAttentionConsume`; spec "(B) Global attention") | `GlobalAttention` (about 240 lines). A soft read over a typed addressable space: input window, STM, LTM and the part/whole/symbol codebooks. Alone, it parks the read unused. With the consumer, the read enters the answer head as a zero-initialized gated residual. `MM_qa` adds a small LTM TruthSet for retrieval QA. | Retrieval through thought: the controller's hard reads over LTM and the taxonomy. Keep `_addressable_spaces`, which the reasoner also reads. |
| The D3 idea decoder (`ideaDecode`; plan of 2026-06-20, now in `doc/old`) | On the old `reverse()` path it skips the chart rebuild and seeds the perceptual reverse with what the shared generate walk produces from the completed clause. GradientFlow already lists the D3 path as retired; the seeding hook (about 60 lines) and the flag remain. `MM_add_verb` also sets it. | Reconstruction through the understanding with tied inverses (universal since §14), and generation as the generate walk itself. |
| Overlapped where tiling (`overlapWhereTiling`) | Instead of one flat tiling into wholes, WholeSpace keeps four scales at once (typed run, word, separator run, enclosing sentence) and refines `.where` over a fixed number of passes. Requires mereology raising. Experimental; never adopted. | Item 6.8's bracket and narrowing, and its word whole: a maximal run of letters (6.8 §3a). |
| The word store (`<PartSpace><wordStore>`; 2026-07-12) | PartSpace's promoted word entries become the candidates that the old reverse's recommenders propose from, in its free-derivation decode. | The reconstruction's own candidate bank, which §15.3 of the 6.9 plan makes the primed symbols. |

`radialStmReduce` rides in the word-store configuration but is separate:
Alec's radial min and max for STM folds ("min should be a radial min",
2026-07-13), which order conjunction and disjunction by signed magnitude.
Unit tests in five files cover the kernels, so it needs no configuration of
its own. It is an operator choice for the operators update, not a retirement
candidate.

**Decided (Alec, 2026-10-02):**

- **Reading attention:** "if 6.8 replaces it, great, but let's make sure we
  aren't deleting useful functionality."
- **Global attention:** "same; let's make sure we are not dropping features
  with the 6.8 integration."

So both stay, with their four configurations, until item 6.8 retires them.
6.8's plan §7 lists, capability by capability, what 6.8 already covers and
what it must add. Among the additions are priming that steers the reading, a
choice across memory spaces, and a learned read trained by the answer.

- **Overlapped tiling:** "in principle, this is the kind of multi-resolution
  .where tiling that makes JPEG efficient." It retires, with a FutureWork
  entry detailed enough to revive it from `d679df2b`.
- **The D3 decoder and the word store** retire as recommended. Alec raised no
  objection.

**(b) Six files exercise features that the training model or the XOR gate
already enables** (mereology raising, serial object meta, symbol tower,
category codebook, the parallel field): `MM_mereology`,
`MM_mereology_serial`, `MM_symbol_tower`, `MM_sparse_concept`,
`MM_masked_semantic`, `MM_20M_grammar`.

*Recommendation:* move their surviving tests onto kept configurations and
delete the files:

- the serial features go to `MM_ladder`;
- the parallel field and the symbol tower go to `XOR_exact` or `MM_20M_xor`;
- `make bench_local` changes its default from `MM_20M_grammar` to
  `MM_ladder`.

**`MM_add_verb`** holds the math-as-syntax proof
(`test_successor_is_learned_as_a_verb`). At 1024 wide it cannot run under the
8 GiB guard: its first batch reached 8.5 GiB. Shrink its geometry so the proof
runs. If the D3 decoder is retired, its `ideaDecode` setting goes too.

### 2.5 Keep (28, plus `MM_add_verb` after shrinking)

| Group | Files |
|---|---|
| Template and training | `model`, `BasicModel`, `BasicModel_answers_tied_benchmark`, `MM_ladder`, `MM_ladder_idiom`, `MM_grammar_wording` |
| XOR table | `XOR_exact`, `XOR_grammar`, `MM_xor`, `MM_grammar`, `MM_20M_xor`, `matrix/MM_20M_xor_noraise` |
| Fixtures | `MM_xor_fixture` (112 sweep cases), `MM_xor_loopback` (77), `MM_ltm_consolidation_serial_fixture` (57), `MentalModel` (32), `MM_ltm_consolidation_fixture` (26), `XOR_pos` (16), `MM_math`, `idempotent`, `POS_smoke`, `RamsifiedModel`, `MM_shamatha`, `MM_query_reasoning`, `MM_add`, `stream_smoke`, `MM_sequence_predict`, `xor` (the `bin/Models.py` command-line default) |

Five kept fixtures are among §14's first-batch exceptions and need repair,
not deletion:

- `MM_add`, `MM_math` and `stream_smoke` exceed the where-space capacity.
  Streaming is live; BasicModel reads 28 corpus streams.
- `MM_ltm_consolidation_fixture` fails the TruthSet closing requirement.
- `MM_query_reasoning` fails the interrogative TruthSet rule.

`MM_sequence_predict` passes, but costs 6.7 GiB and 394 s for one batch of
four. Item 6.8 absorbs its predictor.

**Result, as decided:** 72 files become 36. They are:

- the 28 above;
- `MM_add_verb`, after shrinking;
- `mnist`, `ergodic` and `ergodic-only`;
- the four attention configurations, until item 6.8.

Of the 18 first-batch exceptions, nine are deleted or merged and nine are
repaired. Pulling one LFS file repairs the three MNIST configurations, and
`MM_qa` joins the earlier five. The files left on the MM_20M geometry are
the two XOR-gate configurations and, until 6.8, the four attention ones.

## 3. Test time

The twenty-six sweep cases over 60 s:

- **Five whose subject is the compiled graph (1,454 s).** Move them to the
  weekly tier. Smaller compile tests stay in the sweep.
  - real aligned loop, K=2 (849 s)
  - compiled understanding capture (253 s)
  - packed ends `[True]` (136 s)
  - fullgraph word-loop provenance (132 s)
  - fullgraph query mask (84 s)
- **The nanochat trace (340 s).** It goes with its configuration.
- **The other twenty (about 6,500 s).** Apply the trim's treatment for the
  previous 25: eager loops where compiling is not the subject, with subjects
  and assertions unchanged. That treatment took those 25 cases from 19,743 s
  to 1,713 s.

The expected sweep case time is under 1,000 s, down from 9,083 s. The §14
sweep now running will refresh these numbers; the rule applies to whatever
remains above 60 s.

In the weekly run, three cases time out at 30 minutes without a result:

- `MM_ladder_idiom`'s cold-start boundary test;
- `MM_boolean`'s sentences test, which goes with its file;
- one `MM_ladder` expectation-enable case.

The two that remain need a bounded workload.

## 4. Every configuration

The first two numeric columns give cases and seconds in the closing sweep,
and cases and failures in the weekly run. The last column is the result of
the §14 first batch on the candidate, with peak memory.

| Configuration | Disposition | Sweep cases / s | Weekly cases / red | Candidate first batch |
|---|---|---:|---:|---|
| `BasicModel_answers_benchmark` | delete | 1 / 0 | 0 / 0 | — |
| `BasicModel_expectation_benchmark` | delete | 1 / 0 | 0 / 0 | — |
| `BasicModel_long_tied_benchmark` | delete | 1 / 0 | 0 / 0 | — |
| `BasicModel_output_tied_benchmark` | delete | 1 / 0 | 0 / 0 | — |
| `HeadEmission` | delete | 0 / 0 | 0 / 0 | exception |
| `LM_5M` | delete | 1 / 0 | 0 / 0 | completed, 2.8 GiB |
| `LM_5M_IR` | delete | 0 / 0 | 0 / 0 | exception |
| `MM_400M` | delete | 1 / 0 | 1 / 1 | completed, 7.3 GiB |
| `MM_5M_AR` | delete | 0 / 0 | 0 / 0 | exception |
| `MM_5M_IR` | delete | 0 / 0 | 0 / 0 | exception |
| `MM_boolean` | delete | 2 / 0 | 1 xfail + 1 timeout | completed, 0.7 GiB |
| `MM_init_scale` | delete | 0 / 0 | 0 / 0 | completed, 0.4 GiB |
| `MM_ladder_text` | delete | 2 / 0 | 0 / 0 | — |
| `MM_ladder_textpacked` | delete | 1 / 0 | 0 / 0 | — |
| `MM_nanochat_grammar_gate` | delete | 2 / 340 | 2 / 0 | — |
| `MM_nanochat_grammar_pilot` | delete | 1 / 0 | 1 / 0 | — |
| `MM_symbolic_iter` | delete | 1 / 0 | 1 / 1 | completed, 0.5 GiB |
| `MM_xor_step3` | delete | 0 / 0 | 0 / 0 | completed, 0.5 GiB |
| `MM_xor_step4` | delete | 1 / 0 | 1 / 1 | completed, 0.5 GiB |
| `XOR_recon` | delete | 2 / 0 | 1 xfail | exception |
| `XOR_spaces` | delete | 1 / 0 | 0 / 0 | exception |
| `MM_20M_fineweb` | delete after the defaults move | 7 / 0 | 0 / 0 | — |
| `MM_ltm_consolidation_stateful_fixture` | merge | 6 / 0 | 0 / 0 | exception |
| `ergodic` | §2.3 | 0 / 0 | 0 / 0 | exception |
| `ergodic-only` | §2.3 | 0 / 0 | 0 / 0 | exception |
| `mnist` | §2.3 | 0 / 0 | 0 / 0 | exception |
| `simple` | §2.3 | 0 / 0 | 0 / 0 | exception |
| `tomatoes` | §2.3 | 0 / 0 | 0 / 0 | exception |
| `MM_decode` | §2.4 (a) | 2 / 0 | 2 / 0 | completed, 4.7 GiB |
| `MM_global` | §2.4 (a) | 7 / 0 | 7 / 4 | completed, 4.8 GiB |
| `MM_meronomy_smoke` | §2.4 (a) | 15 / 0 | 15 / 5 | completed, 7.8 GiB |
| `MM_overlap_tiling` | §2.4 (a) | 1 / 0 | 1 / 1 | completed, 4.6 GiB |
| `MM_phrase_decode` | §2.4 (a) | 4 / 0 | 2 / 1 | completed, 7.8 GiB |
| `MM_qa` | §2.4 (a) | 6 / 0 | 5 / 5 | exception |
| `MM_reading` | §2.4 (a) | 8 / 0 | 8 / 5 | completed, 4.7 GiB |
| `matrix/MM_20M_grammar_reading` | §2.4 (a) | 6 / 0 | 4 / 1 | — |
| `matrix/MM_20M_grammar_wordstore` | §2.4 (a) | 2 / 0 | 1 / 0 | — |
| `MM_20M_grammar` | §2.4 (b) | 23 / 0 | 17 / 6 | — |
| `MM_masked_semantic` | §2.4 (b) | 13 / 0 | 13 / 12 | completed, 5.4 GiB |
| `MM_mereology` | §2.4 (b) | 8 / 0 | 8 / 4 | completed, 4.7 GiB |
| `MM_mereology_serial` | §2.4 (b) | 24 / 36 | 18 / 1 | completed, 6.4 GiB |
| `MM_sparse_concept` | §2.4 (b) | 25 / 0 | 23 / 13 | completed, 5.4 GiB |
| `MM_symbol_tower` | §2.4 (b) | 7 / 0 | 7 / 2 | completed, 4.1 GiB |
| `MM_add_verb` | shrink | 1 / 0 | 8 GiB stop | 8 GiB stop |
| `BasicModel` | keep | 48 / 2,403 | 10 / 6 | — |
| `BasicModel_answers_tied_benchmark` | keep | 3 / 0 | 2 / 0 | — |
| `idempotent` | keep | 4 / 4 | 1 / 0 | completed, 0.5 GiB |
| `matrix/MM_20M_xor_noraise` | keep | 6 / 0 | 4 / 1 | — |
| `MentalModel` | keep | 32 / 5 | 0 / 0 | completed, 0.7 GiB |
| `MM_20M_xor` | keep | 76 / 1 | 68 / 20 | — |
| `MM_add` | keep, repair | 1 / 0 | 1 xfail | exception |
| `MM_grammar` | keep | 42 / 3 | 5 / 0 | completed, 0.5 GiB |
| `MM_grammar_wording` | keep | 1 / 0 | 0 / 0 | — |
| `MM_ladder` | keep | 238 / 5,928 | 52 / 9 | — |
| `MM_ladder_idiom` | keep | 3 / 0 | 2 / 1 | — |
| `MM_ltm_consolidation_fixture` | keep, repair | 26 / 1 | 0 / 0 | exception |
| `MM_ltm_consolidation_serial_fixture` | keep | 57 / 9 | 0 / 0 | completed, 0.5 GiB |
| `MM_math` | keep, repair | 10 / 1 | 1 xfail | exception |
| `MM_query_reasoning` | keep, repair | 2 / 0 | 0 / 0 | exception |
| `MM_sequence_predict` | keep | 1 / 0 | 1 / 0 | completed, 6.7 GiB |
| `MM_shamatha` | keep | 3 / 0 | 1 / 1 | completed, 0.5 GiB |
| `MM_xor` | keep | 184 / 45 | 12 / 1 | completed, 0.5 GiB |
| `MM_xor_fixture` | keep | 112 / 58 | 0 / 0 | completed, 0.5 GiB |
| `MM_xor_loopback` | keep | 77 / 5 | 0 / 0 | completed, 0.5 GiB |
| `model` | keep | 621 / 14 | 1 / 0 | — |
| `POS_smoke` | keep | 3 / 15 | 0 / 0 | completed, 0.7 GiB |
| `RamsifiedModel` | keep | 3 / 0 | 0 / 0 | completed, 0.7 GiB |
| `stream_smoke` | keep, repair | 1 / 0 | 1 / 1 | exception |
| `xor` | keep | 0 / 0 | 0 / 0 | completed, 0.5 GiB |
| `XOR_exact` | keep | 18 / 17 | 3 / 0 | — |
| `XOR_grammar` | keep | 28 / 29 | 2 / 1 | completed, 0.5 GiB |
| `XOR_pos` | keep | 16 / 1 | 0 / 0 | completed, 0.5 GiB |

A dash in the last column means the configuration was outside the §14
first-batch audit: the grammar-free and understanding configurations were not
migrated. A weekly "stop" is a process ended by the 8 GiB guard. The
build-all test's single stop is counted against every configuration it names
(`MM_400M`, `MM_shamatha`, `MM_xor`, `MM_xor_step4`).

## 5. Hand-off

One hand-off to Codex follows Alec's answers to §2.3 and §2.4 and the close
of item 6.9. It covers:

- the deletions, defaults and merge (§2.1–2.2);
- the decided parts of §2.3–2.4;
- the repairs of the five kept fixtures;
- the test-time moves (§3).

Recommended order: immediately after 6.9 closes. Every later receipt gets
cheaper, and the operators update then migrates 29 files instead of 72.
