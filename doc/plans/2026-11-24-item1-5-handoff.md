# Item 1.5 integration handoff

**CPU reading rose from 7.04 to 10.23 sentences/s (+45.2%).** The matched reader reached 122/122 in both held-out modes in 44.06 s, down from 97.89 s. It also qualified one epoch earlier, after 36 optimizer updates instead of 54.

| Local main | Commit |
| --- | --- |
| Published starting point | `ad8fafea927839c0d7f92d0aee35c913d02b2612` |
| Reviewed 1.7 ancestor, verified | `d38029113103ce5e9cf1223eaa55c9181010d649` |
| Final training code | `b0b825a2132a2e99e0940d76fb49d23c3f83a9d6` |

## Real training measurements

| Workload | Released 1.7 | Final |
| --- | ---: | ---: |
| CPU reading, two steady replicates, sentences/s | 7.06, 7.02 | 10.22, 10.23 |
| CPU reading, first three updates, seconds | 0.874, 0.614, 0.573 | 0.763, 0.396, 0.401 |
| Metal reading, steady sentences/s | 0.695 | 0.871 |
| Metal reading, first three updates, seconds | 3.867, 5.159, 5.594 | 3.591, 4.557, 5.012 |
| 40 corpus lesson 13 presentations, seconds / steady sentences/s | 53.54 / 0.745 | 50.42 / 0.798 |
| 12-sentence worked math document, seconds / steady sentences/s | 21.46 / 0.516 | 18.76 / 0.597 |

The step benchmark uses the same four pages, batch 4, float32, seed 1505, Python 3.12, PyTorch 2.15 and one CPU thread. Each timed update includes forward, backward and the optimizer. The nine steady updates follow the first three. Metal timings synchronize the device. The lesson uses a separate validation set to stop training, then saves, restores and scores the reader.

| Sentences × explore paths | Steady CPU sentences/s | Seed 1505, seconds and sampled/eval correct out of 122 | Seed 387885085 |
| --- | ---: | ---: | ---: |
| 4 × 1 | 10.22 | 44.06 · 122/122 | 31.99 · 121/122 |
| 4 × 3 | 6.41 | 55.22 · 122/122 | 36.51 · 122/122 |
| 8 × 1 | 12.72 | 50.63 · 121/122 | 27.27 · 119/122 |

The last two configurations each evaluate sixteen candidate walks per update. Three explorations earned perfect held-out scores on both starts and took longer than one exploration on both. Batch eight delivered the most sentences/s and missed the sampled gate once. In a ten-second training budget at seed 1505, the configurations above completed 19, 10 and 12 updates and scored 117, 122 and 122 sampled, respectively. All scored 122 in evaluation. The default remains one exploration.

## What changed and what still costs time

- The sentence bank shares request-independent percept reads, location factors and the opening forecast. The wide-open pass still precedes one whole-field reconstruction, narrowed model-chosen fixations with a joint content/location request, and optional thinking. Every later forecast uses its own stream’s prior observations.
- A release replay had four greedy winners while a rejected path promoted four persistent Sigma entries. Alec approved trial-local growth. The same replay now has zero persistent candidate-table mutations during all four walks. Winning rows replay their observations at close after the optimizer step. Direct fixation and forecast gradients follow their selected rows. The existing policy objective still compares path costs.
- Tensor-to-host calls fell from **23,736 to 16,089** per update and **2,409 to 360** inside looks. Boundary codebook calls fell from 1,079 to 419 and repeated native addresses from 3,696 to 1,721, including padding. Both candidates together took 0.267 → 0.178 s to compose and 0.108 → 0.089 s to score in the separate CPU profile. In a further update timed without cProfile, the explore path’s narrowed walks and scoring callbacks together took 0.194 → 0.099 s. At three explore paths their combined time was 0.342 s.
- The six fixed Pi/Sigma arrays kept their storage identities. The sparse-weight revision reduced `aten::empty` allocation volume from 721 to 517 MB per profiled update. Input-only, unpacked, explicitly staged updates bypass the saved-tensor hook. Packed and supervised training retain versioned saves. Online Sigma admission preserves the occupied weights needed by backward. Sentence-local metadata sharing reduced recursive validation calls from 4,020 to 1,352.
- Native Inductor captured one graph on the first update. Cold time was 3.85 → 4.39 s and steady rate 6.50 → 8.62 sentences/s. The optional fullgraph whole/part fold matched all 595 native replay tensors, compiled on four updates and measured 7.95 sentences/s. Prepared bank lookup and its backward also have fullgraph tests.

## Behavioral verification

- Two independent unseeded restored readers passed 99% and all required pairs at age 0.00. They scored **122 and 121 sampled**, and **122 each in evaluation**, out of 122. The release had two perfect sampled readers.
- Final fresh verbal XOR passed **4/4 ordered pairs into grammar and 4/4 through the inverse** after 400 epochs. The preceding sparse-weight revision also passed both. Earlier ownership revisions scored 0/4 and 2/4 at both boundaries. Release notes document intermittent retention. Numeric XOR convergence and raised-class denotation passed.
- Before the ownership change, release replay matched all **595** CPU tensors, including 26 live gradients. The approved change produced 106 differences in the default replay. With exploration disabled, all **541** release-to-final tensors matched exactly. The last metadata optimization matched all **595** with exploration enabled. The performance-only Metal parameter difference peaked at 4.47e-8 against a release-repeat difference of 5.22e-8.
- Final groups passed **133** path/memory/performance tests, **3** reading/numeric/raised-class tests and **1** verbal XOR test. An earlier overlapping broader run passed **124 with 9 skips**. Three additional diagnostics failed on final. Item 1.8 documents the omitted-word failure. Independent release runs reproduced the word-concept teacher threshold failure (54bd) and sparse/dense lesson mismatch (58, table needed 24 of 16). Existing assertions are intact. Verification was scoped.

## Claude’s integration points

| Interface | What to preserve |
| --- | --- |
| `BasicPercepts.prepare`, `subspace_bounds`, `Spaces.cs_lookup`, `Salience`, `PerceptProperties` | Sentence bank lifetime, live vs stored location wires, place-specific value charts for number and felt percepts. |
| `CandidateGrowth`, `ConceptSpread`, `FieldProjection`, `IntuitionField`, `Layers` | Provisional topology belongs to each logical stream. Case-inventory basis vectors have a different axis. |
| `Models._run_sentence_word_bricks`, `SentenceLooks.Looks`, `LookForecast` | Share the immutable opening, preserve per-path forecasts, and commit only selected rows’ observations. |
| `Meaning`, `Comprehension`, `SentenceBoundary`, `Memory`, `MemoryLink`, `MemberRead` | Validate new grounding metadata on entry. Preserve mutable-source ownership and the lifetime of stored event operands. |

The older unified MTM/promoted-row proposal awaits architecture review. It affects `MediumTermMemory.entries`, `state/restore`, transient IDs and event snapshots used by 1.6. In this profile `PrependedSymbols.stage` took 1.8 ms. Host-side topology and serial candidate suffixes remain. Per-look timestamps and reconstruction remain separate FutureWork experiments.

## Reproduction

Use a fresh output directory and the base archive and environment commands in the [earlier handoff](2026-11-23-item1-5-handoff.md#reproduction). That handoff also records the port from the older preparation branch.

- `bin/bench_item15.py --output DIR` measures real steps. Add `--lesson`, `--explore-paths 3`, `--batch-size 8`, `--lesson --training-seconds 10` or `--corpus lesson-13|math-chain` for the studies above.
- `--steps 1 --explore-paths 3 --path-audit` reproduces the early rejected promotions. `--compiled-step` uses the native entry. Add `--compile-percept-fold` to try the additional fold with a fresh Inductor cache.
- Use `--explore-paths 0` on both sources for release-to-final exact replay. `bin/compare_item15.py BEFORE AFTER` checks parameters, buffers, optimizer, RNG and live gradients. Local JSON receipts, logs and profiles are under `output/item1-5/`, including `solo-*` for the isolated CPU runs.

## Host-read checkpoint, pending architecture review

Local training commit `883fbb21c67d18049ec3af5b95c632507c19c299` follows the measured stopping point `5a393548e30f0e8ba0c5fbacc15b52977b6c6a3b`. Item 1.5 remains open. The proposed comparison-only grammar reader and the later MTM layout are described in the [representation design](2026-11-24-item1-5-device-sentence.md).

| Four-page CPU update | Before this checkpoint | After |
| --- | ---: | ---: |
| Six explicit tensor-read methods over the update | 16,089 | 8,772 |
| Same methods within ordinary sentence execution | 15,962 | 8,645 |
| Same methods inside looks | 360 | 360 |
| Steady sentences/s, nine updates after three warm-ups | 10.33 | 10.24 |
| Fresh lesson to restored 122/122 in both modes, seconds | 44.06 | 44.12 |

The row writers now retain the occupied extent and row names that they publish. Buffer replacement, version changes and compaction invalidate the lookup. The close aligns winners’ fields on device, including duplicate concept IDs and different candidate widths. The residual winner selection still reads the winning paths to choose variable-sized fields, and the ID union has a data-dependent size. All 595 CPU replay tensors agreed, including the 26 live gradients. The focused path, memory and performance suite passed 93 tests. Four learning gates passed. Both newly trained and restored readers (one paired benchmark start, one independent gate start) scored 122/122 in sampled and evaluation modes. Fresh verbal XOR passed 4/4 ordered pairs into grammar and 4/4 through the inverse, and numeric XOR convergence and raised-class denotation passed.

The newly timed Metal control on the archived stopping point reached 0.96 sentences/s while a CPU test was also running. The paired configuration and operator counts are comparable, while machine load limits a precise incremental speed claim. The new Metal run took 2.49, 3.36 and 3.70 s for its first three updates and sustained 1.22 sentences/s, compared with 0.87 at the previous checkpoint and 0.695 on 1.7. Metal replay against the previous checkpoint differed in 45 tensors. The maximum absolute parameter and live-gradient differences were 2.24e-8 and 2.92e-11. Earlier same-code repeat differences reached 5.22e-8 and 1.46e-10.

The six-method census and the native dispatcher census are separate measurements. A strict capture probe on the ordinary entry stops at the input byte decoder’s `ndarray.tobytes()`. On Metal, the archived control and the new training code respectively issued 14,175 → 6,692 scalar extractions, 591 → 584 equality operations and 571 → 487 nonzero operations within sentence execution. Explicit device-to-host copies rose from 2,349 to 2,355. Some dispatches can operate on CPU tensors. PyTorch’s dynamic-shape tag also applies to indexing overloads that accept either integer or Boolean indices. These counts cannot be added to infer synchronization time.

Run `bin/bench_item15.py --output DIR --dispatch-census` with the same environment as the earlier runs. The six-method census includes source file, function and line within `BasicModel._execute_input_sentence`. The native census runs on an extra update after the gradient snapshot. Local receipts are under `output/item1-5/host*`. The changed model files in the pre-commit receipts match the committed hashes. The additional integration surfaces are `Memory.IdeaRows`, `Memory.LTM1` and `SentenceField.select_fields`. The grammar reader, event writers and checkpoint schema are still the existing implementations.
