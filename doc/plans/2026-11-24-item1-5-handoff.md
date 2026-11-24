# Item 1.5 integration handoff

## Current local checkpoint

Item 1.5 is open. Training code is at `d79488b88c10786e73c8e8c5d22d5fe9184dd57a` on local main. The subsequent test and design checkpoint is `29c378be9f0534a522a3210e45c02595fe000306`. The base is `ad8fafea927839c0d7f92d0aee35c913d02b2612`, including reviewed 1.7 commit `d38029113103ce5e9cf1223eaa55c9181010d649`. Alec owns integration and push.

| Four paced pages, seed 1505 | 1.7 base | Current |
| --- | ---: | ---: |
| CPU steady training sentences/s | 6.99 | 10.42 |
| MPS steady training sentences/s | 0.786 | 1.67 |
| CPU first three updates, seconds | 0.938, 0.634, 0.576 | 0.785, 0.404, 0.406 |
| MPS first three updates, seconds | 3.352, 4.464, 4.932 | 2.159, 2.538, 2.857 |
| CPU seconds to restored 122/122 in both modes | 97.89 | 40.40 |
| In-sentence explicit host calls, CPU / MPS | 23,609 / 48,286 | 3,610 / 7,310 |
| Repeated native percept addresses in the audited update | 3,696 | 402 |

The steady rates include backward and optimizer in nine eager updates after the three warm-ups. In a separately instrumented step the two candidate walks together spent 180 ms on looks and 65 ms on path scoring on CPU, or 998 ms and 519 ms on MPS. Receipts are under `output/item1-5/located-close-*`. Strict Inductor on CPU and MPS reproduced both the located-parts lookup and the witnessed-overlap reconstruction. Strict capture of the ordinary sentence still stops in `PartSpace._embed_ladder_word_major` where Python consumes learned span endpoints for omission, admission and reporting.

Two independent fresh starts restored to 122/122 in both modes in 40.40 s (seed 1505) and 26.56 s (seed 1539668869). On seed 1505, four pages with three extra explorations sustained 6.11 sentences/s and reached the same endpoint in 33.36 s. Eight distinct pages with one extra exploration sustained 12.87 sentences/s. At 47.47 s their restored result was 121/122 sampled and 122/122 evaluated. Both larger schedules evaluated sixteen paths per update.

Each of two successive CPU update comparisons matched all 597 tensors, including 26 live gradients. Across the scoped suites there were 87 passes with 5 skips for the lookup, ladder, performance and reconstruction changes, 56 passes for closing and meaning, and 20 passes for the reading-round group. The optional fragment leaf-count assertion also failed on its unchanged predecessor. Two unseeded sampled readers missed the held-out gate at 119 and 120 of 122, both with 122 in evaluation. A fresh verbal XOR run passed both 4/4 boundaries after another fresh start failed its reading prerequisite. Numeric XOR class, numeric convergence and raised-class denotation passed. All existing assertions stand.

Ordinary input now obtains characters and learned runs directly from observed bytes at every offset. The legacy hidden-part, taught nested and surrogate paths still consume host structures. The inverse can assemble a predicted percept with located children by compatible overlap and strict extension. Ordinary forms without located children still use the stored spelling projection. With the underlying byte occurrences available, the native nomination code can propose `shop` from `sho | pping` when attention supplies the four-byte window. Autonomous selection of that window and a frequency-based non-overlapping cover are proposed in the [device sentence design](2026-11-24-item1-5-device-sentence.md).

Claude's closest interfaces are the two Pi lanes and native occurrence geometry in `BasicPercepts` and `PerceptField`, observed part counts and spans in `Spaces.PartSpace` and `WordIdentity`, narrowed reads in `SentenceLooks`, `FragmentRead` and `LookForecast`, and the inverse projection in `FieldReverse`. For the unfinished memory transfer the shared interfaces are `Memory.MediumTermMemory`, `Understanding`, `PartClosing` and `ClauseJournal`. The ordinary grammatical close still relies on the Python reading index, and the per-stream historical root index is open.

## Measurements before the input ladder change

Item 1.5 is open. The measured training commit is `efcb9c1dfe8d79b122c970e00db2da33721c4986`. The base is `ad8fafea927839c0d7f92d0aee35c913d02b2612`, which contains the reviewed `d38029113103ce5e9cf1223eaa55c9181010d649`. Integration and push belong to Alec.

| Four pages, batch four, float32, seed 1505 | 1.7 base | Measured checkpoint |
| --- | ---: | ---: |
| CPU steady sentences/s | 6.99 | 10.51 |
| MPS steady sentences/s | 0.786 | 1.54 |
| CPU warm-up updates, seconds | 0.938, 0.634, 0.576 | 0.779, 0.400, 0.401 |
| MPS warm-up updates, seconds | 3.352, 4.464, 4.932 | 2.402, 2.829, 3.251 |
| CPU seconds to a restored 122/122 in both reading modes | 97.89 | 40.68 |
| CPU explicit host calls in the audited sentence | 23,609 | 3,634 |
| MPS explicit host calls in the audited sentence | 48,286 | 7,368 |
| Repeated native percept addresses | 3,696 | 402 |

These are eager optimizer updates including backward. Nine steady updates follow three warm-ups. The MPS run overlapped a CPU job. The CPU path audit measured 58 ms for greedy looks and 22 ms for scoring, and 89 ms and 22 ms for the alternative. The corresponding MPS measurements were 413/201 and 448/219 ms. The strict whole-sentence entry fails in the input ladder when Python consumes learned byte-span endpoints for `WordIdentity`. Compilation of the overlapping spelling matcher is separately tested.

Two fresh readers restored to 122/122 in sampled and evaluation modes. Five learning tests passed in 304.55 seconds, including verbal XOR at 4/4 in both directions, numeric XOR convergence and class, and raised-class denotation. Compared to the checkpoint before the device spelling scan, a real CPU update matched all 597 tensors, including 26 live gradients. Receipts are under `output/item1-5/device-spans-*`. The benchmark and comparison commands appear below.

Four pages with three additional explorations sustained 5.86 sentences/s and reached restored 122/122 in both modes in 33.22 seconds. Eight independent pages with one additional exploration sustained 12.94 sentences/s and reached 121/122 sampled and 122/122 evaluated in 51.05 seconds. These schedules each evaluate sixteen paths per update. Only the four-page measurement reached the same exact held-out endpoint in this seeded comparison.

The overlapping matcher checks every byte offset for each learned spelling. An attention aperture includes occurrences within its endpoints and wholes enclosing both endpoints. A partial overlap alone does not qualify. The reverse currently expands a predicted row from its known spelling, stored run, or concatenated parts, then pastes the decoded bytes using predicted locations. Changing the chooser's fixation targets or the reconstruction objective is unnecessary for the input lookup optimization.

Other open work includes replacing runtime `ReadingIndex` and event replay with the selected native head, completing ownership through STM → MTM → LTM and addressing each stream’s retained root by its own time. The closest interfaces for 1.6 are `BasicPercepts`, `PerceptField`, `SentenceLooks`, `LookForecast`, `FragmentRead`, `WordIdentity`, `WordSurfaces`, `Memory`, `PartClosing`, `Understanding` and `ClauseJournal`.

## Earlier MTM ownership and native forecast checkpoint

The published base is `ad8fafea927839c0d7f92d0aee35c913d02b2612`. Its reviewed 1.7 ancestor is `d38029113103ce5e9cf1223eaa55c9181010d649`. Local training commit `80150fc8da226099cf59d897b8b3fd805e1ea2de` implements the first STM → MTM → LTM transfer changes on main.

| Four pages, batch four, float32, seed 1505 | 1.7 | MTM checkpoint |
| --- | ---: | ---: |
| Steady CPU sentences/s | 6.99 | 10.09 |
| Steady MPS sentences/s | 0.786 | 1.63 |
| First three updates on CPU, seconds | 0.938, 0.634, 0.576 | 0.825, 0.404, 0.407 |
| First three updates on MPS, seconds | 3.352, 4.464, 4.932 | 2.332, 2.596, 2.921 |
| Wall time to restored 122/122 sampled and evaluation on CPU | 97.89 s | 41.95 s |
| Explicit tensor reads inside the CPU sentence | 23,609 | 4,515 |
| Repeated native percept addresses in an audited update | 3,696 | 402 |

Times include backward and the optimizer. Steady throughput uses the nine steps after three warm-ups. These eager measurements can be reproduced with the benchmark commands below using `mtm-kept-cpu`, `mtm-winner-metal`, and `mtm-kept-reader` under `output/item1-5`. On the separately timed CPU audit the exploit took 59 ms in looks and 25 ms in scoring. One exploration took 81 ms and 25 ms. Two fresh restored readers scored 122/122 sampled and 122/122 in evaluation. All five learning gates passed, including 4/4 ordered pairs into grammar, 4/4 through its inverse, numeric XOR convergence and class, and raised-class denotation. The scoped memory and performance rerun passed 270 tests, including one additional forecast isolation test added after that commit. One matched update compared 597 tensors against the preceding training code, with exact equality across parameters, buffers, optimizer, RNG and all 26 live gradients.

The retained native head goes through MTM at close. An eligible resolved STM0 idea transfers directly to LTM0. A distinct predicted post-action idea stays in MTM while recent noun/event references need its value and signed evidence. Promotion transfers the native row with its original stable name. At expiry MTM releases the row using the existing token lifetime. Event queries consult the owning store, including when an older checkpoint still stores a predicted result in LTM0. When the durable head exists the sentence stack retains its reference and releases the trial copy. General grammatical closing still uses the Python reading index and event topology. The per-stream root/time index is outstanding. Claude may overlap in `Memory.MediumTermMemory`, `PartClosing`, `CaseSelection`, `MemberRead`, and `TimelineRead`.

The physical native percept prior was reused across histories at this checkpoint. On an audited three-exploration update the four paths advanced shared native state 3, 0, 0 and 0 times. Disabling the cache without isolating state changed those counts to 3, 5, 3 and 7. Seven predictor buffers differed even though the live gradients, parameters, optimizer and RNG matched in the ablation. The cached and uncached schedules sustained 6.12 and 5.83 CPU sentences/s. The causal predictor is neural. Shared mean and variance are serialized but have no production reader in the ordinary forecast. The per-stream recent percept and its prior mean feed `ActivationPredictor` together with order-zero concept activity.

For locating parts, a dense signed-bit dot product recovered the same 71 overlapping byte matches as exact windows on the four-page surface. With pattern encoding outside the timed loop and delivered-byte encoding inside, the sweeps took 0.043 versus 0.010 ms on CPU and 0.220 versus 0.054 ms on MPS. This was a matcher probe. The retained graded lanes still fold through Pi. Each fresh occurrence can use the existing powerset of learned parts without allocating all theoretical subsets.

## Earlier status at the spelling audit checkpoint

Item 1.5 is open on local `main`. The starting commit was `ad8fafea927839c0d7f92d0aee35c913d02b2612`, with reviewed handoff `d38029113103ce5e9cf1223eaa55c9181010d649` in its ancestry. The current training commit is `091e5493101496ccef1f96ce7dad9304bf3850d1`. The original preparation branch was ported onto the new layout. All integration and pushes belong to Alec.

| Matched four-page reading, seed 1505 | 1.7 base | Current |
| --- | ---: | ---: |
| CPU steady sentences/s | 6.99 | 10.62 |
| Metal steady sentences/s | 0.786 | 1.69 |
| CPU first three updates, seconds | 0.938, 0.634, 0.576 | 0.789, 0.400, 0.395 |
| Metal first three updates, seconds | 3.352, 4.464, 4.932 | 2.217, 2.516, 2.792 |
| CPU wall time to restored 122/122 in both modes | 97.89 s | 41.88 s |
| Explicit CPU tensor reads inside a sentence | 23,609 | 4,767 |
| Repeated native percept addresses | 3,696 | 402 |
| Native scalar extractions / device-to-host copies during Metal execution | 21,426 / 3,108 | 3,057 / 1,848 |

Times include backward and the optimizer. The steady rates use the nine updates after three warm-ups, with float32, one CPU thread, four pages and the same fixation policy. CPU sustained 1.52 times the base rate, Metal 2.15 times. The ordinary timings used eager execution. With a fresh CPU Inductor cache the separately partitioned training entry took 3.51 s on its first update and sustained 9.73 sentences/s. Its 1.7 control took 3.75 s and sustained 6.80 sentences/s. The compiler captured one partition on the cold update in both runs. See `compiled-base` and `primed-spelling-compiled`.

In the current instrumented 3.02-second Metal sentence call, 2,911 scalar extractions on Metal took 0.456 s, 1,848 Metal-to-host copies took 0.274 s, and 411 Metal equality checks took 0.080 s. The remaining 146 scalar extractions ran on CPU. The measured host calls occupied 27% of instrumented wall time including any previously queued GPU work they waited for. Kernel dispatch, small operations and Python still occupy the rest of the update. The host census, profiler, path audit and uninstrumented steady throughput are separate measurements. Comparing the prior spelling audit to the current one on Metal gave maximum parameter, optimizer and gradient differences of 2.98e-8, 7.45e-9 and 2.91e-11. Repeating the current code gave the same three maxima. Buffers and RNG matched.

Use `bin/bench_item15.py` with `PYTHONHASHSEED=0`, `MODEL_COMPILE=none`, `BASICMODEL_DEVICE=cpu` or `mps` and a fresh output directory. Use `--steps 12 --dispatch-census` for comparable receipts, `--lesson` for time to the reading gate and `--strict-sentence eager|inductor` for the actual capture probe. The current receipts are `output/item1-5/primed-spelling-{cpu-isolated,metal-isolated,reader,strict-cpu,strict-metal}`. The corresponding base receipts are `e2e2-paired-base`, `e2e2-paired-metal-base` and `solo-base-lesson` under the same directory. The configuration SHA-256 starts with `523ba88e7104` and the lesson with `402d10eea7c1`. Each receipt records full hashes, device, thread count and seed.

The ordinary four-page update still fails strict capture on CPU and Metal. The first failure is the learned bracket in `Spaces.PartSpace._embed_ladder_word_major`. A device tensor describes the cuts. Python consumes their endpoints to slice variable-length byte strings for `WordIdentity` admission, nested parts and collision repair. Other remaining host consumers include the grammatical close using `ReadingIndex` and `ClauseJournal`, MTM event history, and the lookup for one durable root per stream and tick. The [device sentence design](2026-11-24-item1-5-device-sentence.md) describes the approved direct STM head, its LTM copy, and the roles of the positive and negative lanes in Pi.

The four-page CPU replay through the last three performance changes matched all 596 tensors, including 86 parameters, 293 buffers, 190 optimizer tensors, the RNG state and 26 live gradients. At `538ec9555c`, 62 look, compose and performance tests passed. At `b254d26e52`, 58 propagation, intuition, growth and performance tests passed, including gradient checks on cycles and explicit external graphs. On CPU and Metal, strict Inductor compiled the native propagation and stage pullback on acyclic and cyclic examples with and without part evidence. On the current commit the combined identity, reconstruction, look, spread, candidate-growth, compile, memory and checkpoint suite passed 139 with 9 skips. This includes the batched-versus-scalar spelling inverse test. The historical groups overlap with this run. The item 1.8 omitted-word failure remains recorded in `suggestions.txt`.

On `b254d26e52`, seed 1505 and the same lesson, the four-page, one-exploration control reached restored 122/122 sampled and evaluated in 36 updates and 41.28 s. Four pages with three explorations reached 122/122 in both modes in 18 updates and 33.58 s. Eight distinct pages with one exploration per page reached 121/122 sampled and 122/122 evaluated in 27 updates and 47.71 s. Both larger schedules evaluate sixteen paths per update. At the current commit the separate twelve-update steady CPU runs sustained 6.28 sentences/s for four pages with three explorations and 12.95 sentences/s for eight pages with one exploration. The three alternative walks together took 277 ms in looks and 67 ms in scoring on the separately audited step. In a separate update at batch four, one exploration spent 80 ms in looks and 24 ms in scoring on CPU, and 405 ms and 244 ms respectively on Metal. The greedy walk took 57 ms and 24 ms on CPU, and 428 ms and 240 ms on Metal. `--path-audit` times the actual callbacks with synchronization at their boundaries.

Two fresh readers at the current training commit scored 122/122 sampled and 122/122 evaluated after restore. One used the preselected timing seed and the other was an independent unseeded gate start. All five learning gates passed, including numeric XOR convergence and class, raised-class denotation, and verbal XOR 4/4 into grammar and 4/4 through the inverse after 400 epochs. The five gates had also passed at `538ec9555c`. On `a8bdee4595`, the same unseeded gate had scored 3/4 at both boundaries. Its missed pair entered grammar in reverse order. Seed 606702945 gave 4/4 on both release and the candidate in a paired diagnostic. The tests keep the required assertions.

The causal word forecast is path-local. The physical native percept cache still reuses the greedy native prior on a later path with a different history. An audited three-exploration update had source order 2, 4, 0 on its third greedy page and 4, 0, 2 on its first exploration. The greedy walk advanced native state seven times, the following three walks zero times. All four rows selected greedy in that update. Before claiming isolation, compare a mixed set of winners to isolated histories including native gradients and persistent state. The open choice is the weighting of simultaneous winning observations in the single saved native mean and variance. Replaying ascending source columns within each look preserves the 1.7 update order. Pooling all winners from one look changes that weighting. Exact-prefix sharing can follow causal parity.

Claude's closest overlap is in `BasicPercepts`, `PerceptProperties`, `Places` and `Spaces` for number, place coordinates and observed part bounds, plus `CandidateGrowth`, `ConceptSpread`, `ReferenceClose`, and the retained event operands in `Memory` for grounding. The rounded-location `BasicPercepts.place_cones` dictionary and the checkpointed native predictor moments are additional model-side state. The native local predictor reads per-stream context and admitted rows. Its shared moment buffers update and serialize, but the ordinary forecast has no direct read of them. `ModelState.when_time` is still a scalar in the implementation. The trace below preserves the intermediate measurements and earlier scoped failures.

## Earlier measurements on the learned morphology and host transfers

Item 1.5 remains open. The verified base is `ad8fafea927839c0d7f92d0aee35c913d02b2612` and the measured local training commit is `a8bdee4595079b385b38a2c974abecb5bc5dbccf`. The base contains the reviewed 1.7 handoff `d38029113103ce5e9cf1223eaa55c9181010d649`.

| Four paced pages, seed 1505 | 1.7 | Local training |
| --- | ---: | ---: |
| CPU sentences/s | 6.99 | 9.92 |
| MPS sentences/s | 0.786 | 1.63 |
| CPU first three updates, seconds | 0.938, 0.634, 0.576 | 0.919, 0.479, 0.434 |
| MPS first three updates, seconds | 3.352, 4.464, 4.932 | 2.252, 2.678, 2.937 |
| Wall time to restored 122/122 sampled and 122/122 evaluation | 97.89 s | 41.54 s |
| CPU in-sentence explicit reads | 23,609 | 4,845 |
| MPS in-sentence native scalar extractions / explicit device-to-host copies | 21,426 / 3,108 | 3,117 / 1,864 |
| Repeated native percept addresses per instrumented step | 3,696 | 402 |

These are eager training steps with backward and optimizer included. Throughput uses the last nine of twelve updates. All runs use float32 and one CPU thread, with synchronization on MPS. The CPU throughput run overlapped scoped tests. The preceding two checkpoints measured 10.26 and 10.48 CPU sentences/s on the same workload. The current count of 402 repeated addresses includes direct reads by the native morphological occurrence fold, which the earlier counter missed. The ordinary sentence still fails strict capture at the learned span comparison in `PartSpace._embed_ladder_word_major`. The matcher still needs a Python byte string for durable `WordIdentity` admission. The matched receipts are `delivered-layout-cpu`, `delivered-layout-metal` and `delivered-layout-reader` under `output/item1-5`. `reading-properties-strict-metal` contains the failing whole-sentence Inductor probe on the preceding spelling revision.

With one fixed fresh seed, four pages and three explorations reached 122/122 in both held-out modes in **18 updates and 36.25 s**. Eight independent pages and one exploration reached 121/122 sampled and 122/122 evaluation in **27 updates and 53.40 s**. Both schedules evaluate sixteen paths per update. The four-page, one-exploration control reached 122/122 in both modes in **36 updates and 44.89 s**. This measures one seed. The separate steady-state comparison is below.

Commit `76faa8a211` shares a prediction among disjoint streams arriving in the same look. Commit `7a52180dfe` replaces the deterministic target's one-hot form fold with exact overlapping window counts, limits that work to the delivered text where its brackets establish the bound, and preserves the full-width spatial loss. Two consecutive CPU optimization replays matched all 596 tensors, including 26 live gradients. The ladder/UTF-8/spelling/separator/performance group passed 78 with 5 skips, the reconstruction/place/scalar group passed 23, and the bank/identity group passed 86. These scoped counts are separate from the learning gates.

The native property fold in `8b34dc162a` matched scalar reference intersections on all 468 test extents on Metal in both eager and strict Inductor. The combined admission, attention and performance group passed 61 tests. Commit `af371b1669` preserves the earlier page's forecast and gradient when a second page reaches its word on a later look. The predictor/looks tests passed 26. The delivered-span transfer bound passed 45 ladder and spelling tests with 5 skips. Each of these three changes matched all 596 state and gradient tensors in its fixed real update replay.

On the three-exploration audit (seed 1505), page 3's greedy native arrivals were source columns 2, 4, 0. Its first alternative read 4, 0, 2. Greedy advanced the shared predictor seven times across the four pages. The three other walks each advanced it zero times and reused the greedy native readings. All four selected rows in this update were greedy. Seeds 1506 through 1509 also selected greedy on all pages of the inspected update. This is direct evidence of shared native history during scoring. It does not establish the effect on a mixed winner. The policy choice for retained native moments remains open: replay the winners' source columns in the existing order within each look, or update the one shared mean and variance from the winners pooled at that look. On the instrumented CPU step one exploration took 80 ms in looks and 24 ms in scoring. The corresponding Metal times in a separate run were 419 and 230 ms.

At `16e3c40fde` five fresh gates passed together: the restored reader scored 122/122 in both modes, numeric XOR class and convergence passed, raised-class denotation passed and verbal XOR scored 4/4 at both boundaries. The optional fragment suite had four passes and one failed leaf-count assertion. With seed 71203915, the previous and revised implementations produced identical leaves, actions, fragment runs and failure. The assertions remain intact.

A fresh paired verbal XOR run with seed 606702945 passed all four ordered pairs into grammar and back through its inverse on both 1.7 and the candidate. The seed was drawn before either run. This diagnostic was taken after the reconstruction and reference edits and before the lexical extent edit. The earlier unseeded 3/4 result remains recorded below.

`Places.meet` maintains `BasicPercepts.place_cones`, a Python dictionary keyed by rounded location wires. The entries retain two location lanes and the common content of occurrences seen there for nomination. The code now transfers all observed rows together. Changing the owner of these cones would affect place learning shared with 1.6. The native percept predictor's running statistics are also still shared between candidate walks. The word-level forecast and the native arrival are distinct interfaces.

## Learned spellings and causal word forecasts

Item 1.5 is open. The base is `ad8fafea927839c0d7f92d0aee35c913d02b2612`, which contains the reviewed `d38029113103ce5e9cf1223eaa55c9181010d649`. The measured local training code is `42c46002f5f42891d8e6e7dc9a322689202b84ec` on `main`. Commit `96f97d7b64` gives each exploration its own causal word forecasts. The later commit reads the morphological occurrence from its native part addresses and whole bounds. The positive and negative property lanes still enter the existing Pi. The ordinary training entry still uses host code.

| Four-page paced reading, batch 4 | Published 1.7 | Measured code |
| --- | ---: | ---: |
| CPU, sentences/s | 6.99 | 9.83 |
| MPS, sentences/s | 0.786 | 1.56 |
| CPU first three updates, seconds | 0.938, 0.634, 0.576 | 0.807, 0.425, 0.443 |
| MPS first three updates, seconds | 3.352, 4.464, 4.932 | 2.873, 2.846, 3.114 |
| Fresh lesson through restored 122/122 in both modes, seed 1505 | 97.89 s | 45.17 s |
| Explicit CPU tensor reads inside the sentence | 23,609 | 5,230 |
| Native scalar extractions / device-to-host copies inside the MPS sentence | 21,426 / 3,108 | 3,438 / 1,988 |
| Repeated native percept addresses on the profiled CPU update, legacy counter | 3,696 | 355 (partial) |

The throughput runs used one CPU thread, float32 and twelve updates including backward and optimizer, with the first three excluded from steady state. The MPS runner synchronizes the device. The timings above use eager training. The new occurrence fold alone passed strict Metal Inductor forward and backward against eager. The complete sentence still fails strict capture in the lexical stem, where variable-length learned intervals are converted to Python for `WordIdentity` and its consumers. Native percept forecasting also shares arrival state between candidate walks, separately from the word forecast corrected in `96f97d7b64`.

With sixteen candidate readings per update, four pages and three exploratory paths delivered 5.96 sentences/s. Eight independent pages with one exploratory path delivered 11.97. The three exploratory walks took 0.287 s to compose and 0.069 s to score on the instrumented four-page update. The Pi and Sigma promotion tables stayed unchanged throughout all four trials. This measures throughput and provisional growth. It does not establish an accuracy advantage for either batch arrangement.

The fresh unseeded gates on `96f97d7b64` passed the restored reader (122/122 sampled and evaluation), numeric XOR class and convergence, and raised-class tests. Verbal XOR scored **3/4 into grammar and 3/4 through its inverse** after 400 epochs. The learned reader supplied “there loving” for “loving there” before the grammar stage. This is a failed gate. On a separate fresh paired seed (387885085), both release and causal code scored 121/122 sampled and 122/122 evaluation, taking 49.99 and 29.79 seconds respectively. Some of those timings overlapped independent tests. The omitted-word issue remains tracked under item 1.8.

For 1.6 integration, review the shared percept staging and grounding interfaces in `BasicPercepts.py`, `WordIdentity.py`, `Spaces.py` and `Interpret.py`. The candidate word forecast touches `SentenceLooks.py` and `Models._run_sentence_word_bricks`. The outstanding grammatical close and native event history overlap `ClauseJournal.py`, `Comprehension.py`, `Meaning.py` and the grounding writers.

CPU replay of the morphological fold against its immediate predecessor matched all 86 parameters, 293 buffers, 190 optimizer tensors, RNG and all 26 live gradients. The morphology, Pi, gather and looks group passed 34 tests, and the two regressions for ordered extents and tied part gradients passed separately. The causal-word group passed 91 tests, including its compiler test with a writable cache. These groups overlap.

## Earlier checkpoint before the learned-span changes

Item 1.5 is open. The training code on local `main` is `e2e2cd472374881b4d225ba7cd13e9a442963422`. Its base is `ad8fafea927839c0d7f92d0aee35c913d02b2612`, which contains the reviewed 1.7 handoff `d38029113103ce5e9cf1223eaa55c9181010d649`. Integration with 1.6 and push remain with Alec.

| Matched metric | Released 1.7 | Current |
| --- | ---: | ---: |
| CPU sentences/s | 6.99 | 10.09 |
| Metal sentences/s | 0.786 | 1.448 |
| CPU first three updates, seconds | 0.938, 0.634, 0.576 | 0.791, 0.414, 0.421 |
| Metal first three updates, seconds | 3.352, 4.464, 4.932 | 2.332, 2.963, 3.262 |
| Native partitioned CPU Inductor, cold first update / steady sentences/s | 3.85 s / 6.50 | 3.41 s / 9.31 |
| Fresh lesson through restored 122/122 sampled and evaluation, seed 1505 | 97.89 s, 54 updates | 42.65 s, 36 updates |
| Explicit CPU tensor reads in sentence / inside looks | 23,609 / 2,409 | 6,200 / 223 |
| Repeated native percept addresses in an update, including padding | 3,696 | 629 |
| Profiled time composing / scoring two candidate paths, CPU seconds | 0.282 / 0.115 | 0.191 / 0.072 |

Both checkouts used the four-page paced reading workload, batch four, float32, one CPU thread, PyTorch 2.15 and seed 1505. All twelve updates included backward and the optimizer. The nine steady timings exclude the first three updates. Metal was synchronized. The default measurements used eager execution. The separate partitioned Inductor run captured one graph. A separate three-exploration CPU run achieved 6.16 sentences/s. Eight independent pages with one exploration achieved 12.47. Both configurations evaluate sixteen candidate paths per update. The three exploratory suffixes and their scoring callbacks together took 0.358 s in the instrumented update. None of the four candidate walks mutated the persistent growth tables. The selected rows in that update were all greedy. Mixed winners are covered separately by the trial tests.

The Metal dispatch census inside the sentence fell from 21,426 to 4,332 native scalar extractions and from 3,108 to 2,024 explicit device-to-host copies. Strict capture of the complete ordinary entry failed on both CPU Dynamo and Metal Inductor at the lexical stem's conversion of learned span endpoints to Python. The tiling, ragged bounds and retained-pair translation passed separate fullgraph Metal checks.

Two fresh restored readers scored **122/122 in sampled and evaluation modes**. Verbal XOR scored **4/4 into grammar and 4/4 through its inverse**. The numeric XOR convergence and raised-class tests passed in the same ten-test group. Scoped verification included 70 look, growth, memory and expectation tests and 61 overlapping retention, performance and reduction tests. The broader expectation, trust and generation group passed 121 with one skip after two compiler tests were rerun with cache access. The prior omitted-word failure and the release-reproduced diagnostics are described below. These counts do not represent a full-suite run.

The performance replay against the preceding winner-only implementation matched all 86 parameters, 293 buffers, 190 optimizer tensors, the RNG state and 25 of 26 live gradients. Summing repeated reads of the same percept changed 46 elements of the WholeSpace primitive-property gradient by at most 1.86e-9. The largest absolute value of that gradient was 0.00463. The separate release-to-winner-only behavioral change is recorded below.

**Remaining work and 1.6 overlap**

- The agreed temporal rule is one typed idea or relation head **per stream per tick**. Four simultaneous streams share the clock advance. `ReadingMemory.commit` copies reachable STM0 ideas and interns STM1 pairs with translated addresses. The normal grammatical close still uses `PartClosing` and `Models._commit_sentence` to write the admitted clause. Its recent relation head consumes an extra occurrence name. The temporal lookup and legacy checkpoint migration are under review. Claude's grounding can meet this interface in `ClauseRow`, `MemberRead`, and the reference and event writers.
- `ChunkJournal`, `ReadingIndex`, `SentenceStrata` and `Comprehension` still use the Python grammar record while scoring the neural chooser. The approved replacement is the operator-prefixed STM head, with the kept head copied to LTM. Bind, transpose, inverse credit and later access to pre-action operands must survive this change. The broader MTM staging review belongs to 1.4.
- `WordIdentity` and `PartSpace._embed_ladder_word_major` use byte-keyed dictionaries during lexical admission. A newly seen form can update parts of earlier forms. Porting spelling and part lookup to the native percept inventory is under review. `BasicPercepts`, `Spaces`, `Places` and `ConceptTrust` overlap with number and felt-world work. The current implementation preserves the binary64 clock and trust calculations.
- `ModelAttention._percept_arrivals` retains per-word diagnostics on device. The local objective reads `_percept_arrival_loss`; a repository search found no reader of the accumulated diagnostic list.

Reproduce the CPU and Metal comparisons with the archive commands below and `--steps 12 --dispatch-census`. Use `--strict-sentence eager` for the failing complete-entry capture probe, `--compiled-step` with `MODEL_COMPILE=inductor` and a fresh cache for the partitioned result, and `--lesson` separately for time to accuracy. Local receipts under `output/item1-5/` include `e2e2-paired-base`, `e2e2-cpu`, `e2e2-paired-metal-base`, `e2e2-metal`, `e2e2-compiled`, `collection-lesson`, `collection-learning.log`, `e2e2-three-audit` and `collection-strict`.

## Earlier performance checkpoint

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

Local training commit `883fbb21c67d18049ec3af5b95c632507c19c299` follows the measured stopping point `5a393548e30f0e8ba0c5fbacc15b52977b6c6a3b`. Item 1.5 remains open. The revised design replaces the grammar reader with direct writes to STM ideas and relations. STM copies the retained head directly to LTM. Each memory holds at most one root idea or relation per model time. MTM stages basic percepts and concepts. See the [representation design](2026-11-24-item1-5-device-sentence.md).

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
