# Powers of two for all constants: audit of the integer constants (Claude, 2026-11-04)

**Status.** From Alec's answer to basic-percepts D1 ([spec §7](2026-11-04-item2-0-basic-percepts.md)): "I would prefer powers of two for all constants." Decided: every constant set by hand is a power of two, and a width made by adding such constants stays a sum (Alec, 2026-11-04: "Sums can stay sums."). This is an audit, not a change. The pass it implies is in todo item 2 (housekeeping), and its measured values are decisions collected at the start of item 0. The audit scanned `data/*.xml`, `data/matrix/*.xml`, `data/model.xsd`, `doc/Params.md`, the `TheXMLConfig` integer defaults in `bin/*.py` and the item 2.0 plans, and was re-checked line by line. Line numbers are from the 2026-11-04 working tree; Codex re-locates them. Seeds, versions, booleans, 0 and 1, rates, weights and ratios are excluded.

## 1. What is already a power of two

Every capacity, horizon, budget and cap in `data/model.xml` and `data/BasicModel.xml`:
- `generateBudget` 32, `ltmWindow` 1024, `forgetMinInterval` 256, `forgetHalfLife` 4096, `intuitionCapacity` 4096, `attentionBudget` 32, `maxResponseLength` 4096, `truthMaxEntries` 1024, `ltmCapacity` 1024, `relationCapacity` 8192, `equalityWitnessCapacity` 8192, `mtmPrepend` 64, `mtmCapacity` 64, `mtmHorizon` 16;
- the three identity parts `identityPairDim` 64, `identityLengthDim` 32 and `propertyDim` 256;
- BasicModel's InputSpace 8192 and 256, PartSpace 32768, ConceptualSpace 65536 with 32768 active, `serialWordCapacity` 256 and `ltmCapacity` 262144.

The item 2.0 plans add one capacity, `basicPerceptCapacity` 262,144 = $2^{18}$.

## 2. Sums, which cannot all be powers of two

Every percept, part and whole event carries a fixed band of 8 (four `.where`, four `.when`; `bin/architecture.py` about 29-46); concepts and outputs carry none. A content that is a power of two of 16 or more therefore gives an event width $\equiv 8 \pmod{16}$, and the only all-power-of-two case is 8 + 8 = 16. `identityLengthDim` is fixed at 32 by code, as the one-hot over exact word lengths of 1 to 32 bytes (`bin/WordIdentity.py` about 36 and 117-120).

The widths most configurations declare do not run. `resolve_identity_layout` (`bin/WordIdentity.py` about 17-59, called at `bin/Models.py` about 1783) rewrites them at load for every embedding configuration with `identityByParts`:
- input, percept and whole events: 360 = 64 + 32 + 256 + 8;
- concepts: 360 plus the declared concept-minus-percept difference, which gives 1256 on `BasicModel`, `BasicModel_corpus` and the tied benchmark, 488 on `MM_grammar` and `XOR_grammar`, and 392 on `MM_corpus_orders`.

Of 172 declared non-power-of-two widths, 152 are overwritten. Declared widths that are powers of two (1024, 64, 16) are overwritten too. The 20 that run are `data/model.xml`'s numeric-configuration 9s (about 444, 469, 497 (ModalSpace), 529), `MM_xor`'s twelve 14s (`identityByParts` false), `MM_grammar.xml` about 64, and the band-free OutputSpace widths at `stream_smoke.xml` about 61 (104), `MentalModel.xml` about 86 and `RamsifiedModel.xml` about 83 (100). `data/BasicModel.xml`'s header (about 13-14) still describes 96- and 104-wide events.

The sums, by family:
- **Fixture events** of 9, 10, 12, 14 and 20 (content 1, 2, 4, 6 and 12 plus the band). The values become 16 with content 8.
- **Declared percept widths** 104 and 136, and **concept widths** 232, 1000 and 1032: dead inputs except for their differences.
- **The percept row** 352 = 64 + 32 + 256 (basic percepts). A power-of-two row of 256 or more needs a remainder part (`identityPairDim` 224 for 512) or padding (512, 0.54 GB at $2^{18}$ rows).
- **The concept width** 1256 = 360 + `meaningWidth` 896. Declaring a 360-wide percept and a 1024-wide concept gives 1024, with `meaningWidth` either 664 or a power of two such as 512 leaving a reserve. `MereologicalCodes` takes `min(meaningWidth, width - 360)` (about 45-48); whether the reserve is harmless to `pole_stride` (`IndependentComponents.py` about 366) is unverified.
- **`conceptLayers` = max order + 1**: 5 on the BasicModel family, `MM_grammar_wording` and the ladders, 3 on `MM_corpus_orders`. `conceptLayers` = `bindingDepth` + 1 is a convention, which no code ties together.
- **Preallocated seats**: 20 = 4 words + 4 covers + 8 mints + 4 spare (`MM_grammar.xml` about 55, `MM_xor_loopback.xml` about 62) and 18 (`XOR_grammar.xml` about 84).

## 3. Free sizes that could simply be powers of two (the item 2 pass)

| Constant | Value | Where |
|---|---|---|
| `maxDocs` | 10000, 2000, 200 | `model.xml` about 229; `BasicModel.xml` about 117, `MM_grammar_wording.xml` about 33; `stream_smoke.xml` about 16; defaults `Models.py` about 25972, `recon_bench.py` about 125 |
| `batchSize` | 12, 6; default 10 | `MM_20M_xor` 49, `MM_add_verb` 39, `MM_global` 51, `MM_qa` 62, `MM_reading` 47; `MM_query_reasoning` 50; `Models.py` about 26025, `recon_bench.py` about 132 (against `model.xml`'s 64) |
| `verbGroups` | 24 | `Layers.py` about 2665, `Params.md` about 415 |
| `placementNegatives` | 5 | `model.xml` about 513, `MereologicalCodes.py` about 71 |
| `identityOnes` | 3 | `model.xml` about 462, `Spaces.py` about 10624, `WordIdentity.py` about 32 |
| `meaning_ones` literal | 6 or 3 | `MereologicalCodes.py` about 52 |
| `bindingDepth` | 3 | sixteen MM fixtures (`MM_20M_xor`, `MM_add`, `MM_add_verb`, `MM_corpus_orders`, `MM_global`, `MM_grammar`, both `MM_ltm_consolidation` fixtures, `MM_math`, `MM_math_chain`, `MM_qa`, `MM_query_reasoning`, `MM_reading`, `MM_sequence_predict`, `MM_xor`, `MM_xor_fixture`) |
| fixture `nVectors` | 200, 40, 10, 6 | `MM_math` 51, `MM_xor` 45, `idempotent` 114; `idempotent` 85; `idempotent` 98; `XOR_pos` 25, 40 |
| fixture slots | 6, 10 | `XOR_grammar` 63, 68, 75, 89 and `XOR_pos` 23, 30, 38, 46; `idempotent` (seven) |
| `stmCapacity` | 7 | `idempotent.xml` about 109 ("7 $\pm$ 2") |
| OutputSpace `nDim` | 100 | `MentalModel.xml` about 86, `RamsifiedModel.xml` about 83 |
| hidden width | 20 | `ergodic.xml`, `ergodic-only.xml` |
| cadences | 100 | `checkpointEveryBatches` (`BasicModel` 187, `BasicModel_corpus` 69, `MM_grammar_wording` 58, both ladders), `branchDiagnosticsEvery` (`BasicModel` 152, `BasicModel_corpus` 46) |
| fixture `numEpochs` | 3 to 400 | `model.xml` 248 and the fixtures |

A fixture whose tests depend on one of these values keeps it, with a comment saying why.

## 4. Measured values (decisions at the start of item 0)

- `batchSize` 28 on the BasicModel family (`BasicModel.xml` about 131, `BasicModel_corpus.xml` about 39, the tied benchmark about 43; literal at `corpus_capacity.py` about 335), the protected-memory operating point on the 36 GB MPS host: 16 or 32.
- `maxDocs` 11,207 (`BasicModel_corpus.xml` about 32), chosen for 1,000,014 training sentences: 8,192 or 16,384 changes the corpus size.
- The concept width 1256 (`meaningWidth` 896), §2.
- `conceptLayers` 5, orders 0 to 4: 4 or 8, §2.
- The recurrence thresholds at 3: `conceptMintCount`, `raiseCount` (schema minimum 2) and `mereologyRefinePatience`: 2 or 4.

## 5. Kept as they are

- Fixed counts: WholeSpace `nVectors` 9, the eight canonical property rows plus the word row (`Spaces.py` about 18910-18920); MNIST's 784, 10 classes and `classificationMax` 9; the decomposition chooser's five weights, which go with its teacher.
- Values that are not sizes: the ports 8001 and 8002, `recurrenceAngle` 15 degrees, the identifier `conceptIds` 20001 and `mathTestDepths`.

## 6. Found on the way (item 2)

- `timeout` 120 (`model.xml` about 333) is read by nothing; delete it.
- `data/model.xsd` (about 1207) gives `mtmPrepend`'s default as eight; the configuration and the code use 64.
- `doc/Params.md`'s source references at about 918 and 930 point one line off and twelve lines off.
- `data/model.xsd`'s comment (about 55-64) and `doc/Spaces.md` (about 196-203) still say `subsymbolicOrder`.
