# Mechanism diagnostics

## G0: why was the actual compose carrier outside the cube?

| Question | Cause and repair |
|---|---|
| Does the bank use production [0, 1] codes? | Yes. `G0.json` measures range [0, 1], with zero negative coordinates. |
| Did the binding carrier preserve that representation? | No. `WordIdentity.projection` mapped native forms into signed coordinates: the observed carrier range was −1.810452 to .984518, with 314 negative coordinates. The identity projection and `Interpret.binding_atoms` route are removed; interpretation now consumes the native code and its activation. |
| Does “the cat” tie richer candidates by distance? | No. Distances to the two richer descriptions were 4.123106 and 3.741657. Their request coverage is exactly 1 in both cases; candidate coverage is .451613 and .5. The production cube adds support for defining parts, so the meet equals the less specific request, not the richer candidate. |
| How is the initial content tie represented? | The existing resolver receives both requested meet ratios. Its initial candidate-coverage weight is 0 and request-coverage weight is 1; both remain trainable. This implementation choice preserves the required underspecified tie without suppressing either feature. There is no distance feature. |
| Does the repaired carrier stay in the cube? | `G0-after.json` repeats the native and actual interpretation-carrier measurements: both are [0, 1], zero negative coordinates. The initial missing-evidence script error is retained separately and is not a measurement. |

## Native close: why did some descriptions still violate the cube?

| Question | Cause and repair |
|---|---|
| What did the native failing journal contain? | A signed relation-address token (±.098058) was lowered and treated as an NP description. The relation had no descriptive point. A separate surface operation's additive numerical carrier exceeded 1 even though its declared semantic operand was the body. Full failing coordinates, frames and journals are in `cube-failure.jsonl` and `cube-journal.jsonl`. |
| What now supplies the description? | The existing closing journal reads its retained semantic operand when the operation declares one. A relation address without a descriptive point cannot create an individual. No coordinates are clipped, no relation token is made into a region, and no operation is replayed. |
| Which invariant checks this? | `test_reference_closing_reads_surface_meaning_and_never_mints_a_relation_address` requires the raw body's point and no individual from a relation address. The whole §14 file and native unlabelled batch subsequently complete in `native-08`. |

## Decomposition: why did the first arm fail before recording its questions?

| Question | Cause and repair |
|---|---|
| What failed? | Both original certificates in `bind-decomposition-zero-candidate` stopped at the unchanged footprint assertion. The unexpected key was `conceptualSpaces.2.syntacticLayer.layers.2._adverb_purchase`. |
| Why did the read mutate the model? | `LiftLayer.apply_adverb` stored a diagnostic tensor measuring nonzero input coordinates. The new native geometry reached this candidate in the ordinary thought search. No consumer read the diagnostic; it was outside the episode's allowed write set. |
| What changed? | The diagnostic member, its constructor initializers and the write are removed. Adverb numerical outputs, the forced scope, the keep rule, the certificate, the seeds and saved initial parameters are unchanged. The existing adverb test now also checks that evaluation leaves the layer's attributes unchanged. |
| Was initialization selected again? | No. `bind_gate.py` requires exact equality against the initial parameter snapshots already saved by the failed run. The enabled arm must match the zero arm as well. The receipt reports differences from the step-1 model separately because the signed identity projection and old resolver matrix were retired. |

## Fixture migration

| Question | Cause and repair |
|---|---|
| Why were early reference tests red? | Their stand-in owner had no retained LTM, or their random signed vectors were fed to a cube meet. The port uses native concept seats and a real retained store. The 18 close/mention assertions, G1–G3 generators, access bounds and revision assertions remain. |
| Why did streaming predictor fixtures differ? | They compared copied observations with reads from committed rows. Streaming and packed forms now bind the same retained rows; direct word/thought context consumers read the address index through `_read_context`. Evidence uses the store's actual `c_plus` / `c_minus` lanes. |
| Does forgetting erase an individual's folded content? | No. Ordinary retention loss leaves the native fold unchanged. A binding membership revision explicitly refolds affected individuals from the remaining retained occurrences. The new regression distinguishes those two operations. |
| Are failed diagnostics hidden by later passes? | No. Every completed attempt and every failed case is retained in `runs.md`, `failed-cases.json` and its original worker reports. The repair checks are mechanism certificates, not a new grade of any failed learning run. |

## Landing sweep: model-time ordering

| Question | Cause and repair |
|---|---|
| Why did the reversed event implication pass as ordered? | The writer had moved to model-time phasors, but `ThoughtLTMCapability.relation_evidence` still decoded them with the store-capacity address encoder. |
| What now owns the scale? | The reader uses `RowTime.encoder`, the same declared ladder as the writer. Unknown zero phasors from the one-time old-clock migration cannot prove an order. |
| Which regression changed? | The existing two-direction event certificate explicitly supplies its event clocks instead of using source positions as clocks. Its assertions are unchanged. A separate regression requires unknown event time to return no evidence and accounts for both endpoint reads. |
| What about clock independence? | Programs, content, answers and every non-clock buffer remain exactly equal across all nine clock offsets. New occurrence stamps must shift by the offset, prior stamps and unused capacity must remain unchanged, and the writer counter must equal the actual maximum write time or its prior value. |
| Which other clock expectations were ported? | Definitions now record when they were learned; re-witnesses keep their stamp; capacity growth preserves phasors bitwise. The live stream certificate checks model processing time and unchanged re-witness time. The reader inventory records the current owners. |

## Landing sweep: cyclic generic binding

| Question | Cause and repair |
|---|---|
| What failed in the native math-chain smoke? | The generic occurrence menu included the unresolved NP itself. A kept bind then tried to make its role reference point back to that NP, and the store rejected a semantic constituent cycle. A containing generic row would have the same defect indirectly. |
| How is the menu repaired? | Before scoring, the resolver excludes a candidate occurrence whose typed role-reference graph reaches the NP being filled or its enclosing query row. Native concept identities remain legal. This is structural eligibility, with no change to score weights, initialization, keep rule or proof scope. |
| What proves the boundary? | The new bind regression offers self and two levels of containing rows with perfect content coverage. With no legal antecedent the site remains open with zero evidence; with an independent occurrence it binds that occurrence and keeps the NP's observed content bitwise unchanged. |

## Landing sweep: remaining fixture ports

| Question | Cause and repair |
|---|---|
| Why did the native symbol-handle certificate lose its altered-value contrast? | Its unit presence now produces the native cube exactly after removal of the signed projection. Explicit nonunit presence creates the intended attenuation; the original two-lane, exact-handle and no-nearest-key assertions remain. |
| Why did a packed context test fail to unpack? | Committed STM now contains `SituationAddress`. The test reads it through `_read_context` and retains its depth, payload and occupied-role assertions. |
| Which stand-ins were stale? | The manually constructed sentence fixture now declares its cube mode; the type-family fixture provides native `individual_ids`; the attention fixture provides the admitted-form interpreter. Production has no fallback to the retired owners. |

The three strict artifact-load failures have a separate
[migration hold](migration-hold.md), decided by Alec. They are not fixture
ports, skips or repaired gates.
