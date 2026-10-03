# Part 4 ports

Complete old and new bodies are in `part4-ports.json`; whole sources are in
`part4-before.zip` and `part4-after.zip`. No gate threshold, seed, production
configuration value or memory ceiling changes in these ports.

| File / case or fixture | Reason and retained contract |
|---|---|
| `test_compile_static_loop.py::test_per_word_body_callable_with_static_signature` | Its direct word call now stages the reconstruction teacher and journal rule maps required by the normal forward. The original callable/state assertion stays. Saved failure: weekly worker 011. |
| `test_compiled_word_chunk.py::_concept_space` | The hand-built ConceptualSpace now includes its Error registry. Saved failure: `chunk-registry-before/`. |
| `test_compiled_word_chunk.py::test_k2_adapter_matches_legacy_loop_stm_loss_and_gradients` | All original STM, raw-loss and gradient comparisons stay; trained relative cost and its gradient must also agree across chunks. Repeated capture is covered by the unchanged one-graph test. |
| `test_concept_readout_l1.py::test_real_runbatch_stages_l1_once_and_reports_it_separately` (both arms) | Spec §10.2a retires the detached inverse for retained meronomy readings, including an XML that previously requested it. The fixture is shortened and numerical because compilation is not its subject. Every applicable step must stage L1 once; both arms must update live readout coefficients; evaluation must leave them unchanged. The separate detached report retains its exact configured strength. The former detached-root assertion is retired by the explicit reconstruction scope decision. |
| `test_reconstruction_bank_contract.py::test_staging_rejects_missing_or_unusable_reconstruction_bank` | Structural corruption still raises. A structurally valid bank with no admitted candidate now counts the unavailable sentence, as explicitly decided in §10.2a. |
| `test_reconstruction_bank_contract.py::test_failed_first_sight_admission_cannot_become_null_reconstruction` | The historical selector's hard-error contract is replaced by explicit unavailable-candidate accounting; the added companion checks the count. No fabricated NULL reconstruction is accepted. |
| `test_word_store.py::test_missing_surface_snapshot_is_a_staging_error_without_input_fallback` | Its historical selector now checks a zero contribution and absent candidate. Missing admission is counted rather than treated as a structural error; no input-text fallback is introduced. |
| `test_supplied_answer_training.py::_sentence` | The minimal trial fixture initializes the reconstruction scope used by the new registry. The answer-reach and same-parameter-state assertions stay. |

The added tests cover the relative baselines, penalty separation, masked and
single-row reductions, missing-candidate accounting, reconstruction precedence,
answer-only projection ownership, sparse gradients, inactive-row prediction,
and clearing registered targets on reset. The ledger distinguishes these new
tests from ports.

The weekly runner's complete old/new function bodies are included too. Its
continuation dispatches only unattempted cases after a resource stop. Killed
cases stay red; it does not repeat a measurement or raise a guard. The frozen
source shares only the existing runtime through `.venv`, so subprocess tests
can find their interpreter.
