# Reconstruction scope by unchanged configuration

Merged XML inspection; numeric configurations do not use text reading. Meronomy grammar readings reconstruct automatically. The remaining text reading modes retain their XML setting until the explicitly deferred migration.

## exempt: old reading mode

| Configuration | Previous tied setting | Gains tied reconstruction |
|---|---|---|
| `data/HeadEmission.xml` | False | False |
| `data/LM_5M.xml` | False | False |
| `data/LM_5M_IR.xml` | False | False |
| `data/MM_20M_legacy.xml` | False | False |
| `data/MM_400M.xml` | False | False |
| `data/MM_5M_AR.xml` | False | False |
| `data/MM_5M_IR.xml` | False | False |
| `data/MM_add.xml` | False | False |
| `data/MM_add_verb.xml` | False | False |
| `data/MM_boolean.xml` | False | False |
| `data/MM_bpe.xml` | False | False |
| `data/MM_decode.xml` | False | False |
| `data/MM_global.xml` | False | False |
| `data/MM_grammar.xml` | False | False |
| `data/MM_init_scale.xml` | False | False |
| `data/MM_ltm_consolidation_fixture.xml` | False | False |
| `data/MM_ltm_consolidation_serial_fixture.xml` | False | False |
| `data/MM_ltm_consolidation_stateful_fixture.xml` | False | False |
| `data/MM_masked_semantic.xml` | False | False |
| `data/MM_math.xml` | False | False |
| `data/MM_mereology.xml` | False | False |
| `data/MM_mereology_serial.xml` | False | False |
| `data/MM_meronomy_smoke.xml` | False | False |
| `data/MM_overlap_tiling.xml` | False | False |
| `data/MM_phrase_decode.xml` | False | False |
| `data/MM_qa.xml` | False | False |
| `data/MM_query_reasoning.xml` | False | False |
| `data/MM_reading.xml` | False | False |
| `data/MM_sequence_predict.xml` | False | False |
| `data/MM_shamatha.xml` | False | False |
| `data/MM_sparse_concept.xml` | False | False |
| `data/MM_symbol_tower.xml` | False | False |
| `data/MM_symbolic_iter.xml` | False | False |
| `data/MM_xor.xml` | False | False |
| `data/MM_xor_fixture.xml` | False | False |
| `data/MM_xor_loopback.xml` | False | False |
| `data/MM_xor_step3.xml` | False | False |
| `data/MM_xor_step4.xml` | False | False |
| `data/MentalModel.xml` | False | False |
| `data/POS_smoke.xml` | False | False |
| `data/RamsifiedModel.xml` | False | False |
| `data/XOR_grammar.xml` | False | False |
| `data/XOR_pos.xml` | False | False |
| `data/XOR_recon.xml` | False | False |
| `data/XOR_spaces.xml` | False | False |
| `data/idempotent.xml` | False | False |
| `data/stream_smoke.xml` | False | False |
| `data/tomatoes.xml` | False | False |
| `data/xor.xml` | False | False |

## perception: no grammar

| Configuration | Previous tied setting | Gains tied reconstruction |
|---|---|---|
| `data/MM_20M_xor.xml` | False | False |
| `data/XOR_exact.xml` | False | False |
| `data/matrix/MM_20M_xor_noraise.xml` | False | False |

## perception: numeric configuration

| Configuration | Previous tied setting | Gains tied reconstruction |
|---|---|---|
| `data/ergodic-only.xml` | False | False |
| `data/ergodic.xml` | False | False |
| `data/mnist.xml` | False | False |
| `data/model.xml` | False | False |
| `data/simple.xml` | False | False |

## understanding

| Configuration | Previous tied setting | Gains tied reconstruction |
|---|---|---|
| `data/BasicModel.xml` | True | False |
| `data/BasicModel_answers_benchmark.xml` | False | True |
| `data/BasicModel_answers_tied_benchmark.xml` | True | False |
| `data/BasicModel_expectation_benchmark.xml` | False | True |
| `data/BasicModel_long_tied_benchmark.xml` | True | False |
| `data/BasicModel_output_tied_benchmark.xml` | True | False |
| `data/MM_20M_fineweb.xml` | False | True |
| `data/MM_20M_grammar.xml` | False | True |
| `data/MM_grammar_wording.xml` | True | False |
| `data/MM_ladder.xml` | False | True |
| `data/MM_ladder_idiom.xml` | False | True |
| `data/MM_ladder_text.xml` | False | True |
| `data/MM_ladder_textpacked.xml` | False | True |
| `data/MM_nanochat_grammar_gate.xml` | False | True |
| `data/MM_nanochat_grammar_pilot.xml` | False | True |
| `data/matrix/MM_20M_grammar_reading.xml` | False | True |
| `data/matrix/MM_20M_grammar_wordstore.xml` | False | True |
