| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.44 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.44 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.44 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.44 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.44 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.44 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.07 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.07 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.56 GiB | ending 0.19424480199813843, calls 21, predictions [0.4268348217010498, 0.546610951423645, 0.5471051931381226, 0.4290872812271118] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.56 GiB | ending 0.25000929832458496, calls 1, predictions [0.5020442605018616, 0.5020178556442261, 0.5021454691886902, 0.5021388530731201] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.56 GiB | ending 0.19917547702789307, calls 88, predictions [0.441373735666275, 0.5562533736228943, 0.5494666695594788, 0.44944387674331665] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.56 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.62 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.56 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.1750003695487976; reconstruction 0.00014008326979819685 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.64 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.64 GiB | exact 1.0; where 1.0; output 0.17612579464912415; reconstruction 0.00022659002570435405 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17676472663879395; reconstruction 6.384231528500095e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1426496058702469; reconstruction 0.00032883230596780777 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.66 GiB | exact 1.0; where 1.0; output 0.17530153691768646; reconstruction 8.33908052300103e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.66 GiB | exact 1.0; where 1.0; output 0.17240190505981445; reconstruction 6.906119961058721e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17105071246623993; reconstruction 0.00014677886792924255 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.19621656835079193; reconstruction 0.00019127983250655234 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17712663114070892; reconstruction 7.197300874395296e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17455093562602997; reconstruction 0.00011245875066379085 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.20453986525535583; reconstruction 0.0002631718816701323 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16727390885353088; reconstruction 0.00028553203446790576 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | failed; peak 4.65 GiB | exact 0.75; where 1.0; output 0.17942090332508087; reconstruction 0.00021192835993133485 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | failed; peak 4.65 GiB | exact 0.75; where 1.0; output 0.1751132607460022; reconstruction 5.937687819823623e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.18664437532424927; reconstruction 0.00022261470439843833 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17769098281860352; reconstruction 1.956608502950985e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | failed; peak 4.65 GiB | exact 0.75; where 1.0; output 0.16055122017860413; reconstruction 0.000290547963231802 |
