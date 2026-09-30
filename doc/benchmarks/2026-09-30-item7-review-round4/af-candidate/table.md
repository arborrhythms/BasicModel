| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.46 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.68 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.56 GiB | ending 0.18470481038093567, calls 20, predictions [0.406843900680542, 0.5525275468826294, 0.55282062292099, 0.41604840755462646] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.56 GiB | ending 0.2500334680080414, calls 1, predictions [0.5008697509765625, 0.5008060336112976, 0.5007516145706177, 0.5008191466331482] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.56 GiB | ending 0.19983349740505219, calls 21, predictions [0.3971468210220337, 0.5524064898490906, 0.5615389943122864, 0.49901944398880005] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.56 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.39 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.48 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.64 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.50 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.34 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17500551044940948; reconstruction 0.00011743458162527531; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.16 GiB | diagnostic exit 0; peak 12.19 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.06 GiB | diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.28 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17532512545585632; reconstruction 0.00015146333316806704; diagnostic exit 0; peak 14.54 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.30 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1767597198486328; reconstruction 2.443423909426201e-06; diagnostic exit 0; peak 14.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.22 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17500342428684235; reconstruction 4.11885921494104e-05; diagnostic exit 0; peak 14.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.31 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.17927032709121704; reconstruction 0.0002265395742142573; diagnostic exit 1; peak 14.57 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.26 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17350400984287262; reconstruction 0.00017739298345986754; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.35 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1593349128961563; reconstruction 0.0002436428767396137; diagnostic exit 0; peak 14.56 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.24 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17434559762477875; reconstruction 4.568704025587067e-05; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.11 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16555556654930115; reconstruction 0.00014743518840987235; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.44 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.1748112440109253; reconstruction 7.186651146184886e-06; diagnostic exit 1; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.45 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17676018178462982; reconstruction 0.00018228746193926781; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.38 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17408227920532227; reconstruction 5.039612005930394e-05; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.43 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18644385039806366; reconstruction 0.0002171038941014558; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.48 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1500653326511383; reconstruction 0.0002841603709384799; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.19 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1769915074110031; reconstruction 7.935212488519028e-05; diagnostic exit 0; peak 14.61 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.03 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1854930967092514; reconstruction 0.00010558986105024815; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.17 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17103761434555054; reconstruction 0.00015791326586622745; diagnostic exit 0; peak 14.34 GiB |
