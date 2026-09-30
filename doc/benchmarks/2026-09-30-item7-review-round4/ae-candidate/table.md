| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.65 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.65 GiB | ending 0.18352971971035004, calls 21, predictions [0.4331952631473541, 0.5834706425666809, 0.5754163861274719, 0.43896788358688354] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.65 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.65 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.65 GiB | ending 0.2500310242176056, calls 1, predictions [0.4994387924671173, 0.49946677684783936, 0.49923211336135864, 0.4993826150894165] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.65 GiB | ending 0.19978061318397522, calls 176, predictions [0.4462878108024597, 0.5559225678443909, 0.5509212613105774, 0.4484117925167084] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.65 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.42 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.48 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.48 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.67 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17500074207782745; reconstruction 0.00012342576519586146; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.37 GiB | diagnostic exit 0; peak 12.19 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.11 GiB | diagnostic exit 0; peak 14.52 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.32 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1770041584968567; reconstruction 0.00020999956177547574; diagnostic exit 0; peak 14.48 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.29 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17682430148124695; reconstruction 0.00010467154788784683; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.66 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.17680276930332184; reconstruction 0.00010158443183172494; diagnostic exit 1; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.44 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17207513749599457; reconstruction 0.0001514504401711747; diagnostic exit 0; peak 14.47 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.58 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18007950484752655; reconstruction 0.0005705986404791474; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.08 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17579285800457; reconstruction 0.0; diagnostic exit 0; peak 14.63 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.57 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17272111773490906; reconstruction 4.29121391789522e-05; diagnostic exit 0; peak 14.29 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.32 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1750219464302063; reconstruction 0.00012053463433403522; diagnostic exit 0; peak 14.31 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.56 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18048463761806488; reconstruction 0.00017826177645474672; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.24 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1750274896621704; reconstruction 9.156729356618598e-05; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.18 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17505933344364166; reconstruction 2.6908501240541227e-05; diagnostic exit 0; peak 14.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.43 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.1841352880001068; reconstruction 0.0003722933470271528; diagnostic exit 1; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.07 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17439575493335724; reconstruction 0.00011258952144999057; diagnostic exit 0; peak 14.62 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.38 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16889502108097076; reconstruction 0.00018087706121150404; diagnostic exit 0; peak 14.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.55 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17685218155384064; reconstruction 6.544577627209947e-05; diagnostic exit 0; peak 14.57 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.31 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1853475719690323; reconstruction 0.00015522161265835166; diagnostic exit 0; peak 14.59 GiB |
