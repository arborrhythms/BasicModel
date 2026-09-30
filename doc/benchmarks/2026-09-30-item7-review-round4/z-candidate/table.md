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
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.58 GiB | ending 0.19531139731407166, calls 22, predictions [0.4636852443218231, 0.5465946793556213, 0.571231484413147, 0.42050299048423767] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.58 GiB | ending 0.2500094771385193, calls 1, predictions [0.5048120617866516, 0.5047405362129211, 0.5046858191490173, 0.5045636892318726] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.58 GiB | ending 0.19958671927452087, calls 34, predictions [0.5309670567512512, 0.5664649605751038, 0.6084302067756653, 0.4184989035129547] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.58 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.29 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.49 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.23 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.43 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1750103384256363; reconstruction 0.00014601046859752387; diagnostic exit 0; peak 14.48 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.58 GiB | diagnostic exit 0; peak 12.18 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.29 GiB | diagnostic exit 0; peak 14.55 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.57 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17294643819332123; reconstruction 0.00021311230375431478; diagnostic exit 0; peak 14.57 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.15 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17926837503910065; reconstruction 9.93496723822318e-05; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.37 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1733735352754593; reconstruction 2.9771928893751465e-05; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.41 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1723041832447052; reconstruction 0.00017090316396206617; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.38 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.17712992429733276; reconstruction 0.00021400918194558471; diagnostic exit 1; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.38 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1811203956604004; reconstruction 0.00019228355085942894; diagnostic exit 0; peak 14.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.64 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18801452219486237; reconstruction 0.00010254625522065908; diagnostic exit 0; peak 14.29 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.12 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16719995439052582; reconstruction 0.00015089864609763026; diagnostic exit 0; peak 14.33 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.39 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16928529739379883; reconstruction 0.00024902611039578915; diagnostic exit 0; peak 14.27 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.11 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17555607855319977; reconstruction 3.6956396797904745e-05; diagnostic exit 0; peak 14.26 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.36 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.17504574358463287; reconstruction 1.9181754396413453e-05; diagnostic exit 1; peak 14.27 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.63 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17688941955566406; reconstruction 3.996311716036871e-05; diagnostic exit 0; peak 14.26 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.67 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1744270920753479; reconstruction 9.83166100922972e-05; diagnostic exit 0; peak 14.32 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.71 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1787457913160324; reconstruction 1.822689409891609e-05; diagnostic exit 0; peak 14.33 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.30 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17833998799324036; reconstruction 0.00016103207599371672; diagnostic exit 0; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.03 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18192440271377563; reconstruction 0.00012373375648166984; diagnostic exit 0; peak 14.27 GiB |
