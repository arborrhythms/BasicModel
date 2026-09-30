| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.80 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.80 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.63 GiB | ending 0.18416281044483185, calls 18, predictions [0.43124037981033325, 0.5779790878295898, 0.5590787529945374, 0.4221016466617584] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.63 GiB | ending 0.24996691942214966, calls 1, predictions [0.49676257371902466, 0.49680450558662415, 0.4970497786998749, 0.49692055583000183] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.63 GiB | ending 0.19624775648117065, calls 19, predictions [0.5060674548149109, 0.5966975092887878, 0.5254200100898743, 0.37550994753837585] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.63 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.32 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.30 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.82 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.1750011444091797; reconstruction 0.00014094988000579178 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.75 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.01 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.07 GiB | exact 1.0; where 1.0; output 0.17756129801273346; reconstruction 0.0001510891452198848 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17596662044525146; reconstruction 2.6177323888987303e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1840590536594391; reconstruction 8.366430847672746e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17711172997951508; reconstruction 1.6943986338446848e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1915062516927719; reconstruction 0.000253749021794647 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.18190932273864746; reconstruction 3.420058055780828e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17783354222774506; reconstruction 6.416600081138313e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1718793362379074; reconstruction 4.8435023927595466e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.1445956528186798; reconstruction 0.00011340927449055016 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17554575204849243; reconstruction 4.6390018724196125e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.18153835833072662; reconstruction 6.115862197475508e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.15717104077339172; reconstruction 8.376774349017069e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.18331226706504822; reconstruction 3.530718095134944e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16833578050136566; reconstruction 6.704731640638784e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17418186366558075; reconstruction 1.0601473150018137e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17123612761497498; reconstruction 2.9159285986679606e-05 |
