| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.46 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.89 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.89 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 5.80 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 5.80 GiB | ending 0.19308695197105408, calls 17, predictions [0.4224267601966858, 0.5476679801940918, 0.5375045537948608, 0.418804407119751] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 5.80 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 5.80 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 5.80 GiB | ending 0.2500753104686737, calls 1, predictions [0.48850229382514954, 0.4885140657424927, 0.48861512541770935, 0.4883999228477478] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 5.80 GiB | ending 0.1994420289993286, calls 665, predictions [0.4483393430709839, 0.5545111298561096, 0.5538788437843323, 0.44640281796455383] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 5.80 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.30 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.30 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.82 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.00 GiB | exact 1.0; where 1.0; output 0.17499849200248718; reconstruction 0.00013128970749676228 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.74 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.12 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.08 GiB | exact 1.0; where 1.0; output 0.17724767327308655; reconstruction 0.00021992299298290163 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1495179533958435; reconstruction 0.00023117045930121094 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1746395081281662; reconstruction 5.039346797275357e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17560388147830963; reconstruction 2.6238007194478996e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17616450786590576; reconstruction 5.072907515568659e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1822776198387146; reconstruction 7.464749796781689e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17524354159832; reconstruction 0.0 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.1673785150051117; reconstruction 6.515237328130752e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17583215236663818; reconstruction 2.965058411064092e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.1933697611093521; reconstruction 9.834719094214961e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17561282217502594; reconstruction 3.666101474664174e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.1517469435930252; reconstruction 0.00013588223373517394 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17632125318050385; reconstruction 9.698565008875448e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1795935034751892; reconstruction 2.627815047162585e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16912786662578583; reconstruction 0.00010645405563991517 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1743229180574417; reconstruction 2.485579352651257e-05 |
