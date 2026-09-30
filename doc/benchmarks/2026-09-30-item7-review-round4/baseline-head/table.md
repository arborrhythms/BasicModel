| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.87 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.87 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 1.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 1.63 GiB | ending 0.19773384928703308, calls 18, predictions [0.4482755661010742, 0.5612292289733887, 0.5632734298706055, 0.4546807110309601] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 1.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 1.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 1.63 GiB | ending 0.25022631883621216, calls 1, predictions [0.5122796297073364, 0.5121416449546814, 0.5122642517089844, 0.5124284029006958] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 1.63 GiB | ending 0.19923581182956696, calls 144, predictions [0.44641420245170593, 0.5542888045310974, 0.5528691411018372, 0.44617611169815063] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 1.63 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.30 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.82 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17501792311668396; reconstruction 0.00013889677939005196 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.73 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.12 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17853574454784393; reconstruction 0.00012870448699686676 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17497101426124573; reconstruction 2.422953912173398e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1771426945924759; reconstruction 3.4493485145503655e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17648115754127502; reconstruction 9.981883340515196e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16151097416877747; reconstruction 0.00011666395585052669 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16144493222236633; reconstruction 9.38265657168813e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.18951751291751862; reconstruction 0.0001556811184855178 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.19056063890457153; reconstruction 6.453294190578163e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.19462847709655762; reconstruction 0.0001240348647115752 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1885152906179428; reconstruction 5.2033468818990514e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16984659433364868; reconstruction 5.215244891587645e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17587608098983765; reconstruction 5.2739600505447015e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17870646715164185; reconstruction 6.724393460899591e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17517420649528503; reconstruction 3.8615689845755696e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1658589392900467; reconstruction 9.944626071956009e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17673292756080627; reconstruction 1.4202602869772818e-05 |
