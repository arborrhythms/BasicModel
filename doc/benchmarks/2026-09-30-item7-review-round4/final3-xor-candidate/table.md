| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.43 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.06 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.58 GiB | ending 0.19440315663814545, calls 20, predictions [0.44929391145706177, 0.5647382736206055, 0.5660005807876587, 0.44490373134613037] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.58 GiB | ending 0.25000816583633423, calls 1, predictions [0.4997284412384033, 0.49973779916763306, 0.4996429979801178, 0.4996846914291382] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.58 GiB | ending 0.19981920719146729, calls 166, predictions [0.44814324378967285, 0.5538368821144104, 0.5524408221244812, 0.44617682695388794] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.58 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17500050365924835; reconstruction 0.00012056879495503381 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.64 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17476189136505127; reconstruction 0.0001759253063937649 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1758655607700348; reconstruction 1.0738917808339465e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.1803109347820282; reconstruction 9.384981967741624e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.51 GiB | exact 1.0; where 1.0; output 0.1717180609703064; reconstruction 8.754589362069964e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.58 GiB | exact 1.0; where 1.0; output 0.15101847052574158; reconstruction 0.0002503293508198112 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.64 GiB | exact 1.0; where 1.0; output 0.18199269473552704; reconstruction 0.00021236453903838992 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.17156565189361572; reconstruction 9.984229109250009e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.17597293853759766; reconstruction 0.00011480670218588784 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.19021838903427124; reconstruction 0.00013097930059302598 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.17658255994319916; reconstruction 2.127148945874069e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.18025921285152435; reconstruction 0.0001403751812176779 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.17111067473888397; reconstruction 0.00010014647705247626 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17483527958393097; reconstruction 3.861065488308668e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.1606767177581787; reconstruction 0.0002535766689106822 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.43 GiB | exact 1.0; where 1.0; output 0.17721498012542725; reconstruction 0.0001838705939007923 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | failed; peak 4.54 GiB | exact 0.5; where 1.0; output 0.17787253856658936; reconstruction 2.1758309230790474e-05 |
