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
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.56 GiB | ending 0.19262900948524475, calls 20, predictions [0.4308502972126007, 0.5531465411186218, 0.5500357151031494, 0.4274789094924927] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.56 GiB | ending 0.24999988079071045, calls 1, predictions [0.502221941947937, 0.5022463798522949, 0.5021994709968567, 0.5022037029266357] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.56 GiB | ending 0.19886988401412964, calls 137, predictions [0.4453272819519043, 0.553560733795166, 0.5540938377380371, 0.4461196959018707] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.56 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499999701976776; reconstruction 0.00012098257866455242 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.24 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.56 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.64 GiB | exact 1.0; where 1.0; output 0.17560677230358124; reconstruction 0.00018404463480692357 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17966251075267792; reconstruction 8.475525100948289e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.60 GiB | exact 1.0; where 1.0; output 0.17317314445972443; reconstruction 3.8759993913117796e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17764514684677124; reconstruction 0.00016948056872934103 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | failed; peak 4.53 GiB | exact 0.5; where 1.0; output 0.15535186231136322; reconstruction 0.00021632626885548234 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.47 GiB | exact 1.0; where 1.0; output 0.17629072070121765; reconstruction 0.00012392060307320207 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.1845996379852295; reconstruction 0.0001239324192283675 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17959541082382202; reconstruction 0.000141725322464481 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.17779971659183502; reconstruction 5.3592251788359135e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | failed; peak 4.61 GiB | exact 0.5; where 1.0; output 0.18873700499534607; reconstruction 0.00012201728532090783 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17458391189575195; reconstruction 3.2718555303290486e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.48 GiB | exact 1.0; where 1.0; output 0.1706695258617401; reconstruction 0.0002606557682156563 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.48 GiB | exact 1.0; where 1.0; output 0.17002330720424652; reconstruction 6.53502211207524e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | failed; peak 4.53 GiB | exact 0.75; where 1.0; output 0.1616716980934143; reconstruction 0.00021330757590476424 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.56 GiB | exact 1.0; where 1.0; output 0.17250530421733856; reconstruction 0.0001358073204755783 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.56 GiB | exact 1.0; where 1.0; output 0.18686015903949738; reconstruction 0.00021322145767044276 |
