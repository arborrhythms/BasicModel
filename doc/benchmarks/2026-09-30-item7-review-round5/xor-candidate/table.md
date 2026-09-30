| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.43 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.05 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.05 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.58 GiB | ending 0.18885986506938934, calls 21, predictions [0.4130237400531769, 0.5700827240943909, 0.5674607753753662, 0.46144533157348633] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.58 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.58 GiB | ending 0.2500127851963043, calls 1, predictions [0.4989422559738159, 0.49900349974632263, 0.4987398087978363, 0.49884721636772156] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.58 GiB | ending 0.19979216158390045, calls 65, predictions [0.4451220631599426, 0.553688645362854, 0.5743532776832581, 0.4697510600090027] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.58 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.62 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17500270903110504; reconstruction 0.00011110683408332989 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.51 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17461806535720825; reconstruction 0.00016578436770942062 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.17138570547103882; reconstruction 0.00020295688591431826 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.60 GiB | exact 1.0; where 1.0; output 0.17985041439533234; reconstruction 0.00011371499567758292 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.51 GiB | exact 1.0; where 1.0; output 0.19117620587348938; reconstruction 0.0001505530090071261 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.17472036182880402; reconstruction 1.5482075468753465e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.43 GiB | exact 1.0; where 1.0; output 0.18606238067150116; reconstruction 0.00031094608129933476 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.64 GiB | exact 1.0; where 1.0; output 0.17326423525810242; reconstruction 1.6780208170530386e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.63 GiB | exact 1.0; where 1.0; output 0.1744389533996582; reconstruction 0.00013320895959623158 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.66 GiB | exact 1.0; where 1.0; output 0.16718100011348724; reconstruction 0.00020832850714214146 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.18572688102722168; reconstruction 0.00022209710732568055 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.49 GiB | exact 1.0; where 1.0; output 0.17580023407936096; reconstruction 7.652641943423077e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.16767725348472595; reconstruction 0.000139815776492469 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.45 GiB | exact 1.0; where 1.0; output 0.17996089160442352; reconstruction 1.8566670405562036e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.17437347769737244; reconstruction 5.480252730194479e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.61 GiB | exact 1.0; where 1.0; output 0.17342936992645264; reconstruction 7.515129982493818e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.18301765620708466; reconstruction 0.00011295809963485226 |
