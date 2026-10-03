| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.43 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.10 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.10 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.70 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.70 GiB | ending 0.1854601353406906, calls 21, predictions [0.41637319326400757, 0.5582385659217834, 0.5622906684875488, 0.4262995421886444] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.70 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.70 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.70 GiB | ending 0.24996884167194366, calls 1, predictions [0.5024938583374023, 0.5026171803474426, 0.5025903582572937, 0.5025627017021179] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.70 GiB | ending 0.19748157262802124, calls 114, predictions [0.4408310651779175, 0.5547088384628296, 0.556244969367981, 0.44765108823776245] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.70 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17500731348991394; reconstruction 0.00013001465413253754 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.65 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.51 GiB | exact 1.0; where 1.0; output 0.17322386801242828; reconstruction 0.0002099520352203399 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.58 GiB | exact 1.0; where 1.0; output 0.17905418574810028; reconstruction 0.00010108743299497291 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17437319457530975; reconstruction 1.9690834960783832e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17481191456317902; reconstruction 7.885624654591084e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.172098308801651; reconstruction 0.00018587934027891606 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | failed; peak 4.65 GiB | exact 0.0; where 1.0; output 0.16961292922496796; reconstruction 0.00020973214122932404 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17482565343379974; reconstruction 2.1845011360710487e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16894151270389557; reconstruction 0.0002157421549782157 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | failed; peak 4.65 GiB | exact 0.75; where 1.0; output 0.16281403601169586; reconstruction 0.00025182089302688837 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.58 GiB | exact 1.0; where 1.0; output 0.1751382201910019; reconstruction 2.0470283743634354e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17651700973510742; reconstruction 5.2051480452064425e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1934911608695984; reconstruction 0.00033263041405007243 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17496740818023682; reconstruction 2.989046333823353e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1928168088197708; reconstruction 0.00015597668243572116 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1829942762851715; reconstruction 4.611438271240331e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17239098250865936; reconstruction 3.0069013519096188e-05 |
