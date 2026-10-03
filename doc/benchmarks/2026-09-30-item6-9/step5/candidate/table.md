| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.43 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.20 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.20 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.55 GiB | ending 0.19993332028388977, calls 19, predictions [0.4246658682823181, 0.5345437526702881, 0.5325782895088196, 0.42925477027893066] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.55 GiB | ending 0.25008276104927063, calls 1, predictions [0.5089526176452637, 0.5089451670646667, 0.5088590383529663, 0.5088654160499573] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.55 GiB | ending 0.1998700648546219, calls 7, predictions [0.6059771180152893, 0.4934445321559906, 0.642413318157196, 0.21864426136016846] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.55 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.1750027984380722; reconstruction 0.00014734832802787423 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.58 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.174367755651474; reconstruction 0.00016066321404650807 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | failed; peak 4.65 GiB | exact 0.5; where 1.0; output 0.1576959639787674; reconstruction 0.0003082170442212373 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17013120651245117; reconstruction 0.00012104306370019913 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1790265440940857; reconstruction 6.551103433594108e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.15256212651729584; reconstruction 0.00016950858116615564 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17532481253147125; reconstruction 7.313847163459286e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17241862416267395; reconstruction 9.592436981620267e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16443680226802826; reconstruction 0.00014243641635403037 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | failed; peak 4.65 GiB | exact 0.75; where 1.0; output 0.17634858191013336; reconstruction 1.4399837709788699e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.18149016797542572; reconstruction 0.0001410824916092679 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.18129201233386993; reconstruction 0.00014960781845729798 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | failed; peak 4.65 GiB | exact 0.0; where 0.3333333333333333; output 0.15105365216732025; reconstruction 0.00030975486151874065 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | failed; peak 4.65 GiB | exact 0.5; where 1.0; output 0.18057559430599213; reconstruction 5.591372973867692e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17783446609973907; reconstruction 0.00027710216818377376 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17179882526397705; reconstruction 0.00021726462000515312 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17343024909496307; reconstruction 0.0001528364809928462 |
