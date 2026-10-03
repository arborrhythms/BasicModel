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
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.55 GiB | ending 0.19457091391086578, calls 20, predictions [0.4329751133918762, 0.5510165691375732, 0.5498551726341248, 0.43197181820869446] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.55 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.55 GiB | ending 0.2499912977218628, calls 1, predictions [0.4998455345630646, 0.49984338879585266, 0.4997358024120331, 0.4996986389160156] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.55 GiB | ending 0.19838228821754456, calls 6, predictions [0.5246915221214294, 0.5497045516967773, 0.5516437292098999, 0.3382878303527832] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.55 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.09 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.60 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.52 GiB | exact 1.0; where 1.0; output 0.17499713599681854; reconstruction 0.0001246813335455954 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.51 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.64 GiB | exact 1.0; where 1.0; output 0.1743328720331192; reconstruction 0.00018593193090055138 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.1555212438106537; reconstruction 0.00013793639664072543 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.19228990375995636; reconstruction 0.00014672830002382398 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.58 GiB | exact 1.0; where 1.0; output 0.17314931750297546; reconstruction 9.778379899216816e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | failed; peak 4.65 GiB | exact 0.5; where 1.0; output 0.17867222428321838; reconstruction 0.00010973684402415529 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17252834141254425; reconstruction 7.642676791874692e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1745905727148056; reconstruction 1.4470599126070738e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.1763753890991211; reconstruction 4.5019740355201066e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.48 GiB | exact 1.0; where 1.0; output 0.17533457279205322; reconstruction 4.6778532123425975e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.44 GiB | exact 1.0; where 1.0; output 0.1799984872341156; reconstruction 0.00025383057072758675 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.20091259479522705; reconstruction 0.0001644625881453976 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17498137056827545; reconstruction 4.5207361836219206e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.48 GiB | exact 1.0; where 1.0; output 0.17568956315517426; reconstruction 6.723413389408961e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | failed; peak 4.59 GiB | exact 0.75; where 1.0; output 0.1699540764093399; reconstruction 0.00010991351882694289 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.57 GiB | exact 1.0; where 1.0; output 0.17497016489505768; reconstruction 3.0820116080576554e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.51 GiB | exact 1.0; where 1.0; output 0.1687082201242447; reconstruction 9.201166540151462e-05 |
