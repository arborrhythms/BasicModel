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
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.62 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.62 GiB | ending 0.182596355676651, calls 22, predictions [0.4320341944694519, 0.5915329456329346, 0.5843105912208557, 0.451761931180954] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.62 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.62 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.62 GiB | ending 0.2500123679637909, calls 1, predictions [0.5035778284072876, 0.5034931302070618, 0.5035107135772705, 0.5034263730049133] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | failed; peak 0.62 GiB | ending 0.24595004320144653, calls 900, predictions [0.495912104845047, 0.5040433406829834, 0.5040469169616699, 0.49591219425201416] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.62 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.53 GiB | exact 1.0; where 1.0; output 0.1749967485666275; reconstruction 0.00013658565876539797 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.64 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17738337814807892; reconstruction 0.0002365977707086131 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | failed; peak 4.65 GiB | exact 0.5; where 1.0; output 0.1703559011220932; reconstruction 0.00010404046770418063 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | failed; peak 4.65 GiB | exact 0.5; where 1.0; output 0.17227517068386078; reconstruction 0.0001085484036593698 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.58 GiB | exact 1.0; where 1.0; output 0.17707626521587372; reconstruction 6.435659452108666e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.16378407180309296; reconstruction 0.0003551968256942928 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.14918844401836395; reconstruction 0.0006065507768653333 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17313112318515778; reconstruction 0.00010102523083332926 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17320616543293; reconstruction 0.00016752969531808048 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16746091842651367; reconstruction 0.00026323218480683863 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17530451714992523; reconstruction 1.5535566490143538e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.1792149841785431; reconstruction 0.0004052158910781145 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.17083700001239777; reconstruction 0.0001821104233385995 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.165715754032135; reconstruction 0.00027750435401685536 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.18110816180706024; reconstruction 7.00781965861097e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16460652649402618; reconstruction 0.0001342340838164091 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.65 GiB | exact 1.0; where 1.0; output 0.16687530279159546; reconstruction 0.00012698554201051593 |
