# XOR: final, every named proof

All guarded attempts are reported. A guard stop remains a guard stop even when its separate unguarded diagnostic passes. Fifteen exact round trips per tree are included, with no seed selected.

[HEAD values and raw links](final-xor-head/table.md); [candidate values and raw links](final3-xor-candidate/table.md).

| Proof | HEAD | Candidate |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; 0.45 GiB | passed; 0.43 GiB |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; 1.83 GiB | passed; 1.06 GiB |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; 1.83 GiB | passed; 1.06 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; 0.66 GiB | passed; 0.58 GiB |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; 0.66 GiB | passed; 0.58 GiB |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; 1.29 GiB | passed; 1.10 GiB |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; 1.29 GiB | passed; 1.10 GiB |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; 0.61 GiB | failed; 0.61 GiB |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; 0.61 GiB | failed; 0.60 GiB |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; 0.41 GiB | passed; 0.41 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; 3.99 GiB | passed; 4.53 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; 4.84 GiB | passed; 4.55 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; 4.00 GiB | passed; 4.53 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; 3.73 GiB | passed; 4.23 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; 4.07 GiB | passed; 4.64 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; 3.99 GiB | passed; 4.65 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; 4.17 GiB | passed; 4.65 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; 4.17 GiB | passed; 4.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; 4.17 GiB | passed; 4.51 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; 4.17 GiB | passed; 4.58 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; 4.17 GiB | passed; 4.64 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | failed; 4.17 GiB | passed; 4.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; 4.17 GiB | passed; 4.50 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; 4.17 GiB | passed; 4.55 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; 4.18 GiB | passed; 4.57 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; 4.17 GiB | passed; 4.50 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; 4.17 GiB | passed; 4.50 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; 4.18 GiB | passed; 4.53 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; 4.17 GiB | passed; 4.57 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; 4.17 GiB | passed; 4.43 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; 4.18 GiB | failed; 4.54 GiB |

XOR_grammar and the intermittent MM_20M_xor exact-round-trip cause remain item 6.9; neither is repaired in this pass.
