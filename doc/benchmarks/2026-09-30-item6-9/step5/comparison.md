# Named XOR measurements

Fresh unseeded attempts on the frozen HEAD and candidate source manifests. Each table includes all fifteen exact round trips and the explicit slow proofs.

| Proof | HEAD | Candidate | Candidate peak GiB |
|---|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed | passed | 0.43 |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed | passed | 0.43 |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed | passed | 0.43 |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed | passed | 0.43 |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed | passed | 0.43 |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed | passed | 0.43 |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed | passed | 1.20 |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed | passed | 1.20 |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed | passed | 1.20 |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed | passed | 1.20 |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed | passed | 1.20 |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed | passed | 1.20 |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed | passed | 1.20 |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed | passed | 1.20 |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed | passed | 1.20 |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed | passed | 1.20 |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed | passed | 0.55 |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed | passed | 0.55 |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed | passed | 1.10 |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed | passed | 1.10 |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed | failed | 0.60 |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed | failed | 0.60 |
| test_basicmodel.py::TestSPNN::test_xor_training | passed | passed | 0.41 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed | passed | 4.54 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed | passed | 4.55 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed | passed | 4.53 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed | passed | 4.23 |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed | passed | 4.58 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed | passed | 4.57 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed | failed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed | failed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | failed | failed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | failed | failed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed | passed | 4.65 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | failed | passed | 4.65 |

[HEAD details](head/table.md); [candidate details](candidate/table.md).

head: {'passed': 44, 'failed': 5}; exact round trips {'passed': 12, 'failed': 3}.

head, `TestXorGrammarLearnsXor`: MSE 0.2500030761, 3/4 answers correct. Answers [0.5028591752052307, 0.5003606081008911, 0.5005866885185242, 0.49808812141418457]; reported reconstructions ['', '', '', ''].

head, `TestXorGrammarReconstruction`: MSE 0.2414720809, 3/4 answers correct. Answers [0.475458025932312, 0.5011643767356873, 0.49442651867866516, 0.48516643047332764]; reported reconstructions ['hello hello', '', '', ''].
candidate: {'passed': 43, 'failed': 6}; exact round trips {'failed': 4, 'passed': 11}.

candidate, `TestXorGrammarLearnsXor`: MSE 0.2844835867, 2/4 answers correct. Answers [0.4388618469238281, 0.33069509267807007, 0.8455155491828918, 0.6881134510040283]; reported reconstructions ['there', '', '', ''].

candidate, `TestXorGrammarReconstruction`: MSE 0.2534145598, 2/4 answers correct. Answers [0.5227459669113159, 0.47221577167510986, 0.5183346271514893, 0.47941336035728455]; reported reconstructions ['hello hello', '', '', ''].
