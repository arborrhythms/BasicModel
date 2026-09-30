| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.68 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.68 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.56 GiB | ending 0.19833029806613922, calls 19, predictions [0.4656599164009094, 0.5819024443626404, 0.5747597217559814, 0.4699437916278839] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.56 GiB | ending 0.2499701827764511, calls 1, predictions [0.4985657036304474, 0.49868521094322205, 0.4985244870185852, 0.4985165596008301] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.56 GiB | ending 0.1996452361345291, calls 362, predictions [0.4467729926109314, 0.5533133149147034, 0.5529821515083313, 0.44678956270217896] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.56 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.37 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.48 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.49 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.21 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499560117721558; reconstruction 0.0001281373988604173; diagnostic exit 0; peak 14.48 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.56 GiB | diagnostic exit 0; peak 12.18 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.12 GiB | diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.64 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17264127731323242; reconstruction 0.00015896341938059777; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.39 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17494851350784302; reconstruction 2.724627620409592e-06; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.64 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17340007424354553; reconstruction 3.2067775464383885e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.04 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17545826733112335; reconstruction 0.00013766062329523265; diagnostic exit 0; peak 14.34 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.01 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 0.8333333333333334; output 0.17592746019363403; reconstruction 0.00019212854385841638; diagnostic exit 1; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.32 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16721254587173462; reconstruction 0.00022642992553301156; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.29 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17566487193107605; reconstruction 3.730725438799709e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.72 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17958404123783112; reconstruction 9.52095870161429e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.06 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.15376347303390503; reconstruction 0.00023752174456603825; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.30 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1664573699235916; reconstruction 0.0001642876013647765; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.17 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1734451949596405; reconstruction 8.990257629193366e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.41 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.177021786570549; reconstruction 3.53349132637959e-05; diagnostic exit 1; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.58 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17637072503566742; reconstruction 6.10711140325293e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.70 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17575503885746002; reconstruction 0.0001105711271520704; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.17 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16638341546058655; reconstruction 0.00014416199701372534; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.55 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1761285662651062; reconstruction 3.2981257390929386e-05; diagnostic exit 0; peak 14.35 GiB |
