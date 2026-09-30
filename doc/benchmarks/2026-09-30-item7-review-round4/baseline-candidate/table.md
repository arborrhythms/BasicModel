| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.63 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.63 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.56 GiB | ending 0.19471952319145203, calls 20, predictions [0.45944228768348694, 0.567602276802063, 0.5861825942993164, 0.457797110080719] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.56 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.56 GiB | ending 0.24997808039188385, calls 1, predictions [0.503516435623169, 0.503582775592804, 0.5037209391593933, 0.5036472678184509] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.56 GiB | ending 0.19914641976356506, calls 65, predictions [0.4462016820907593, 0.5465344786643982, 0.5496429204940796, 0.4347841441631317] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.56 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.21 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.49 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.64 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.50 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.29 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17500708997249603; reconstruction 0.00014140970597509295; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.15 GiB | diagnostic exit 0; peak 12.19 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.61 GiB | diagnostic exit 0; peak 14.55 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.20 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17457281053066254; reconstruction 0.00018782679399009794; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.46 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1686694473028183; reconstruction 9.175886225420982e-05; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.56 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1699558049440384; reconstruction 0.0002610747469589114; diagnostic exit 0; peak 14.34 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.40 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17621926963329315; reconstruction 2.484843753336463e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.05 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.15890410542488098; reconstruction 0.0003965821524616331; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.58 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.17641036212444305; reconstruction 0.0; diagnostic exit 1; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.63 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17631220817565918; reconstruction 0.00016945168317761272; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.68 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17839659750461578; reconstruction 0.0001740767911542207; diagnostic exit 0; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.01 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.18023791909217834; reconstruction 7.196491060312837e-05; diagnostic exit 1; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.32 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.16815967857837677; reconstruction 0.00011471487960079685; diagnostic exit 1; peak 14.26 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.00 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18333026766777039; reconstruction 0.00012773448543157429; diagnostic exit 0; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.05 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.17677828669548035; reconstruction 3.095339343417436e-05; diagnostic exit 1; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.21 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16134867072105408; reconstruction 0.00012880364374723285; diagnostic exit 0; peak 14.27 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.16 GiB | UNGUARDED DIAGNOSTIC: exact 0.75; where 1.0; output 0.18638047575950623; reconstruction 2.939517071354203e-05; diagnostic exit 1; peak 14.23 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.36 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18131670355796814; reconstruction 0.00011906847794307396; diagnostic exit 0; peak 14.59 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.77 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.18341390788555145; reconstruction 0.00023874537146184593; diagnostic exit 0; peak 14.59 GiB |
