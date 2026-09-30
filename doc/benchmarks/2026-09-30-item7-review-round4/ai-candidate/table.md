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
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.57 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.57 GiB | ending 0.1987568438053131, calls 21, predictions [0.4538569748401642, 0.5622209310531616, 0.5636581778526306, 0.45496872067451477] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.57 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.57 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.57 GiB | ending 0.24999581277370453, calls 1, predictions [0.4963933825492859, 0.49645206332206726, 0.49666500091552734, 0.49665912985801697] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.57 GiB | ending 0.19919723272323608, calls 98, predictions [0.4480970799922943, 0.5520753860473633, 0.5565277934074402, 0.4457509219646454] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.57 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.11 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.63 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.62 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | memory; peak 8.27 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758; diagnostic exit 0; peak 14.48 GiB |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | memory; peak 8.55 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876; diagnostic exit 0; peak 14.49 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | memory; peak 8.69 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17499691247940063; reconstruction 0.000137266208184883; diagnostic exit 0; peak 14.48 GiB |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | memory; peak 8.74 GiB | diagnostic exit 0; peak 12.18 GiB |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | memory; peak 8.53 GiB | diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | memory; peak 8.77 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17785745859146118; reconstruction 0.0001769691880326718; diagnostic exit 0; peak 14.60 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | memory; peak 8.54 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.15507744252681732; reconstruction 0.00018183809879701585; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | memory; peak 8.37 GiB | UNGUARDED DIAGNOSTIC: exact 0.25; where 0.5; output 0.18118271231651306; reconstruction 0.0003382221912033856; diagnostic exit 1; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | memory; peak 8.16 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16567374765872955; reconstruction 0.0001090693476726301; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | memory; peak 8.33 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1753266304731369; reconstruction 3.509546149871312e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | memory; peak 8.22 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1808544397354126; reconstruction 0.00015460254508070648; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | memory; peak 8.34 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.161550834774971; reconstruction 0.00019327201880514622; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | memory; peak 8.49 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16024497151374817; reconstruction 0.00012042604794260114; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | memory; peak 8.50 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16519275307655334; reconstruction 0.00027845200384035707; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | memory; peak 8.75 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.16929414868354797; reconstruction 0.0003188930859323591; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | memory; peak 8.18 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17395463585853577; reconstruction 0.00021300389198586345; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | memory; peak 8.22 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17380040884017944; reconstruction 2.7377127480576746e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | memory; peak 8.48 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17287014424800873; reconstruction 1.8762319086818025e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | memory; peak 8.54 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.17483548820018768; reconstruction 6.905067129991949e-05; diagnostic exit 0; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | memory; peak 8.70 GiB | UNGUARDED DIAGNOSTIC: exact 0.5; where 1.0; output 0.17016571760177612; reconstruction 0.00011409091530367732; diagnostic exit 1; peak 14.35 GiB |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | memory; peak 8.40 GiB | UNGUARDED DIAGNOSTIC: exact 1.0; where 1.0; output 0.1748555302619934; reconstruction 7.358766015386209e-05; diagnostic exit 0; peak 14.35 GiB |
