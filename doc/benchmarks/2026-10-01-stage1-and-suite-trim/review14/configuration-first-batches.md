# Migrated configurations: first numerical batch

One unseeded attempt each, production batch and all other configuration values unchanged. CPU, compilation disabled: the process time includes construction; training time (when available) excludes construction. Every worker has an 8 GiB ceiling. An exception has no completed-batch timing. These measurements precede removal of the retired student runtime; their exact source hashes are in `mode-first-batches/manifest.json`.

| Configuration | Batch | Process s | Training s | Peak GiB | Outcome |
|---|---:|---:|---:|---:|---|
| `data/XOR_grammar.xml` | 64 | 4.109 | 0.605 | 0.503 | completed |
| `data/HeadEmission.xml` | — | 3.354 | — | 0.378 | ValueError('unresolved generate rule: S -> C') |
| `data/LM_5M.xml` | 128 | 435.961 | — | 6.426 | RecursionError('maximum recursion depth exceeded') |
| `data/LM_5M_IR.xml` | — | 3.334 | — | 0.375 | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activ |
| `data/MM_400M.xml` | 64 | 60.598 | 45.261 | 7.248 | completed |
| `data/MM_5M_AR.xml` | — | 3.596 | — | 0.376 | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activ |
| `data/MM_5M_IR.xml` | — | 3.610 | — | 0.376 | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activ |
| `data/MM_add.xml` | 32 | 4.112 | — | 0.478 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/MM_add_verb.xml` | — | 14.980 | — | 8.441 | memory |
| `data/MM_boolean.xml` | 16 | 4.905 | 1.294 | 0.729 | completed |
| `data/MM_decode.xml` | 12 | 11.630 | 3.423 | 4.808 | completed |
| `data/MM_global.xml` | 12 | 10.350 | 2.088 | 4.898 | completed |
| `data/MM_grammar.xml` | 64 | 4.375 | 0.891 | 0.517 | completed |
| `data/MM_init_scale.xml` | 64 | 3.860 | 0.367 | 0.495 | completed |
| `data/MM_ltm_consolidation_fixture.xml` | 64 | 3.858 | — | 0.444 | ValueError('TruthSet input did not produce a selected grammatical closing') |
| `data/MM_ltm_consolidation_serial_fixture.xml` | 64 | 4.657 | 1.211 | 0.531 | completed |
| `data/MM_ltm_consolidation_stateful_fixture.xml` | 64 | 3.859 | — | 0.447 | ValueError('TruthSet input did not produce a selected grammatical closing') |
| `data/MM_masked_semantic.xml` | 12 | 10.595 | 2.295 | 5.556 | completed |
| `data/MM_math.xml` | 64 | 3.860 | — | 0.483 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/MM_mereology.xml` | 12 | 10.103 | 1.730 | 4.820 | completed |
| `data/MM_mereology_serial.xml` | 12 | 18.109 | 9.509 | 6.409 | completed |
| `data/MM_meronomy_smoke.xml` | 12 | 15.478 | 7.047 | 7.699 | completed |
| `data/MM_overlap_tiling.xml` | 12 | 10.108 | 1.753 | 4.819 | completed |
| `data/MM_phrase_decode.xml` | 12 | 15.500 | 7.070 | 7.773 | completed |
| `data/MM_qa.xml` | 12 | 9.079 | — | 3.130 | ValueError('TruthSet input did not produce a selected grammatical closing') |
| `data/MM_query_reasoning.xml` | 6 | 11.346 | 7.501 | 6.009 | completed |
| `data/MM_reading.xml` | 12 | 10.091 | 1.843 | 4.833 | completed |
| `data/MM_sequence_predict.xml` | 4 | 4.118 | — | 0.499 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/MM_shamatha.xml` | 64 | 4.648 | 1.146 | 0.534 | completed |
| `data/MM_sparse_concept.xml` | 12 | 10.573 | 2.316 | 5.556 | completed |
| `data/MM_symbol_tower.xml` | 12 | 9.287 | 0.936 | 4.216 | completed |
| `data/MM_symbolic_iter.xml` | 4 | 4.112 | 0.410 | 0.542 | completed |
| `data/MM_xor.xml` | 64 | 3.869 | 0.371 | 0.504 | completed |
| `data/MM_xor_fixture.xml` | 64 | 3.866 | 0.374 | 0.505 | completed |
| `data/MM_xor_loopback.xml` | 4 | 4.391 | 0.830 | 0.511 | completed |
| `data/MM_xor_step3.xml` | 64 | 4.396 | 0.751 | 0.510 | completed |
| `data/MM_xor_step4.xml` | 64 | 4.907 | 1.210 | 0.547 | completed |
| `data/MentalModel.xml` | 64 | 6.996 | — | 0.895 | RuntimeError('compose has no legal alternative at the selected exploration round') |
| `data/POS_smoke.xml` | 16 | 3.871 | — | 0.447 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/RamsifiedModel.xml` | 64 | 13.211 | 9.526 | 3.552 | completed |
| `data/XOR_pos.xml` | 64 | 3.858 | 0.366 | 0.496 | completed |
| `data/XOR_recon.xml` | — | 3.616 | — | 0.375 | ValueError('Config requirement failed: WholeSpace requires WS event width == CS width (got CS width=8, WS event width=10). Fix: set <WholeSpace><nDim> to match <ConceptualSpace><nOutputDim> if present, else <ConceptualSpace><nDim>.') |
| `data/XOR_spaces.xml` | 64 | 3.857 | — | 0.431 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/ergodic-only.xml` | — | 3.599 | — | 0.567 | TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.") |
| `data/ergodic.xml` | — | 3.618 | — | 0.568 | TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.") |
| `data/idempotent.xml` | 1 | 3.858 | 0.337 | 0.462 | completed |
| `data/mnist.xml` | — | 3.611 | — | 0.566 | TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.") |
| `data/simple.xml` | — | 3.612 | — | 0.565 | TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.") |
| `data/stream_smoke.xml` | 4 | 4.114 | — | 0.483 | RuntimeError('occurrence exceeds its configured where-space capacity') |
| `data/tomatoes.xml` | — | 4.648 | — | 0.374 | HfUriError("Invalid HF URI 'hf://datasets/rotten_tomatoes@aa13bc287fa6fcab6daf52f0dfb9994269ffea28/.huggingface.yaml'. Repository id must be 'namespace/name', got 'rotten_tomatoes'.") |
| `data/xor.xml` | 64 | 4.121 | 0.377 | 0.496 | completed |

The numeric configurations retain perceptual reconstruction because they have no grammar understanding. Their common first-batch harness raised on an object-typed NumPy input; no timing is claimed for a completed numeric batch. `data/model.xml` is the inherited template, not an executable dataset. `MM_bpe.xml` and `MM_20M_legacy.xml` existed to select retired paths and were removed with those paths; their complete XML remains in `retired-configurations/`.

`MM_add_verb` reached 8.441 GiB at the guard's next sample. The ceiling remains 8 GiB. Geometry, occurrence-capacity, external-data and grammatical-closing failures were retained without changing a configuration, inference policy, or capacity. They are measurement failures, not assumed successes or proven pre-existing failures.

`LM_5M` exposed Python recursion in two host tree traversals. The original and repaired attempts, including the second traceback, remain in `mode-first-batches/` and `mode-repair-batches/`; the small probes retain the same postorder/actions and bounded meaning recovery. The second repair is measured separately in `mode-repair-batches-2/`.
