# Final migrated configurations: first batch

One unseeded first training batch per configuration, at its own configured batch, on the final candidate. CPU numerical execution with compile backend none excludes graph capture and evaluation. Each fresh worker has the unchanged 8 GiB ceiling and 30-minute deadline; two workers share 16 GiB. Configurations without grammar retain perceptual reconstruction. These are timing attempts, not gate results. Earlier failed attempts and repair probes remain in the adjacent receipt directories.

Outcomes: {'exception': 18, 'completed': 32, 'memory': 1}; 51 configurations; source matched: True; wall time 8.81 minutes.

| Configuration | Batch | Scope | Training seconds | Worker seconds | Peak GiB | Outcome |
|---|---:|---|---:|---:|---:|---|
| `data/HeadEmission.xml` | — | not built | — | 3.708 | 0.375 | exception |
| `data/LM_5M.xml` | 128 | understanding | 3.4302 | 7.366 | 2.782 | completed |
| `data/LM_5M_IR.xml` | — | not built | — | 3.724 | 0.374 | exception |
| `data/MM_400M.xml` | 64 | perception: no grammar | 41.3450 | 56.371 | 7.274 | completed |
| `data/MM_5M_AR.xml` | — | not built | — | 3.733 | 0.377 | exception |
| `data/MM_5M_IR.xml` | — | not built | — | 3.704 | 0.375 | exception |
| `data/MM_add.xml` | 32 | perception: no grammar | — | 4.259 | 0.482 | exception |
| `data/MM_add_verb.xml` | — | not built | — | 15.665 | 8.507 | memory |
| `data/MM_boolean.xml` | 16 | understanding | 1.2021 | 5.349 | 0.712 | completed |
| `data/MM_decode.xml` | 12 | perception: no grammar | 3.4879 | 12.418 | 4.695 | completed |
| `data/MM_global.xml` | 12 | perception: no grammar | 2.1139 | 10.705 | 4.763 | completed |
| `data/MM_grammar.xml` | 64 | understanding | 0.5624 | 4.181 | 0.504 | completed |
| `data/MM_init_scale.xml` | 64 | perception: no grammar | 0.3473 | 4.183 | 0.407 | completed |
| `data/MM_ltm_consolidation_fixture.xml` | 64 | perception: no grammar | — | 4.191 | 0.418 | exception |
| `data/MM_ltm_consolidation_serial_fixture.xml` | 64 | understanding | 0.7034 | 4.191 | 0.518 | completed |
| `data/MM_ltm_consolidation_stateful_fixture.xml` | 64 | perception: no grammar | — | 4.200 | 0.414 | exception |
| `data/MM_masked_semantic.xml` | 12 | perception: no grammar | 2.2494 | 10.648 | 5.432 | completed |
| `data/MM_math.xml` | 64 | perception: no grammar | — | 4.275 | 0.485 | exception |
| `data/MM_mereology.xml` | 12 | perception: no grammar | 1.7485 | 10.161 | 4.694 | completed |
| `data/MM_mereology_serial.xml` | 12 | perception: no grammar | 9.5688 | 18.323 | 6.354 | completed |
| `data/MM_meronomy_smoke.xml` | 12 | understanding | 7.1213 | 16.161 | 7.779 | completed |
| `data/MM_overlap_tiling.xml` | 12 | perception: no grammar | 1.7531 | 10.162 | 4.619 | completed |
| `data/MM_phrase_decode.xml` | 12 | understanding | 7.2604 | 15.608 | 7.778 | completed |
| `data/MM_qa.xml` | 12 | perception: no grammar | — | 10.723 | 3.129 | exception |
| `data/MM_query_reasoning.xml` | 6 | understanding | — | 4.811 | 0.833 | exception |
| `data/MM_reading.xml` | 12 | perception: no grammar | 1.8495 | 10.162 | 4.731 | completed |
| `data/MM_sequence_predict.xml` | 4 | understanding | 393.9137 | 398.114 | 6.704 | completed |
| `data/MM_shamatha.xml` | 64 | understanding | 0.5356 | 4.275 | 0.503 | completed |
| `data/MM_sparse_concept.xml` | 12 | perception: no grammar | 2.2736 | 10.770 | 5.445 | completed |
| `data/MM_symbol_tower.xml` | 12 | perception: no grammar | 0.9088 | 9.687 | 4.091 | completed |
| `data/MM_symbolic_iter.xml` | 4 | perception: no grammar | 0.4139 | 4.263 | 0.543 | completed |
| `data/MM_xor.xml` | 64 | perception: no grammar | 0.3855 | 4.270 | 0.507 | completed |
| `data/MM_xor_fixture.xml` | 64 | perception: no grammar | 0.3862 | 4.272 | 0.507 | completed |
| `data/MM_xor_loopback.xml` | 4 | understanding | 0.5378 | 4.265 | 0.497 | completed |
| `data/MM_xor_step3.xml` | 64 | understanding | 0.5498 | 4.265 | 0.501 | completed |
| `data/MM_xor_step4.xml` | 64 | understanding | 0.5805 | 4.266 | 0.504 | completed |
| `data/MentalModel.xml` | 64 | understanding | 1.0142 | 4.822 | 0.742 | completed |
| `data/POS_smoke.xml` | 16 | understanding | 1.1294 | 4.803 | 0.728 | completed |
| `data/RamsifiedModel.xml` | 64 | understanding | 0.8784 | 4.820 | 0.699 | completed |
| `data/XOR_grammar.xml` | 64 | understanding | 0.5240 | 5.266 | 0.497 | completed |
| `data/XOR_pos.xml` | 64 | perception: no grammar | 0.3771 | 4.255 | 0.495 | completed |
| `data/XOR_recon.xml` | — | not built | — | 3.724 | 0.376 | exception |
| `data/XOR_spaces.xml` | 64 | perception: no grammar | — | 4.276 | 0.433 | exception |
| `data/ergodic-only.xml` | — | not built | — | 3.702 | 0.413 | exception |
| `data/ergodic.xml` | — | not built | — | 3.730 | 0.425 | exception |
| `data/idempotent.xml` | 1 | perception: no grammar | 0.3489 | 4.269 | 0.458 | completed |
| `data/mnist.xml` | — | not built | — | 3.719 | 0.425 | exception |
| `data/simple.xml` | — | not built | — | 3.722 | 0.421 | exception |
| `data/stream_smoke.xml` | 4 | perception: no grammar | — | 4.277 | 0.486 | exception |
| `data/tomatoes.xml` | — | not built | — | 4.793 | 0.377 | exception |
| `data/xor.xml` | 64 | perception: no grammar | 0.3759 | 4.271 | 0.493 | completed |

## Incomplete attempts

An exception is a measured inability to finish the batch, not a successful reconstruction. Numeric configurations were also audited because they inherit the template; the four numeric attempts fail while loading MNIST numpy.object_ data, before model construction, so they supply no model timing or evidence about arithmetic. No data, threshold, capacity or optimizer setting was changed to obtain a passing row.

- `data/HeadEmission.xml`: ValueError('unresolved generate rule: S -> C'). See `mode-final-batches/HeadEmission.log`.
- `data/LM_5M_IR.xml`: ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).'). See `mode-final-batches/LM_5M_IR.log`.
- `data/MM_5M_AR.xml`: ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).'). See `mode-final-batches/MM_5M_AR.log`.
- `data/MM_5M_IR.xml`: ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).'). See `mode-final-batches/MM_5M_IR.log`.
- `data/MM_add.xml`: RuntimeError('occurrence exceeds its configured where-space capacity'). See `mode-final-batches/MM_add.log`.
- `data/MM_add_verb.xml`: memory. See `mode-final-batches/MM_add_verb.log`.
- `data/MM_ltm_consolidation_fixture.xml`: ValueError('TruthSet input did not produce a selected grammatical closing'). See `mode-final-batches/MM_ltm_consolidation_fixture.log`.
- `data/MM_ltm_consolidation_stateful_fixture.xml`: ValueError('TruthSet input did not produce a selected grammatical closing'). See `mode-final-batches/MM_ltm_consolidation_stateful_fixture.log`.
- `data/MM_math.xml`: RuntimeError('occurrence exceeds its configured where-space capacity'). See `mode-final-batches/MM_math.log`.
- `data/MM_qa.xml`: ValueError('TruthSet input did not produce a selected grammatical closing'). See `mode-final-batches/MM_qa.log`.
- `data/MM_query_reasoning.xml`: ValueError('a TruthSet cannot assert an interrogative clause'). See `mode-final-batches/MM_query_reasoning.log`.
- `data/XOR_recon.xml`: ValueError('Config requirement failed: WholeSpace requires WS event width == CS width (got CS width=8, WS event width=10). Fix: set <WholeSpace><nDim> to match <ConceptualSpace><nOutputDim> if present, else <ConceptualSpace><nDim>.'). See `mode-final-batches/XOR_recon.log`.
- `data/XOR_spaces.xml`: RuntimeError('occurrence exceeds its configured where-space capacity'). See `mode-final-batches/XOR_spaces.log`.
- `data/ergodic-only.xml`: TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool."). See `mode-final-batches/ergodic-only.log`.
- `data/ergodic.xml`: TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool."). See `mode-final-batches/ergodic.log`.
- `data/mnist.xml`: TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool."). See `mode-final-batches/mnist.log`.
- `data/simple.xml`: TypeError("can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool."). See `mode-final-batches/simple.log`.
- `data/stream_smoke.xml`: RuntimeError('occurrence exceeds its configured where-space capacity'). See `mode-final-batches/stream_smoke.log`.
- `data/tomatoes.xml`: HfUriError("Invalid HF URI 'hf://datasets/rotten_tomatoes@aa13bc287fa6fcab6daf52f0dfb9994269ffea28/.huggingface.yaml'. Repository id must be 'namespace/name', got 'rotten_tomatoes'."). See `mode-final-batches/tomatoes.log`.

## Exception groups

These are immediate causes, not claims about when the failures began. No HEAD run was performed.

- unresolved generation rule: `data/HeadEmission.xml`.
- unchanged configured space geometry: `data/LM_5M_IR.xml`, `data/MM_5M_AR.xml`, `data/MM_5M_IR.xml`, `data/XOR_recon.xml`.
- fixed input-address capacity: `data/MM_add.xml`, `data/MM_math.xml`, `data/XOR_spaces.xml`, `data/stream_smoke.xml`.
- 8 GiB worker guard: `data/MM_add_verb.xml`.
- TruthSet clause/closing requirements: `data/MM_ltm_consolidation_fixture.xml`, `data/MM_ltm_consolidation_stateful_fixture.xml`, `data/MM_qa.xml`, `data/MM_query_reasoning.xml`.
- MNIST data loading before model construction: `data/ergodic-only.xml`, `data/ergodic.xml`, `data/mnist.xml`, `data/simple.xml`.
- dataset lookup: `data/tomatoes.xml`.
