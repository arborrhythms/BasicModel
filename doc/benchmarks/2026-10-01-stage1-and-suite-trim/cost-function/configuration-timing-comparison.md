# Universal-reconstruction prototype: first-batch timings

These compare the verified bank-only candidate with the saved mandatory-serial-reconstruction prototype. The prototype is NOT the current production source: its new failures remain unresolved, and its patch has been restored out. One fresh unseeded process per configuration and phase; unchanged XML batch, CPU, no-compile backend in both phases, one first training batch including cold work, no endpoint evaluation. This is a timing probe, not a gate campaign or a steady-state epoch benchmark. No failed run is replaced.

The worker guard is 8 GiB / 30 minutes throughout. A memory stop is censored, not a timing. Configuration/data failures before training are retained. The defaults file model.xml was included conservatively in the initial static audit, but its actual model is parallel and it does not gain a serial inverse; it is explicitly excluded from the gain claim.

| Configuration | Before seconds | Prototype seconds | Delta seconds | Before outcome | Prototype outcome |
|---|---:|---:|---:|---|---|
| BasicModel_answers_benchmark | 3.5543 | — | — | completed | memory |
| BasicModel_expectation_benchmark | — | — | — | ValueError('complete sentence exceeds the packed word capacity: source index 23968 has 72 words, capacity is 64; refusing to clip it') | ValueError('complete sentence exceeds the packed word capacity: source index 23968 has 72 words, capacity is 64; refusing to clip it') |
| HeadEmission | — | — | — | ValueError('unresolved generate rule: S -> C') | ValueError('unresolved generate rule: S -> C') |
| LM_5M | 2.3437 | — | — | completed | RuntimeError('tied reconstruction bank requires staged word and sentence masks') |
| LM_5M_IR | — | — | — | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') |
| MM_20M_fineweb | — | — | — | memory | RuntimeError('tied reconstruction has no surface candidates after concept admission; populate the concept inventory and stage its WORD surfaces before reconstruction') |
| MM_20M_grammar | — | — | — | memory | memory |
| MM_400M | — | — | — | AssertionError("Embedding.insert: lexicon is full (65536 >= capacity=65536); the where-space slice allocated for this Embedding only covers ``lexicon_capacity`` prototypes. Raise <nVectors> on the owning Space's XML (or the Embedding._LEXICON_DEFAULT_CAPACITY default) and reload.") | AssertionError("Embedding.insert: lexicon is full (65536 >= capacity=65536); the where-space slice allocated for this Embedding only covers ``lexicon_capacity`` prototypes. Raise <nVectors> on the owning Space's XML (or the Embedding._LEXICON_DEFAULT_CAPACITY default) and reload.") |
| MM_5M_AR | — | — | — | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') |
| MM_5M_IR | — | — | — | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') | ValueError('XML config inconsistencies:\n  - flat-slab invariant violated: PS.nOutput*content (1024*6=14336) must equal CS.nOutput*content (8*1030=8240), except when aligned serialObjectMeta uses equal native PS/WS peers followed by sparse sigma/codebook activation. That activation must already emit CS-width events. The configured geometry is PS 1024x14, WS 1024x1030, and CS input 1024x14 -> output 8x1030; IS may be larger (PS scopes the input down).') |
| MM_add_verb | — | — | — | memory | memory |
| MM_boolean | 0.9422 | 1.1029 | 0.1607 | completed | completed |
| MM_bpe | 0.3432 | — | — | completed | RuntimeError('tied reconstruction bank requires staged word and sentence masks') |
| MM_decode | 3.2027 | — | — | completed | RuntimeError('tied reconstruction requires completed sentence state') |
| MM_grammar | 0.5314 | 0.5187 | -0.0127 | completed | completed |
| MM_ladder | 2.3811 | 2.5550 | 0.1738 | completed | completed |
| MM_ladder_idiom | 4.1725 | 4.0333 | -0.1392 | completed | completed |
| MM_ladder_text | — | — | — | memory | memory |
| MM_ladder_textpacked | — | — | — | memory | memory |
| MM_ltm_consolidation_serial_fixture | 0.5545 | — | — | completed | RuntimeError('tied reconstruction has no surface candidates after concept admission; populate the concept inventory and stage its WORD surfaces before reconstruction') |
| MM_mereology_serial | 9.0158 | — | — | completed | RuntimeError('tied reconstruction requires completed sentence state') |
| MM_meronomy_smoke | 4.7538 | 4.9501 | 0.1963 | completed | completed |
| MM_nanochat_grammar_gate | 8.4452 | — | — | completed | RuntimeError('tied reconstruction has no surface candidates after concept admission; populate the concept inventory and stage its WORD surfaces before reconstruction') |
| MM_nanochat_grammar_pilot | — | — | — | memory | memory |
| MM_phrase_decode | 4.7373 | 4.9219 | 0.1846 | completed | completed |
| MM_query_reasoning | 4.1779 | — | — | completed | ValueError('a TruthSet cannot assert an interrogative clause') |
| MM_sequence_predict | 100.9239 | — | — | completed | RuntimeError('tied reconstruction bank requires staged word and sentence masks') |
| MM_shamatha | 0.4380 | 0.4749 | 0.0369 | completed | completed |
| MM_xor_loopback | 0.4311 | 0.4797 | 0.0486 | completed | completed |
| MM_xor_step3 | 0.4372 | 0.4814 | 0.0442 | completed | completed |
| MM_xor_step4 | 0.4693 | 0.5117 | 0.0424 | completed | completed |
| MentalModel | 0.7623 | 0.8187 | 0.0564 | completed | completed |
| POS_smoke | 0.9566 | — | — | completed | RuntimeError('tied reconstruction has no surface candidates after concept admission; populate the concept inventory and stage its WORD surfaces before reconstruction') |
| RamsifiedModel | 0.6892 | 0.7389 | 0.0496 | completed | completed |
| XOR_grammar | 0.4397 | 0.4810 | 0.0414 | completed | completed |
| model | — | — | — | AssertionError() | AssertionError() |
