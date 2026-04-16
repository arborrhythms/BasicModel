# TODO

================================== Active Projects ==================================

## 1. Remove `ramsified` parameter and code; replace with `useButterflies` + `useGrammar`

The `ramsified` flag has been superseded by two orthogonal flags. The code still
carries `ramsified` everywhere — remove it. The four quadrants and their intended
architectures:

|                     | **No Grammar**                                 | **Grammar**                               |
|---------------------|------------------------------------------------|-------------------------------------------|
| **No Butterflies**  | Large flattened sigma (original)               | Grammar-directed composition              |
| **Butterflies**     | Pairwise mixing                                | ❌ Excluded (permutations fight constituency) |

### Scope
- **Models.py (55 mentions)**: delete `self.ramsified`, `self._ramsified_uses_grammar`,
  `self._ramsified_pair_enabled()`; rename `_ramsified_state_*` → `_butterfly_state_*`;
  restructure forward/reverse dispatch around (`useButterflies`, `useGrammar`);
  drop the legacy `elif self.ramsified and not self._ramsified_uses_grammar:` branch
  (no config uses it any longer); drop `use_stages` key from butterfly_config.
- **Spaces.py (5 mentions)**: remove butterfly code folded into `ButterflyStage` (see
  parallel cleanup below). Drop `butterfly=` kwargs from `_CSLevelView` / `_SSLevelView`
  and main `forward`/`reverse`, remove early-return branches, fold `butterfly_stages`
  into `self.sigmas` / `self.pi_layers`.
- **data/model.xsd**: remove `<ramsified>` element.
- **data/*.xml (11 files)**: remove `<ramsified>` element. All live configs are:
  - `useButterflies=true, useGrammar=false` → butterflies (MM_5M, MM_400M, MM_shamatha,
    MM_xor, MM_xor_step4, RamsifiedModel)
  - `useButterflies=false, useGrammar=false` → flattened sigma (MM_bpe, MentalModel,
    model, MM_grammar, MM_xor_grammar)
  - No live config is `useButterflies=false, useGrammar=true` yet — but that's the
    intended slot for grammar-directed composition; verify MM_grammar / MM_xor_grammar
    should flip `useGrammar=true`.
- **test/** (4 files, ~40 mentions): `test_hierarchical.py`, `test_mm_xor.py`,
  `test_config_scoping.py`, `test_reasoning.py` — rename `_ramsified_config()` helpers,
  drop `model.ramsified` checks in favor of `useButterflies`/`useGrammar`.
- **doc/** (5 files, ~9 mentions): `Spaces.md`, `Params.md`, `Architecture.md`,
  `Reasoning.md` — update terminology to useButterflies/useGrammar.

### Current status (entering this work)
- Orthogonal flags `useButterflies` and `useGrammar` already parsed in
  Models.py with legacy fallback (lines ~2008–2020). Exclusion of
  butterflies+grammar already raises `ValueError`.
- `ButterflyStage` class exists in `bin/Layers.py` and is wired through
  `butterfly_config["use_stages"]: True`. The `use_stages=False` legacy
  butterfly path in Spaces.py is dead — no live config uses it.
- Test suite: 639 pass, 1 pre-existing flaky (test_sigmapi XOR convergence,
  platform-specific local minimum 0.1298 vs 0.1 threshold; unrelated).

---

## 2. Stage 2: sequential concept→symbol mapping via accumulate pattern

(Deferred from butterfly refactor — to resume after Project 1.)

### Goal
Stage 1 established pairwise sigma/pi (butterflies) with N-halving, keeping D
constant. Stage 2 makes the conceptual codebook **wide AND deep** so a large
STM (short-term memory) activation pattern can drive sequential symbolic
emission.

### Architecture sketch
- **Wide conceptual codebook**: ConceptualSpace's codebook holds many vectors,
  each of moderate dimension. The *forward* pass produces an activation
  pattern across this codebook — this *is* STM.
- **STM = concept_states**: treat the per-batch concept activation vector as
  the STM. No separate STM memory; it's an interpretation of what's already
  computed.
- **LTM = TruthLayer snapshots**: truths stored in TruthLayer are the LTM.
  Pi + L1 bridge STM→LTM (commit highly-active concepts as truths).
- **Sequential symbolic emission**: instead of one-shot Pi over the whole
  STM, call `SymbolicSpace.forward(..., accumulate=True)` repeatedly. Each
  call emits the *next* symbol and accumulates it (running average or
  winner-take-all) into the emitted sequence. This matches how a decoder
  AR-generates tokens but the "tokens" here are symbols grounded in the
  conceptual codebook.
- **Context**: with Percept/Symbol/Word spaces all using the same nDim
  (nDim=4 + nWhere=2 = 6-dim vectors), we can concatenate
  `[percept + symbol + word, nDims + nEncoding]` and feed as a wide input
  to ConceptualSpace. nActive is the LLM context window; nWhere
  auto-increments from 0 (sequential symbol indices, not byte offsets).

### What's in place
- `SymbolicSpace.accumulate_symbol_objective(sym_vectors, target=...)`
  exists and caches nearest codebook targets.
- InputSpace nActive / nWhere semantics documented in
  `memory/project_inputspace_context.md`.
- Span-table architecture is live (see
  `memory/project_span_table_architecture.md`).

### What's needed
- Add `accumulate: bool = False` kwarg to `SymbolicSpace.forward()` that
  routes emission through a running-average accumulator instead of emitting
  a full symbol grid in one shot.
- MentalModel loop: instead of one Pi call per conceptual stage, drive Pi
  with `accumulate=True` in a sequential inner loop, pulling one symbol at
  a time from the STM.
- Define how the Wide codebook is trained: currently codebook is
  nSymbols-wide; Stage 2 wants nConcepts-wide (with nConcepts >> nSymbols).
  This likely means a separate ConceptualBasis distinct from SymbolicBasis,
  with Pi doing the codebook lookup.

### Open questions
- Should accumulate be per-symbol (one call → one symbol) or per-batch
  (one call → whole sequence, internally AR)?
- How does STM decay interact with accumulation? (STM_decay is in the
  config but currently only bleeds into concept_states, not symbolic
  emission.)

================================== April 24 ==================================

# Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

# Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

# Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov’s guardrails on the output, which is a famous failure mode in terms of it’s loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.


## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d² / 2(σx² + σy²))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere — a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform σ as a proxy — high gradient variance (unstable region) → larger σ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

## Implement forgetting
If someone has contributed information to an LLM and asked for the LLM
to learn from that data, even when the data is revoked, it will be remembered
in some vague way by the weights. So implement non-destructive forgetting:
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).

## Meaningful Prediction
Since we predict both words and sentences, the best prediction of future content is *meaningful*. Predictions of the next sentence should be ConceptualSpace predictions of the head (as a token), then of the NP+VP (as two tokens), iteratively refining until the full sentence has been specified as a grammatical tree — rather than strictly AR-1 over words (which is a surface-level prediction, not a meaningful prediction).

For example, instead of predicting "the fast dog jumped" token-by-token left to right, predict an XML-encoded version of `(((dog) fast) the) jumped` — the head of the sentence first, then iterative refinement of that conceptual space. This takes the same number of production steps as a current LLM, but the iterative refinement of the next-sentence production is conceptually much different, and closer to human reasoning where there is a core truth (S) with spatial NP and temporal VP that are successively refined by adjectives and adverbs that scope the conceptual space of that kernel sentence.

* Ensure words are aligned with sentences when there is a single unambiguous head.
* Head-first prediction requires that the head can be reliably identified from the composition.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.
