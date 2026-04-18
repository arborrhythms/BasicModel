# AR Sentence Prediction Integration

**Goal:** Predict the current sentence from past sentence history before each ARLM next-word step. The prediction seeds the intuitive (perceptual→conceptual) pathway, guiding word generation via `prediction × sentenceConfidence`.

---

## Components

### 1. LTM — SymbolicSpace sentence store
- Add `ltm_s` and `ltm_w` persistent ring buffers to `SymbolicSpace` (or extend `WordSpace.discourse`)
- `ltm_store(s_vec, w_vec)`: commit the current sentence's S+W snapshot to LTM at each sentence boundary
- `ltm_retrieve(query, top_k=3)`: cosine-similarity lookup over the LTM ring; returns a top-K weighted mean for use as context
- Config: `<ltmSize>` (default 64 sentences)
- Buffer is non-persistent (transient per run), kept detached — no backprop through history

### 2. STM — ConceptualSpace gamma trace
- Add `stm` buffer to `ConceptualSpace`: exponentially decaying trace over concept activations
- Update rule at end of each forward pass: `stm = γ * stm + (1−γ) * concept_act`
- `γ` configurable via `<stmGamma>` (default 0.9)
- Reset to zero at document boundaries, not sentence boundaries — the trace carries the conceptual thread across sentences within a document
- Dim matches `concept_dim`; non-persistent buffer

### 3. SentencePredictor module
- New `SentencePredictor` layer (add to `Layers.py`)
- Input: `concat([ltm_retrieve(stm), stm])` → Linear → Tanh → `predicted_sentence_vec`
- Runs **per-batch, once per sentence** — not per token
- `sentenceConfidence = clamp(cosine_similarity(predicted_sentence_vec, actual_sentence_vec), 0, 1)`
- Prediction loss: `L_sentence = 1 − cos(predicted_sentence_vec, actual_S_W_snapshot)`
- Separate from the ARLM per-symbol predictor (which runs per token over S and W)

### 4. ARLM integration
- **Before** the ARLM next-word generation loop: run `SentencePredictor`
- Compute seed: `seed = predicted_sentence_vec × sentenceConfidence`  
- Project `seed` to `concept_dim` and **add as a bias to the ConceptualSpace input**:  
  `concept_input[:, :nPercepts, :] += seed_projected`
- The sentence-level prediction provides the "gist"; the per-symbol ARLM predictor handles token-level details
- `sentenceConfidence` is used as a scalar gate — low confidence means the seed has little effect

### 5. Training
- `L_sentence` added to total loss, scaled by `sentencePredictionScale` (existing config key)
- STM and LTM contents are always detached — no BPTT through history
- Sentence predictor parameters included in the main optimizer
- Loss is only computed when `_recent` buffer is non-empty (skip first sentence, consistent with existing discourse loss guard)

### 6. Config additions (XML)
```xml
<training>
  <!-- existing -->
  <sentencePrediction>true</sentencePrediction>
  <sentencePredictionScale>0.1</sentencePredictionScale>
  <!-- new -->
  <ltmSize>64</ltmSize>           <!-- LTM ring buffer depth (sentences) -->
  <stmGamma>0.9</stmGamma>        <!-- STM exponential decay factor -->
</training>
```

---

## Implementation Order

1. **STM**: Add `stm` buffer + update step to `ConceptualSpace.forward()` — gated on `stmGamma > 0`
2. **LTM**: Add `ltm_store` / `ltm_retrieve` methods to `SymbolicSpace` (or `InterSentenceLayer`) — ring buffer over S+W snapshots
3. **SentencePredictor**: New layer in `Layers.py` — takes `(stm, ltm_context)`, outputs `predicted_sentence_vec` + `sentenceConfidence`
4. **Wiring**: In `BasicModel.forward()`, run `SentencePredictor` before the ARLM loop; compute and inject `seed` into `concept_input`
5. **Loss**: Add `L_sentence` to `runBatch()` total loss via the existing `sentence_prediction_scale` path
6. **Config**: Add `<ltmSize>` and `<stmGamma>` to XML schema and `TheXMLConfig` reads
7. **Tests**: `test_ltm_store_retrieve`, `test_stm_decay`, `test_sentence_predictor_forward`, `test_seed_injection`, `test_ar_sentence_loss`

---


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
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.


## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
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
Since we predict both words and sentences, the best prediction of future content is *meaningful*. Predictions of the next sentence should be ConceptualSpace predictions of the head (as a token), then of the NP+VP (as two tokens), iteratively refining until the full sentence has been specified as a grammatical tree -- rather than strictly AR-1 over words (which is a surface-level prediction, not a meaningful prediction).

For example, instead of predicting "the fast dog jumped" token-by-token left to right, predict an XML-encoded version of `(((dog) fast) the) jumped` -- the head of the sentence first, then iterative refinement of that conceptual space. This takes the same number of production steps as a current LLM, but the iterative refinement of the next-sentence production is conceptually much different, and closer to human reasoning where there is a core truth (S) with spatial NP and temporal VP that are successively refined by adjectives and adverbs that scope the conceptual space of that kernel sentence.

* Ensure words are aligned with sentences when there is a single unambiguous head.
* Head-first prediction requires that the head can be reliably identified from the composition.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.
