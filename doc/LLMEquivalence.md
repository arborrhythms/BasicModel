# LLM equivalence: compose, predict, invert, emit

**Purpose (Alec, 2026-09-26).** State once, with a proof, the sense in
which a sentence-level model — partition a text on sentences, combine each
sentence's words into a representation by invertible transforms, predict
the next sentence's representation, invert, emit the words — is equivalent
to a token-level language model; and state exactly where the equivalence
stops, because each stopping point is a design fact of BasicModel.

## The theorem

Let `q` be any language model: a distribution over finite word sequences
that factors by the token chain rule. Let a boundary rule partition every
sequence into sentences `s_1 … s_n`. The chain rule regroups:

```
q(s_1 … s_n) = ∏_k q(s_k | s_1 … s_{k-1})
```

Let `f` map sentences to vectors injectively, with inverse `f⁻¹` on its
image. Define the sentence model `g` by pushing each conditional through
`f`:

```
g(e_{k+1} | e_1 … e_k) := q(f⁻¹(e_{k+1}) | f⁻¹(e_1) … f⁻¹(e_k))
```

**Claim.** Compose (`f`), predict (`g`), invert (`f⁻¹`), emit induces the
same distribution over word sequences as `q`.

**Proof.** A change of variables through a bijection preserves a
distribution; emission given `e` is deterministic; all randomness is in
`g`. Conversely, a sentence model with bijective `f` induces a token model
whose conditionals are marginals of the sentence conditional:
`p(w_t | w_{<t}, s_{<k}) = Σ_{s ⊇ w_{≤t}} p(s | s_{<k}) / Σ_{s ⊇ w_{<t}} p(s | s_{<k})`.
So the two classes are equally expressive, and partitioning on sentences
is a change of factorization, not of model. ∎

## Where it stops holding

Each step of the pipeline carries one condition. Where BasicModel meets
it, and where it deliberately does not, is recorded beside each.

1. **Invertibility must hold on the image the predictor reaches.** `f` is
   a bijection only onto its image; a predicted `e` will not in general
   lie on it, so inversion needs a projection onto the image. That is the
   tied reconstruction's candidate-bank search
   ([Architecture, tied reconstruction](Architecture.md#loop-and-parameter-ownership-tied-reconstruction-2026-09-16)).
   A bijection from discrete sequences to vectors is trivial to state and
   useless without metric structure — small errors in `e` must be small
   errors in `s` — which is the **generativity** requirement of the
   accessible-mind spec
   ([§2.0](specs/2026-09-20-accessible-mind-subsystems.md#20-fields-codes-and-ideas)).
2. **Prediction must be a distribution, not a point.** `g` is a
   conditional distribution over representations. A mean-squared-error
   predictor supplies the conditional *mean*, and `f⁻¹` of a mean is not
   the mode; it is usually not a sentence at all. Under MSE the
   equivalence holds only when the conditional is nearly deterministic —
   the regime of **reading**, where the next sentence is given and the
   predictor's job is expectation
   ([production objective §8.1, §8.7](plans/2026-09-15-next-sentence-as-the-production-objective.md#81-objectives-and-phase-boundaries);
   [expectation §2.6](specs/2026-09-20-accessible-mind-subsystems.md#26-expectation)),
   and not the regime of generation. Item 9's null on ordered prediction
   is consistent with asking a regressor for a distribution's worth of
   information. Closing the gap in the generative regime means a
   distribution over the next representation: a mixture, a diffusion, or
   the discrete form BasicModel already has, the chooser over candidate
   rows.
   In 6.8-1 the word level of `BracketExpectation` supplies that discrete
   distribution over a native candidate bank. The frozen next-word pilot uses
   its probabilities. This meets the distribution requirement at the word
   bracket; it does not make the sentence-level regressor a distribution over
   all sentences or establish the language learning gates.
3. **The history must be the same.** A token model conditions on every
   token in its window; the sentence model conditions on what it keeps.
   With LTM rows as a lossy statistic
   ([§2.7](specs/2026-09-20-accessible-mind-subsystems.md#27-ltm-semantic-and-episodic-memory);
   [forgetting](specs/2026-09-16-forgetting.md)), equivalence holds up to
   the information retained — a truncated context of a different shape
   from the window.
4. **Evaluation by perplexity is not available.** The token conditionals
   are marginals of the sentence conditional and are intractable from `g`
   unless `g` factorizes. A sentence model can be equivalent as a
   distribution and still be unscorable on perplexity, which is why the
   [NanoChat comparison](NanoChatGrammarPilot.md#question-and-falsifiable-first-milestone)
   scores a held-out next-word choice on common items and never compares
   reconstruction cost to bits per byte.

## The version BasicModel is

BasicModel's `f` is not on words but on meanings: compose after
`interpret`. Paraphrases map to one idea, so `f` is not injective; it is a
**quotient**. The theorem then holds on the quotient: the model is a
language model over meaning classes, and emission chooses a wording within
the class. Equivalence *modulo paraphrase* is weaker than equivalence, and
it is the intended weakness — the two truths are built on it
([ideas and relations](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided)).
The `<compose>`, `<thought>` and `<generate>` catalogs are the three faces
of `f`, its use, and `f⁻¹`
([Language](Language.md#grammar)).

## Empirical anchor

Meta's Large Concept Models (December 2024) are this pipeline literally:
sentences to fixed SONAR embeddings, next-embedding prediction, a decoder
back to text. Their plain MSE predictor underperformed, and they moved to
diffusion and quantized variants to obtain a distribution over the next
embedding — condition 2 observed in practice. Meta published no successor
as of September 2026; FAIR was reorganised into Meta Superintelligence
Labs in August 2025 and LeCun left in November 2025 for a world-model
lab. The line was continued elsewhere: SONAR-LLM (FusionBrain Lab, August
2025, revised May 2026) keeps the sentence-embedding state but trains it
with token-level cross-entropy propagated through the frozen SONAR
decoder — dropping the diffusion sampler and restoring a likelihood
signal, which is conditions 2 and 4 addressed together: the sentence
model is made scorable and trainable by routing the objective through
the token marginals it otherwise cannot compute.
