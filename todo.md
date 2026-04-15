# TODO

This machine answers the question "what is the truest thing we can say of this experience?".
The way it does that is to preserve energy from the input as light or illumination.
If the machine tries to maximize illumination directly, it might push weights high, so weight vectors are normed: they must match the input.  
This is another way of saying "match the input within a constrained set of praameters such that the difference between the input and the prediction is zero".

If there is no truth that the machine can see, it says nothing.
The one execption to this is that if the machine sees a contradiction, whihc would otherwise result in darkness, it can offer up one of the alternatives as a question.

Currently, each layer has a contract not to exceed -1..1
If we attempt to maximize the norm of the signal at any level, we will obviously need a strong guarantee of signal stablility first.



Create 1B model

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
