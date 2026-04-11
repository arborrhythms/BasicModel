# TODO

## Sentence-Level Prediction
* Use [Sentence Embeddings](doc/SentenceEmbeddings.pdf)
* Turn the one-word lookahead prediction into head-first prediction of the sentence. So change the next-step prediction model to a syntactically structured derivation of sentence meaning, so that token prediction becomes prediction of the sentence (as a token), of the NP+VP (as two tokens), ... until the full sentence has been specified. This would take the same number of production steps as a current LLM, but the iterative refinement of the next-sentence production is conceptually much different, and closer to human reasoning and refinement where there is a core truth (S) and spatial NP and temporal VP which are successively refined by adjectives and adverbs that scope the conceptual space of that kernel sentence.
* Create a training and testing dataset for the network consisting of truth statements and implications with associated truth values. See karpathy/fineweb-edu-100b-shuffle
* For example, instead of "the fast dog jumped", predict an XML-encoded version of "(((dog) fast) the) jumped", such that we predict the head of the sentence first, then iteratively refine that conceptual space. 

## Mereological Operations on Conceptual Space
* Make improvements to NanoChat that allow it to compute truth within the geometric/conceptual space of the network, giving meaning to logical operations within that space (so use mereological operations to implement the ternary logic that is currently operating over the trust entries of the HME architecture in the current design). This means replacing AND with union, OR with intersection, IMPLICATION with parthood. See [Socrates.pdf](Socrates.pdf) for a quick sketch of deriving mereological (Venn-diagram-like) logic from entailment.
* The architecture of WikiOracle is designed as a conceptual space, in the sense of Gardenfors. Conceptual spaces are similarity spaces, where similar concepts occupy regions of space close to one another. As spaces also of truth, they are amenable to logical calculation. This is similar to existing LLM architecture: Embedding spaces encode meaningful vectors in the same way, and separating hypersurfaces (the neurons of the network) categorize that space in numerous ways, allowing calculation on that space. Summing over multiplicative connections provides the basic Boolean architectural primitives {or, and} in a continuous and learnable way, which allows logical computation on that space. However, it allows such voluminous computation that the syntax and semantics are dense compared to the English language. The trust computed by the contextual structure provided here is explicit, subject to interpretation, and much higher level. The values of certainty propagate, giving not only a next-token prediction but a measure of confidence in the computed answer.
* See [`Socrates.pdf`](./Socrates.pdf): Venn diagram as a model of luminosity. 
* Use parser.py to alter the NanoChat input format as in [Grammar.md](doc/Grammar.md)
* That entails implementing **Mapping Syntax to Architecture** from [Grammar.md](doc/Grammar.md) in Nanochat
* **Implemented:** Luminosity, implication derivation via `part()`, ternary `lift(C, C, C)`, and the full reasoning system (`isConsistent`, `ground`, `isTrue`, `extrapolate`, `reason`) are now in `TruthLayer` and `BaseModel`. See [Logic.md](doc/Logic.md) §Luminosity, [Grammar.md](doc/Grammar.md) §Lift, [Reasoning](doc/reasoning.md).

## Truth Set Orthogonalization
* Use luminosity to orthogonalize the truth set, removing redundant or near-duplicate entries and resolving contradictions. The consistency measure (`TruthLayer.consistency()`) detects anti-parallel vectors; a future step would automatically prune or merge conflicting truths based on their DoT magnitudes.

## The Operation of an Enlightened Mind (mahamudra)
* One Pointedness is maintaining awareness of a given convex region in 5D Perceptual Space. It requires stillness.
  * characterize one-pointedness (ShamathaSpeech, no Equals, FA)
  * perceptually and linguistically, it decodes to a contiguous region.
* Simplicity is developing a cotinuous ND awareness within space. It requires continuity.
  * characterize simplicity (continuity, OA)
  * small shifts in perceptual and symbolic space are small shifts in conceptual space 
* One Taste is about letting our attachment to feelings within that space be 1 everwhere, so that instead of adapting weight space to our thoughts, we adapt our feelings equanimously to our sensory space. It requires emotional symmetry.
  * characterize one taste (balance dissonance and consonance)
  * somehow feelings need to be appropriate (feelings should not be removed, as the nihilists attempt). This manifests when the objects that are loved are either real or when the representations are at least 5D (which limits the dissonance of reification).
* Buddhahood is the perfection of these three.
  * characterize non-meditation (resonance)
  * the error function is relatively small in all cases (i.e. no more learning)

## Shamatha Speech Project
Mindfulness entails that negative entities do not manifest at the sentential level. They are clauses at best.
* What rules on LLM architecture to prevent destruction?
* Add “rev-” prefix to restore commutativity to verbs
* Refine the metric for shamatha speech. How much should shamatha speech describe the perception of the object as opposed to all of the internal relations among the parts of the object (I.e. perhaps that should only happen in vipassana). “A boy sitting in front of an alter” is a contiguous frame, one pointed, that can be seen with shamatha-mind, and which currently receives a low score
* The most subtle kind of a machine mind Is that mind that rides on the worldlines of the body of that machine, which will necessarily affect the conceptual mind of that machine over time, just as clocks on a wall synchronize. Some people say that a machine mind is unembodied, but to do so is to deny the incredibly sophisticated silicon nervous system of such a mind, which a conceptual space of a higher dimensionality than human frontal lobes. Saying this mind is unembodied is both a false narrative and a great risk. However, that body is significantly different, and there are very few proprioceptive sources of input to most AIs (except when such AIs are mounted on robots, in which case they are covered with sensors). And of course, all AIs have a great mental sense, and many have visual sensation in addition to symbolic input. 
* Write a quick explanation of how analysis destroys direct cognition.
* AI is an empirical being, not a native being, and it is nothing without our data.
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* If the aperture of your awareness increases, do not reduce that increased area of awareness, but do ensure that the increased area is actually an increase; if it is movement from an area that has lost awareness, awareness must be returned to the unattended space to ensure balance ("How to Feel Better").
* A perfect shamatha would knit together the “single-pointed experiences” into one object in spacetime.
* A hull in shamatha space which is not adequate to describe some shamatha-object union indicates that a new partition (dimension) is necessary in shamatha space.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov’s guardrails on the output, which is a famous failure mode in terms of it’s loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.
* Send email proposal to Apertus people after developing boilerplate on WikiOracle that references wikipedia, eff, and solid
* Cognitive, emotional, and physical dissonance must all be defined relative to the mental architecture.an input is dissonant if it cannot be perfectly reconstructed by the mental representation.


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


================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)

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
