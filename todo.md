
* WholeSpace property mechanism — I made a scoping call worth confirming. You said properties "are" WholeSpace.what. But that codebook currently holds the symbol/truth prototypes wired into the codebook-snap machinery; making properties the live .what semantics would rip that out and move the basin. So I built the property capability as opt-in/additive (Codebook.property_basis) alongside the existing symbol codebook, not as a wholesale replacement. If you intended the live cutover, that's a separate deliberate step.

* Genuinely future (engine built/tested but not wired into training): the model/train-level two-pass driver that runs the forward twice and applies two_pass_loss; the optional MLPTransformChooser; the soft-codebook option. Flag-on paths (neuralToolUser, symbolicComposition, intent-driven top-k) run but their semantics are yours to validate via training.
* Subsymbolic: learning higher-order words

* We should also include the TruthLayer, which ascertains the truth of all propositions before encoding them over symbols if they are consistent. In other words, besides the subsymbolic system that we have already built to lear the extension of words, we may learn a taxonomy that defines words in virtue of their intension. So, learning the syntax of a RelativeTruth is different than learning the syntax of an AbsoluteTruth because it encodes a relation over propositions which should be stored in a Taxonomy (e.g. "men" < "mortal"). That can be encoded as a part/whole relationship over symbols (the symbol for mortal is a part of the symbol for mortal: this can be stored in the representation (i.e. geometrically), and what it means is that the equivalence class of one is a subset of the equivalence class of the other). In terms of tokens that are assigned to types, such as socrates, if socrates is categorized as mortal, he should be promoted to a token of the type man (tokens define their nearest containing symbolic wholes).   


* Grammar Definition
  * NP and VP need masks on conceptual space that restrict their operation
  * Verbs are temporal things, nouns are spatial things, the full expression within conceptual space is possible only after lifting. 
  * Lifting twice is required for modality.

* grammatical reverse() operations

* Make the chart parser predict the words, taking into account the part of speech

* Learning a new concept
  * This requires iterative sigma/pi application over layers that turn an input vector into a symbol. This implies several things:
  * A symbol’s basis is expressed at one of several orders, and therefore the symbolic codebook must be ramsified
  * The process of naming/identifying requires traversing multiple orders before a symbol can be looked up
  * Full LLM expressivity requires multiple layers of distinct Sigma/Pi matrices. 

Sentences are sometimes composites of ideas (over relations isEqual and isPart).
* For example, questions relate two ideas:
  * Is subject predicate ?
  * part( x, y ) ?
* The IS of definition: equals ( x,y )
  * Store explicit parthood on WordSpace's Mereonomy when encountering a definiton (please rename from MereologicalTree to mereonomy) 
  * This should only happen when some measure of sentence confidence is high.

Memory of previous sentences requires prediction relating one to the next
* Store the sentences explicitly 
* predict from one sentence to the next

* Process truth statements
  * truth statements should become conceptual bivectors (ideas) and/or meronymic relationships
  * score the user's query in terms of the change in luminousity() vs the Truth

* Currently nWhere on percepts is unnecessary, because percepts are dense on input space (the network wiring densely covers the input).
  If perception is guided by attention, it can roam on input space, in which case the .where is particularly useful.

* .where dimensionality must scale with abstraction level: 1D suffices for a word token's input span, but accurate object (NP) representations need a 3D spatial extent (location / body / extent), and a VP needs a low-dim path/control manifold. Current .where is 2-dim; widening it for objects/VPs while keeping word-level uses smaller is a future change. (From the modality re-architecture: doc/plans/2026-06-03-modality-architecture-design.md.)

================================== April 24 ==================================


### Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

### Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

### Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

### Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

### Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

