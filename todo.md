* doc/plans/2026-06-25-part-b-handoff.md

* The "Codebook.property_basis" is a hack that needs to be removed. Please summarize the WholeSpace property mechanism. You said properties "are" WholeSpace.what. But that codebook currently holds the symbol/truth prototypes wired into the codebook-snap machinery; making properties the live .what semantics would rip that out and move the basin. So I built the property capability as opt-in/additive (Codebook.property_basis) alongside the existing symbol codebook, not as a wholesale replacement. If you intended the live cutover, that's a separate deliberate step.

* Pelase make sure that the folllowing is wired into training: the model/train-level two-pass driver that runs the forward twice and applies two_pass_loss; the optional MLPTransformChooser; the soft-codebook option.

* The TruthLayer had been a container for user truth, and to some degree it has been replaced by the LTM.
For simplicity, it would be better to add user Truth to the LTM, and have TruthLayer be an interface to that tensor.

* Make sure the chart parser predicts masked words, so that we can train it predictions. To do so, it will be useful to consider the cateogry of the word and attention over all concepts.

* Learning a new concept is a parallel symbolic operation that may not be active at the same time as grammatical processing (serial mode). 
Ensure that we spend enough tim learning the definitons of words (I think we do Subsymbolic before Symbolic order, can we do symbolic order in addition to serial processing?).

* The "persisted episodic store of perceptual events" is an LTM store that corresponds to the words instead of the ideas composed of objects. Right now we treat words symbolically; that is their primary use case. However, the parser needs to extract word context to become a fluent symbol manipulator.
So the encoding of the words themselves, qua words, is not being done except in the mereology of the partSpace and wholeSpace.
So we need to store information about the context of words; that is where all of the power of word2vec comes from, which is a significant source of knowing that should not be ignored.
How might we well-capture that in this architecture? There is no context learning happening above symbolic encoding; might we regard the conceptual taxonomy, currently storing pairwise associations, as a bit of a markoff chain? Is there a way to do that which maintains pairwise relations, but which allows a nonlinearity at each association to add information to higher network levels?

The gap, stated precisely. A word here currently has three encodings, all hard/structural: its symbolic identity (codebook index), its orthographic mereology (PartSpace bytes/radix → WholeSpace word-as-whole), and its grammatical/taxonomic relations (the pairwise edges). What's missing is the fourth, soft/distributional one — meaning-from-company — which is the whole of word2vec's contribution. And you're right that it's a major source of knowing: the taxonomy tells you men ⊑ mortal; the distributional geometry tells you king − man + woman ≈ queen — graded analogical structure that no pairwise edge-set encodes.

* Formal Concept Analysis, DisCoCat

* A code's basis is expressed at one of several orders, and therefore the codebooks must be ramsified. Please analyse this. 

* The process of naming/identifying may require traversing multiple orders before a symbol can be matched with its meaning.

* Full LLM expressivity requires multiple layers of DISTINCT Sigma/Pi matrices. Lets enable that. 

* Ensure that we are store explicit Taxonymic knowledge when processing a definiton.
This should only happen when some measure of sentence confidence is high.

* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

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

