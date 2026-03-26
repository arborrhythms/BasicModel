# TODO

## use subspace activation
* introduce an objectActivation which is used when the subspace data is stored in object
* create methods 
  * setActivation(whatA, whereA, whenA) to store a cross-product activation
  * setObjectActivation(objectA) # compute the (scalar) subspace activation and assign to the subspace
  * materialize() converts these to an objectA and then returns object[objectA] # select at most nOutput entries zero the other activations

## symbol processing
* symbols map bijectively to concepts
* when calling materialize, extracts only [activation, index] (the numeric indices of the concepts)
* when reverse() is called, indices are used to look up the concepts that were passed in
* its subspace consists only of an (ordered) vector of activations
* It passes [nBatch, nOutput, 1] to all consumers (OutputSpace and SyntacticSpace)
* Integrate the Activation and Syntactic Layer classes

## AttentionLayer
* re-write the attention layer

================================== April 1 ==================================

## Percept and concept vectors
* vectors that learn positions and vectors that learn boundaries
* make sure concepts and percepts both quantize nicely

## Implement equals in symbolicSpace
* "equals"

## Implement input-energy reduction in MentalModel
* implement "Truth is accurately characterizing what exists"
* "union, intersection, part, not, nor"

## Map input sentences to operations
* map mental operators to SyntacticSpace
* implemment symmetry (the golden rule)
* implemment isomorphism
* map TruthSets to SyntacticSpace (top-down reasoning)

## Make sure that the mind is balanced
* characterize one-pointedness (ShamathaSpeech, no Equals, FA)
* characterize simplicity (continuity, OA)
* characterize one taste (balance dissonance and consonance)
* characterize non-meditation (resonance)

================================== April 17 ==================================

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

# Implement forgetting
* if someone has contributed information to an LLM and asked for the LLM 
to learn from that data, even when the data is revoked, it will be remembered 
in some vague way by the weights. So implement non-destructive forgetting: 
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).

================================== April 24 ==================================

## Do lots of training
* optimize
* buy a fast machine

================================== May ? ==================================
