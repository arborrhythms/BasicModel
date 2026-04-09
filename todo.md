# TODO

Mapping from ConceptualSpace to SymbolicSpace has to happen only one vector at a time, since the SymbolicSpace uses a factored representation of each conceptual vector that is not stored. In other words, every concept has a nearest word with a factored representation, and processing into a new activation vector on conceptual space can be done by accumulating the symbolic activations serially. 

* ShortTermMemory can be implemented by a set of Symbols that have a high LR. They should exist on top of ConceptualSpace, at the last ConceptualOrder
* ConceptualOrder recursion should be ramsified.

* Luminousity should take account of the potential for lack of orthogonality of symbols to create dissonance, which can be approximated by conceptual overlap that causes interference within conceptual space, or that fails to illuminate some area of awareness.

* Universality should be tested for accurately identifying the subject and object of a transitive verb.

================================== April 17 ==================================

## Make sure that the mind is balanced
* characterize one-pointedness (ShamathaSpeech, no Equals, FA)
* characterize simplicity (continuity, OA)
* characterize one taste (balance dissonance and consonance)
* characterize non-meditation (resonance)

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

## Do lots of training
* optimize
* buy a fast machine

================================== May ? ==================================

* Implement forgetting  
If someone has contributed information to an LLM and asked for the LLM 
to learn from that data, even when the data is revoked, it will be remembered 
in some vague way by the weights. So implement non-destructive forgetting: 
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).
