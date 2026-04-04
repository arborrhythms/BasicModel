# TODO

* ProjectConcepts() : 
Mereology
Implement parthood, union, intersection on the embedded vectors
parthood needs to exist on both concepts (as activation containment) and percepts (as a simpler "<" operator)

================================== April 9 ==================================

* SyntacticLayer() : 
the logical operations on symbols form a basis for propositional knowledge over all symbols.
symbol-symbol words can be dynamically added (represented) as new symbols in SymbolicSpace
ensure that EQUALS and other symbolic logical operators in sentences equate the meaning of those symbols.
add LIFTING/LOWERING to handle VP / ARTICLE

* Verify() : map TruthSets to ConceptualSpace (top-down reasoning) 
a given sentence can be measured against a corpus of truth by measuring the osel after apoha of the TruthSet.
negation of the non-concept from the truthset leaves a space of truth and a background of falsity.
it will be sparse: not all turths will be covered.

* Universalize() : implement universality (the golden rule) by measuring osel after round-robin projection of the given action
Next we need to determine some measure of harm or destructiveness osel(x).
Finally if K(X,Y) + K(Y,X) leads to less osel(X’)+osel(Y’) than osel over the original X and Y, then the kronecker in question is not universalize, and thus not ethical.
Osel(x) is the weak spot in this architecture, the rest is math. One way to describe it is through subjective measure; another might be to describe its “perceptive volume”. 
If we follow Augustine is saying that all percepts are good, then any increase in volume is better. That does capture growth and destruction as being relatively good and  bad for the things in question.

* Implement forgetting
If someone has contributed information to an LLM and asked for the LLM 
to learn from that data, even when the data is revoked, it will be remembered 
in some vague way by the weights. So implement non-destructive forgetting: 
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).

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
