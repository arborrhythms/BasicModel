# TODO

## Reasoning
* <facts> should not /bias/ learning, they should form a logical basis for it.
* Logical operations on concepts create new conceptual spaces that can be symbolized.
* Arbitrary conceptual spaces can be decoded into symbols.
* Logical inference over the set of truth should result in non-contradictory propositions, and should not reduce luminousity.
* LLMS minimize prediction loss. 
  It would be better to maximize truth: so integrate a loss penalty for false propositions against the TruthSet as weighted by DoT

## Make sure that the mind is balanced
* characterize one-pointedness (ShamathaSpeech, no Equals, FA)
  * perceptually and linguistically, it decodes to a contiguous region.
* characterize simplicity (continuity, OA)
  * small shifts in perceptual and symbolic space are small shifts in conceptual space 
* characterize one taste (balance dissonance and consonance)
  * somehow feelings need to be appropriate (feelings should not be removed, as the nihilists attempt). This manifests when the objects that are loved are either real or when the representations are at least 5D (which limits the dissonance of reification).
* characterize non-meditation (resonance)
  * the error function is relatively small in all cases (i.e. no more learning)


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


================================== May ? ==================================

* Implement forgetting  
If someone has contributed information to an LLM and asked for the LLM 
to learn from that data, even when the data is revoked, it will be remembered 
in some vague way by the weights. So implement non-destructive forgetting: 
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).
