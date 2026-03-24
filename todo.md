# TODO

## test Model.py
* ensure that Ergodic works on LULayer
* rename LULayer -> InvertibleLinearLayer

## percept and concept vectors
* vectors that learn positions and vectors that learn boundaries
* make sure concepts and percepts both quantize nicely

## use subspace activation
* migrate all forward/reverse methods to pass subspaces (and remove the forward_subspace() method)

* all spaces call Materialize() at the top to extract their input vector (uses activation to select specified fields: activation/what/where/when)
* assert [nBatch, nInput, nInputDims]
* if flatten
  * flatten
* do Space-specific processing
* compute the (scalar) subspace activation
* if not flatten
  * assert nShape >= nOutput
  * do Space-specific processing
  * where nShape != nOutput, select the nOutput most active for propogation (creating [nBatch, nOutput, ?]) creating a new subspace
  * if quantized=true, select the nearest vectors from the Codebook
* if flatten
  * unflatten
* assert [nBatch, nOutput, nOutputDims]

## symbol processing
* symbols map bijectively to concepts
* they preserve only the activation of those concepts

================================== April 1 ==================================

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

================================== April 24 ==================================

## Do lots of training
* optimize
* buy a fast machine

================================== May ? ==================================
