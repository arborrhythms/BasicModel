# TODO

## test Model.py
* ensure that Ergodic works on LULayer
* rename LULayer -> InvertibleLinearLayer

## use subspace activation
* migrate all forward/reverse methods to pass subspaces (and remove the forward_subspace() method)

* For all spaces:
* call Materialize() # extract the input vector in a suitable form (select specified fields: activation/what/where/when)
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
* when calling materialize, extracts only [activation, index] (the numeric indices of the concepts)
* when reverse() is called, indices are used to look up the concepts that were passed in
* its subspace consists only of an (ordered) vector of activations
* It passes [nBatch, nOutput, 1] to all consumers (OutputSpace and SyntacticSpace)

## percept and concept vectors
* vectors that learn positions and vectors that learn boundaries
* make sure concepts and percepts both quantize nicely

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
