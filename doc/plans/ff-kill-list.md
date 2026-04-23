# Feed-Forward Refactor Kill-List

Generated: 2026-04-22 (Task 1 of 2026-04-22-pipeline-ff-architecture.md).
Removed at Task 22.

## basicmodel/bin/Spaces.py

ConceptualSpace:
- `__getitem__` (line ~5704)              -> delete
- `_CSLevelView` inner class (line 5707)   -> delete
- `sigmas: ModuleList`                     -> replaced by single `.sigma` per-instance
- `level_shapes`, `_hierarchical` flags    -> delete
- `self._last_svo` (line 5680, 5799)       -> moved to vspace.wordSpace.last_svo
- `@property last_svo` (line 5809)         -> delete
- `self.sentence_primer = SentencePrimingLayer(...)` (line 5691) -> delete
- `from Layers import ..., SentencePrimingLayer` (line 226) -> remove SentencePrimingLayer

SymbolicSpace:
- `__getitem__` (line ~6367)               -> delete
- `_SSLevelView` inner class (line 6370)   -> delete
- `pi_layers: ModuleList`                  -> replaced by single `.layer` per-instance
- `level_shapes`, `_hierarchical` flags    -> delete
- `self._symbol_objective_terms` (lines 6177, 6328-6329, 6334, 6340, 6414-6416, 6759-6791) -> writes go to vspace.errors; dict deleted
- `reset_symbol_objective` (line 6176)     -> deleted
- `accumulate_symbol_objective`            -> deleted
- `symbol_objective_loss` (line 6354)      -> deleted
- `symbol_objective_terms` method          -> deleted
- `self.reset_symbol_objective()` in __init__ (line 5947) -> delete

## basicmodel/bin/Models.py

MentalModel.create:
- `self.conceptualSpace = ConceptualSpace(..., level_shapes=...)` -> replaced by ModuleList `self.conceptualSpaces`
- same for `symbolicSpace`

MentalModel.forward (line 1750, 3164):
- `self.symbolicSpace.reset_symbol_objective()` -> delete

MentalModel.reverse (lines 3355, 3359, 3372, 3376):
- `self.conceptualSpace[t]` -> `self.conceptualSpaces[t]`
- `self.symbolicSpace[t]`   -> `self.symbolicSpaces[t]`

MentalModel.build_pipelines (lines 2832, 2879-2880, 2896):
- `AdditiveFeedbackGlue` insertion -> delete (linear chain)
- `self.conceptualSpace[t]` -> `self.conceptualSpaces[t]`
- `self.symbolicSpace[t]`   -> `self.symbolicSpaces[t]`

BaseModel.runBatch (line 2257):
- `self.symbolicSpace.symbol_objective_loss()` -> replaced by summing `self.outputSpace.subspace.errors.terms()`
- `self.symbolicSpace.symbol_objective_terms()` -> same

## basicmodel/bin/Layers.py

- `class SentencePrimingLayer(Layer):` (line 3148) -> delete

## basicmodel/bin/Pipeline.py

- `AdditiveFeedbackGlue` class (line 181) stays but docstring notes "unused by MentalModel.build_pipelines"

## basicmodel/test/

- `test_sentence_priming_layer.py` -> delete (Task 11)
- `test_phase2_pipeline_primitives.py` (lines 88, 181, 188, 201, 210, 222) -> AdditiveFeedbackGlue unit tests stay (class still exists)
- `test_basicmodel.py` (lines 3341, 3362, 3382) -> reset_symbol_objective / symbol_objective_loss callers updated
- `diag_where.py` (line 59) -> `_last_svo` reference updated
- `test_legacy_removed.py` -> add guard for `_symbol_objective_terms`, `_CSLevelView`, `_SSLevelView`, `class SentencePrimingLayer`
