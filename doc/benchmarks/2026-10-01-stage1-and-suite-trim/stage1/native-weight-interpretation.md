# Native stage-1 weights and their roles

This describes the measured pre-normalization candidate, not the later cost-function
implementation. Both native arms use the unchanged production configuration. The
cut arm omits only the trial's answer term. Numerical term distributions and physical
parameter membership are in each completed arm's JSON files.

| Setting / term | Value | Where and what it does |
|---|---:|---|
| `reconstruction_scale` | 0.5 | Multiplies each sentence trial's owned byte reconstruction through its understanding. At batch end it weights `lossIn` and any additional `lossRev`; `1-r=0.5` weights `lossOut`. This is the existing reconstruction/output blend. |
| Trial supplied answer | 1 | Adds the supplied answer's error with its graph intact; omitted in the cut arm. Both trials are costed before either optimizer step. |
| `what_scale` / `where_scale` / `when_scale` | 0.7 / 0.2 / 0.1 | Weight content, location and time in the event error. A bandless numeric output uses the content weight only. These internal factors precede the batch-end blend. |
| `inter_loss_weight` | 0.1 | Weights the structured next-sentence prediction. Content MSE, presence BCE and kind BCE have internal weights 1. The cold first sentence has no prior prediction, so this production run's expectation term is zero; that is not a claim of perfect prediction. |
| `inter_contrastive_weight` | 0 | Disables the optional next-sentence contrastive term. Its configured temperature is 0.1, a score temperature rather than an additive cost weight. |
| `intra_loss_weight` | 0 | Disables the older within-sentence predictor cost. |
| `arma_scale` | 0 | Disables the older discourse prediction cost. |
| `grammar_lesson_weight` | 1 | Weights supplied grammar lessons when present; this benchmark supplies no lesson cost. Batch reports of sentence-trained lessons are detached, preventing a second training owner. |
| `embedding_scale` | 0.25 | Batch-end SBOW embedding objective, when present; no SBOW term was produced in the measured native batch. |
| `conceptual_similarity_scale` | 0 | Disables conceptual SBOW. Its batch-end branch is for a parallel reading. |
| `definition_sparsity_scale` | 0 | Disables the definition soft-L0 penalty. The producer already applies this factor when enabled. |
| `output_policy_weight` | 0 | Disables the answer walk's sampled-action credit at the batch end. Trial answer generation does not own that policy cost. |
| `expectation_policy_weight` | 0 | Disables optional expectation policy credit. |
| `selected_thought_policy_weight` | 0 | Disables selected thought-episode answer credit. |
| `leaf_distill_weight` | 0 | Disables the older leaf-distillation objective. |
| `gate_l1_lambda` | 0 | Disables operator gate sparsity. The producer includes the factor when enabled; an already weighted penalty is added once. |
| Concept-readout L1 | 0.01 per each of four levels | Optimizer-owned proximal penalty on each concept readout. The batch total includes its detached report with outer weight 1; there is no duplicate L1 backward term. |
| `luminosity_weight` / `universality_weight` | 0.1 / 0.1 | Existing detached truth modulation of the assembled batch total. The observed batch has no active modulation. |
| `truth_loss_weight` | 0 | Disables the independent truth penalty. The truth owner's balance factor is 0.1; no balance penalty was produced here. |
| Pipeline auxiliary costs | Producer's individual weight, outer sum 1 | The shared pipeline Error registry is consumed once at batch end. None were present in this native batch. |

Each trial reduces over its active sentence rows. Batch terms use their producer's
reduction; the receipt preserves raw and weighted values separately and reports the
accounting residual. The native trial and batch weights are not interchangeable:
for example, the answer has trial weight 1, while the batch answer additionally has
the `0.5` blend and applicable event-band weight.

The contextual code-learning rate `0.01` and situation/expectation weights `0/0`
control buffer rotation rather than autograd loss terms. The byte-assignment
softmax temperature `0.1` controls assignment scores rather than weighting a cost.
`reconstruct_in_loop=true`, `detached_reverse=false`, `answer_synthesis=true` and
`legacy_prediction_enabled=false` select the measured paths; none was changed.

The gradient manifest distinguishes owned trainable parameters from native object
code buffers. Generation and the reading map overlap in this native configuration;
physical parameter IDs and the overlap are recorded, so their norms must not be
added as if they were disjoint groups. The exact cached-perception pullback is
included. Zero expectation gradients at the cold state do not establish that the
expectation path is disconnected.
