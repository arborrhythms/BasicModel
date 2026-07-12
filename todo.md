

## Deliverable: create a dedicated masked semantic reconstruction training config.
  * Add a new XML training file derived from the masked IR config, but with the parser-side semantic context enabled. It should train masked-word reconstruction/prediction, not plain byte reconstruction: set a nonzero training.maskRate, use reconstruction-heavy loss (reconstructionScale high enough to dominate), keep sentencePrediction enabled, and enable the symbolic/category machinery needed for the parser to use word category evidence during reconstruction.
  * The config should also enable bottom-up concept attention: use the sparse conceptual loop path with symbolicOrder > 0, parallel symbolic execution, mereologyRaise, symbolTower, and the concept inventory sized so snap rows plus relation-pool rows cannot overflow. The goal is that a masked word is predicted from both its parser category/semantic-whole evidence and bottom-up attention over all active concepts.
  * Acceptance criteria: running this config performs masked IR training; masked positions contribute reconstruction loss; parser/category evidence is visible in the reconstruction path; concept attention/wave is live, not dark; and a targeted smoke test confirms masked-word predictions change when category evidence or concept-attention state is disabled.


## Implement doc/plans/2026-07-04-attention-to-relation-promotion.md
  * Sentences create explicit knowledge if they pass a criterion:
  learn_score = children_in_codebook * is_truth_obvious * resolves_contradiction
  accept iff learn_score >= truth_criterion and truth_criterion < 1


## Operational integration of the QUERY TOOLS (isEqual, isPart, exist, and
  the reasoner's equal/part/query/quantize/wholes/parts/arma) — the
  invocation design is now SPECIFIED in
  doc/plans/thinking_kernel_spec.md (the Thinking Kernel: lookup/part/
  think/query/answer over truth intervals + trust, STM frame stack,
  runtime-enforced execution loop). It extends the reasoner-driven,
  post-parse model already LIVE-wired per
  doc/plans/2026-06-23-reasoning-live-wiring.md (serve.py answer_query ->
  Models._detect_query/reason_about -> NeuralToolUser over reasoning.py's
  TruthGroundedReasoner); the tools build NO parse structure and live in
  <Queries>. REMAINING = (1) implement the kernel spec; (2) the
  grammar-consistency cleanup left by the 2026-07-05 relocation:
    (a) exist still holds the absolute-truth START role in <compose> while
        its cousins moved to <Queries>; decide if it relocates too.
    (b) default.grammar / shamatha.grammar did not follow complete.grammar's
        relocation (generalizing there broke the MM_grammar XOR convergence
        bar — reverted); they are inconsistent.
    (c) NP-R-NP (3-concept) relative-sentence form lost its grammar-level
        producer (4 integration tests skip at the finder).

