

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

