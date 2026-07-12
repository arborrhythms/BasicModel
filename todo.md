

## Operational integration of the QUERY TOOLS — status 2026-07-12

  DONE (1) the Thinking Kernel spec (doc/plans/thinking_kernel_spec.md) is
  IMPLEMENTED: bin/thinking.py (TruthInterval, STM Frame stack, the
  runtime-enforced lookup/part/think/query/answer loop, closure rules,
  grounded materialize/incorporate as the only LTM writes, reward + trace
  compilers), gated by <architecture><thinkingBudget> (absent/0 = off,
  byte-identical; ON at 16 in MM_query_reasoning.xml — the kernel result
  rides answer_query's payload under "kernel"). Execution notes:
  doc/plans/2026-07-12-thinking-kernel-execution.md. Tests:
  test/test_thinking_kernel.py (50).
    - Follow-ons LANDED (second pass, same day): §12.6/12.7 next-op
      learning (NextOpPolicy + thinkingLossWeight, behavior-cloned on
      grounded store traces; consulted only at explore-vs-stop points);
      §7.1/§14.2 testimony in the live loop (leaf policy queries
      addressees once; numeric testimony folds at source trust);
      provisioning fix (<serial>true — the observe-site push lives only
      in the per-word body; MM_query_reasoning landed 0 rows before).
    - Remaining seams (blocked on other tracks): the full §13
      next-sentence route pends the decode round-trip; relation-endpoint
      parse fidelity for provisioned truths (the depth-3 NP/VP/NP split)
      pends the word-grain track — the untrained parse collapses each
      truth text to a depth-1 absolute row.

  Grammar-consistency cleanup left by the 2026-07-05 relocation:
    (a) DECIDED 2026-07-12: exist STAYS in <compose> — it carries the
        absolute-truth START (a structural role; the 1-concept sentence
        form) and its introspective half is isTrue(X) in <Queries>, now
        operational via the kernel's lookup(). Recorded in
        complete.grammar's header.
    (b) RESOLVED 2026-07-12 (second pass): default.grammar /
        shamatha.grammar relocated to complete.grammar's OPERATOR form —
        relations (part/equal) stay assertive compose rules, the is*
        boolean-predicate queries move to <Queries>, the interrogative
        query="true" twins retire. Re-measured GREEN against the same
        MM_grammar XOR bar the 2026-07-05 FULL-removal broke
        (test_mm_grammar_learns_xor_signal converges as at baseline).
        Rider fix: _detect_query now keys on <Queries> declarations
        (query_ops) — complete.grammar has had NO query="true" rules
        since the relocation, so detection had gone dead.
    (c) FIXED 2026-07-12: the NP-R-NP (3-concept) producer was RESTORED by
        the 2026-07-05/06 "relations as operators" pass (part/whole/equal
        compose rules heading the relative_truth starts); the 4 integration
        tests skipped only because their finders still required the RETIRED
        is* method names — finders now key on the grammar-driven relative
        rule set; all 4 run green.
