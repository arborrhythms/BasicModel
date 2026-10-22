# Review the remaining 2b test ports

User-authorized Claude review. Read only: do not run tests or experiments,
edit files, implement or publish. Recommend dispositions for Alec; do not
decide on his behalf.

The repository handoff is complete: main is published through
`c39ef9f3ab9dc334fbee3b6e83d520a256423005`, including both preflight commits
and the requested unchanged commit of your seven documents. No WikiOracle
bump was made for documentation alone. Production and tests still match
published 2a, `0b030e3eb`. No 2b implementation or gate has run.

Read the step-2 spec `doc/plans/2026-10-21-step2-one-ltm.md`, especially
the decisions in §§20–23 and the new audit in §24. Read this directory's
README, proposed-fixed-ports.patch, ltm0-refs-audit.json and review-input.json,
and the cited tests and implementation as needed. Do not reopen the owner,
tags, worked layout, evidence, stored-mode removal, postings/adjacency
removal or staged-forgetting decisions.

The latest user handoff authorizes:

- standalone-memory posting restore ports;
- retirement of tests solely checking adjacency/nudging;
- the two evidence tuple ports;
- the two-pass coarsening port and §19's snapshot port;
- the tag-aware role-0 read in test_live_parent_pins_its_child.

It also says: "Any other assertion change, or a failure in an existing test
or in production, stops for Claude and Alec." Spec §22 explicitly says
"any existing assertion that depends on a stored mode comes back to Alec."
We found that assertion before implementation. No failed run is being
reported; no new-test setup exception is being invoked.

Please check the concrete proposed ports and separate genuinely additional
authorizations from ports already covered by the handoff. Avoid asking Alec
to authorize an already-decided change again.

1. P1: test_nested_retention.py:56 checks a retained question's stored
   interrogative mode. Does replacing only that assertion with its zero
   evidence pair correctly preserve the no-fact-authority obligation of
   §22, with all other checks unchanged? Is there another stored LTM0 mode
   assertion missed by the audit? Live parse and WhatInteractionMemory mode
   checks stay unchanged.
2. P2: the exact LTM1.columns tuple and two parametrized column-count cases
   require the seven-column tagged layout. Verify the proposed tuple/count
   edits, including the decision to place ref_kinds immediately after refs.
3. P3: forced_prefix_certificate has an assertion on symbol_centroid inside
   a mixed descent/reconstruction/snapshot certificate. Verify removing
   only that conjunct rather than retiring the test or retaining an inert
   switch. Is this already authorized by removal of the mechanism, or does
   the user's explicit assertion rule require the narrow port?
4. P4: inspect the direct LTM0 refs inventory. Is the proposed port rule
   precise enough to preserve content, antecedents, evidence, addresses,
   bank isolation, immutability, restoration and rejection while changing
   the physical representation? Flag any listed case that cannot be a
   mechanical representation port under the decided layout. The already
   authorized coarsening/snapshot cases are not being re-requested.
5. P5: the complete-model learning-readiness test corrupts serialized
   posting_codes and requires loud rejection. The larger test must remain.
   Assess replacing only that corruption subcase with malformed ref_kinds
   in the decided LTM1 checkpoint, preserving the earlier assertions and
   loud rejection. Is this distinct from the authorized standalone restore
   ports, or already within the authorized checkpoint removal?

Correct wrong code facts with locations. State the minimum additional
decision Alec needs and any flaw in the proposed ports. If you notice
another unavoidable existing assertion conflict in these same changed
mechanisms, identify it without broadening this into a sweep or running
tests. A proposed representation adapter must not preserve fake empty
columns or untyped role edges solely for an assertion.

This is a pre-implementation assertion audit, not the frozen-source review
or a publication request. The two R10 arrival notes stay in 2c, and step 2's
single full sweep stays at its end. Item 3.0 has not been reached.
