# Step 2b preflight — 2026-10-22

Base: published BasicModel `0b030e3ebdeca87f9ebf80739237437c3c75ea8f`,
WikiOracle `baa2e7fd130434cf619fb061d9035e853016b80c`. Step 2a's rename
and sole cull are complete. Its [receipt](../2026-10-22-item3-5-step2a-where/README.md)
retains the passing gates and Claude's approval of source `d4f316411`.

[Step-2 spec §18](../../plans/2026-10-21-step2-one-ltm.md#18-codexs-2b-preflight--2026-10-22-for-claude-and-alec)
raises the three representation questions required by §16: the leaf
postings' restore contract, adjacency witnesses, and legacy/live
`unspecified` mode. It also proposes the two implicit-evidence expected
values and flags the old coarsening test's conflict with the decided
protection policy. Each question has a recommendation for Alec.

This is a read-only code review and a documentation change. No production
or test file is changed; no 2b gate or experiment is run. No existing
failure has occurred or been relabeled, and no assertion port is applied.
The next frozen implementation, its scoped gates and the one full sweep
at step 2's end remain future work. The two R10 arrival notes stay in 2c.
The new 2a review observation about bounded recall's `SentenceEndState.where`
is carried to the 5.5 context-wholes spec, with no code change here.

The review input is **`d5fc19791558c8f8320788e0dd20e9f3430a31ac`**.
[Claude's read-only answer](claude-review.md), complete tool stream and run
metadata are retained. [Spec §19](../../plans/2026-10-21-step2-one-ltm.md#19-claudes-2b-preflight-review--2026-10-22-decisions-still-open)
records the corrections and the outstanding decisions. Claude recommends
keeping the three representations and confirms the two implicit-pair
expected values and the coarsening-policy conflict. He also identifies:

- two additional tag-aware reindex readers and restore-order constraints;
- an existing adjacency-remap gap (the earlier claim that it remapped was
  wrong), with its follow-up destination still for Alec;
- a second swap in `write_clause` and an unspecified relation-polarity
  boundary;
- missing participant-edge/reading-retention details and the meaning of
  one forgetting pass across both tables.

The preflight's code-fact correction is marked in §18 and explained in
§19. Recommendations are not decisions: no production or test change has
been made and no assertion has been ported. Implementation stops for
Alec's dispositions and the completed layout specification, under the
handoff's new-question rule. This is not an implementation receipt or an
item-3.0 transition. The review documents are retained on the local
`codex/item3-5-step2b-spec` branch; both published mains remain at the 2a
landing above.
