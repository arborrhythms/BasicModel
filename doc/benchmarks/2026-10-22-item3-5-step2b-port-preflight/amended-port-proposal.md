# Amended 2b port proposal — 2026-10-22

No production or test edit is applied. This amends P1–P5 after the actual
[Claude review](claude-review.md), and adds the limited closing/layout
audit requested by that review. It seeks test dispositions, not a change
to the decided 2b design.

## Fixed ports

1. **P1:** replace the stored-mode assertion at
   `test_nested_retention.py:56` with a check of the inner row's retained
   `kind == "question"` **and** `evidence == (0., 0.)`. The original
   evidence-only proposal is withdrawn: origin withdrawal zeros evidence,
   so it did not distinguish the retained question. All other assertions
   stay. Preserve the incoming live-mode admission check. Stored reads
   use `ConceptualMeaning`'s existing default, `assertive`, as LTM1 reads
   already do; add no replacement mode or inferred mode. Remove the
   redundant stored-mode conjunct in `reasoning.py:170`, preserving the
   kind, metadata, shape, evidence and content guards. This follows the
   decided removal; it is not a proposal to reclassify questions as facts.
2. **P2:** the exact LTM1 column tuple gains `ref_kinds` after `refs`,
   and the two parametrized column-count checks become seven. Zero is an
   invalid/empty tag; retained pre-2b symbolic pairs restore as SYMBOL.
3. **P3:** remove only the `symbol_centroid` conjunct. Keep the per-forward
   form snapshot and its vocabulary-revision cache, so the existing
   `[2, 2]` snapshot assertion remains unchanged. Remove the adjacency
   revision and switch from the key and remove the unused config/schema
   entry with the mechanism. Preserve `centroid_audit()`'s non-mutation
   check using the unmodified forms; no nudge or inert enabled switch.
4. **Additional absence ports:** the idea branch in
   `test_truth_text.py:47` checks absence of LTM0 `refs`; its relation
   branch keeps the exact two-reference assertion. In
   `test_selected_nested_meaning.py:81`, remove only `where` from the
   compared columns; retain the full `.when` and other value comparisons.
   The latter is a static conflict with published 2a, not a newly observed
   failed run. Do not rerun 2a or change `.when` to diagnose it.

## P4: representation checks and fixtures

The proposed rule retains every semantic obligation and lists each final
port in the implementation receipt. It covers direct LTM0 `refs` accesses,
row-export `['refs']` accesses and readers that use `role_refs[0]` as the
old derivation handle.

- Snapshot **pre-existing** LTM1 rows by handle, including `refs` and
  `ref_kinds`. Specify the newly written reading/derivation rows separately.
  Comparing the formerly all-`-1` LTM0 column was vacuous; the new comparison
  strengthens coverage and any failure is a production finding, not
  permission to weaken the assertion.
- Cross-path comparisons (`test_selected_nested_meaning`) normalize tagged
  graphs through the occurrence-address mapping and relation structure;
  they do not compare allocator-specific handle integers as identities.
- Replace derivation-root reads with the reading-row accessor. Preserve
  the existing participant/antecedent, content, evidence, bank, address,
  open-slot and restore outcomes. Any literal row/edge count changed by
  the new reading/mention rows must state the exact decided layout in the
  receipt; do not turn it into a lower bound.
- The native observer in `identity_measurement.py` will start observing a
  retained reading rather than the unused LTM0 column. Declare that change
  of measurement. Do not reinterpret earlier captures or claim learning.
- `test_concept_forgetting` plants valid reading rows in place of the old
  column. Reclaim counts must stay unchanged; a changed count stops.
- In `test_operators_round4a0`, rewrite the compaction and legacy-migration
  fixtures to carry a declared typed derivation whose participant is the
  same child. In `test_nested_retention`'s repeated-edges fixture, use a
  declared typed derivation with the same child in the same two roles.
  Keep their role-0 child, repeated-role order, address, content-key,
  evidence, stream, timing, remapping and bounded-read assertions. This is
  a fixture-meaning port requiring Alec, not a mechanical column rename.
  Do not introduce an untyped occurrence edge or guess a legacy head.

## P5: full-owner checkpoint checks

Retain the complete learning-readiness case and all its earlier assertions.
Replace only the obsolete saved-posting corruption subcase with malformed
`ref_kinds` in the one exported `part_store` extras. Require LTM1's own
validation to reject both a bad tag value and a bad shape with an error
naming `ref_kinds`, rather than accepting a generic PyTorch shape error.

In `test_part_read.py:25–28`, remove only the stale serialized-posting
injection. Keep load, configure, and both final `leaf_terms` assertions,
including exclusion of 999. Rebuild occurs after both tables load.
`leaf_complete` can stay as an in-memory, non-persistent derived buffer;
its two existing assertions need no port.

## P6: close-time minting assertions

The round is already decided: **§11.1 ends close-time argument minting in
2b**, when OCC arguments become available. It is not a question whether
to defer it to 2c. The following existing checks need explicit dispositions:

| Test | Obsolete obligation | Proposed replacement obligation |
| --- | --- | --- |
| `test_sentence_identity::test_referenced_phrase_is_minted_as_a_part_argument` | A seat for the newly referenced phrase, and symbol-only traversal of its derivation. | The same phrase is referenced as OCC, its point and typed derivation remain readable, and no argument seat is allocated. |
| `test_sentence_identity::test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically` | A full working bank forces argument-mint rejection. | With the same full bank and capacities, the occurrence arguments close without adding seats. Retain separate atomic failure coverage for actual LTM/table exhaustion. |
| `test_part_closing::test_new_relation_argument_is_minted_and_unseatable_argument_fails` | Zero order-1 seats force rejection, then a restored seat budget permits a mint. | With the same zero-seat setup, the argument closes as an occurrence with its content and no seat. Do not resize the fixture. |
| `test_part_closing::test_nested_terminal_mint_keeps_its_word_and_admitted_occupation` | A new native object and definition for a previously objectless word. | Retain the word and admitted occupation through the decided word/occurrence reading; no new object symbol or fabricated definition. Preserve the occupation restore check. |
| `test_part_closing::test_minted_surface_phrase_keeps_the_unminted_heads_word` | A seat and definition for the new phrase. | Preserve the same head word through the occurrence's reading and derivation; no argument seat or fabricated definition. |
| `test_relation_closing::test_three_slot_relation_closes_unindexed_numerical_operands` | Resolve every numerical argument with `_concept_point(symbol)`. | Dispatch by tag and compare the same operand points exactly, preserving the two retained occurrence rows and the existing slot equality. |

The same representation rule covers positive-symbol-only assertions for
unindexed arguments in the two related `test_relation_closing` cases:
`test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`
and `test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows`.
Validate each tag and its live target; do not substitute an unconditional
positivity check for reference validity. Keep their content, evidence,
unverified status, no-fabricated-definition and fixture-size assertions.
The native OBJECT carrier certificate stays unchanged.

## P7: root/pending layout and defensive cycles

The limited root-read audit also finds uses in `test_part_closing`,
`test_part_read`, `test_definition_taxonomy`, `test_operators_final`,
`test_medium_term_memory` and `test_reference_close`. Route reads that
mean the derivation root through the reading row; do not reinterpret the
root participant as that root. Preserve the original content and matching
obligations. The anonymous-name checks at `test_reference_arrivals.py:76`
and `:184` currently require no raw role-0 reference; the new pending
reading is the objectless word. Their proposed port checks that reading
while retaining kind None, pending/open behavior, token membership, zero
content and bank/row immutability checks.

`test_part_read::test_part_membership_terminates_on_cycles_and_retains_reachable_leaves`
currently creates a cycle with two public `add` calls. Port its setup to
explicitly inject a corrupt tagged graph, with its reachable-leaf assertion
unchanged. Normal writes must still reject a self/forward reference before
mutation. This is a defensive-reader fixture, not permission for production
cycles.

This is a scoped static inventory, not a claim to have executed or exhausted
the test suite. Any other semantic change, any unresolved layout, or any
existing-test/production failure still stops for Claude and Alec. The
same-round setup exception does not apply to these existing fixtures.
