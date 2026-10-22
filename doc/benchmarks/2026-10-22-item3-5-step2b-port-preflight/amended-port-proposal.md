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
   kind, metadata, shape, mask and content guards. This follows the
   decided removal; it is not a proposal to reclassify questions as facts.
2. **P2:** the exact LTM1 column tuple gains `ref_kinds` after `refs`,
   and the two parametrized column-count checks become seven. Zero is an
   invalid/empty tag; only occupied pre-2b pairs (`refs > 0`) restore as
   SYMBOL, while empty rows keep zero tags.
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
- In `test_operators_round4a0`, the compaction and legacy-migration
  fixtures use `e = lift(OCC child, SYM predicate)` and the parent's
  reading `Part(OCC parent, SYM e)`. `lift`'s declared subject is the same
  child in role 0. In `test_nested_retention`'s repeated-edges fixture, use
  `e = equal(OCC child, OCC child)` and `Part(OCC parent, SYM e)`.
  Both role-0 and role-2 edges come from `equal`'s two declared arguments,
  with no structural reference retained in the sidecar.
  Keep their role-0 child, repeated-role order, address, content-key,
  evidence, stream, timing, remapping and bounded-read assertions. This is
  a fixture-meaning port requiring Alec, not a mechanical column rename.
  Do not introduce an untyped occurrence edge or guess a legacy head.
  `records_scanned == 2` stays verbatim: `Layers.py:9060–9064` defines this
  field as the LTM0 address-scan count, and these fixtures still have two
  LTM0 rows. No node/depth/record budget is increased. This does not exempt
  LTM1 work from the separate existing query work meter.

## P5: full-owner checkpoint checks

Retain the complete learning-readiness case and all its earlier assertions.
Replace only the obsolete saved-posting corruption subcase with malformed
`ref_kinds` in the one exported `part_store` extras. Require LTM1's own
validation to reject both an out-of-range tag value on an occupied row
and a bad shape, each with `pytest.raises(ValueError, match='ref_kinds')`.
Validate before `super()._load_from_state_dict` so the shape case does not
fall through to PyTorch's generic error. A missing OCC target is not the
bad-value case: that belongs to the owner after both tables load.

In `test_part_read.py:25–28`, replace the stale serialized-posting
injection with an assertion that no `posting_*` key is serialized.
Keep load, configure, and both final `leaf_terms` assertions, including
exclusion of 999. The absence check replaces the removed injection's
obligation; the 999 assertion alone would be vacuous after removing it.
Rebuild occurs after both tables load.
`leaf_complete` can stay as an in-memory, non-persistent derived buffer;
its two existing assertions need no port.

## P6: close-time minting assertions

The round is already decided: **§11.1 ends close-time argument minting in
2b**, when OCC arguments become available. It is not a question whether
to defer it to 2c. The following existing checks need explicit dispositions:

| Test | Obsolete obligation | Proposed replacement obligation |
| --- | --- | --- |
| `test_sentence_identity::test_referenced_phrase_is_minted_as_a_part_argument` | A seat for the newly referenced phrase, and symbol-only traversal of its derivation. | Argument 1 is OCC to the existing constituent (`len(f.store) == 1` stays). Its reading reaches edge `e`, whose two SYM arguments remain `chases` and `mouse`; `_csw_row_of(e) is None`. Compare the occurrence point to the same child point at the existing tolerance. |
| `test_sentence_identity::test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically` | A full working bank forces argument-mint rejection. | With the same full bank and capacities, the occurrence arguments close. Inventory `rows` and `placement` stay unchanged; `next_id`, LTM0 and LTM1 deltas match the declared layout exactly. Atomic table exhaustion remains covered by `test_complete_derivation_capacity_failure_has_no_prefix` at `test_part_closing.py:47–55`. |
| `test_part_closing::test_new_relation_argument_is_minted_and_unseatable_argument_fails` | Zero order-1 seats force rejection, then a restored seat budget permits a mint. | One write under the same zero-seat setup succeeds and `csw_rows` stays unchanged. The bare argument has a new occurrence and no reading row because it has no root. Remove the restore-caps second write; do not resize the fixture. |
| `test_part_closing::test_nested_terminal_mint_keeps_its_word_and_admitted_occupation` | A new native object and definition for a previously objectless word, with symbol-keyed occupation counts. | Preserve the word and its grammatical position through the occurrence reading and typed derivation, with no new object symbol or fabricated definition. **Alec must decide the occupation disposition below; the original promise to preserve its symbol-keyed count/restore check is withdrawn.** |
| `test_part_closing::test_minted_surface_phrase_keeps_the_unminted_heads_word` | A seat and definition for the new phrase. | Preserve the same head word through the occurrence's reading and derivation; no argument seat or fabricated definition. |
| `test_relation_closing::test_three_slot_relation_closes_unindexed_numerical_operands` | Resolve every numerical argument with `_concept_point(symbol)`. | Dispatch by tag and compare the same operand points at the existing `assert_close` tolerance, preserving the two retained occurrence rows and the existing slot equality. |

**Occupation decision for Alec.** `PartClosing.py:265–271` keys admitted
occupations by symbol and deliberately skips a bare word. Once this
argument is OCC, there is no symbol key. Claude recommends storing no
separate occupation for this occurrence-only argument: its typed
derivation records the position. Explicitly retire this case's stored
`counts['verb_argin2'] == 1` and count-restore checks, and check the
retained word/position without indexing its reading as a definition.
The alternative is recording the occupation on the word, which changes
the bare-word rule and requires Alec's decision. Do not insert an untagged
OCC into the symbol map or introduce a new tagged occupation layout by
implication. No option is implemented before Alec answers.

The same representation rule covers positive-symbol-only assertions for
unindexed arguments in the two related `test_relation_closing` cases:
`test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`
and `test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows`.
Validate each tag and its live target; do not substitute an unconditional
positivity check for reference validity. Keep their content, evidence,
unverified status, no-fabricated-definition and fixture-size assertions.
Reading rows must be excluded from `relations('definition')`. The stale
native-seat comment at `test_relation_closing.py:65–66` changes; its
`concept_rows=64` does not.
The native OBJECT carrier certificate stays unchanged.

## P7: root/pending layout and defensive cycles

The limited root-read audit also finds uses in `test_part_closing`,
`test_part_read`, `test_definition_taxonomy`, `test_operators_final`,
`test_medium_term_memory` and `test_reference_close`. Route reads that
mean the derivation root through the reading row; do not reinterpret the
root participant as that root. Preserve the original content and matching
obligations. The anonymous-name checks at `test_reference_arrivals.py:76`
and `:184` currently require no raw role-0 reference; the new pending
reading is the objectless word. Their proposed port checks raw reading
`SYM words['x']`, with `role_refs[1:] == (None, None)`, while retaining
kind None, pending/open behavior, token membership, zero content and
bank/row immutability checks. The bound use at `:184` gets its own reading
`Part(OCC used, SYM x)`; MTM carries its token link. It does not create an
OCC-to-OCC edge. Snapshot the pending reading before the arrival and
check it is unchanged afterwards.

**Reader precedence preserves the existing antecedent.** For an
occurrence in MTM, `_derivation_symbol` reads the token's `kind` first;
an unrefined token returns None, not the pending word. Only unpooled
occurrences use the reading's derivation root. Thus the immutable word
reading remains unchanged while an arrival makes the token read `three`,
as the existing assertions require. This is an implementation consequence,
not permission to port a wrong antecedent.

`test_part_read::test_part_membership_terminates_on_cycles_and_retains_reachable_leaves`
currently creates a cycle with two public `add` calls. Port its setup to
explicitly inject a corrupt tagged graph into `refs`, `ref_kinds` and
`count`, with its reachable-leaf assertion unchanged. Do not use the writer
or checkpoint loader to admit that corruption. Normal writes must reject
a self/forward reference before mutation, including an explicit `symbol=`
handle already referenced by an existing row. This is a defensive-reader
fixture, not permission for production cycles. The existing sidecar-
corruption helpers in `test_nested_retention.py:147–196` need equivalent
corruption-only setup ports wherever those structural references move;
their diagnostics and bounded-read assertions stay unchanged, otherwise
the round stops for a further disposition.

This is a scoped static inventory, not a claim to have executed or exhausted
the test suite. Any other semantic change, any unresolved layout, or any
existing-test/production failure still stops for Claude and Alec. The
same-round setup exception does not apply to these existing fixtures.
