# P4, P6 and P7 against §31 — exact proposed dispositions, 2026-10-22

**Review only. Nothing here is applied or authorized by this document.**
Source is `e0d474090`; production/tests equal published 2a. R1–R8 refer to
[spec §32](../../../plans/2026-10-21-step2-one-ltm.md#32-codexs-review-of-31-against-the-code--2026-10-22-stopped-for-claude-and-alec).
P1, P2, P3 and P5 remain authorized exactly as amended. In particular,
P5's exact leaf sets and its 999 exclusion remain; R2 must reconcile the
new word source with them without changing their assertions.

The numbered entries below quote original AST source, not reconstructed
paraphrases. Each entry gives the exact proposed replacement or an explicit
retirement. A fixture-only entry retains the assertion verbatim while
proposing a changed source for its inputs. A held entry proposes **no**
assertion edit; it identifies a contract choice that must be answered
before a further port can be written. These are not blanket permissions
to adapt any test until green. Unlisted assertions stay, and an additional
affected assertion needs its own disposition.

## Proposed read notation and layout assumptions

The following are review notation, **not implemented runtime APIs**:

- `reading(store, occurrence)` returns the sole raw mention reading as
  `('sym', target)`, or None. It never reads MTM, reconstructs a sentence,
  returns an asserted Part, or creates a definition. A sentence has no
  reading row. If Alec chooses the sidecar alternative, this same
  observation uses the authenticated mention-reading binding instead;
  all reading-edge count proposals then need a revised exact ledger.
- `membership_words(store, occurrence)` is the set of literal SYM word
  targets in the sentence's **owned bag**. R3 must decide how this use is
  distinguished from an independent assertion. It is not an inverse read.
- `tagged_arguments(parts, edge)` returns the two typed references in
  argument order. `valid_tagged_arguments(store, edge)` means that each
  tag is valid and each value resolves in that tag's live namespace;
  OCC keys are signed addresses, not necessarily positive integers.
- `reference_point` dispatches to the existing symbol point or whole
  occurrence point according to the tag; `occurrence_point` requires OCC.
  Both preserve the original point comparison's tolerance.
- `tagged_snapshot(parts, handles=...)` returns exact detached copies of
  `refs` and `ref_kinds` at those stable handles. Equality means equal
  handle sets, dtype, shape and tensor values (rtol=atol=0); it does not
  compare only a hash or only rows that happen to survive. Omission of
  `handles` means all occupied rows. `before_parts` is captured before the
  operation. Existing slots, bank, time, poles and metadata checks stay.
- `sentence_for(node)`, `mention`, `argument_occurrence` and
  `child_sentence` come from the close's written address map, not a
  vector lookup or the integer of a newly allocated relation.
- `normalized_layout` compares exact LTM0 addresses/content and exact
  tagged LTM1 pairs using the paired fixture's source-address mapping.
  `normalized_part_columns` additionally compares every originally tested
  Part column plus ref_kinds, at normalized handles. It does not erase
  argument order, evidence or timing. Repeated references remain repeated.

Counts in this proposal assume R1's source-bearing S occurrences, distinct
word bags for each nested S, R3's mention-only READING rows, and the end of
close-time minting. **They describe final 2b.** Claude must specify the
intermediate layout commit if it intentionally retains a mint path;
these final counts must not be used to bless that different source.
There is no allowance to turn an exact count into a lower bound.

For `test_selected_nested_meaning`'s owner fixture, `AnswerProgram` carries
no `word_ids`: do not invent words from its numerical `word_rows` or
concept_ids. `expected_nested_layout` is the existing initial Part set,
plus `Part(SYM entry.concept_ids[0], SYM entry.concept_ids[1])`,
`Part(SYM entry.concept_ids[2], OCC child_sentence)` and two opaque S rows
at their written source addresses. There are no bag or mention-reading
rows in this fixture. Keep the child assertion separately named so its
old exact content, ignorance and no-gradient checks remain meaningful.
All three writer paths must produce that same layout. If a source path
does not carry a fused S point, that is R1's code question, not permission
to manufacture a point in the test.

## Fixture rewrites, independently of assertion text

1. **P4 snapshots:** `test_identity_rows` original= at 105;
   `test_medium_term_memory.snapshot` at 15 and tuples at 59/68/96;
   `test_reference_close.current_identities` at 79/83 and snapshots at
   194/213; `test_reference_arrivals` at 93;
   `test_reference_mentions` at 74; `test_reference_occurrence_views` at
   16/26. Replace the removed LTM0 refs tensor with a pre-existing-handle
   tagged snapshot. Keep all other tensors and existing full Part-state
   immutability checks. Restore fixtures load both tables under their
   model owner before index rebuild; standalone semantic-only tests need
   no fabricated leaf reader.
2. **P4 symbol protection:** `test_concept_forgetting` at 53/85 plants
   `store.refs` only to pin a native symbol. Replace each plant with a
   valid mention reading to `weak`/`referenced` on that same retained
   occurrence. No extra LTM0 row or seat; all reclaim counts, identity
   and survivor assertions stay. This assumes R3; it is not a word
   membership on an unrelated sentence.
3. **P4 compaction/legacy:** `test_operators_round4a0`'s two fixtures
   carry the child through `bindings={'referent': reference}` instead
   of the superseded proposed lift/read-root graph. The legacy input
   still explicitly contains its old raw refs wire column and old
   namespace/address; keep that obsolete **input** to test migration,
   not an attempted access to the new store attribute. Rewrite its
   authenticated context and fingerprint consistently. The two exact
   binding reads below replace the old role-0 reads. This depends on R5's
   semantic-binding boundary and R8's migration decision.
4. **P4 cross-path helper:** remove its LTM0 refs-based address traversal;
   normalize the actual tagged assertions, word bags and mention readings
   instead. Drop only `where` in the LTM0 comparison loop; `.when` stays.
5. **P4 measurement:** `identity_measurement.native_observer:197`
   observes `reading(store, occurrence)` rather than the unused refs
   column. It reports None for an opaque sentence. Declare that measurement
   change; do not reinterpret old captures as measurements of the new view.
6. **P6 nested terminal:** right is `verb` argument 2. Stamp that actual
   participant order 1 in this unit fixture (the permitted grammar-order
   forcing), retain `word_id=word`, and obtain its mention from written
   addresses. No forced resolver choice or new fixture capacity. Replace
   the old stored symbol occupation check with the **live kept** position:
   `assert right in participant_terms(((root, False),))[0].values()` and
   `assert root.children[1] is right`. Capture the definition index before
   close and require `assert definition_pairs_after == definition_pairs_before`.
   This is the precise proposed replacement for §27's now-lapsed retained
   occupation read; it is not an occupation stored on the occurrence.
7. **P6 surface phrase:** the phrase is the **predicate head** (argument 1
   of verb) in this fixture. The old P6 proposal to retain it as an NP
   was wrong. Leave its part of speech alone. Its word survives only in
   the bag; require `assert written_participant_occurrences == ()` for
   this all-native/order-0 fixture. Snapshot seats and definition pairs;
   neither gains a newly minted object. The literal word-membership edge
   is the only new row for this manual one-word journal.
8. **P6 full bank/zero seats:** one successful close with the original
   exhausted setup. Remove the recovery write that restored seats. New
   unseated relation handles may extend allocator `placement` and
   `relate_idx`; compare the existing entries exactly and the additions
   against the explicitly enumerated layout. Do not assert equality of
   the whole placement dictionary as the earlier P6 proposal did. Atomic
   LTM table exhaustion still has its existing separate test; preserve it.
9. **P7 definition taxonomy:**
   `test_selected_anaphor_form_binds_to_its_own_end_particular` at line 187
   obtains `subject` from the written
   mention reading for the selected word, not the enclosing operation's
   first argument. Its four word/identity assertions at 188–191 stay.
10. **P7 pending names:** raw word reading is x, working MTM kind is None
    until refined, then three. Snapshot the original mention reading and
    require exact equality after arrival. Preserve token count/membership,
    all `matches`, no-open-slot, content, evidence and bank assertions.
    The pending **parent** role assertions are held for R5, separately.
11. **P7 nested metadata:** `_meaning(child=...)` in the origin-withdrawal,
    restore and invalid-link fixtures carries an explicit semantic
    binding, not a parse child edge. `_corrupt_child` corrupts that binding
    and its authenticated fingerprint; retain dangling-link atomicity.
    The opaque structure view contains only its root. R5 must approve this
    changed fixture meaning. Working thought references and live root
    ownership are not automatically ported or retired.
12. **P7 exact component units:** the four lift/head/operand assertions
    move verbatim to a controlled bounded reconstruction-adapter unit
    fixture. The marker before/after checks move to a live kept-position
    fixture, retaining the same definition change. For the named-composite
    matching test, explicitly provide the native composite concept and its
    existing name before close; its mention reading names that concept,
    and `matches(..., 'bound-name')` stays. For the composed-marker match,
    exercise the transient inverse adapter with a controlled inverse and
    a grounded native content reference; `matches(..., 'content')` stays.
    This changes the source of component evidence and needs approval; it
    must not force the inverse in any live/learning certificate.
13. **P7 defensive Part cycle:** the original `leaves(parts, 10)` test's
    reachable set `{2, 3}` stays; plant its two corrupt edges and SYMBOL
    tags directly, not through the public writer or load. It now covers
    the defensive traversal of explicit Part edges, not sentence word
    membership. No production self/forward reference is admitted.

## New stage certificates replacing the old coarsening/cascade test

This is a proposed retirement of **one test function, both loop arms**,
not a claim that the old behavior passes. Its exact removed assertions
are listed below. Keep the original capacity and content/evidence values;
use the common owner and tagged references instead of hand-written refs.

- Bag stage: an unprotected sentence with a word bag survives the first
  selected coarsening pass as an opaque gist. Snapshot its raw slots,
  evidence and address before the pass. Require
  `assert store.location(sentence) is not None`,
  `assert membership_words(store, sentence) == set()`,
  `torch.testing.assert_close(store.raw_meaning_of(store.location(sentence)).roles, before_roles, rtol=0, atol=0)`,
  `assert store.row(store.location(sentence))['evidence'] == before_evidence`,
  and `assert store.occurrence_of(store.location(sentence)) == sentence`.
  An independently low-valued mention is not pinned by this sentence.
- Release stage: an explicit low-valued assertion references a child.
  At the first boundary require `assert not store.parts.contains(edge)`
  and `assert store.location(child) is not None`. Only after the second
  boundary require `assert store.location(child) is None`. The child's
  slots/evidence are identical after pass one, and no surviving sentence
  role is rewritten. The separate live-parent test uses this explicit
  assertion as its live root and keeps its surviving-child check.

R7 must state the stage selection order under the existing low-water
policy before this fixture is final. Do not change weights/capacities or
increase passes until the result passes; if the decisions imply a
different schedule, return an exact revised port for review.

## Proposed retirement of the stored sentence-cycle test

`test_nested_retention::test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`
asserts that a persisted sentence parse cycle restores successfully.
Retire that test (all seven checks below), not its safety obligations.
Replace it with two scoped cases: corrupt a tagged LTM1 graph directly and
require a bounded explicit incomplete/cycle diagnostic; loading that same
invalid graph must raise `ValueError` before either table's previous state
changes. A self/forward reference through the public writer also fails
before mutation, as already decided. No new graph budget or loader
permission to accept cycles. The exact component-view result fields for
this new test depend on R5 and must be specified before implementation.

No other whole test is proposed for retirement here. The two nested-
terminal occupation assertions are separately retired as already decided;
their replacement live position/word checks are proposed above.

## Assertions held for decisions

The raw pending-parent view, nested thought writes, constituent capacity
checks and the four occurrence-view cases are explicitly held below.
Their semantic outcomes are not weakened. `test_part_read`'s existing
direct `_forward_references`/equality-witness tests and related
`test_item6_2_thinking` checks also must not be silently converted to the
new inverse: they exercise `read_reference_bindings`, not merely a stored
operation root. Their provenance/MTM interpretation needs R5's answer.
No failure of any of these has been executed or diagnosed in this review.

The unit inverse tests in `test_stored_idea_generation`, working
`test_thought_binding` components, the carrier certificate, and all
antecedent/content/evidence assertions outside the explicitly listed
ports stay unchanged. The receipt of any later implementation must list
what was actually approved and changed by source, not claim that this
proposal itself authorizes a port.

## Exact assertion ledger

Generated from the saved AST spans and the explicit proposal mapping.
The JSON beside this document is the machine-readable source of this
ledger. Names and line numbers refer to the reviewed source, not a future
implementation.

### P4

**1. `test/test_clause_acceptance.py:113` — `test_01_compound_vp_has_one_idea_and_readable_cat_chase_mouse`**

Original:

```python
assert f.store.refs[row].tolist() == [-1, -1, -1]
```

Proposed:

```python
assert not hasattr(f.store, 'refs')
```

**2. `test/test_clause_storage.py:20` — `test_end_row_has_shared_references_and_field_coordinates`**

Original:

```python
assert store.refs.shape == (8, 3)
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

**3. `test/test_clause_storage.py:95` — `test_compound_idea_stores_only_fused_point_without_factored_target`**

Original:

```python
assert store.refs[row].tolist() == [-1, -1, -1]
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

**4. `test/test_clause_storage.py:114` — `test_recursive_absolute_clauses_each_write_one_idea`**

Original:

```python
assert bool((store.refs[:3, 2] == -1).all())
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

**5. `test/test_part_closing.py:42` — `test_absolute_close_keeps_only_the_concept_and_derivation_pairs`**

Original:

```python
assert f.store.refs[row].eq(-1).all()
```

Proposed:

```python
assert not hasattr(f.store, 'refs')
```

**6. `test/test_sentence_end_state.py:21` — `test_reading_an_idea_returns_its_one_occupied_slot`**

Original:

```python
assert store.refs[row].tolist() == [-1, -1, -1]
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

**7. `test/test_sentence_retention.py:62` — `test_narrow_idea_retains_no_implicit_operand_reference`**

Original:

```python
assert store.refs[second].tolist() == [-1, -1, -1]
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

**8. `test/test_clause_storage.py:21` — `test_end_row_has_shared_references_and_field_coordinates`**

Original:

```python
assert store.refs.dtype == torch.int64
```

Proposed:

```python
assert 'refs' not in store.state_dict()
```

**9. `test/test_clause_storage.py:22` — `test_end_row_has_shared_references_and_field_coordinates`**

Original:

```python
assert bool((store.refs == -1).all())
```

Proposed:

```python
assert 'refs' not in store.row(opaque_row)
```

Add one opaque row in this schema fixture solely for the export check; retain the original capacity and .when-shape assertion.

**10. `test/test_clause_storage.py:201` — `test_clause_end_state_and_evidence_checkpoint_roundtrip`**

Original:

```python
torch.testing.assert_close(restored.refs, store.refs)
```

Proposed:

```python
assert tagged_snapshot(restored.parts) == tagged_snapshot(store.parts)
```

Restore both tables through the owner; exact refs and ref_kinds by handle.

**11. `test/test_sentence_end_state.py:49` — `test_no_clause_derivation_or_factored_target_is_checkpointed`**

Original:

```python
torch.testing.assert_close(restored.refs, store.refs, rtol=0, atol=0)
```

Proposed:

```python
assert tagged_snapshot(restored.parts) == tagged_snapshot(store.parts)
```

Restore both tables through the owner; exact refs and ref_kinds by handle.

**12. `test/test_reference_close.py:421` — `test_kept_journal_supplies_np_content_predicate_and_positional_provenance`**

Original:

```python
torch.testing.assert_close(restored.refs, f.store.refs, rtol=0, atol=0)
```

Proposed:

```python
assert tagged_snapshot(restored.parts) == tagged_snapshot(f.store.parts)
```

Restore both tables through the owner; exact refs and ref_kinds by handle.

**13. `test/test_identity_rows.py:109` — `test_forgetting_and_reinterpretation_never_refold_a_native_identity`**

Original:

```python
torch.testing.assert_close(store.refs,original[1],rtol=0,atol=0)
```

Proposed:

```python
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Capture pre-existing handles before the operation; new detail rows are checked separately.

**14. `test/test_reference_arrivals.py:113` — `test_arrival_bind_refines_and_reads_three_through_the_token`**

Original:

```python
torch.testing.assert_close(store.refs[:pending+1], refs[:pending+1], rtol=0, atol=0)
```

Proposed:

```python
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Capture pre-existing handles before the operation; new detail rows are checked separately.

**15. `test/test_reference_close.py:199` — `test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised`**

Original:

```python
torch.testing.assert_close(store.refs[:store.index_of_row(missing)], before[:store.index_of_row(missing)])
```

Proposed:

```python
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Capture pre-existing handles before the operation; new detail rows are checked separately.

**16. `test/test_reference_close.py:217` — `test_empty_close_has_no_policy_gradient_or_store_side_effect`**

Original:

```python
torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)
```

Proposed:

```python
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Capture pre-existing handles before the operation; new detail rows are checked separately.

**17. `test/test_reference_mentions.py:94` — `test_later_fill_trains_the_same_occurrence_menu_without_a_live_graph`**

Original:

```python
torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)
```

Proposed:

```python
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Capture pre-existing handles before the operation; new detail rows are checked separately.

**18. `test/test_reference_close.py:198` — `test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised`**

Original:

```python
assert int(store.refs[store.index_of_row(missing), 0]) == -1
```

Proposed:

```python
assert reading(store, store.occurrence_of(store.index_of_row(missing))) is None
```

This is a gap, not a named mention; retain missing_reason and antecedent assertions.

**19. `test/test_medium_term_memory.py:69` — `test_bound_use_reads_as_its_introduction_without_rewriting_rows`**

Original:

```python
torch.testing.assert_close(current[:use], old[:use], rtol=0, atol=0)
```

Proposed:

```python
torch.testing.assert_close(current[:use], old[:use], rtol=0, atol=0)
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Original tensor comparison stays for slots/time/bank; remove only refs from the iterated tuple and compare the saved tagged rows separately.

**20. `test/test_medium_term_memory.py:97` — `test_lru_eviction_keeps_rows_and_a_later_mention_opens_a_new_token`**

Original:

```python
torch.testing.assert_close(current[:retained_rows], before[:retained_rows], rtol=0, atol=0)
```

Proposed:

```python
torch.testing.assert_close(current[:retained_rows], before[:retained_rows], rtol=0, atol=0)
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Original tensor comparison stays for slots/time/bank; remove only refs from the iterated tuple and compare the saved tagged rows separately.

**21. `test/test_reference_close.py:84` — `current_identities`**

Original:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Proposed:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Original tensor comparison stays for slots/time/bank; remove only refs from the iterated tuple and compare the saved tagged rows separately.

**22. `test/test_reference_occurrence_views.py:27` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Proposed:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
assert tagged_snapshot(f.store.parts, handles=before_parts.keys()) == before_parts
```

Original tensor comparison stays for slots/time/bank; remove only refs from the iterated tuple and compare the saved tagged rows separately.

**23. `test/test_selected_nested_meaning.py:80` — `test_10_packed_pending_and_eager_ends_write_identical_rows`**

Original:

```python
assert addresses(store) == addresses(stores[0])
```

Proposed:

```python
assert normalized_layout(store) == normalized_layout(stores[0])
```

Normalize actual OCC addresses through the paired source-address map and SYM relation handles through their tagged arguments; compare exact bags/readings/assertions, no parse graph.

**24. `test/test_selected_nested_meaning.py:84` — `test_10_packed_pending_and_eager_ends_write_identical_rows`**

Original:

```python
torch.testing.assert_close(getattr(store.parts, name), getattr(stores[0].parts, name), rtol=0, atol=0)
```

Proposed:

```python
assert normalized_part_columns(store) == normalized_part_columns(stores[0])
```

Include refs, ref_kinds, truth_plus, truth_minus, surprise and when at normalized handles; retain exact tensor equality/tolerances. The preceding LTM0 loop loses only where, already authorized.

**25. `test/test_truth_text.py:47` — `test_supplied_trust_preserves_both_possible_grammatical_fields`**

Original:

```python
assert store.row(row)['refs'].numel() == (3 if field.relation is None else 2)
```

Proposed:

```python
assert ('refs' not in store.row(row) if field.relation is None else store.row(row)['refs'].numel() == 2)
```

**26. `test/test_part_closing.py:13` — `test_relative_close_is_one_part_edge_and_rewitnesses_its_two_lanes`**

Original:

```python
assert len(f.store) == 0
```

Proposed:

```python
assert len(f.store) == 1
```

R1: opaque part(a,b) sentence; same returned assertion.

**27. `test/test_part_closing.py:14` — `test_relative_close_is_one_part_edge_and_rewitnesses_its_two_lanes`**

Original:

```python
assert len(f.store.parts) == before + 1
```

Proposed:

```python
assert len(f.store.parts) == before + 3
```

Two word memberships + one assertion.

**28. `test/test_part_closing.py:41` — `test_absolute_close_keeps_only_the_concept_and_derivation_pairs`**

Original:

```python
assert root[0] == 'sym' and f.store.parts.contains(root[1])
```

Proposed:

```python
assert reading(f.store, f.store.occurrence_of(row)) is None
assert membership_words(f.store, f.store.occurrence_of(row)) == {f.words[n][0] for n in ('a', 'v', 'b')}
```

Opaque sentence, exact lexical bag.

**29. `test/test_part_closing.py:43` — `test_absolute_close_keeps_only_the_concept_and_derivation_pairs`**

Original:

```python
assert len(f.store.parts) == before + 2
```

Proposed:

```python
assert len(f.store.parts) == before + 3
```

Three word memberships; no lift/verb pairs.

**30. `test/test_clause_acceptance.py:114` — `test_01_compound_vp_has_one_idea_and_readable_cat_chase_mouse`**

Original:

```python
assert len(f.store.parts) == 10
```

Proposed:

```python
assert len(f.store.parts) == 10
```

Assertion text unchanged; exact layout becomes four existing definitions + four word memberships + two mention readings. Replace old explanation, assert exact tagged set.

**31. `test/test_clause_acceptance.py:129` — `test_04_absolute_clauses_close_bottom_up_with_their_part_edges`**

Original:

```python
assert len(f.store.parts) == 14
```

Proposed:

```python
assert len(f.store.parts) == 24
```

Seven definitions + 4 inner + 6 middle + 7 outer memberships; no operations. Three sentence rows stay.

**32. `test/test_clause_acceptance.py:142` — `test_05_embedded_part_is_unasserted_and_propagates_operator_kind`**

Original:

```python
assert f.store.parts.refs[verb, 1] == child
```

Proposed:

```python
assert membership_words(f.store, sentence_for(clause.children[0])) == {f.words[n][0] for n in ('cats', 'animals')}
```

Delete setup read of outer refs[1]. child assertion/evidence still read independently; sentence_for uses the close result.

**33. `test/test_clause_acceptance.py:143` — `test_05_embedded_part_is_unasserted_and_propagates_operator_kind`**

Original:

```python
assert not len(f.store)
```

Proposed:

```python
assert len(f.store) == 2
```

Opaque outer and inner S; outer no longer becomes a structural operator pair.

**34. `test/test_sentence_identity.py:25` — `test_closing_idea_is_narrow_and_the_operation_has_a_native_part`**

Original:

```python
assert inventory(f.cs)['next_id'] == before['next_id'] + 1
```

Proposed:

```python
assert inventory(f.cs)['next_id'] == before['next_id'] + 2
```

Two word memberships.

**35. `test/test_sentence_identity.py:26` — `test_closing_idea_is_narrow_and_the_operation_has_a_native_part`**

Original:

```python
assert f.store.parts.find(f.noun('cat'), f.noun('runs')) is not None
```

Proposed:

```python
assert membership_words(f.store, f.store.occurrence_of(row)) == {f.words[n][0] for n in ('cat', 'runs')}
```

**36. `test/test_sentence_identity.py:113` — `test_full_inventory_can_record_part_of_already_seated_arguments`**

Original:

```python
assert after['next_id'] == before['next_id'] + 1
```

Proposed:

```python
assert after['next_id'] == before['next_id'] + 3
```

Two memberships + asserted Part; seat assertion stays.

**37. `test/test_operators_final.py:158` — `test_definition_sentence_joins_identities_beside_the_interpret_definition`**

Original:

```python
assert len(f.store) == 0 and first[0] == 'sym'
```

Proposed:

```python
assert len(f.store) == 1 and first[0] == 'sym'
```

Source reading now has an opaque S; returned equality assertion stays.

**38. `test/test_operators_final.py:210` — `test_determiner_kept_closing_is_the_only_mint_and_lowers_once`**

Original:

```python
assert root[0] == 'sym' and f.store.parts.contains(root[1])
```

Proposed:

```python
assert reading(f.store, f.store.occurrence_of(row)) is None
assert reading(f.store, f.store.occurrence_of(child)) == ('sym', int(program.concept_ids[1]))
```

Read lower mention, not sentence root.

**39. `test/test_selected_nested_meaning.py:57` — `test_nested_observation_writers_retain_children_without_certifying_them`**

Original:

```python
assert len(store) == 0
```

Proposed:

```python
assert len(store) == 2
```

Two opaque S rows.

**40. `test/test_selected_nested_meaning.py:60` — `test_nested_observation_writers_retain_children_without_certifying_them`**

Original:

```python
assert set(store.parts.indices().tolist()) == store._test_initial_parts | {child_id, parent_id}
```

Proposed:

```python
assert normalized_layout(store) == expected_nested_layout
```

Enumerated fixture in the ledger: two assertions + two S rows, no operation pairs. Do not compute expectations from the result.

**41. `test/test_selected_nested_meaning.py:65` — `test_nested_observation_writers_retain_children_without_certifying_them`**

Original:

```python
assert parent['meaning'].role_refs[2] == child['occurrence']
```

Proposed:

```python
assert tagged_arguments(store.parts, parent_id)[1] == child_sentence
```

The parent assertion refers to the whole inner occurrence. Keep child variable as its original assertion for evidence/content checks; child_sentence is the independently captured opaque occurrence.

**42. `test/test_clause_acceptance.py:185` — `test_06_implication_keeps_two_unasserted_relation_operands`**

Original:

```python
assert not len(f.store)
```

Proposed:

```python
assert len(f.store) == 3
```

Two child sentences and the implication sentence; no change to their asserted relation operands/evidence.

**43. `test/test_operators_round4a0.py:102` — `test_references_and_content_keys_survive_compaction_and_checkpoint`**

Original:

```python
assert store.meaning_of(1).role_refs[0] == reference
```

Proposed:

```python
assert dict(store.meaning_of(1).bindings)['referent'] == reference
```

R5-dependent fixture port: authenticated semantic binding, no invented lift tree. Retain remapping, content key, stream and time assertions.

**44. `test/test_operators_round4a0.py:103` — `test_references_and_content_keys_survive_compaction_and_checkpoint`**

Original:

```python
assert int(store.refs[1, 0]) == int(store.address_keys[0])
```

Proposed:

```python
assert not hasattr(store, 'refs')
```

R5-dependent fixture port: authenticated semantic binding, no invented lift tree. Retain remapping, content key, stream and time assertions.

**45. `test/test_operators_round4a0.py:140` — `test_legacy_checkpoint_rekeys_native_and_typed_edges`**

Original:

```python
assert restored.meaning_of(1).role_refs[0] == reference
```

Proposed:

```python
assert dict(restored.meaning_of(1).bindings)['referent'] == reference
```

R5-dependent fixture port: authenticated semantic binding, no invented lift tree. Retain remapping, content key, stream and time assertions.

**46. `test/test_operators_round4a0.py:141` — `test_legacy_checkpoint_rekeys_native_and_typed_edges`**

Original:

```python
assert int(restored.refs[1, 0]) == reference[2]
```

Proposed:

```python
assert not hasattr(restored, 'refs')
```

R5-dependent fixture port: authenticated semantic binding, no invented lift tree. Retain remapping, content key, stream and time assertions.

**146. `test/test_forgetting.py:151` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
assert store.index_of_row(child[2]) is None
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**147. `test/test_forgetting.py:153` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
assert store.index_of_row(parent[2]) is None and report['cascade'] == 1
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**148. `test/test_forgetting.py:156` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
assert index is not None and store.refs[index, 0] == -1
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**149. `test/test_forgetting.py:157` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
assert store.meaning_of(index).role_refs[0] is None
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**150. `test/test_forgetting.py:158` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
torch.testing.assert_close(store.meaning_of(index).roles, root.roles)
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**151. `test/test_forgetting.py:159` — `test_idea_coarsens_parent_and_relation_cascades`**

Original:

```python
assert report['coarsened'] == 1
```

Proposed:

RETIRE with the old child-first coarsening/cascade fixture; replace with the two §31 stage certificates specified below.

This is a newly proposed disposition. Removing an independent mention must not rewrite/cascade its enclosing sentence; an asserted edge releases its child on the next pass.

**152. `test/test_forgetting.py:127` — `test_protected_and_live_roots_survive_and_block_unsafe_target`**

Original:

```python
torch.testing.assert_close(before[key], store.state_dict()[key])
```

Proposed:

```python
torch.testing.assert_close(before[key], store.state_dict()[key])
assert tagged_snapshot(store.parts, handles=before_parts.keys()) == before_parts
```

Remove only refs from snapshot key loop, add separately saved tagged rows.

**153. `test/test_forgetting.py:166` — `test_live_parent_pins_its_child`**

Original:

```python
assert store.meaning_of(store.index_of_row(parent[2])).role_refs[0] == child
```

Proposed:

```python
assert child in tagged_arguments(store.parts, parent[1])
```

Rewrite linked_store for this test as an explicit asserted relation with OCC child; parent is its SYM handle. The surviving-child assertion stays. Do not claim that an opaque sentence protects a mention.

### P6

**47. `test/test_sentence_identity.py:36` — `test_referenced_phrase_is_minted_as_a_part_argument`**

Original:

```python
assert f.cs._csw_row_of(reference) is not None
```

Proposed:

```python
assert tagged_arguments(f.store.parts, row[1])[1][0] == 'ltm'
```

reference is the tagged whole inner S argument. There is no derivation root, mention reading or seat for that S.

**48. `test/test_sentence_identity.py:37` — `test_referenced_phrase_is_minted_as_a_part_argument`**

Original:

```python
assert f.store.parts.refs[reference].tolist() == [f.noun('chases'), f.noun('mouse')]
```

Proposed:

```python
assert reading(f.store, reference) is None
assert membership_words(f.store, reference) == {f.words[n][0] for n in ('chases', 'mouse')}
```

reference is the tagged whole inner S argument. There is no derivation root, mention reading or seat for that S.

**49. `test/test_sentence_identity.py:38` — `test_referenced_phrase_is_minted_as_a_part_argument`**

Original:

```python
torch.testing.assert_close(f.store._concept_point(reference), clause.children[0].point)
```

Proposed:

```python
torch.testing.assert_close(occurrence_point(f.store, reference), clause.children[0].point)
```

reference is the tagged whole inner S argument. There is no derivation root, mention reading or seat for that S.

**50. `test/test_sentence_identity.py:39` — `test_referenced_phrase_is_minted_as_a_part_argument`**

Original:

```python
assert len(f.store) == 1
```

Proposed:

```python
assert len(f.store) == 2
```

reference is the tagged whole inner S argument. There is no derivation root, mention reading or seat for that S.

**51. `test/test_sentence_identity.py:91` — `test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically`**

Original:

```python
with pytest.raises(RuntimeError, match='concept row pool'):
            f.store.write_clause(clause)
```

Proposed:

```python
row = f.store.write_clause(clause)
```

Remove expected pool failure; one write, same full bank and capacities.

**52. `test/test_sentence_identity.py:93` — `test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically`**

Original:

```python
assert inventory(f.cs) == before
```

Proposed:

```python
after = inventory(f.cs)
assert after['rows'] == before['rows']
assert {key: after['placement'][key] for key in before['placement']} == before['placement']
assert after['next_id'] == before['next_id'] + 6
```

Correct earlier P6: unseated edge handles enter placement too. Existing entries stay; check all new handles against the six expected edges, not whole-dict equality.

**53. `test/test_sentence_identity.py:94` — `test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically`**

Original:

```python
assert len(f.store) == rows and len(f.store.parts) == edges
```

Proposed:

```python
assert len(f.store) == rows + 2 and len(f.store.parts) == edges + 6
```

Outer/inner S; outer bag of three words + inner bag of two + one asserted inner Part, no disjunction pair. Preserve allocator relate_idx entries and require any additions to name exactly those edges.

**54. `test/test_part_closing.py:72` — `test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`**

Original:

```python
assert a == participant and minted != word
```

Proposed:

```python
assert left.reference == participant
assert reading(f.store, mention) == ('sym', word)
```

Supply the actual right participant order-1 stamp in this unit fixture. Inspect position in the live kept journal; no stored occupation or native object. Capture the definition index before and require no additions (fixture note).

**55. `test/test_part_closing.py:73` — `test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`**

Original:

```python
assert f.cs.definitions.words(minted) == (word,)
```

Proposed:

```python
assert f.cs.definitions.words(word) == ()
```

Supply the actual right participant order-1 stamp in this unit fixture. Inspect position in the live kept journal; no stored occupation or native object. Capture the definition index before and require no additions (fixture note).

**56. `test/test_part_closing.py:74` — `test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`**

Original:

```python
assert f.store.parts.find(minted, word) is not None
```

Proposed:

```python
assert f.cs.definitions.objects(word) == ()
```

Supply the actual right participant order-1 stamp in this unit fixture. Inspect position in the live kept journal; no stored occupation or native object. Capture the definition index before and require no additions (fixture note).

**57. `test/test_part_closing.py:76` — `test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`**

Original:

```python
assert counts['verb_argin2'] == 1
```

Proposed:

RETIRE this symbol-keyed occupation-count assertion.

Supply the actual right participant order-1 stamp in this unit fixture. Inspect position in the live kept journal; no stored occupation or native object. Capture the definition index before and require no additions (fixture note).

**58. `test/test_part_closing.py:81` — `test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`**

Original:

```python
assert restored.counts[minted] == counts
```

Proposed:

RETIRE this symbol-keyed occupation-restore assertion.

Supply the actual right participant order-1 stamp in this unit fixture. Inspect position in the live kept journal; no stored occupation or native object. Capture the definition index before and require no additions (fixture note).

**59. `test/test_part_closing.py:129` — `test_unseated_native_argument_does_not_acquire_the_working_value_at_close`**

Original:

```python
assert f.store.parts.find(participant, word) is not None
```

Proposed:

```python
assert word in membership_words(f.store, f.store.occurrence_of(row))
```

Set word_id=word on this known lexical leaf; bag membership, not participant-predicate pair. Existing no-seat and symbol-type checks stay.

**60. `test/test_part_closing.py:271` — `test_new_relation_argument_is_minted_and_unseatable_argument_fails`**

Original:

```python
with pytest.raises(RuntimeError, match='concept row pool: order=1'):
        f.store.write_clause(clause)
```

Proposed:

```python
row = f.store.write_clause(clause)
```

Keep zero order-1 capacity. Remove restore-caps/second-write setup.

**61. `test/test_part_closing.py:273` — `test_new_relation_argument_is_minted_and_unseatable_argument_fails`**

Original:

```python
assert (alloc.next_id, len(alloc.parts), len(f.store)) == before
```

Proposed:

```python
assert (alloc.next_id, len(alloc.parts), len(f.store)) == (before[0], before[1], before[2] + 2)
```

Manual journal has no word IDs: sentence and bare order-1 argument, no edges/readings/native mint.

**62. `test/test_part_closing.py:278` — `test_new_relation_argument_is_minted_and_unseatable_argument_fails`**

Original:

```python
assert f.cs._csw_row_of(minted) is not None
```

Proposed:

```python
assert reading(f.store, argument_occurrence) is None
assert dict(f.cs._csw_rows) == seats_before
```

Read argument_occurrence from written_rows, not an invented symbol.

**63. `test/test_part_closing.py:298` — `test_minted_surface_phrase_keeps_the_unminted_heads_word`**

Original:

```python
assert f.cs._csw_row_of(minted) is not None
```

Proposed:

```python
assert dict(f.cs._csw_rows) == seats_before
```

Old P6 correction: this surface phrase is the verb predicate head, not an NP. Do not force a mention; its actual word survives in the sentence bag. Check no new native seat/definition.

**64. `test/test_part_closing.py:299` — `test_minted_surface_phrase_keeps_the_unminted_heads_word`**

Original:

```python
assert f.cs.definitions.words(minted) == (word,)
```

Proposed:

```python
assert word in membership_words(f.store, f.store.occurrence_of(row))
```

Old P6 correction: this surface phrase is the verb predicate head, not an NP. Do not force a mention; its actual word survives in the sentence bag. Check no new native seat/definition.

**65. `test/test_part_closing.py:300` — `test_minted_surface_phrase_keeps_the_unminted_heads_word`**

Original:

```python
assert f.store.parts.find(minted, word) is not None
```

Proposed:

```python
assert f.cs.definitions.objects(word) == ()
```

Old P6 correction: this surface phrase is the verb predicate head, not an NP. Do not force a mention; its actual word survives in the sentence bag. Check no new native seat/definition.

**66. `test/test_relation_closing.py:34` — `test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`**

Original:

```python
assert len(f.store) == len(unknown)
```

Proposed:

```python
assert len(f.store) == len(unknown) + (2 if nested else 1)
```

Unindexed operands plus all completed S occurrences.

**67. `test/test_relation_closing.py:35` — `test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`**

Original:

```python
assert (f.store.parts.refs[f.store.relations()] > 0).all()
```

Proposed:

```python
assert all(valid_tagged_arguments(f.store, edge) for edge in f.store.relations().tolist())
```

Validate tag and live target, including signed OCC addresses.

**68. `test/test_relation_closing.py:36` — `test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`**

Original:

```python
assert not f.store.trust[:len(f.store)].any()
```

Proposed:

```python
assert not f.store.trust[operand_rows].any()
```

operand_rows is the exact len(unknown) written children; the newly retained asserted sentence is outside this unverified operand obligation.

**69. `test/test_relation_closing.py:39` — `test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`**

Original:

```python
assert f.store.row(index)['kind'] == 'unverified'
```

Proposed:

```python
assert f.store.row(index)['kind'] == 'unverified'
```

Loop over those operand_rows, not all ideas (which now also contains S). Content/evidence/unverified outcomes stay.

**70. `test/test_relation_closing.py:54` — `test_three_slot_relation_closes_unindexed_numerical_operands`**

Original:

```python
assert len(f.store) == 2 and isinstance(row, tuple)
```

Proposed:

```python
assert len(f.store) == 3 and isinstance(row, tuple)
```

Two operands + source S; R1 must specify the fused source value for the three-slot path.

**71. `test/test_relation_closing.py:56` — `test_three_slot_relation_closes_unindexed_numerical_operands`**

Original:

```python
torch.testing.assert_close(f.store._concept_point(symbol), point)
```

Proposed:

```python
torch.testing.assert_close(reference_point(f.store, reference), point)
```

Iterate tagged arguments on the same assertion. Keep the following slots[:2,0] equality/tolerance verbatim.

**72. `test/test_relation_closing.py:93` — `test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows`**

Original:

```python
assert (store.parts.refs[clauses] > 0).all()
```

Proposed:

```python
assert all(valid_tagged_arguments(store, edge) for edge in clauses.tolist())
```

Same 64-row fixture; update only its stale native-seat explanation. Carrier certificate unchanged.

### P7

**73. `test/test_part_closing.py:152` — `test_new_referent_closes_as_an_unminted_occurrence`**

Original:

```python
assert f.store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
```

Proposed:

```python
assert reading(f.store, f.store.occurrence_of(row)) == ('sym', kind)
```

Raw mention reading; preserve the distinction from its working MTM kind.

**74. `test/test_part_closing.py:186` — `test_same_description_tokens_keep_distinct_occurrence_identities`**

Original:

```python
assert f.store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
```

Proposed:

```python
assert reading(f.store, f.store.occurrence_of(row)) == ('sym', kind)
```

Raw mention reading; preserve the distinction from its working MTM kind.

**75. `test/test_reference_close.py:117` — `test_recent_token_returns_at_its_kinds_native_conceptual_order`**

Original:

```python
assert store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
```

Proposed:

```python
assert reading(store, store.occurrence_of(row)) == ('sym', kind)
```

Raw mention reading; preserve the distinction from its working MTM kind.

**76. `test/test_reference_close.py:414` — `test_kept_journal_supplies_np_content_predicate_and_positional_provenance`**

Original:

```python
assert f.store.raw_meaning_of(f.store.location(identity)).role_refs[0] == ('sym', program.concept_ids[1].item())
```

Proposed:

```python
assert reading(f.store, identity) == ('sym', program.concept_ids[1].item())
```

Raw mention reading; preserve the distinction from its working MTM kind.

**77. `test/test_medium_term_memory.py:66` — `test_bound_use_reads_as_its_introduction_without_rewriting_rows`**

Original:

```python
assert entry.kind == store.raw_meaning_of(store.index_of_row(address)).role_refs[0][1]
```

Proposed:

```python
assert entry.kind == reading(store, store.occurrence_of(store.index_of_row(address)))[1]
```

**78. `test/test_part_closing.py:156` — `test_new_referent_closes_as_an_unminted_occurrence`**

Original:

```python
assert (f.cs._concept_allocator.next_id, len(f.store.parts)) == before
```

Proposed:

```python
assert (f.cs._concept_allocator.next_id, len(f.store.parts)) == (before[0] + 1, before[1] + 1)
```

One mention-only reading; manual fixture carries no sentence word IDs.

**79. `test/test_part_closing.py:195` — `test_same_description_tokens_keep_distinct_occurrence_identities`**

Original:

```python
assert (alloc.next_id, len(f.store.parts)) == before
```

Proposed:

```python
assert (alloc.next_id, len(f.store.parts)) == (before[0] + 2, before[1] + 2)
```

Two mention readings; all distinct-address, identity, activation and bank assertions stay.

**80. `test/test_part_closing.py:230` — `test_pronoun_at_occurrence_closes_verb_over_antecedent_kind_without_seating`**

Original:

```python
assert f.store.parts.refs[edge].tolist() == [kind, verb]
```

Proposed:

```python
assert resolver._derivation_symbol(address) == kind
assert reading(f.store, f.store.occurrence_of(row)) is None
```

Replace parent operation-pair assertion; keep audit antecedent, seats and bank verbatim.

**81. `test/test_reference_arrivals.py:76` — `test_named_first_bind_opens_anonymous_without_row_or_bank_writes`**

Original:

```python
assert store.raw_meaning_of(row).role_refs == (None, None, None)
```

Proposed:

```python
assert reading(store, token) == ('sym', words['x'])
assert store.raw_meaning_of(row).role_refs == (None, None, None)
```

Word is in the mention reading; anonymous kind/open state unchanged. Snapshot the reading before arrival.

**82. `test/test_reference_arrivals.py:184` — `test_bound_use_does_not_assign_a_kind_before_arrival`**

Original:

```python
assert store.raw_meaning_of(used).role_refs[0] is None
```

Proposed:

```python
assert reading(store, store.occurrence_of(used)) == ('sym', words['x'])
assert store.raw_meaning_of(used).role_refs[0] is None
```

Own word reading, MTM token link. No OCC-to-OCC identity edge.

**83. `test/test_nested_retention.py:93` — `test_repeated_edges_keep_role_order_without_evidence_inheritance`**

Original:

```python
assert view["occurrences"] == (root, child)
```

Proposed:

```python
assert view["occurrences"] == (root,)
```

Opaque two-row fixture: remove repeated stored parse roles; inspect child evidence separately. incomplete==() and records_scanned==2 stay.

**84. `test/test_nested_retention.py:94` — `test_repeated_edges_keep_role_order_without_evidence_inheritance`**

Original:

```python
assert view["edges"] == ((root, 0, child), (root, 2, child))
```

Proposed:

```python
assert view["edges"] == ()
```

Opaque two-row fixture: remove repeated stored parse roles; inspect child evidence separately. incomplete==() and records_scanned==2 stay.

**85. `test/test_nested_retention.py:96` — `test_repeated_edges_keep_role_order_without_evidence_inheritance`**

Original:

```python
assert view["rows"][1]["kind"] == "question"
```

Proposed:

```python
assert store.row(store.location(child))["kind"] == "question"
```

Opaque two-row fixture: remove repeated stored parse roles; inspect child evidence separately. incomplete==() and records_scanned==2 stay.

**86. `test/test_nested_retention.py:97` — `test_repeated_edges_keep_role_order_without_evidence_inheritance`**

Original:

```python
assert view["rows"][1]["trust"] == 0
```

Proposed:

```python
assert store.row(store.location(child))["trust"] == 0
```

Opaque two-row fixture: remove repeated stored parse roles; inspect child evidence separately. incomplete==() and records_scanned==2 stay.

**87. `test/test_nested_retention.py:98` — `test_repeated_edges_keep_role_order_without_evidence_inheritance`**

Original:

```python
assert view["rows"][1]["meaning"].scope == (("quote", True),)
```

Proposed:

```python
assert store.row(store.location(child))["meaning"].scope == (("quote", True),)
```

Opaque two-row fixture: remove repeated stored parse roles; inspect child evidence separately. incomplete==() and records_scanned==2 stay.

**88. `test/test_nested_retention.py:114` — `test_nested_read_limits_are_separate_and_explicit`**

Original:

```python
assert reason in view["incomplete"]
```

Proposed:

```python
assert view["incomplete"] == (("record_limit",) if reason == "record_limit" else ())
```

An opaque root fits max_nodes=1 and max_depth=0. Keep non-mutation and separately cover explicit relation/reconstruction budgets, with no budget increase.

**89. `test/test_nested_retention.py:128` — `test_retained_structure_and_reference_identity_survive_checkpoint_restore`**

Original:

```python
assert view["incomplete"] == () and view["occurrences"] == (root, child)
```

Proposed:

```python
assert view["incomplete"] == () and view["occurrences"] == (root,)
```

R5: semantic binding protects child through origin withdrawal; opaque structure view does not expose that binding as a parse edge.

**90. `test/test_nested_retention.py:129` — `test_retained_structure_and_reference_identity_survive_checkpoint_restore`**

Original:

```python
assert view["rows"][1]["kind"] == "unverified"
```

Proposed:

```python
assert restored.row(restored.location(child))["kind"] == "unverified"
```

**91. `test/test_nested_retention.py:169` — `test_structural_restore_rejects_dangling_local_links_atomically`**

Original:

```python
assert good["records"][0]["context"]["role_refs"] == (None, None, None)
```

Proposed:

```python
assert good["records"][0]["context"]["bindings"] == ()
```

Move dangling corruption to semantic binding; retain exact atomic-load failure and context/text immutability.

**92. `test/test_operators_round4a0.py:289` — `test_explicit_rewitness_keeps_a_self_reference_without_inventing_evidence`**

Original:

```python
assert kept == row and store.meaning_of(row).role_refs[0] == reference
```

Proposed:

```python
assert kept == row and dict(store.meaning_of(row).bindings)['referent'] == reference
```

R5: explicit self binding in semantic context is distinct from a self edge in the acyclic LTM1 graph; needs Alec decision.

**93. `test/test_operators_round4a0.py:290` — `test_explicit_rewitness_keeps_a_self_reference_without_inventing_evidence`**

Original:

```python
assert store.read_structure(reference)['incomplete'] == ('cycle',)
```

Proposed:

```python
assert store.read_structure(reference)['incomplete'] == ()
```

No stored parse edge; evidence comparisons unchanged.

**94. `test/test_nested_retention.py:176` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
assert view["incomplete"] == ("cycle",)
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**95. `test/test_nested_retention.py:177` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
assert len(view["rows"]) == 2 and view["records_scanned"] == 2
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**96. `test/test_nested_retention.py:178` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
assert view["rows"][1]["kind"] == "question"
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**97. `test/test_nested_retention.py:183` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
assert reread["incomplete"] == ("cycle",)
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**98. `test/test_nested_retention.py:184` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
assert reread["rows"][1]["kind"] == "question"
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**99. `test/test_nested_retention.py:185` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
torch.testing.assert_close(restored.c_plus, store.c_plus, rtol=0, atol=0)
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

**100. `test/test_nested_retention.py:186` — `test_cycle_is_reported_without_unbounded_read_or_evidence_transfer`**

Original:

```python
torch.testing.assert_close(restored.c_minus, store.c_minus, rtol=0, atol=0)
```

Proposed:

RETIRE with the stored sentence-parse-cycle test.

Proposed replacement is a separate LTM1 DAG rejection/defensive-reader certificate (see fixture ledger); a checkpoint loader must not admit a cycle. This retirement is newly proposed, not covered by old P7.

### Fixture-only

**101. `test/test_part_read.py:118` — `test_typed_part_read_follows_attachments_and_keeps_the_clause_subject`**

Original:

```python
assert not incomplete
```

Proposed:

```python
assert not incomplete
```

Move verbatim to the bounded transient reconstruction component-adapter unit fixture; root denotes the stored opaque occurrence. Existing inverse is controlled in this unit fixture only; no cold-start gate forcing. R5 must specify the adapter before this move.

**102. `test/test_part_read.py:119` — `test_typed_part_read_follows_attachments_and_keeps_the_clause_subject`**

Original:

```python
assert terms[root]['kind'] == 'lift'
```

Proposed:

```python
assert terms[root]['kind'] == 'lift'
```

Move verbatim to the bounded transient reconstruction component-adapter unit fixture; root denotes the stored opaque occurrence. Existing inverse is controlled in this unit fixture only; no cold-start gate forcing. R5 must specify the adapter before this move.

**103. `test/test_part_read.py:120` — `test_typed_part_read_follows_attachments_and_keeps_the_clause_subject`**

Original:

```python
assert terms[root]['operands'][0] == ('sym', f.noun('a'))
```

Proposed:

```python
assert terms[root]['operands'][0] == ('sym', f.noun('a'))
```

Move verbatim to the bounded transient reconstruction component-adapter unit fixture; root denotes the stored opaque occurrence. Existing inverse is controlled in this unit fixture only; no cold-start gate forcing. R5 must specify the adapter before this move.

**104. `test/test_part_read.py:121` — `test_typed_part_read_follows_attachments_and_keeps_the_clause_subject`**

Original:

```python
assert terms[root]['reference'] != root
```

Proposed:

```python
assert terms[root]['reference'] != root
```

Move verbatim to the bounded transient reconstruction component-adapter unit fixture; root denotes the stored opaque occurrence. Existing inverse is controlled in this unit fixture only; no cold-start gate forcing. R5 must specify the adapter before this move.

**105. `test/test_part_read.py:140` — `test_a_later_word_definition_does_not_retype_a_retained_marker`**

Original:

```python
assert not missing and before[reference]['reference'] == ('sym', content)
```

Proposed:

```python
assert not missing and before[reference]['reference'] == ('sym', content)
```

Move to a live kept-parse/position fixture; before and after come from its working view. Remove stored product/surface pair setup; lexical definition changes still cannot reclassify the marker.

**106. `test/test_part_read.py:143` — `test_a_later_word_definition_does_not_retype_a_retained_marker`**

Original:

```python
assert not f.cs.interpret.no_object(marker)
```

Proposed:

```python
assert not f.cs.interpret.no_object(marker)
```

Move to a live kept-parse/position fixture; before and after come from its working view. Remove stored product/surface pair setup; lexical definition changes still cannot reclassify the marker.

**107. `test/test_part_read.py:146` — `test_a_later_word_definition_does_not_retype_a_retained_marker`**

Original:

```python
assert not missing and after[reference]['reference'] == ('sym', content)
```

Proposed:

```python
assert not missing and after[reference]['reference'] == ('sym', content)
```

Move to a live kept-parse/position fixture; before and after come from its working view. Remove stored product/surface pair setup; lexical definition changes still cannot reclassify the marker.

**108. `test/test_part_read.py:164` — `test_answer_read_checks_a_named_composite_before_descending_its_head`**

Original:

```python
assert matches(model, value, 'bound-name')
```

Proposed:

```python
assert matches(model, value, 'bound-name')
```

Matching outcome stays; see separate named-composite/surface fixture proposals. Do not substitute nonemptiness or nearest-vector agreement.

**109. `test/test_part_read.py:183` — `test_answer_read_follows_admitted_surface_with_a_composed_marker`**

Original:

```python
assert matches(model, value, 'content')
```

Proposed:

```python
assert matches(model, value, 'content')
```

Matching outcome stays; see separate named-composite/surface fixture proposals. Do not substitute nonemptiness or nearest-vector agreement.

### Held

**110. `test/test_item6_2_repair.py:28` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert resolved.constituents == (child,)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**111. `test/test_item6_2_repair.py:29` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert not open_slots(resolved)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**112. `test/test_item6_2_repair.py:30` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert evidence_pair(resolved) == (.8, .2)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**113. `test/test_item6_2_repair.py:34` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert len(store) == 2
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**114. `test/test_item6_2_repair.py:35` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert store.row(index)['kind'] == 'inference'
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**115. `test/test_item6_2_repair.py:36` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert store.meaning_of(index).role_refs[0] == store.occurrence_of(0)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**116. `test/test_item6_2_repair.py:37` — `test_saved_construction_binds_and_writes_the_child`**

Original:

```python
assert bindings(store.meaning_of(index))['_producing_operation'] == 'ask'
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**117. `test/test_item6_2_repair.py:50` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert resolved.role_refs == (('constituent', 1), None, ('constituent', 0))
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**118. `test/test_item6_2_repair.py:51` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert resolved.constituents == (existing, nested)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**119. `test/test_item6_2_repair.py:52` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert nested.constituents == (leaf,)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**120. `test/test_item6_2_repair.py:53` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert nested.role_refs[0] == ('constituent', 0)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**121. `test/test_item6_2_repair.py:54` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert goal.constituents == (existing,) and goal.role_refs[0] is None
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**122. `test/test_item6_2_repair.py:58` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert len(store) == 4
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**123. `test/test_item6_2_repair.py:60` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert root.role_refs[2] == store.occurrence_of(0)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**124. `test/test_item6_2_repair.py:61` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert root.role_refs[0] == store.occurrence_of(2)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**125. `test/test_item6_2_repair.py:62` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert store.meaning_of(2).role_refs[0] == store.occurrence_of(1)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**126. `test/test_item6_2_repair.py:63` — `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`**

Original:

```python
assert store.meaning_of(2).scope == (('child', store.occurrence_of(1)),)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**127. `test/test_item6_2_thinking.py:493` — `test_thought_nested_writer_is_restricted_and_preflights_capacity`**

Original:

```python
assert len(store)==2
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**128. `test/test_item6_2_thinking.py:494` — `test_thought_nested_writer_is_restricted_and_preflights_capacity`**

Original:

```python
assert all(store.row(i)['kind']=='question' for i in range(len(store)))
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**129. `test/test_item6_2_thinking.py:495` — `test_thought_nested_writer_is_restricted_and_preflights_capacity`**

Original:

```python
assert store.meaning_of(1).role_refs[0]==store.occurrence_of(0)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**130. `test/test_item6_2_thinking.py:498` — `test_thought_nested_writer_is_restricted_and_preflights_capacity`**

Original:

```python
with pytest.raises(OverflowError,match='forgetting'):
        write(model,parent,row=0)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**131. `test/test_item6_2_thinking.py:500` — `test_thought_nested_writer_is_restricted_and_preflights_capacity`**

Original:

```python
assert len(store)==0
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**132. `test/test_reference_arrivals.py:203` — `test_anonymous_parent_keeps_its_raw_open_role_and_query_predicate`**

Original:

```python
assert raw.role_refs[0][0] == 'ltm' and open_slots(raw) == (('referent', 0),)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**133. `test/test_reference_arrivals.py:205` — `test_anonymous_parent_keeps_its_raw_open_role_and_query_predicate`**

Original:

```python
assert bindings(query)['_query_occurrence'] == store.occurrence_of(row)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**134. `test/test_reference_arrivals.py:206` — `test_anonymous_parent_keeps_its_raw_open_role_and_query_predicate`**

Original:

```python
torch.testing.assert_close(query.roles[1], raw.roles[1])
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**135. `test/test_reference_occurrence_views.py:14` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert address == f.store.occurrence_of(target)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**136. `test/test_reference_occurrence_views.py:15` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert not f.cs.components.individual_ids
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**137. `test/test_reference_occurrence_views.py:19` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert address in candidates and f.identities[distractor] in candidates
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**138. `test/test_reference_occurrence_views.py:20` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert f.resolver.prediction(address, f.atoms[4], 0) > 0
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**139. `test/test_reference_occurrence_views.py:21` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert f.resolver.prediction(f.identities[distractor], f.atoms[4], 0) == 0
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**140. `test/test_reference_occurrence_views.py:23` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert context['identity'] == address and context['predicate'] == f.predicates[tuple(f.atoms[4].tolist())]
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**141. `test/test_reference_occurrence_views.py:25` — `test_occurrence_candidates_use_retained_part_participants`**

Original:

```python
assert result.value.role_refs[0] == address
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**142. `test/test_selected_nested_meaning.py:120` — `test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write`**

Original:

```python
with pytest.raises(OverflowError, match='forgetting'):
        store.append_meaning(meaning, kind='observation')
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**143. `test/test_selected_nested_meaning.py:122` — `test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write`**

Original:

```python
assert len(store) == 0
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**144. `test/test_selected_nested_meaning.py:124` — `test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write`**

Original:

```python
with pytest.raises(ValueError, match='constituent'):
        store.bind_constituents(malformed)
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.

**145. `test/test_selected_nested_meaning.py:126` — `test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write`**

Original:

```python
assert len(store) == 0
```

Proposed:

KEEP VERBATIM; no assertion edit proposed.

R4–R6 must supply the source of this semantic/participant obligation. A new fixture port requires a further exact proposal; no fallback to a weaker assertion.
