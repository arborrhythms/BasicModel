# Step 1b: implementation and proposed ports for Claude's review

Prepared by Codex, 2026-10-21, at Alec's request to describe the steps
carefully before extending the three declared test ports. This is an
implementation report and a proposed next sequence, not a review verdict
or authorization to change the remaining assertions.

## 1. Source and review boundary

Work is on `codex/item3-5-first-pass`, from the §14 base `93788f9d2`.

| Commit | What it contains |
|---|---|
| `eb75de56ab62a04553553d62f8edb5c898d992cc` | D2's name-scoped reverse assertion and the existing plan amendments |
| `f9faae610f35d5ca660d0ac49249637cb23e7adf` | The §10.2 implementation, closing regressions and three declared reference-close ports; receipts for the two D2 replays on the preceding source |
| `67c5b203f34e4299703d4c7801b34e06c1045eb6` | Follow-up repairs to the admitted OBJECT-ID carrier, lexical reference staging, and occurrence participant reads; new regressions |
| `28e928404fb0a3f7246c57949173139c1f798381` | Completed receipts, failed-run evidence and the current carry ledger; no further production/test changes |

The complete production/test delta is reviewable with
`git diff 93788f9d2..67c5b203f -- bin test data`.
The two implementation commits can also be reviewed separately with
`git show f9faae610` and `git show 67c5b203f`.

The additional ports below are **proposed, not applied**. Publication is
held under the user's §14 instruction: extra assertion changes are outside
the three declared ports, and the remaining mechanism gates are not green.
BasicModel main is still `204e1be55`; WikiOracle main is still `a89c958`.
The October 20 amendments are retained on the work branch.

## 2. D2: restrict the reverse assertion to the requested name

In [test_math_chain_section14.py](../../../test/test_math_chain_section14.py),
the reverse arm builds the same `word_concepts('x')` name set as the forward
arm. Every non-estimate row's working read must have no forward reference
with an identity in that set. A row whose raw journal has such a pending
name must also have no open referent slot in its working read.

The saved row 5 / word 14 (`is`) is the reason for the scope change: it is
an unrelated name introduced by exploration. The `carries_three` check,
minimum two readings, departure check, forward assertion, forced scope,
budget and keep rule are unchanged. Both requested saved reverse states
pass on `eb75de56a`: arrival-14 and fresh-both-reverse-5. See the
[D2 receipt](../2026-10-20-item3-5-stm1/review3/d2-replays.md) and
[closing-read-causes §6.2](../2026-10-20-item3-5-stm1/closing-read-causes.md).

## 3. §10.2: implement location-based tokens and inline lowering

The following is the implementation in `f9faae610`, in the plan's order.

| Plan change | Concrete edit and intended consequence |
|---|---|
| 1. Keep the operand; inline `lower` | `ClauseJournal.composition` retains the operand reference when a mint/order request is present. `PartClosing._plan_one` reuses the declared head child's symbol and types for `lower`, requires exact equality of the child's and result's points, and allocates no lower edge or seat. Preflight still records all three lower argument/result occupations. |
| 2. Index ordinary tokens by occurrence | `ClauseRows.write_clause` records an individual symbol only if it belongs to `alloc.individuals`. Otherwise the activation is `(occurrence, None, role)`; the occurrence address is the working candidate's identity. |
| 3. Keep addresses through reference closing | `ReferenceClose.commit` uses the written occurrence for an ordinary mint. The new-candidate map, audit and held candidates carry that address. Training exposes the form to the addressed row's point; a genuinely minted symbol still uses its bank point. |
| 4. Read an addressed operand's derivation | A located leaf in `PartClosing._plan_one` uses raw `role_refs[0]` when it supplies a symbol. Thus an addressed pronoun's verb edge uses the antecedent's retained kind/fold symbol. A row without a retained derivation follows the existing seat-or-loud-failure path. |
| 5. Remove implicit individual promotion | `_individual_terms` and promotion from `clause.minted` are removed. Seating an order-1 fold handle does not itself add an individual or singleton. The existing explicit individual writers remain. |
| 6. Keep the eight-seat capacity gate | `MM_grammar.xml` remains at eight seats, 900 epochs and the .20 threshold. The saved gate passed initially; the first fresh draw failed. The exact diagnosis and repair are in §4 below. |
| 7. Closing tests and declared ports | The ordinary-token test now requires its kind in the raw derivation, no Part allocation and no bank allocation. Added tests require two same-description tokens to retain distinct addresses, and an addressed pronoun to form a verb edge without seating. A further regression requires altered-content lowering to fail before any partial write. The existing new-relation-argument seat-or-loud-failure test remains. |
| 8. Keep the later work separate | Recurrence minting, LTM2-to-LTM1 edges, `search_exhausted`, and the non-lower composed-argument fold-cache rule are unchanged. |

The three declared reference-close ports are named in the
[receipt's port table](README.md#declared-reference-close-ports): the
selected-member-order case and the two native-order `[0]`/`[3]` cases.
Their order and distinctness requirements are retained; their ordinary
token is now an occurrence rather than an implicit bank individual.

## 4. Diagnose and repair the native capacity failure at the same draw

The initial saved-state pass did not replace the fresh failure. The latter
was captured in `mm-fresh-02` and replayed in `mm-fresh-diagnosis-03` and
`mm-fresh-identity-04`.

The failing plans appeared to require three new relation operands. A
read-only audit showed that their points exactly equaled already admitted
native IDs 1, 2 and 3. This disproved the initial interpretation that the
fixture had exhausted its pool with genuinely new content. That initial
interpretation and all failed attempts remain in
[capacity-causes.md](capacity-causes.md).

Two source-reference losses caused the duplicate seating:

1. `_stage_reading_grammar_references` already provided OBJECT IDs beside
   the grammar rows and atoms. `Models._capture_reading_programs` consumed
   only the WORD identity carrier, which is absent on this mixing path,
   and substituted -1. It now carries those already-staged OBJECT IDs
   where the corresponding grammar row is valid. The exact-code search
   was diagnostic only; no distance search was added to production.
2. The carrier-only replay, `mm-fresh-repair-05`, still failed:
   `concept_ids=[1,2]`, but `reference_ids=[-1,-1]`.
   `Language.resolve_lexical_references` independently erased a known
   non-individual kind when the selected order was 1. It now preserves
   that reference and stamps the requested order. This is the remaining
   instance of the derivation cut removed by §10.2.

The combined repair is committed in `67c5b203f`. Both rng-119
(`mm-saved-repair-07`) and the original fresh draw (`mm-fresh-confirm-08`)
pass with **zero seats and zero lower edges**, at the unchanged inventory
and budget. The captured capacity-02 lower plan has no `seat: true` entry.
No fresh initialization was substituted. A new native mixing regression
checks the OBJECT IDs, absence of fabricated WORD IDs/definitions and
absence of duplicate seating. The accepted relative-closing boundary is
unchanged by this follow-up.

## 5. Read participant expectation for occurrence candidates

The scoped bind failures also exposed a mechanism defect:
`ReferenceClose.prediction` returned zero for every occurrence candidate,
and `accessible` compared a Part argument with the frame's intentionally
empty individual-symbol field. The occurrence had retained usable Part
evidence, but these reads did not reach it.

In `67c5b203f`, `_derivation_symbol` reads the raw derivation of a located
occurrence. `prediction` uses that symbol with the existing rebuildable
participant cache; `accessible` uses it to recover the occurrence's
predicate context. The candidate remains the occurrence address. No new
scorer, association table, objective term or stored identity label is
introduced.

The four new cases in
[test_reference_occurrence_views.py](../../../test/test_reference_occurrence_views.py)
check the expected participant against a distractor, the recovered
predicate, exact selected occurrence and unchanged store/bank. They pass.
These are additional regressions, not replacements for G1–G3.

## 6. Proposed test ports: exact scope, not yet applied

The [port inventory](review-required.md#1-additional-representation-ports-need-a-declared-scope)
lists each affected family and the invariants to preserve. The intended
steps, subject to review, are:

1. In the four `test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding`
   cases, require available native occurrence addresses instead of ints.
   Keep every expected partition and content comparison.
2. In `test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised`,
   count the retained first token occurrence and require zero implicit
   individuals. Keep no-access, no-new-mint and immutable-row checks.
   In `test_revision_merges_three_accessible_groups_transitively_across_cued_documents`,
   count the four initial tokens by address instead of bank membership.
   Its currently failing one-group assertion is **not** part of this port.
3. In both `test_kept_journal_supplies_np_content_predicate_and_positional_provenance`
   cases, check the order-1 occurrence and its retained derivation instead
   of expecting an individual bank symbol and symbol witness. Preserve
   the cue, predicate, role, source leaves, content and immutable reload.
4. In bind-thought G1's 12 cases, G2's 12 cases, the four accessible G3
   cases and `test_episode_has_one_bind_per_site_and_kept_write_preserves_np_content`,
   compare the result with the supplied expected occurrence itself rather
   than wrapping that tuple in `('sym', expected)`. Do not change the
   expected antecedent. Some G1/G2 failures on `f9faae610` also selected
   the wrong occurrence; the wrapper port cannot cure that. They must
   pass their original semantic comparisons after the participant repair.
5. Give the five tests specifically about a native individual a genuinely
   minted native concept in their fixture. The exact cases are
   `test_individual_axes_in_the_change_residual_exclude_form_and_its_gradient`,
   `test_native_row_is_the_only_content_and_witnesses_are_live_queries`,
   `test_forgetting_and_reinterpretation_never_refold_a_native_identity`,
   and both `test_loader_permanently_rejects_signed_identity_without_modifying_the_row`
   cases. Preserve native content, gradient, witness, immutable-store and
   strict-rejection checks. Their ordinary-mint setup no longer creates
   the native object their assertions examine. No empty-axes fallback
   has been added to `innovation` to conceal the first failure.
6. Port the five assertions requiring the retired -1 lexical cut:
   `test_selected_grammar_context_selects_order[particular-1,name-1,pronoun-1]`,
   `test_capture_owns_grammar_resolution_without_changing_reconstruction`,
   and `test_determiner_kept_closing_is_the_only_mint_and_lowers_once`.
   Require the preserved source-kind ID and unchanged order request,
   leaves, closing behavior and reconstruction. An ordinary token must
   still close without allocating an individual.

This does not classify all 43 scoped failures as harmless ports. A changed
representation check must expose the same semantic assertion behind it;
a wrong antecedent, lost merge or changed evidence remains a failure.

## 7. Unresolved mechanism: grouping uses after a correct bind

`working-groups-08` on `67c5b203f` preserves two unchanged failures:

| Question | Observed close | Next working read | Required result |
|---|---|---|---|
| Common later evidence | Two distinct introductions; each later use selects the intended earlier occurrence | Four separate groups | All four reconcile into one group |
| Transitive revision | Three introductions and three uses; each use selects its intended introduction | Six accessible groups; the outside occurrence stays excluded | One accessible group, with the outside occurrence still excluded |

The [row tables](working-groups-08.md) link the actual closing audits and
working reads. `individuals()` initially keys each unminted row by its
own occurrence. Its old merge pass skips non-int identities and reads
native individual baselines. Correcting the participant lookup therefore
does not by itself repair grouping.

The issue for architectural review is the retained evidence for that
grouping. The chosen occurrence is present in the closing audit, while
the raw derivation records the kind/fold symbol. Two independent
introductions of the same kind must stay distinct, yet a later bound use
must be interpretable with its selected antecedent, and the working
grouping must remain revisable. Which retained fact should license this
read, consistently for both composed and inline-lowered tokens?

No grouping repair is implemented yet. In particular, using a shared
kind/fold seat as identity would bypass the distinction required by the
new two-token regression. The rule must respect §6.6's prohibition on
writing working identity back into LTM. Neither merge assertion has been
weakened or relabeled a fixture port.

## 8. Evidence by source, including failures

| Source | Gate | Result |
|---|---|---|
| `eb75de56a` | D2, both saved reverse states | 2 passes |
| `f9faae610` | D1, both exact saved forward states | 2 passes; both formerly deciding comparisons tie at R=.036067377775907516, A=0 and keep greedy |
| `f9faae610` | Six fresh initializations per order, every first run retained | 12/12 passes; twelve distinct initial parameter fingerprints |
| `f9faae610` | One scoped set | 70 passed, 43 failed, one copular prerequisite skip |
| `f9faae610` | Section 14 within that set | 12 passed, one prerequisite skip |
| `f9faae610` | Paired zero-cost reconstruction invariant | Pass |
| `f9faae610` | Both decomposition lengths, both gain arms | Four passes; all eight companion bindings/proof chains correct |
| `67c5b203f` | Saved and original fresh eight-seat MM states | Both pass; zero seats and lower edges |
| `67c5b203f` | `address-read-07` | 37 passed, five failed; the five failures assert the retired -1 cut |
| `67c5b203f` | `working-groups-08` | Both merge certificates fail, as described above |

D1's [paired readings and tables](arrival-keep-tables.md),
[initial-state validation](arrival-keep-validation.json),
[all twelve fresh captures](fresh-final.md),
[every scoped failure](scoped-final.md), and
[one table per decomposition question](decomposition-final-questions.md)
are retained. [receipt-manifest.json](receipt-manifest.json) maps completed
runs to source, including exact file blobs for intermediate repair attempts.

The twelve-fresh and full scoped/decomposition results belong to
`f9faae610`. They are **not** final validation of the later `67c5b203f`
source. All failed draws, diagnostic attempts and the prior D1 failures
remain failures in their original receipts.

## 9. Proposed remaining sequence

First, review the port scope in §6 and settle the grouping read in §7.
Then implement the agreed ports and mechanism, preserving all named
semantic assertions. Freeze the resulting source in a candidate commit
before collecting its final evidence.

On that source, retain the requested six fresh arrivals per order with
full captures; diagnose any failure from its capture, including the keep
probe for a forward failure, before further changes. The forward contract
goes back to Alec and Claude if such a failure remains. Run the requested
scoped set and both decomposition arms once on that frozen source. The
two MM gates and targeted regressions remain tied to their recorded
source; if a further production edit affects those paths, their evidence
must cover the new source as well.

Publication follows only when the contract questions and gates are closed:
BasicModel and WikiOracle main, the WikiOracle submodule bump, the October
20 amendments and a clean BasicModel tree. There is no second full sweep,
fixture resize, replacement draw, pool-based chooser mask, seat reservation
or extra training in this sequence. Item 0 learning findings, separate 5.5
gates and the later 3.5 steps remain explicitly carried in the existing
ledger; none is claimed complete by these mechanism receipts.
