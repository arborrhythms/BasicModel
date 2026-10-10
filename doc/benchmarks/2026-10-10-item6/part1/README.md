# Item 6 part 1 — reviewed repair pass

Alec authorized this mechanism landing after one repair pass and one full
sweep, under [plan §1.6](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#16-hand-off-to-codex).
This is part 1, not the closing of item 6. The complete
[identity-from-data amendment](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#2-identity-from-data-amendment-alec-2026-10-10)
defines part 2: corpus and measurements before mechanism changes, with each
identity rule retired only after its lesson passes without it.

## Repairs

- Reconstruction audits missing and excess leaves after the free walk. Counts
  never choose inverse actions or STOP. Dropped support marks the reconstruction
  incomplete and contributes a separate `reconstruction.coverage` term to the
  actual keep cost when free-byte reconstruction is reporting-only. Both exact
  and within-tolerance projections have regressions.
- Free candidate search rejects self-reproducing pairs before ranking. An
  unspelled whole can otherwise choose itself twice, then fail the progress
  test and hide an available lexical split. Search and lexical STOP retain
  separate masks. The cleared-cache word-drivability assertion is unchanged.
- The walk owns independent copies of every explicit captured tensor bank,
  mask, initial depth and case-bank field. Overlapping input views therefore
  cannot alias across `torch.while_loop` inputs. The unnamed-concept regression
  explicitly enables eager capture, regardless of a previous test's backend.
- Families describe candidate types, without S/VP/adverb side masks. Every
  family is admitted on either side. A noncommutative recomposition regression
  exercises all 16 family pairs and both orientations, eager and compiled.
- Sparse encoding greedily adds a column only while it reduces the remainder.
  Admission normalizes the dictionary-unexplained part of the prediction
  residual (the observed row when no prediction exists). Witnessing still reads
  the whole row. The recurrence gate and existing admission modes remain;
  retiring modes waits for part 2. A known column re-witnessed in a new sentence
  seeds no pending prototype. A correlated second signature now correctly uses
  the known column plus its newly admitted remainder.
- `LanguageSpace.reverse_inverses` and obsolete test stubs are deleted. The
  materialization fixture now requires one supported word, exact text and no
  truncation for a representable resolved answer.

The diagnostic cleared-cache bank retained the spellings `hello` and `world`;
the empty read did not arise from clearing those spellings. The self-pair
regression exercises the obstruction directly. The original cleared-cache
failure was **new with the candidate relative to accepted 6.1**; the earlier
receipt's characterization as unchanged was incorrect.

## Fixed source and affected checks

The [765-file manifest](source.json), [source metadata](source-metadata.json),
[source archive](source.tar.gz) and [candidate patch](candidate.patch) identify
the repaired source. Archive SHA-256:
`e36f183876e98952b7aad544f9a717e8bc2d8e367e039aafbb9e2498d925bdc4`.

The affected group passes **102**, skips **2**, and retains **1 expected failure**
([log](validation/affected-final.log.gz)). Its materialization fixture was then
strengthened and passes its own [check](validation/materialization.log.gz).
The isolated default-backend probes pass **2** with `MODEL_COMPILE` unset
([log](validation/default-backend.log.gz)); one explicitly enables the default
capture backend to prevent test-order leakage. Both explicit compiled checks
pass ([log](validation/compiled.log.gz)), including packed retention. That pass
does not explain the earlier intermittent compiled mismatch.

Development diagnostics remain under `validation/`, including the original
default-backend alias failure and the pre-repair empty cache read. Intermediate
affected logs are development snapshots, not final-source full sweeps. The
corrected greedy-support fixture uses a relevance scale that actually increases
the remainder; its first attempted scale still improved it.

## Recovery measurement

The [gates](gates/results.json) use one unseeded initialization, no retries,
the existing predeclared fit/plateau/budget conditions and frozen forward
artifacts. Only the generate policy is trained. Missing/excess support is
audited after decoding with the same function as production reconstruction;
the raw walk's termination is retained as `walk_complete`, separately from
audited `complete` and `coverage_cost`.

The lift policy fits at update **3,982**, CE **.0009970585**. Lower plateaus at
update **6,688**, CE **.4180057049**. Lower at **2/3/5** emits its surviving head
and is **incomplete**, with support costs **.5 / .666667 / .8**, in both contexts.
It is never reported complete-wrong. These costs do not recover the discarded
modifiers; that ambiguity remains visible. Full gate details are in the JSON.

| Operator | Words | Candidates | Forced raw | Forced bounded | Audited free |
|---|---:|---|---|---|---|
| lift / lower | 1 | either | exact | exact | exact |
| lift | 2 | either | wrong | exact | exact |
| lift | 3, 5 | stored dictionary | wrong | wrong | incomplete, no leaves |
| lift | 3, 5 | live wholes | wrong | exact | exact |
| lower | 2 | either | wrong | exact | incomplete, one leaf |
| lower | 3, 5 | either | wrong | wrong | incomplete, one leaf |

Exact matches for a lossy or commutative operator do not establish uniquely
recovered orientation: numerical ties can retain bank order. The A/B columns
are untrained in this probe. The forced two-word lower match is such a tie,
not evidence that the discarded marker was identified.

## Full sweep against accepted 6.1

The single source-matched bounded sweep completed all **5,745** cases:
**5,454 passed, 287 skipped, 3 failed,
1 expected failure**, exit 1, in 609.22 seconds.
Peak aggregate memory was 8.56 GiB; limits were 28 GiB aggregate and 8 GiB per worker.
The [summary](full/summary.json), [failure messages](full/failures.json),
[worker archive](full/worker-receipts.tar.gz) and
[baseline comparison](full/baseline-comparison.json) retain the result.
No failed initialization was retried and no assertion was relaxed.

Accepted 6.1 (`7c9aeac67`, mechanism `4d9d9ca8a`) completed **5,719** cases:
**5,428 passed, 286 skipped, 4 failed, 1 non-strict XPASS**. Its four failure
nodes compare as follows; a passing current initialization does not establish
a fix for an intermittent historical failure.

| Accepted 6.1 failure | Part-1 sweep |
|---|---|
| `test_expectation_off_keeps_every_packed_observation_in_ltm` | passed |
| `test_eight_corpus_sentences_train_at_distinct_fork_rounds` | passed |
| `test_forced_ordinary_answer_fills_committed_question_without_its_own_episode` | failed |
| `test_forced_ordinary_pending_premise_in_both_orders[False]` | passed |

**New failure outcomes relative to accepted 6.1:**

- `test/test_math_chain_section14.py::test_ordinary_initial_binding_distribution_includes_every_retained_candidate`.
- `test/test_reconstruction_scope.py::test_missing_packed_sentence_does_not_dilute_the_owned_reconstruction`.

The new coverage-scoping failure observes **1.0** where the unchanged
assertion requires **0.0** for a packed sentence with no available candidates.
The support charge sits outside the existing candidate-availability mask;
it remains open alongside item 6. b reaches **1.101915** against the unchanged
**1.1** upper bound. That outcome is new relative to the accepted sweep, which
passed b, but its flakiness was already documented (earlier maximum 1.153045).
Neither classification claims a new causal diagnosis for the initialization-sensitive b test.

The cleared-cache word-drivability regression is **passed**. Its earlier empty
inverse was **new with the candidate** and has been repaired in this measured
source. All **19** stored-idea regressions and both new component regressions
pass in this sweep, including the explicitly isolated default-backend test.

The existing word-overlap marker changes from accepted 6.1's non-strict
**XPASS** to **XFAIL** here: **0.000** versus the unchanged **.800** bar.
That adverse outcome remains in the [expected-failure report](full/expected-failures.json).
It is not counted as a new strict failure or hidden by the repaired drivability check.

The requested single sweep and its frozen source are retained; no post-sweep
production or test change is represented as tested. The weekly slow campaign
was not rerun; the runner reports its existing age warning.

## Open work carried to part 2 and alongside item 6

Stored-idea recovery beyond two words requires actual unmixing through trained
A/B dictionaries. Order-lesson CE remains constant with zero scorer gradient.
MM grammar recovery is blocked before evaluation by the 1,024-row LTM capacity.
The intermittent compiled-retention mismatch has no established cause. The
earlier curriculum and MM runs belong to their preserved sources; they were
not rerun or relabelled as results of this repair pass. Keep b's unseeded bound,
c, fork diversity and the accepted 6.1 retention findings visible. Item 5 must
not drop derivations on this evidence. No identity rule is retired in part 1.


## Reproduction and landing

Mechanism commit: `324b84d67ffe85e31861a440d088290ee99abc9b` ([landing record](acceptance.json)).

The accepted repair decision is in plan §1.6; it authorizes commit/push and
WikiOracle's bump directly after this measured pass. The new coverage failure,
b's failed bound and the carried correctness/learning findings remain open;
this is not acceptance of item 6 as a whole.

Commands (repository root; CPU):

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python -m pytest -q test/test_stored_idea_generation.py test/test_independent_components.py test/test_stm_recon_from_cleared_cache.py test/test_native_attention_field.py test/test_free_reconstruction.py test/test_output_walk.py test/test_sentence_compose.py
env -u MODEL_COMPILE BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest -q test/test_stored_idea_generation.py::test_unnamed_primed_concept_cannot_complete_as_a_word test/test_stored_idea_generation.py::test_unspelled_idempotent_whole_cannot_hide_its_lexical_inverse
RUN_SLOW=1 BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python -m pytest -q 'test/test_sentence_compose.py::test_real_packed_ends_train_before_the_next_sentence[True]' test/test_query_phase_fullgraph.py::test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python test/stored_idea_gates.py output/item6-part1/gates
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python test/test_report.py --workers 4 --run-dir output/item6-part1-full
```

The two explicit integration checks use fullgraph eager capture; the structural
candidate/orientation tests also exercise the Inductor backend. These are
separate from the default-backend isolation probe. All **362** documentation-link
checks pass ([log](validation/documentation.log.gz)); staged source and archive
blobs match all 765 entries in the measurement manifest. The strengthened lexical
materialization case is included in the frozen full sweep. The full receipt's
repair-check rows are the final-source outcomes for all new regressions.

Implementation and documentation whitespace checks pass. Raw patch context lines
and the original curriculum XML retain their whitespace as measurement evidence;
they are not normalized to satisfy a source-style check. The saved candidate
patch passes the reverse-application check.
