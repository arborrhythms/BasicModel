# Item 6 — preserved pre-review generativity candidate

This is the preserved pre-review candidate receipt. Its measurements and
source archives remain unchanged. The review accepted a **part-1 mechanism
landing after repairs**, not item 6's closing; the
[repair-pass receipt](part1/README.md) records that landing and compares it
with accepted 6.1. The cleared-cache empty inverse below was **new with this
candidate**. The earlier description of all three failures as unchanged was
incorrect. Item 5 must not discard derivations on these probes.

## Implemented boundary

`Generative.inverse_menu` now owns the same bounded forward-kernel candidate
search for the tensor output walk and `MemoryIndex.unfold_idea`. The latter
previously passed an unmasked two-dimensional word bank and therefore used
the affine/reference-free fallback instead of the intended candidate search.
Every binary candidate is checked against its actual forward operation.

Words are terminal candidates. The production priming bank can also contain
unnamed concepts, so lexical STOP uses `PrimedSymbols.terminal_valid`: the row
must have an actual spelling. A new regression demonstrates the former false
completion on an unnamed primed concept and verifies explicit incompleteness
after the repair. A second [failing regression](validation/terminal-search-failing.log.gz)
verifies that an unnamed intermediate
still participates in compound decomposition: the search mask and lexical
STOP mask are separate. Learned A
columns, B columns and other native higher-order property concepts form
separate nonterminal families. The S
inverse searches noun/property and verb families in its respective roles;
VP searches verb and noun/property, and adverb modification searches verb
and property. Words retain learned grammatical uses, rather than receiving
a lexical POS table. Existing priming and the reconstruction candidate bound
apply; staging neither allocates columns nor scans LTM episodes.
An unnamed copy in the primed bank or live STM inherits its native family's
constraints. A deterministic reversed-bank probe exposed a bypass through
those copies; the [failing probe](validation/type-alias-probe-failing.log.gz)
and [16 passing mechanism checks](validation/type-alias-affected.log.gz),
including eager and fullgraph compiled alias cases, are retained.

During a sentence, a detached, STM-sized set of composed forward values is
also available. It contains no source positions, actions, child pointers or
desired word sequence. Each trial owns its snapshot, which ends with the
invocation. Stored reads receive native type structure only. There is no new
stored derivation or input trace. Family availability is a mechanism, not
evidence that the dictionaries have learned sufficient phrase structure.

Binary completion now requires recomposition within the existing relative
`1e-4` numerical tolerance, as well as supported, progressing children. This
prevents the previously observed false success in which a three-word root
decoded to its final two words and was marked complete. The bound applies to
the whole inverse, not only its emitted words. Unavailable structure remains
an explicit incomplete result. This is a free-decoder completion contract.
When the sentence field is active, `reconstruction.free_bytes` remains a
reporting term; the actual derivation field supplies the keep cost and leaves
undefined inverse coordinates uncharged. This change therefore does not yet
prove that every dropped-phrase reading receives a reconstruction charge.

The cued-read taxonomy adapter also rejects non-taxonomic `meta` references
before they reach `Taxonomy.concept_reference`. A regression test preserves
the actual exception reported during 6.1's list curriculum.

Stored unfolding restores the native operators' diagnostic activations after
candidate evaluation. The preceding full sweep exposed a lift activation
escaping into an ordinary episode's state footprint. A
[deterministic native probe](validation/inverse-read-state-probe.log.gz)
reproduces that mutation while recovering both words correctly. The scoped
repair preserves the same recovery and passes
[42 affected checks](validation/inverse-read-state-affected.log.gz), including
the native regression, stored-index cases and the ordinary fork certificate.

## Determiners, order and identity

Alec's October 10 clarification is implemented at the stored-idea boundary:
lowering may leave the numerical description unchanged while changing its
order. An order mismatch can therefore require a determiner. Increasing the
description's order by one is inverse progress, so this expansion does not
need a fabricated numerical residual and cannot repeat at the same stamp.

Binding proposals participate in compose choices; only the selected closing
commits identities. A stored filled-referent fact licenses the definite
operation. Other records leave the choice to the learned grammar. Identity
alone does not determine the original article: familiarity also depends on
the generation context. Formation probabilities and recorded operation
choices are not generator inputs.

The complete grammar now declares both mint and definite inverses, sharing
the existing lower kernel. Their policy keys distinguish semantic binding
modes. The original mint key survives, and checkpoint migration adds the
definite row without moving learned existing rows. Candidate marker words
are scored by the ordinary compose chooser on the proposed marker/description
pair. No English word-to-operation map is introduced. The lexical tensor
walk without order/reference context still cannot reconstruct a discarded
determiner from a bare numerical point.

## Fixed-artifact recovery measurement

[Protocol and raw results](gates/results.json) use one unseeded initialization,
no retries, fixed native word/forward/operator artifacts, chains of 1, 2, 3
and 5 words, and a 32-node inverse allowance. Only the generate policy is
trained. Its predeclared stop conditions are a fitted policy (CE at most
`.001`, accuracy 1, five consecutive updates), a loss plateau (range at most
`1e-6` over 200 updates after update 1,000), or the 10,000-update limit.
There is no recovery pass threshold and no assertion is made true by a seed.

The lift policy fits at update **3,983**, CE **.000997409**. For chains of
length 2/3/5, the raw forced inverse is wrong in all three cases. The bounded
forced inverse and free policy both recover **all three exactly** when live
forward wholes are supplied. With only the word dictionary, both recover
length 2. At lengths 3 and 5, the forced inverse is wrong and free decoding
is **incomplete**, with no emitted codes. This is the missing-structure
control, not a test of a trained A/B dictionary. These ordered matches do
not establish unique surface order: the search retains dictionary order
to break equal-residual ties.

The bare-point lower policy reaches a **loss plateau**, CE **.418005705**, at
update **6,697**, rather than fitting contradictory split/STOP labels for the
same projected point. Free decoding emits the final word in every compound
case. The two-word forced bounded result is exact in this bank order, but
all discarded-marker candidates have the same forward value; it does not
establish identifiability. The contextual order/identity tests are separate
mechanism checks, not a reinterpretation of these nulls.

The word codes, forward trees, initial inverse/operator state and initial/final
policy tensors are retained in `gates/forward-artifacts.pt`. Its metadata
identifies the larger original model artifact by path and SHA-256; the full curves retain every update. The earlier
development campaign, before the whole-inverse completion check, remains in
`development/` and is not a final-source result.

## Validation and known limits

The [765-file source manifest](source.json), [source archive](source.tar.gz),
and [candidate patch](candidate.patch) identify the final code and tests.
Archive SHA-256:
`4d398c0203c9a48e321c7f3c3ba45fac59f2aabc26d5e299f8d44e561e9615e2`.
The [final source-matched full sweep](full/summary.json) completes all
**5,737 cases: 5,446 passed, 287 skipped, one expected failure and three
failures**, in 622 seconds (peak aggregate memory 7.67 GiB). The
failures are:

- The eight-sentence fork certificate selects only departure round **12**.
- c closes the question's open slot but its retained answer does **not**
  match `five`.
- **New with the candidate:** the cleared-cache owned inverse returns **no indexed word**.

The [failure messages](full/failures.json),
[worker receipts](full/worker-receipts.tar.gz), and complete result are
retained. The existing [word-overlap expected failure](full/expected-failures.json)
still observes **0.000** against its unchanged **.800** threshold. No failure
was removed, skipped, rerun to select a passing initialization, or relaxed.
The final source and archive match all 765 validated files.

Both explicit
[compiled checks pass on this candidate](validation/review-candidate-compiled.log.gz)
(79.98 seconds). The isolated compiled source matches this manifest.
The [mechanism validation record](validation/mechanisms.json) identifies the
42 affected checks, including all 17 new stored-generation cases, and both
compiled commands and their source.
On the source before the read-state repair, both explicit
[compiled checks pass](validation/ready-compiled.log.gz): the packed-sentence
training test and the query fullgraph forward/backward test (85.75 seconds).
The fixed-artifact
measurement above uses this same source. Earlier long measurements retain
their own archives and [source bridge](source-bridge.json).

Every complete development sweep is retained:

| Source stage | Selected / completed | Passed | Skipped | Expected failure | Failed |
|---|---:|---:|---:|---:|---:|
| [Initial](development/initial-full/summary.json) | 5,731 / 5,731 | 5,350 | 287 | 1 | 93 |
| [Cold-priming guard and fixture ports](development/corrected-full/summary.json) | 5,732 / 5,732 | 5,437 | 292 | 1 | 2 |
| [Expanded b coverage](development/pre-terminal-full/summary.json) | 5,732 / 5,732 | 5,442 | 287 | 1 | 2 |
| [First lexical mask](development/terminal-filter-full/summary.json) | 5,733 / 5,733 | 5,442 | 287 | 1 | 3 |
| [Separate search mask and type aliases](development/pre-read-state-full/summary.json) | 5,736 / 5,736 | 5,445 | 287 | 1 | 3 |

The first lexical mask also filtered unnamed compounds out of the search.
Its unchanged cleared-cache drivability assertion failed. Separate search
and STOP masks restore the missing intermediate candidates, but the later
full sweep still produces an empty owned word surface at that assertion.
Its targeted pass does not close this failure. The first lexical-mask sweep
also failed b and c. The
next [sweep was interrupted](development/pre-alias-interrupted-full/summary.json)
after 3,948 of 5,734 cases when the type-alias bypass was demonstrated. It is
explicitly incomplete, with its source and worker receipts retained; it is
not counted as a full receipt.

The separate-mask/type-alias full sweep has three failures: b's initial
distribution reaches **1.153045** times uniform; the cleared-cache inverse
returns no word; and the fork test detects the native lift activation
mutation repaired above. It passes c in that initialization. Its complete
[failure messages](development/pre-read-state-full/failures.json) are
retained; passing c once does not close its historical correctness finding.

The preceding expanded-b sweep failed the unforced fork-diversity
certificate and c's supplied-answer closing. c left the question's referent
open; the fork certificate selected no compose departures in its eight rows.
Its [failures](development/pre-terminal-full/failures.json) and
[worker receipts](development/pre-terminal-full/worker-receipts.tar.gz) remain.
The existing [cleared-cache wording expected failure](development/pre-terminal-full/expected-failures.json)
observed 0.000 word overlap against the unchanged .800 threshold; its marker
was not changed.

The first [full development sweep](development/initial-full/summary.json)
completed all 5,731 cases: 5,350 passed, 287 skipped, one expected failure and
93 failures. Of these, 89 were the same new cold-priming error: definitions
can be indexed before a priming tensor exists. The adapter now returns an
explicit incomplete read in that state. A runtime ingestion failure followed
an interrupted provisioning in that worker; its complete integration
selection passes after the guard. The other three were c's supplied-answer
failure and the two completion assumptions discussed below. All original
failures and their exact source are retained.

Two existing mechanism fixtures were changed explicitly:

- `test_public_reading_fuses_the_reference_before_capture_and_write` still
  checks the fused reference, retained reading and gradients. Its arbitrary
  unnamed prior referent has no lexical realization; it now expects an
  incomplete lexical reconstruction instead of accepting an approximate pair
  as complete.
- `test_answer_materialises_as_its_own_conceptual_idea_and_realises_through_the_walk`
  passes the record's structural candidates and checks finite output with
  either an emitted word or explicit pending work. A fresh off-dictionary
  answer root does not guarantee a word. Its materialization assertions stay
  intact. This is not a trained wording claim.

The echoic-cost stub now accepts the decoder's optional context keywords;
its assertions are unchanged. The curriculum measurement helper now forwards
the record's structural candidates and terminal mask, matching production
decoding. The completed curriculum below predates that helper port.

The [corrected full sweep](development/corrected-full/summary.json) completes
all **5,732** cases: **5,437 passed, 292 skipped, one expected failure and two
failures** (c and fork diversity), in 669 seconds. This isolated copy lacked
the existing `sentence.pt` fixture, so five embedding probes skipped; that
read-only fixture is restored for the final sweep, with its digest recorded
in [the fixture record](validation/embedding-fixture.json). No new skip marker
was introduced. The [source bridge](source-bridge.json) records subsequent changes: b's
expanded test, separate lexical and search masks, native type-alias constraints,
and their regressions. Earlier
measurements are not relabelled as final-source measurements.

The native determiner/catalogue/checkpoint and index selection passes **26
checks**. Both explicit real compiled checks pass on the preceding production source;
its [log](validation/corrected-compiled.log.gz) is retained. After the terminal
repair the [affected selection](validation/terminal-affected-final.log.gz)
passes **72 tests**, with one existing skip. The new regression previously
[failed](validation/terminal-probe-failing.log.gz). The first lexical-mask
[compiled run](validation/terminal-compiled.log.gz) has **one pass and one
failure**: the packed-sentence test executes its compiled path, then its
stored-meaning comparison differs in 64 of 312 elements (maximum absolute
difference .255649). The query fullgraph forward/backward test passes. This
retention failure remains recorded. After the separate search-mask repair,
both explicit compiled checks pass in the
[pre-alias run](validation/pre-alias-final-source-compiled.log.gz), and its
[affected selection](validation/pre-alias-terminal-search-affected-final.log.gz)
passes 59 tests with one skip. These subsequent results do not establish
the cause of the earlier retention failure. The
50-check earlier affected selection is supporting development evidence, not
the final full receipt.

On the preceding production source, the [unchanged MM grammar recovery assertion](validation/mm-grammar/summary.json)
fails before evaluation: **LTM capacity 1,024 is exhausted during the second
training epoch**, after 608 seconds. Neither its input, epochs, assertion nor
capacity was changed. This is a capacity-blocked measurement, not a new 0/4
score or a successful recovery. Its historical 0/4 and current missing
remeasurement remain visible; forgetting is outside this change.

### Completed list curriculum on the initial source

Both 64-presentation modes complete through stage 3 with a
[matching frozen source](curriculum/source_check.json). Identity permanence
and identifying words pass in both modes. All four lists select the intended
read order, but the enabled mode fails its unchanged lesson CE threshold
(`.01`): final CE is `.346574` for two words and `.794513` for four.

| Reconstruction mode | List length | Exact derivation readback | Exact free readback | Stage passes |
|---|---:|---:|---:|---|
| Disabled | 2 | 4/4 | 0/4 | Yes |
| Disabled | 4 | 2/4 | 0/4 | Yes |
| Enabled | 2 | 2/4 | 0/4 | No: lesson CE |
| Enabled | 4 | 0/4 | 0/4 | No: lesson CE |

The [summary](curriculum/summary.json), per-mode reports and complete curves
retain every outcome. No native-reference retrieval exception recurs. This
teacher-forced reading protocol attempts and keeps zero departures; it is
not a demonstration of compose exploration. It used the initial candidate,
before the cold-priming guard, terminal-mask, type-alias, stored-read state and measurement-helper
repairs, as the source bridge records. These scores are not relabelled as
measurements of the final candidate. Free list recovery remains an active
inverse limitation.

### Carried correctness findings

The expanded b fixture uses **32 documents**, history lengths 1–4 and four
variable names at each length. One unseeded development run observed **6,044
menus**, including **4,788 with retained candidates**, and menu sizes 2, 4,
5, 6, 7, 8 and 9. The unchanged .9–1.1 conditional probability bound fails
at **.937935–1.125857** times uniform. The
[report](diagnostics/b/distribution.json), [menus](diagnostics/b/menus.json.gz)
and [failure](validation/b-coverage-verified.log.gz) are preserved. More
coverage exposed the defect; no seed or band changed. The preceding full sweep's
fresh initialization passes at **.952373–1.040006** across **5,160 menus**;
its [report](diagnostics/b/pre-terminal-distribution.json) and
[menus](diagnostics/b/pre-terminal-menus.json.gz) are also retained. That pass is not
an initialization fix and does not erase the failed development run.
The final source's b run passes at **.972679–1.061050** across **6,072 menus**;
its [report](diagnostics/b/final-distribution.json),
[menus](diagnostics/b/final-menus.json.gz) and
[origin](diagnostics/b/final-origin.json) are retained. The intervening
pre-read-state full sweep failed at **1.153045** times uniform. b therefore
remains an initialization correctness finding despite the final pass.

The retained c diagnostic records the answer's read order as
`. is five answer the`. The forced adjacency grammar then constructs a
binding to an unnamed minted row, whose graph has no lexical `five`
association. The [binding graph](diagnostics/c/binding.json.gz) and
[compose choices](diagnostics/c/forced.json.gz) preserve that failure.
Neither the answer comparison nor the read order was changed to conceal it.
The unforced eight-sentence diversity certificate also fails in the corrected
sweep when the selected compose departures all use round 24. Its native
sampling and assertion are unchanged.

Passing a fresh initialization does not close those historical failures.
Packed retention and pending-premise retention pass in both corrected full
sweeps. No targeted repair of those historical failures is claimed. These remain correctness findings, not
million-sentence learning gates. No derivations may be dropped on the
strength of this candidate.
