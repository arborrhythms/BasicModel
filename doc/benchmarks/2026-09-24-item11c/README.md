# Item 11c: located concept composition and native perception

This landing implements the [decided plan](../../plans/2026-09-24-item-11c.md)
and the September 24 working-tree review. Conceptual requirements take min
over nonzero contributions separately on each pole; alternatives and extent
readout use max. Negation swaps poles. Native properties use max over allowed
primitive memberships and min pervasion over observed positions. Absence
contributes no counterevidence.

Pi edges exist only within the order-0 field. Higher orders union symbols of
the previous order; symbolization raises order. Located definitions retain
field brackets until the joint witness matches. Descent selects cases by
intersecting with an observed field, or chooses a case without one, then
attributes native memberships.

## Review corrections

- The XOR target is a learned row's positive pole. The observer sees all four
  unlabelled primitive cases; their pi definitions match located positive and
  negative literals before readout. A zero-initialized order-1 sigma row learns
  the two mixed cases. Pool sizes four and eight each run three times without
  seed selection. Initial error must be nonzero, and the tests inspect that
  every pi edge is at order 0. The pooled both corner is diagnostic only.
- Sentence-boundary witnessing writes admitted present features at each active
  concept pole. Partial observations update literals in place and cannot add
  sufficient word alternatives. Absent features are never written. Zero
  candidates retain their brackets, do not veto matches, and cannot skip an
  order. The parameter getter stays pure; normalization applies only to
  assigned provisional disjunctive rows.
- A bound field can contain unwritten inventory addresses beyond the feature
  matrix's current extent without an out-of-bounds gather. Concept ids remain
  persistent through transient field bindings.
- Pass-back scales content by `0.5 + 0.5*attribution`. Novel content keeps
  half strength; attributed content approaches full strength. Location bands
  and the retained reconstruction carrier are unchanged. Both attention
  producers write the normalized scope owned by the shared conceptual field.
- Native perceptual folds are deleted. `subsymbolicLoop` selects attention
  passes within the retained loop bound. Word codes are max over constituents:
  anagrams share a code, while ordered containment and the retained
  reconstruction witness distinguish them.
- The lexicon indexes a form to a set of concept ids across orders. Selected
  grammar roles request an event, particular/name/pronoun, kind, or numeric
  order. Missing or ambiguous references remain unknown. The captured program
  owns the resolved semantic ids alongside the original reconstruction
  provenance, and checkpoints preserve the form index.

## Failing probes

The [initial affected diagnostic](red-result.json.gz),
[workers](red-workers.log), and [source map](red-source-manifest.json)
preserve the obsolete fold assertions and review failures. That diagnostic
was interrupted by subsequent source edits and is not a validation receipt.
Focused probes separately recorded the [located-match, focus and gather
failures](probes/review-red.log), [unimplemented lexical resolution](probes/lexical-orders-red.log),
[zero-candidate and order-boundary failures](probes/boundary-candidates-red.log),
and [primed scope producer mismatch](probes/primed-scope-red.log).
The [fixed boundary probes](probes/review-boundaries-fixed.log) pass 236
cases with three existing skips; [rewritten primitive probes](probes/primitive-contract-fixed2.log)
pass all 14. The final source-matched receipts below supersede these local
intermediate checks. The program-reference correction also keeps unresolved
identities implicit while preserving owned explicit resolutions; its
[nested and surface probes](probes/program-reference-ownership-fixed.log)
pass all 16 cases.

The complete [discovery run](full-discovery-result.json.gz) preserved
13 failures among 4,815 cases on its [source snapshot](full-discovery-source-manifest.json).
The [failure logs](full-discovery-failures.log) identify retired fold geometry,
old probabilistic-union values, mixed-order chain assumptions, a compile fixture
that still expected folds, and state-dependent chronology/question checks.
The chronology regression now supplies distinct observations. The dependent
question check was redundant with the existing causal training gate and
incorrectly depended on that earlier slow test having run. The latter gate
is unchanged. The stronger [self-reference probe](probes/self-reference-partial-write-red.log)
then exposed a partial write before rejection; identity validation now precedes
all sparse mutation. These are failing probes, not the final receipt below.

An extended all-slow diagnostic also re-exercised the already-open item 8
`XOR_grammar.xml` W=6 configuration failures, the missing external enwiki
word-vector fixture, and corpus-learning workloads that exceeded the 8 GiB
worker cap. Those are not capability passes. The final complete selection
uses the existing default slow-test policy; the 11c slow gates have their
own explicit receipt. No skip or expected-failure rule was added.

## Serial reconstruction and priors

The [preserved driver](run_measurements.py) compares `MM_ladder.xml` with
conceptual pi off and on, then freezes primitive memberships for an ablation.
Seed 42 belongs to this measurement only. All four guarded probes exit zero;
[comparison](final-source/comparison.json) records the results without a
tolerance gate:

| Reconstruction mean | `d4dc385` | 11c off / on / frozen |
| --- | ---: | ---: |
| Before training | 0.1005932558 | 0.1006144527 |
| Five measured updates | 0.0927930698 | 0.0879226878 |
| After training | 0.0923267286 | 0.0891618710 |

The after-training mean is 3.43% lower on this fixed measurement. The three
current runs agree. The [prior comparison](final-source/priors.json) keeps
all eight rows over 256 bytes at zero error and all 32 sentences / 160
constant-signature runs unchanged. The [measurement manifest](final-source/manifest.json)
records 644 validated files, unchanged during all probes, with SHA-256 of
the sorted source-map JSON:
`78d4d76d694e3c17a7cc3203a8a45e296880a3d071ed632ff65bc3653fcfb077`.

## Validation

The single final [full receipt](full-result.json.gz) completes **4,812 cases: 4,485 passed, 326 existing skips, and 1 existing expected failure**,
exit zero. Its [workers](full-workers.log), [summary](full-summary.json), and
[source map](full-source-manifest.json) preserve the complete default selection.
The bounded run takes 1101.2 seconds and peaks at
11.87 GiB aggregate memory.

The final [affected correction receipt](affected-result.json.gz) completes
**98 cases: 80 passed and 18 existing skips**, exit zero. It covers all nine files corrected after the
discovery run; the full receipt covers the complete implementation and those
corrections. The [workers](affected-workers.log), [summary](affected-summary.json),
and [source map](affected-source-manifest.json) are preserved.

The [explicit slow receipt](explicit-result.json.gz) passes **19/19**:
the relevant smoke, priming, focus, native-geometry and frozen-definition checks
that the full suite's existing default policy skips. Its
[workers](explicit-workers.log), [summary](explicit-summary.json), and
[source map](explicit-source-manifest.json) match the full run and measurements.
No skip or expected-failure rule was added.

All six [unseeded XOR runs](xor-full.json) in the final full receipt improve
from MSE **0.5 to 0**, reading `[0,1,1,0]` at the learned row's positive pole.
Unrelated controls are exactly zero at every scope. The restored
[raw smoke](smoke.log) reads all four present words as `(1,0)` at extent and
symbol scope and asserts their retained position and carrier poles.

All three final test receipts and the reconstruction/priors probes match the
same 644-file validated source map byte for byte. Source
hashes are checked again against the implementation commit before publication.
