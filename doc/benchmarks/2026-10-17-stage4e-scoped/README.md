# Lesson-scoped nested admission, then stage 4e

Starts from landed main `552c626ae`. **Nested-admission certificate passes;
the single stage 4e run fails all four gates.** Learning stops here. No
clock, chain, surface or stage-3 change; no identity run or learning claim.

Nested admission defaults off. `WordIdentity.nested_lesson(marker)` permits
one named suffix at a time; a word and its remaining stem must both have
been witnessed. No shared-substring search or token-count admission remains.
The lesson permission is transient; admitted identity parts persist.
The default certificate restores the stage-3 artifact below: the 17 4e
words and walk/walked/walking/walks/cat/cats/went remain whole without a
pattern. With `ed` named, only `walked` gains `walk` + `ed`, with native
sub-word spans and whole-word attention. The focused certificate passes.

**Artifacts:** restores `output/stage4e-student-order-20261017/order.ckpt`
(SHA-256 `e19d84e8e708…`); produces
`output/stage4e-scoped-20261017/trained.ckpt` (`19b8d791398f…`). The restored
file is unchanged. The protocol records full artifact and corpus hashes;
the 783-file source manifest matches the completed run. Stage 3 is neither
retrained nor re-evaluated.

**Declared run:** seed 20261010, LR .003, B4, two passes over the same 72
training sentences: **144 presentations, 72 paired updates**, no batch-end
update (two attempts were blocked). Each pass follows descriptions → adjective NPs → intransitive →
transitive → prepositional, shuffling the unchanged counterbalanced cases
within each family. No nested pattern is named. Transpose and unary faces
remain legal and receive no lesson targets. The same 40 held-out sentences
were read once at the end; held-out labels went only to the grader. All
40/40 kept held-out readings preserve source order. Elapsed: 2,084.58 s.

| Held-out gate | Fixed requirement | Result |
| --- | --- | --- |
| Exact main brackets | ≥90% | **Fail: 1/32 (3.125%)** |
| Singular-NP order-1 occurrences | ≥95%; zero duplicates, collisions or unexpected occurrences | **Fail: 4/48 (8.333%); 0 duplicates, 0 collisions, 30 unexpected** |
| Lowered NPs | Every NP has its own order-1 occurrence | **Fail: 1/48** |
| Unlowered descriptions | No occurrence for all eight controls | **Fail: 3/8 clean** |

Trajectory, eight batches per block (final remainder four). A lesson call is
one row in one native trial. CE uses observed pre-update probabilities on
reachable student states, with reused prefixes counted per call. The visited
states and family mix change; these numbers are not a matched-state learning
claim. No fold target was unreachable.

| Batches | Calls | Fold targets/call | Wait CE (reachable; unreachable) | Fold CE (reachable) |
| --- | ---: | ---: | ---: | ---: |
| 1–8 | 64 | 104/64 = 1.625 | 2.583 (973; 7) | 3.598 (104) |
| 9–16 | 64 | 111/64 = 1.734 | 1.626 (518; 31) | 2.733 (111) |
| 17–24 | 64 | 62/64 = .969 | 1.354 (370; 18) | 2.354 (62) |
| 25–32 | 64 | 75/64 = 1.172 | 1.218 (477; 31) | 2.294 (75) |
| 33–36 | 32 | 84/32 = 2.625 | 1.222 (280; 10) | 2.413 (84) |

First source-pair accuracy on kept training readings, correct/total. A dash
means no presentation in that block. Missing first pairs count as incorrect:
five descriptions, three adjective NPs and one transitive overall.

| Batches | Descriptions | Adjective NPs | Intransitive | Transitive | Preposition |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–8 | 0/8 | 0/16 | 0/8 | — | — |
| 9–16 | — | — | 0/8 | 2/16 | 2/8 |
| 17–24 | 1/8 | 0/16 | — | — | 3/8 |
| 25–32 | — | — | 0/16 | 3/16 | — |
| 33–36 | — | — | — | — | 8/16 |

Kept-program step totals. Every binary, unary and transpose step pays
`WHAT_STEP_COST` in the keep comparison; unary faces and transpose stayed
legal but received no lesson targets.

| Batches | Sentences | Leaf | Binary | Unary | Transpose | All steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–8 | 32 | 96 | 64 | 386 | 0 | 546 |
| 9–16 | 32 | 160 | 128 | 0 | 0 | 288 |
| 17–24 | 32 | 112 | 80 | 0 | 0 | 192 |
| 25–32 | 32 | 144 | 112 | 0 | 0 | 256 |
| 33–36 | 16 | 96 | 80 | 0 | 0 | 176 |
| Total | 144 | 608 | 464 | 386 | 0 | 1,458 |

All 386 unary steps occur in the first 32 presentations; subsequent kept
programs contain none.

Records: `output/stage4e-scoped-20261017/{protocol,results,trajectory,
trajectory-batches,kept-programs,protocol-verification}.json`. The final
audit confirms each training ID once per pass in family order, all held-out
IDs once after training, and unchanged corpus/artifact/source hashes.

Focused checks: nested certificate **1 passed**; lesson/trajectory checks
**6 passed, 2 existing skips**. The single final-source default sweep:
**5,768 cases — 5,431 passed, 335 skipped, one XFAIL, one carried c failure**
(1,274.63 s). All eight mechanism certificates and the trajectory check pass.
c still closes the question but misses the expected `five` referent in
`test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`.
The paired-trial check passes; its intermittent `.0517` finding stays carried.
Sweep and verification: `output/stage4e-scoped-default-20261017/`.

Claude approved the source before the run and the measured result/receipt
before landing, with no source or interpretation blocker. Both final
conditions are met: no new sweep failure, and all 783 source hashes match
the run, sweep and current source. Both checkpoint hashes still match.
Reviews: `output/stage4e-scoped-review-20261017/` and
`output/stage4e-scoped-landing-review-20261017/`.

Disposition: nested-admission fix and preparation; 4e remains failed. No
budget extension, seed selection, automatic rerun or identity run followed.
The enclosing commit is the archive.
