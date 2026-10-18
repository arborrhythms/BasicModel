# Distributional parts of speech and the NP prerequisites

Starts from main `85b39a4c2`, under [the specification](../../specs/2026-10-18-parts-of-speech.md)
§2a and Alec's clarification: **do not introduce multiple unknowns at the same
time; the data teaches the distinctions, without new UG accommodations.**
The categorizer's mechanism certificates pass. **The NP learning sequence
stopped at its first failed lesson. This is preparation, not a completed
NP stage or an NP–VP fixture. No identity run.**

Sparse counts on 92 fixed formal arguments replace the learned category
codebook, its context and category priors. Only kept programs supply input
and output word-head counts. The stored label is the recomputed argmax;
the chooser adds each operand's argument probability with coefficient one.
Unseen words have a uniform floor. Cosines report occupant overlap without
training or merging prototypes. The noun prior is recorded only.

No new lexical head rule was added. Existing declared heads and sole unary
operands carry their word; an unheaded binary result has no word occupant,
zero head probabilities, and no word counts on its output or subsequent
operand slots. The initially proposed first-word fallback was removed
before learning: it would have credited adjectives as determiner operands.
Output typing remains deferred. Stage 3 was restored, not retrained; clock,
chain and surface operations were unchanged.

The committed factorial has 56 sentences: 40 determiner NPs and 16 bare
descriptions, split 42/14 by the declared counterbalance. The clarification
changed the proposed monolithic run into prerequisite lessons, each gated
once on its own held-out subset before the next could begin. Their maximum
combined budget was still 84 presentations / 44 paired updates / 14 held-out,
seed **20261010**, LR **.003**, two passes, B4 with partial bare-NP batches.
STOP, lower, definite and intersection remained legal throughout; reference
variants were unchanged. There were no noun-aware masks or membership labels.

| Executed lesson | Training / paired updates | Held-out exact brackets | Order-1 occurrences | Expected operation sequence | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Bare `a` NPs | 6 / 4 | 1/1 | 1/1; no extras | 1/1 `lower` | Pass on this one case |
| Bare `the` NPs | 6 / 4 | 1/1 | 1/1; no extras | 1/1 `definite` | Pass on this one case |
| `a` + adjective + noun | 24 / 12 | **2/4** | **2/4; 2 unexpected** | **0/4** `intersection`, `lower` | **Fail; stop** |

Thresholds were ≥90% brackets, ≥95% one occurrence, no duplicates,
collisions or unexpected occurrences, and the taught operation sequence.
Duplicates and collisions were zero throughout. The four adjective questions
were answered correctly, despite the structural failures. **Totals: 36
training presentations, 20 paired updates, six held-out readings.** Each
held-out item appeared once, without its labels or answer entering a model
call; occupation/component learning was frozen for evaluation. No unlowered
control gate ran, and none is claimed passed. These are lesson-end results,
not a final-model evaluation of the full held-out quarter.

**Curriculum error, not a learning claim:** Codex still introduced
intersection inside a new nested NP. That combines an untaught operation
with its new scheduling problem and does not fully apply Alec's rule.
The first 40 of its 48 lesson calls had no fold targets on reachable student
states. Source now requires **bare descriptions/intersection before nested
adjective NPs**: lower → definite → description → adjective → definite adjective.
This correction has not been run. Resume a future lesson from the passed
definite artifact, not the failed nested artifact. No budget extension,
seed selection or learning rerun followed the failure.

A future continuation requires Alec's decision on budget and held-out reuse.
Finishing the corrected sequence from the definite artifact would require
72 more presentations / 36 updates: **108 presentations / 56 updates in
total**, including this discarded branch. Its adjective gate would reuse
four already-graded cases. The original 84/44 budget does not cover that
cumulative work, even though no failed-branch state would carry forward.

[Tables](tables.md) give every word's label and counts before/after, the
noun prior, occupant cosines, and trajectory per eight batches. All 4,186
argument pairs are in [collapse.json](collapse.json); full measurements,
first-pair total logits and the separate occupation term are in
[measurements.json](measurements.json). There were no unidentified word
leaves. All 60 kept compose steps paid `WHAT_STEP_COST` in the keep comparison.

Artifacts (complete hashes in the measurements):

| Role | Artifact | SHA-256 prefix |
| --- | --- | --- |
| Restored stage 3, unchanged | `output/stage4e-student-order-20261017/order.ckpt` | `e19d84e8e708` |
| Passed first lesson | `output/np-lower-20261018/trained.ckpt` | `5bc093fa263f` |
| Last passed prerequisite; next lesson's fixture | `output/np-definite-restored-20261018/trained.ckpt` | `b52392d9fc05` |
| Produced, failed nested lesson; not ready for NP–VP | `output/np-adjective-20261018/trained.ckpt` | `f1f3fe803554` |

Each continuation restored the previous training artifact before its held-out
readings; word identities, counts, labels, noun-prior and collapse tables
matched exactly. These experiment artifacts omit optimizer moments, so each
lesson used a fresh optimizer. Model RNG state continued from the artifact;
the separate data-order RNG exactly reproduced the recorded permutations.
No held-out state carried into the next lesson.

A pre-presentation attempt to restore the first artifact exposed a loader
shape check on variable-length occupation metadata. It ran **zero** examples
or updates. The native metadata check was fixed; the certificate now restores
nonempty counts into a fresh empty model with a different serialized size.
The measurement adapter compares decoded metadata, not pickle-envelope size,
and cannot repair occupation evidence. The final produced artifact also
passed exact fresh-model restoration, including every measured table.

Provenance is the per-run source manifest, not `source_commit` alone (HEAD was
the starting commit on a dirty tree). Lower precedes the loader/adapter fixes
and runner partitioning; the other two lessons precede only the corrected
curriculum order and the later test-fixture ports. No forward, chooser or
occupation-counting code changed after lower began. The manifests and exact
differences are recorded with the measurements.

Validation: all five default mechanism certificates and five formal-argument
checks pass; native full-graph forward/backward across runtime lengths passes.
The first sweep attempt stopped after 1,748/5,747 cases: nine failures,
eight from one shared test double missing argument configuration and one historical
link to the retired category smoke test. The fixture now uses the real
configuration method; **151 dependent tests pass**, and the link check passes.
No production fallback or assertion weakening was added. That incomplete
attempt remains at `output/pos-np-default-20261018/`.

The second attempt stopped after 2,920/5,747 cases: one exact state-key
fixture needed the new occupation metadata key; the other failure was
carried c. Its record remains at `output/pos-np-default-final-20261018/`.
Checking the remaining direct language-owner fixtures exposed two missing
grammar declarations; those were ported explicitly. The four-file group
passes **58 tests, three existing slow skips**. No production change followed
either incomplete sweep.

The completed final-source sweep at
`output/pos-np-landing-default-20261018/` ran **5,749 cases: 5,413 passed,
333 skipped, one XFAIL, two failures**, in 1,336.4 s. All five new
certificates, five formal-argument checks and eight 5.5 certificates pass;
all 788 validated source hashes match. Slow contracts remain in the weekly
tier; the latest weekly record is 16.6 days old. The two incomplete attempts
are disclosed above. The two additional selected cases check this receipt's
relative links.

Failures are carried c and the lexicon's chunked-versus-full exact
index-order check. The latter's kernel and test are byte-identical to
landed main. A separate seed-free tied-row diagnostic gives equal scores
and distances but different index orders. The sweep's inputs were not
saved, so this does **not** establish its failure's cause or its distance
agreement; that new finding remains unresolved. Diagnostic and source hashes
are in the measurements. No seed selection or sweep rerun followed it.

Claude reviewed the final sweep addendum and approved the preparation
landing, with no remaining source or protocol blocker. The lexicon finding
remains a separate follow-up. The enclosing commit is the archive.
