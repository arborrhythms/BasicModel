# Item 6.9 — §25 closing baseline (2026-10-03)

One existing working tree, published HEAD `d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`,
with all earlier rounds retained. Claude accepted this baseline; Alec authorized
commit and push on October 3. The measurements below are the reviewed results;
no training is repeated for publication. This implements
[§25](../../plans/2026-09-29-item-6-9-xor-grammar.md#25-decoding-from-conceptual-space-and-closing-69-alec-2026-10-03).
The proposed §24.4 pre-write snapshot is withdrawn: echoic priming remains
snapshotted after the sentence's seen write.

## Implementation

Reconstruction and output call the same conceptual generate walk. Its inputs
are the root, occupied end slots and the echoic shortlist. The generate chooser
infers a binary undo, unary undo or STOP from the current top. It never receives
the journal's rule sequence. Binary undo searches both children in the primed
bank; unary undo invokes the operator's generate face. A missing bank makes a
requested binary undo unavailable. The gate spells exactly the emitted leaves,
using the same activation × cosine × priming scoring as output and byte loss.

The hard generate choice determines stack topology. A straight-through softmax
over candidate numerical transitions provides reconstruction's derivative to
the chooser, preserving the chosen transition's exact forward value. This uses
the existing unit softmax scale, adds no objective or policy-reward coefficient,
and never reads a compose action as a teacher. Compose-only grammars expose
the generate faces of their declared operators. XOR_grammar previously had no
generate chooser because construction was gated on outputInLoop; this is now
created for reconstruction too, without editing its configuration.

Reconstruction owns the decoder chooser, parameterized generate faces, codes,
compose chooser, operators/tied inverses and perception. Answer synthesis owns
its question conditioner/readers; XOR_grammar owns its numeric head. Its
zero-initialized affine record reader sees root D, occupied end slots 3D and
the echoic weighted code sum D. It cannot see original word values, rule IDs,
arities, operand positions or journal columns. The old answer-owned generate
policy reward is removed. Restricted backward enforces ownership for trial and
batch answers, and grammar lessons train only their respective chooser.

SentenceUnderstanding contains root, end slots/depth, packed roots/depths,
per-word target metadata, sentence id and PrimedSymbols (rows, codes, weights,
own-word mask, surface bytes and validity). The two trial consumers receive
the same object. Neither reader nor decoder reads the target metadata.
The journal still supplies ClauseJournal.finish_clause with actual selected
operation inputs/output and reference addresses/flags, to build the clause's
performed meaning. Decoding no longer reads even its integer rule sequence.
Witness offsets remain absent. No new journal frame is written.

Reconstruction has two relative terms, each weighted by reconstructionScale:

- free_bytes: log-space byte/EOW error, baseline log 256. The target's
  emitting candidates and the uniform null byte are combined by log-sum-exp;
  no probability clamp. Missing words and excess emitted leaves are scored.
  Target positions align scoring only after decoding. Packed sentences retain
  their own denominator.
- antipode: align the selected surface-word code with each emitted leaf and
  repel every other valid shortlist code plus hashed inventory rows. The
  selected error is softplus(-10 cosine); the negative errors are
  softplus(10 cosine), the existing SBOW negative form. Mean binary error has
  baseline log 2. Both operands stay live. Non-surface concepts can be
  negatives but cannot displace the byte reader's selected surface word.

The hash reuses the old rotation's constants (1103515245, 2654435761,
2246822519, 3266489917), selected row, leaf position, sample offset and training
step, modulo the active inventory. The configured conceptualContextNegatives
value/default 4 is retained. A sampled selected row advances by one modulo the
inventory. These are ordinary gradient updates through live code lookups, not
rotation writes. There is no co-activation pull-together term, norm constraint,
EMA refresh or extra seed. Both terms are reported at batch end from the kept
trial; expectation remains detached and outside trial selection. Explore wins
only for strictly lower reconstruction; ties keep greedy.

## Saved-array check before editing

[saved_merging_probe.py](saved_merging_probe.py) and
[saved-merging.json](saved-merging.json) use only the §24-reviewed arrays in the
[correction receipt](../2026-10-03-item6-9-corrections/README.md). They rebuild
pre-step codes from all 800 saved displacements, matching the endpoint within
9.32e-10. World/there cosine changes from .36373164 to essentially 1.

The raw loss gradient points toward (there-world) in 598/800 steps; **descent is
its negative**, so only 202/800 descent directions point toward that full vector.
Code-length changes dominate this Euclidean comparison. Projecting toward
there's direction on world's tangent plane, descent points toward there in
774/800 steps, and the actual momentum displacement does so in 573/800. Most
late angular cosines are tiny because the codes are already nearly parallel.
Thus angular merging is supported; the literal statement about the gradient's
sign is not a statement about gradient descent. These aggregate gradients do
not isolate one candidate pair's causal contribution. The implemented old
search had a hard least-residual forward value with a soft blend derivative,
not a blended forward value. No model or training was run for this check.

## Focused validation and ports

The focused suite has 28 passes, plus the final surface-selection regression
passes separately (29 distinct focused cases). It covers eager and fullgraph
Inductor decoding, inferred binary/unary actions without a journal, live
chooser/code gradients, byte cost/gradient parity, no-witness reconstruction,
shared record identity, the answer cut, sole writers, and four trial/batch
answer ownership cases. The audit was validated on one focused batch before
starting the one 400-epoch measurement. This is fixture verification, not an
extra gate training or a repeated receipt measurement.

[ports.json](ports.json) contains complete old/new bodies for the reconstruction
observer, four parameterized output-ownership cases and the observational audit.
The byte/reference assertions and answer cut stay; the retired answer-policy
contract is replaced by actual owned-backward assertions, including nonzero
answer gradients and absent decoder gradients. All saved failure logs and
source hashes are under probes/. The initial failures, intermediate observer
shape mismatch, disabled-chooser discovery, packed cost dilution, output
shortlist mismatch and non-surface antipode selection are preserved.

[before.zip](before.zip), [source-final.zip](source-final.zip) and
[closing.patch](closing.patch) preserve this round's full source delta;
[source-final.json](source-final.json) freezes the measurement source.
No data configuration, seed, bar, guard or environment version changes.
The measurement runs on CPU, MODEL_COMPILE=none, with the unchanged 8 GiB
per-process guard and 1,800-second limit. No HEAD run, ten-run gates,
attribution, full sweep, MM_grammar or native run.

## Closing measurement

The two unchanged tests used **one model and one 400-epoch training** (the
observer records identical model identities). Both gates are red. All four
thresholded classes are correct; the class gate fails its MSE < .05 condition.
Reconstruction returns none of the four complete word multisets. All four walks
are available and untruncated; failure is early stopping, not an unavailable
inverse. [measurement.json](measurement.json) retains the machine-readable result.

| Input | Target | Answer | Shared decoder read-back |
|---|---:|---:|---|
| hello world | 0 | .2751798630 | world |
| hello there | 1 | .6365581751 | there |
| loving world | 1 | .6935124397 | loving |
| loving there | 0 | .3965403438 | loving |

MSE **.1147481948**, in §20.5's **between** band. Class bar **0/1**,
reconstruction bar **0/1**, joint bar **0/1**; 4/4 class labels and 0/4 complete
word read-backs. Checkerboard contrast is **−.6583504081**. The separate
perceptual rendering report is `hello; hello; loving; loving`; those strings
are preserved in the observations and are not substituted for the grammar
read-back. The gate and log-space byte objective use the shared decoder's
emitted concepts and echoic word scoring.

The bounded process took **72.15 seconds**, peak **675,104,256 bytes (.629 GiB)**.
It exited 1 because the two gate assertions failed. There was no retry.
[The source check](xor/complete.json) and [audit completion](xor-ownership/complete.json)
confirm the frozen source throughout. No tests, fixtures, bars or training
parameters changed after this run.

## Audit

[audits.md](audits.md) and [audit-summary.json](audit-summary.json) contain the
cost/reach, geometry, displacement and stability reports. The raw
[events](xor-ownership/events.jsonl) and [decoder trajectory](decoder-trajectory.json)
record every training trial's recovered-leaf/shortlist-code cosines, priming
weights, selected generate operations and independently observed compose
rules/arities. [Displacement arrays](xor-ownership/displacements/) retain each
step's gradients and changes; the audit performs no extra optimizer step.

- **Zero ownership conflicts**, 20 active and 55 inactive parameters across
  1,200 backwards. Both generate-policy parameters have reconstruction as their
  only declared and observed writer. The codebook and decoder have gradients
  and displacement on all 800 reconstruction steps.
- Every one of the **3,200 training row/trial decodes chose STOP immediately**.
  Each emitted one leaf. The decoder's weights learned, but no hard action
  changed in this run. It therefore never matched the compose derivations,
  which contain binary/unary operations. This directly explains the missing
  second word; useful operation inference remains unproven.
- Median gradient/displacement norms: codes **.226981 / .00828632**;
  generate weight **.00972535 / .000715267**; generate bias **.0134868 / .00103855**.
  Median displacement/gradient cosines are **−.5500, −.8164, −.9339** respectively.
  Every chooser anchor also moves; full per-parameter data are in the audit.
- World/there cosine **.151700 → .623919**; hello/loving **−.046033 → .517130**
  (range −.121779 to .760255 during training). Codes did not become parallel as
  in §24, but this one unpaired run does not establish a causal improvement.
  Dictionary mean squared off-diagonal cosine **.0718581 → .367688**;
  norm range at the end **.336167–1.894742**, with no norm constraint.
  Dictionary Frobenius displacement **2.687247**. Every VQ cluster size remains
  **1**, and EMA refresh is false in both physical codebooks.
- Centered root singular values: **[.744750, .593537, .397149, 4.68e-8] →
  [.516001, .181629, .154045, 2.72e-8]**. Full roots, code rows and pairwise
  matrices are saved beside geometry-start/end.json.
- Modal compose-derivation fractions by sentence: **.885, .8275, .4375, .44**;
  distinct sequences **7, 6, 6, 6** over 400 epochs.
- Explore wins **173/1,600** row comparisons; zero reconstruction-selection
  violations. Expectation and answer are not compared. There are no activated
  extra surface candidates in this fixture's measured banks, so it supplies
  no evidence on their competition with own words.

## Closing baseline

**6.9 is closed as this baseline, by §25.3.** Closing does not assert that its
learning gates pass. The preceding repeated measurement is
[§22](../2026-10-03-item6-9-free-readback/README.md): class **9/10**,
reconstruction **5/10**, sum control **10/10**, named table **33/34**. Those are
prior measurements, not new counts for this candidate. This round's baseline
is the single result above, with both gate bars intact and red.

MM_xor::test_convergence remains red by §17: its former convergence exploited
percepts promoted across word boundaries, and meronomy removed that lookup
shortcut. Its unchanged word-level XOR proof belongs to 6.8. MM_grammar's known
occasional .25 stop and the other previously recorded XOR/round-trip results
remain on record; none was rerun here. The earlier MM_ladder learning failures
also remain unwaived and unmeasured in this minimal round. Focused checks pass;
a full-suite pass is not claimed. Collection succeeds with **5,019 cases**.
**Documentation links: 272 passed** (2.38 seconds of pytest time; 4.12 seconds bounded wall time).

The standing XOR rule for later items is **no regression against this record**,
including the explicit red gates and preserved earlier measurements. The
catalog assigns the remaining work:

| Later work | Expected improvement |
|---|---|
| Operators update, §§20.3 and 25.2–25.4 | Identities from conceptual space and the remaining operator catalogue; useful decoder operation inference and improved XOR class/reconstruction learning. The antipode pull-apart term is implemented; co-activation attraction remains deferred. |
| 6.8, §17 | Open reading and MM_xor's genuine word-level XOR proof. |
| Surface markers, §24.5 | Operand-order recovery and order-sensitive surface reconstruction; commutative binding alone cannot retain order. |

Next: **conference freeze → operators update → 6.8**. No further measurement
or repair is authorized by this closing baseline. Nothing is committed; the
working tree and receipt are ready for review.

## Acceptance and landing

Claude accepted plan §26; Alec authorized commit and push on October 3. All
672 source files still match `source-final.json`. The historical run statuses
and audit arrays are retained unchanged. The former long todo entry is preserved
in [todo-history.md](todo-history.md); item 6.9 moves to Done. Next is decoder
exploration at the start of the operators update, followed by item 6.8. There
is no conference freeze. Publication checks do not repeat the training campaign.
