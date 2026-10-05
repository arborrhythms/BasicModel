# NanoChat-sized grammar pilot

> Historical gate record. These fixed W=64 comparison fixtures predate the
> current `data/BasicModel.xml` architecture. Use `basicmodel.txt` and
> `data/BasicModel.xml` for current training; retain this cohort only for
> reproducing the original held-out evaluation.

## Question and falsifiable first milestone

The comparison-sized experiment asks whether the serial grammar architecture
can acquire predictive language structure from FineWeb-Edu, measured against
Karpathy's default NanoChat speedrun tier, depth 20 (Alec, 2026-09-25;
replaces the earlier matched-parameter depth-4 reference). The two are not
parameter-matched: the comparison is by bytes seen and wall time on the
common held-out item set, and the depth-20 result is the ceiling reference —
what the default $100 speedrun does on these items — not a matched curve.

The first milestone is deliberately narrower:

> After training only on the train documents, use the preceding words to pick
> the true next word from 16 surface-word candidates in held-out FineWeb-Edu
> documents.

For BasicModel, the 6.8 evaluator uses the shared expectation owner's
**word-level categorical distribution**, conditioned on the observed prefix.
All 16 candidates share one bank and are scored by negative log probability.
Candidate completions never enter the predictor's prefix history. NanoChat
scores its same candidates by conditional log probability. Top-1 and reciprocal
rank remain directly comparable; this finite-bank likelihood is not token
perplexity. The retired `IntraSentenceLayer` point-distance scorer is retained
only in historical receipts.

**Scope clarified by Alec, 2026-10-03:** 6.8 checks the evaluator's mechanism
on a small configuration over a few unchanged manifest items. The language gate
waits for item 4's trained checkpoint. It does not call for scoring the fresh
BasicModel now. The three-item mechanism check and earlier stopped attempts
are preserved in the [6.8 receipt](benchmarks/2026-10-03-operators-attention/README.md).

The evaluator stages the normal native word readings, then scores the complete
candidate bank under `no_grad`. It does not build sentence composition or the
answer output, which cannot change the staged word snapshot. Tests compare its
scores exactly with the full-forward boundary. Native evidence and priming
omit inactive padding during eager staging; the returned word buffers retain
their configured dimensions. No item, candidate, or live word is removed.

This is a better language gate than reconstruction loss.  An invertible model
can learn to copy an ill-formed sentence, and its raw MSE is not a normalized
language likelihood.  The multiple-choice task requires context to identify an
unseen held-out word.  A shuffled-prefix control tests whether word order, not
only unigram frequency, supplies the gain.

Use 500 fixed held-out items.  Each has a configured 4--16 word prefix, the true
next word, and 15 frequency/length-matched distractors sampled only from the
held-out documents.  The checked-in realization has 4--7 word prefixes because
it was frozen while the prototype still spent one flat 32-percept slab across
the sentence.  Keep it frozen for comparison, but that restriction is no longer
architectural: the corrected model has an independent 64-word sentence axis and
set-synthesizes every raw constituent inside that word's iteration. PS, WS, CS,
and STM remain eight wide/deep. The evaluator rejects an item if its candidate
falls beyond the outer word cap or if raw-constituent staging ever reports a
cut (which is now an implementation failure, not a PS-capacity condition). The
initial gate is:

- top-1 at least 12.5% (twice the 6.25% chance rate),
- at least 5 percentage points above the same initialized model,
- shuffled-prefix accuracy below intact-prefix accuracy, and
- the result repeats for three seeds before scaling the model.

Also record candidate-bank negative log likelihood, candidate mean reciprocal rank,
and the variance/effective rank of target ideas.  Those last checks catch a
latent-collapse shortcut.  Exact surface reconstruction is useful smoke-test
telemetry, but is not a language pass criterion.

## Two model tiers

| Tier | Config | Parameters | W / F / D | Percept rows | FineWeb-Edu documents |
|---|---|---:|---:|---:|---:|
| Architecture gate | `data/MM_nanochat_grammar_gate.xml` | 8,518,478 | 64 / 8 / 256 | 8,192 | 200 |
| Comparison pilot | `data/MM_nanochat_grammar_pilot.xml` | 41,679,182 | 64 / 8 / 512 | 32,768 | 2,000 |
| NanoChat reference | Karpathy's default speedrun, depth 20 | about 560M (per the nanochat defaults; record the exact count from the run) | context 2048 / -- / 1,280 | 65,536 BPE vocab | the speedrun's own FineWeb-Edu shards, about 11B tokens |
| Existing FineWeb config | `data/MM_20M_fineweb.xml` | 118,256,431 | legacy 64 / -- / 1,024 | 65,536 | 10,000 default |

Counts include the bounded STM reducer that is registered during the standard
pre-training warmup. The pilot is roughly one-thirteenth the size of the
depth-20 reference and sees roughly one five-hundredth of its bytes; the
comparison is deliberately unmatched, and the numbers are reported with
those ratios beside them. The old "20M" name is not a reliable budget: its
current live graph is about 118M parameters. Nothing NanoChat has been built
locally yet: the reference tier is trained on rented hardware (eight H100s,
about four hours for the depth-20 speedrun), and the evaluator's NanoChat
scorer — conditional log-probability of each of the 16 candidates given the
prefix, top-1 and reciprocal rank on the same item file — is still to be
written.

Here `W=64` is only the outer serial-loop bound. `F=8` is the simultaneous PS,
WS, and CS field width and the STM workspace depth. Raw radix width is not a
model capacity: at iteration `w`, PartSpace gathers `[B,P_raw,D]` for the
complete current word and applies its configured sigma set-fold. PS and WS then
bind in CS; one current-word concept, carrying its derived order, enters STM.
Concepts remain a
growing relation inventory, commonly one-part/one-whole identities, rather than
being capped at the instantaneous 8+8 percept field.

Both tiers therefore use:

- `subsymbolicOrder=4`, `symbolicOrder=4`, uniform eight-wide conceptual
  stages, and unbounded `syntacticOrder=0`;
- serial word traversal and `mereologyRaise=true`;
- the learned MLP transform chooser and category codebook;
- one routing pass (`learning=false`) for the first gate; and
- reconstruction, intra-sentence prediction, inter-sentence prediction, and
  contrastive discourse losses.

The tier table and configuration choices above are historical. Current runs
use `data/BasicModel.xml`: the mandatory paired grammar walks, one attention
budget and level-indexed expectation replace the old mode and order switches.
The frozen 500-item manifest and all numerical learning bars remain unchanged.
No fresh-model gate is scheduled in 6.8.
Qualification still requires item 9's million-sentence prerequisite and
ordered/shuffled/context-free controls at equal updates for the three
predeclared seeds. See the [6.8 receipt](benchmarks/2026-10-03-operators-attention/README.md).

## Corpus ladder

All counts below come from the currently downloaded
`data/fineweb/shard_00000.parquet`; the loader splits whole documents
deterministically 8/1/1 into train/validation/test.

| Stage | Documents | Sentences | Whitespace words | Purpose |
|---|---:|---:|---:|---|
| Launch check | 20 | 769 | 13,090 | finite forward/backward and obvious overfit |
| Language gate | 200 | 8,452 | 141,228 | held-out next-word retrieval |
| Comparison pilot | 2,000 | 89,189 | 1,504,570 | first NanoChat-sized learning curve |

Do not move to 2,000 documents because training loss merely decreases.  Move
only after the 8.52M gate beats initialization, chance, and shuffled context on
the fixed held-out item manifest.

The frozen evaluator now lives at `bin/eval_nanochat_grammar.py`; its checked-in
manifest is `data/eval/nanochat_grammar_gate.json`.  The manifest contains 500
items from all 20 test documents in the 200-document gate, has randomized answer
positions, and is pinned by item SHA-256
`ad594aa33ae1ac2f0bc49cc16c2822144fe26c73897f6b17228f641abc9773d7`.
`train.py --test` remains reconstruction telemetry; use this evaluator for the
language claim after item 4 provides the trained checkpoint. The commands below
are historical reproductions, not instructions to run a fresh gate in 6.8:

```sh
cd basicmodel
.venv/bin/python bin/eval_nanochat_grammar.py generate

# Deterministic initialized baseline; explicitly bypass any XML autoload file.
.venv/bin/python bin/eval_nanochat_grammar.py score \
  --device mps --fresh --item-batch-size 32 \
  --output output/nanochat_grammar_gate_stm8_initialized.json

# Trained model. The reducer is prewarmed before checkpoint loading so all
# learned reducer/router weights are restored rather than ignored as lazy keys.
.venv/bin/python bin/eval_nanochat_grammar.py score \
  --device mps --checkpoint output/MM_nanochat_grammar_gate_stm8.ckpt \
  --item-batch-size 32 \
  --output output/nanochat_grammar_gate_stm8_trained.json
```

All prior initialized/trained results and checkpoints used either the flat
sentence-percept axis or the rejected 64-deep STM. They are diagnostic only and
must not be compared with this model. The current trained-checkpoint comparison
and its controls belong to item 4.

Before training or scoring, run the explicit reduction audit:

```sh
.venv/bin/python bin/eval_nanochat_grammar.py trace --device mps --words 64
```

Add `--timeline` only when the per-word confidence/depth sequence is needed;
the default audit prints the compact aggregate.

The audit counts a reduction only when STM depth decreases and separates soft
occupancy-pressure decisions, hard capacity demands, grammatical boundary
closure, and any unlicensed fallback. The controller removes the binary-rule
count prior before deciding SHIFT versus REDUCE, then lowers the configured
`0.75` threshold as STM fills. At full depth it demands the best grammatical
operator; absence of such an operator is an error rather than a dropped word.

On a fresh MPS build, 64 pushes produced 61 soft-pressure reductions, zero
capacity demands, two grammatical boundary reductions, and zero unlicensed
reductions. Peak depth was 4 and final depth was 1, so all 63 required depth
decreases were accounted for inside STM8. The fresh operator mix (60 online
`conjunction`, one online and two boundary `disjunction`) is initialization
telemetry, not evidence of learned grammar diversity.

The depth-20 reference trains on the speedrun's own shards, not on the
BasicModel manifest. Two consequences. First, **contamination**: the speedrun
consumes the first FineWeb-Edu shards, and the checked-in gate draws its
held-out items from `shard_00000`; regenerate the item manifest from a shard
outside the speedrun's shard list before scoring the reference, or state
plainly that the reference has seen the test documents. Second, the
**matched curve is a separate, cheap run**: a small NanoChat (depth 4 to 8)
trained on the exact BasicModel document manifest exported to NanoChat train
and validation parquet files, minutes on one H100, reported beside the
depth-20 ceiling. Budget every run by source UTF-8 bytes seen and wall time,
not optimizer steps or tokenizer tokens.  Report parameter count, peak
memory, bytes/second, top-1/MRR on the common word-choice set, and learning
curves.  Do not compare BasicModel's reconstruction MSE to NanoChat
bits-per-byte or cross-entropy as though they were the same quantity.

## ArborStudio MPS launch posture

The corrected architecture runs locally on `ArborStudio.local` from
`/Users/arogers/github/WikiOracle/basicmodel`; MPS tensors are on `mps:0`. The
command-approval boundary does not move execution to another host or checkout.

Fresh measurements with the independent 64-word axis, eight-wide fields,
STM8, rule-count-neutral pressure controller, and no checkpoint load/save are:

| Gate operation / device | Result | End-to-end wall time |
|---|---:|---:|
| 64-word reduction audit / MPS | peak depth 4; 63 grammatical reductions; 0 unlicensed | 5.2 s |
| One optimizer update / MPS, batch 1 | reconstruction loss 0.5954 | 7.87 s |

These are launch checks, not learning results. The optimizer smoke used 20
FineWeb-Edu documents, stopped after one batch, and wrote no checkpoint. Batch
4 remains the XML's conservative training default but has not yet been measured
on this corrected pressure-driven model; the older batch 4/8/16 measurements
belonged to rejected flat-axis or STM64 prototypes and do not set current
limits.

The axes have independent truncation telemetry. Over all 7,360 complete
batch-4 corpus rows previously audited, 30 sentences (0.41%) exceed the outer
64-word traversal. Raw word-constituent width is dynamically staged rather
than capped at PS width; the observed corpus maximum was 17, and the regression
trace deliberately passes a 20-constituent word through PS8 without a cut.

The serial path currently contains a host `Tensor.tolist()` boundary that
strict `torch.compile(fullgraph=True)` rejects.  Run eagerly until that boundary
is hoisted out of the captured forward:

```sh
cd basicmodel
BASICMODEL_DEVICE=mps MODEL_COMPILE=none .venv/bin/python bin/train.py \
  --model data/MM_nanochat_grammar_gate.xml --data text --batch-size 1
```

After the behavioral gate passes, launch the comparison tier conservatively
and raise batch size only after observing MPS memory:

```sh
cd basicmodel
BASICMODEL_DEVICE=mps MODEL_COMPILE=none .venv/bin/python bin/train.py \
  --model data/MM_nanochat_grammar_pilot.xml --data text \
  --batch-size 4 --test 20
```

The XML defaults retain resumable, periodic checkpoints. Both tiers still need
batch-size probes before a long run; only gate batch 1 is established for the
corrected controller.
