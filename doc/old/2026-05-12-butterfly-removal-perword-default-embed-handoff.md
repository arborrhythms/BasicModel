# Butterfly Removal → Per-Word Stem Default → Embed.py Per-Word Integration

## Context

Three coupled refactors land the per-word symbolic pipeline as the *only* code path:

1. **Remove butterfly mode entirely.** The pairwise N-halving via `<useButterflies>true</useButterflies>` is incompatible with per-word slicing (`N=1` breaks the butterfly pack). Production configs (MM_5M, MM_400M, RamsifiedModel) currently rely on butterfly for throughput; this refactor retires that path. The resulting architecture is uniform — no `N/2` halving across stages, no orthogonal mode-pair to validate.

2. **`perWordStem` defaults to `true`.** Each forward call runs the per-word P→C→S→C round trip and accumulates ideas on `ConceptualSpace.stm`; the chart fires at C over the STM buffer. The legacy chart-at-stem launch site retires. This was the "Step 7" of the original [serial-parser handoff](2026-05-12-serial-parser-handoff.md), deferred there because butterfly incompatibility forced opt-in.

3. **`embed.py` runs through the model pipeline.** Today `embed.py` is an offline standalone that builds a BPE codebook via `ChunkLayer.train_step` over raw text shards, then writes a bundle for the model to load. After this refactor, `embed.py` instantiates the model and feeds text one word at a time through the per-word stem, so the codebook + lexicon + C-tier vector space are co-trained through the same forward path that AR learning uses.

The three refactors are sequential — each depends on the previous landing first.

## Resolved decisions (carried forward from prior work + this conversation)

| # | Decision | Rationale |
|---|---|---|
| 1 | `useButterflies` XML knob retired (not deprecated — removed wholesale). | Single code path; no orthogonal flag-pair to maintain. The throughput optimization butterfly bought is recoverable via other means (kernel fusion, fp16 + monotonic) without the architectural cost. |
| 2 | `perWordStem` flag itself is removed after flip (consistent with [serial-parser handoff Step 7](2026-05-12-serial-parser-handoff.md#step-7-default-flip--cleanup)). | The chart-on-LTM design strictly subsumes the chart-on-stem path once butterfly is gone; no fallback needed. |
| 3 | AR microbatch (K>1) compatibility resolved by **forcing K=1 in the per-word stem driver**. | Per the [original serial-parser plan](2026-05-12-serial-parser-handoff.md#known-limitations--future-follow-ups), the per-word stem's "outer time" is the AR position itself. The AR pipelining (K progressive prefixes) loses its purpose under per-word processing because each word *is* one position. |
| 4 | `embed.py` becomes a thin CLI on top of the model's forward pass — it loads the configured XML, sets `wordLearning>=1`, and feeds text through `model.forward(text_batch)`. | The user directive: "embed.py is the only stage where the vocab grows." Reusing the model's forward path means the BPE chunker, C-tier transform, S codebook, and (in time) the meronomy all train together. |
| 5 | Embed pretraining uses the same per-word STM accumulation but with a different loss (CBOW/SBOW). | Don't reinvent the per-word loop; just swap the loss head. |

## Phase 1 — Remove butterfly mode

### Scope

Drop everything related to butterfly mode:

- XSD element `<useButterflies>` removed from [data/model.xsd:61](../../data/model.xsd).
- Python attribute `Model.useButterflies` removed at [bin/Models.py:1935-1936](../../bin/Models.py) and [bin/Models.py:4230-4231](../../bin/Models.py).
- Validation guard `useButterflies + useGrammar=="all"` excluded at [bin/Models.py:4266-4268](../../bin/Models.py) — removed alongside the flag.
- Butterfly-pack machinery in [bin/Layers.py:1795-2012](../../bin/Layers.py): `_butterfly_pack`, `_butterfly_unpack`, `_butterfly_perm` buffer, packed/unpacked forward/reverse branches in PiLayer / SigmaLayer (and the parallel branches at lines 3338-3407).
- Butterfly schedule check at [bin/Models.py:5922-5930](../../bin/Models.py) (`nPercepts` divisibility for butterfly stages).
- `butterfly=...` constructor kwarg at [bin/Models.py:4471](../../bin/Models.py).
- All 13 XML configs with `<useButterflies>true</useButterflies>`: LM_5M.xml, LM_5M_IR.xml, MM_5M.xml, MM_5M_AR.xml, MM_5M_IR.xml, MM_400M.xml, MM_xor.xml, MM_xor_bivector.xml, MM_xor_step4.xml, MM_grammar.xml, RamsifiedModel.xml, model.xml, idempotent.xml.

### Implementation outline

1. **Audit consumers**: 144 in-code references to `butterfly` across `bin/`. Catalog by type (XML read, attribute access, layer-internal, validation, doc string).
2. **Delete the unconditional-non-butterfly path's gate** — in PiLayer/SigmaLayer, the butterfly branches sit alongside the non-butterfly path under `if self.butterfly_n_t > 0`. Inline the non-butterfly branch and delete the butterfly branch.
3. **Delete the dimensional bookkeeping**: the butterfly path doubled `D` and halved `N` per stage. After removal, every stage processes the same `[B, N, D]` shape (constant across body iterations).
4. **Update `ModelFactory.validate_config`**: drop the butterfly-divisibility check; relax the validator that asserts `nPercepts * state_dim == nSymbols * symbol_width`.
5. **Remove butterfly tests**: any test that asserts butterfly behavior is dead. Tests that load `<useButterflies>true</useButterflies>` configs should have the flag removed from the loaded XML (or fail-loud if they're explicitly testing butterfly).
6. **Doc sweep**: [doc/Architecture.md:137-144](../../doc/Architecture.md) describes the "Butterfly mode" section — delete. Other `doc/*.md` references trimmed accordingly.

### Risks / open questions

- **Memory footprint**: butterfly doubled D per stage; without it, hidden activations might be larger at the latter stages. Validate: does MM_5M still fit in GPU memory at `batchSize=128` after butterfly removal? If not, propose `nDim` reductions in the same pass.
- **Stage shape contract**: `_level_shapes` (hierarchical mode) computed per-level shapes that assumed butterfly halving. Confirm it's a no-op when butterfly is off, or update to flat shapes.
- **`ChartCompose` interaction**: butterfly mode excluded `useGrammar="all"`. With butterfly gone, this exclusion vanishes — the cross-mode tests need to be reviewed.

### Verification

- All chart / grammar / STM / per-word tests still pass.
- A representative MM_5M-style config builds, forward runs, reverse runs, and trains for 10 steps without error.
- No file in `bin/` mentions `butterfly` (verifier: `grep -r butterfly bin/ | wc -l == 0`).

## Phase 2 — `perWordStem` defaults to `true`

Depends on Phase 1.

### Scope

- Flip the default at [bin/Language.py:2097-2107](../../bin/Language.py): `per_word_stem` defaults `True`.
- Remove the flag entirely (no longer needed):
  - `<perWordStem>` element in [data/model.xsd:481](../../data/model.xsd).
  - The `<perWordStem>` reader at [bin/Language.py:2095-2099](../../bin/Language.py).
  - The `<iterationsPerWord>` reader stays (still meaningful: controls C→P loop count per word). Default `1`.
  - All branches in [bin/Models.py `_forward_stem`](../../bin/Models.py) keyed on `self.wordSpace.chart.per_word_stem` become unconditional per-word.
- Retire the legacy chart-at-stem launch site:
  - Delete `Models._chart_compose` and `Models._chart_generate` modules at [bin/Models.py:4741-4756](../../bin/Models.py).
  - Delete the `ChartCompose` / `ChartGenerate` classes in [bin/Language.py](../../bin/Language.py) (or thin them to a no-op).
  - Delete the `self._chart_compose(sub)` call site in `_forward_stem`. Both reverse and forward paths use `_stem_input_to_percepts` followed by `_forward_stem_per_word`.
- STM auto-sizing logic at [bin/Spaces.py:8184-8208](../../bin/Spaces.py) becomes the only path — `wMax` is the canonical STM capacity, `stmCapacity` XML override stays for sub-symbolic configs that want a wider buffer.

### Implementation outline

1. **Confirm K=1 contract** in `_forward_stem_per_word` ([bin/Models.py:4828](../../bin/Models.py)): the `wordSpace.ensure_microbatch(B, 1)` call is mandatory after the InputSpace forward. Document why; add a runtime assertion that K==1 after the call.
2. **Drop the legacy chart firing**:
   - `_chart_compose` deleted, body's chart-at-C ([bin/Models.py:4895](../../bin/Models.py) `_chart_compose_at_C`) is the only firing site.
   - Reverse path: `_chart_generate` → C-tier reverse via the same chart's `generate` method called from the body's reverse pass.
3. **Update at-risk tests**:
   - `test_per_word_stem_flag_off_path_unchanged` ([test/test_per_word_stem.py:139](../../test/test_per_word_stem.py)) — delete (no flag-off path).
   - `test_chart_wordspace_wiring.py::test_chartcompose_and_chartgenerate_in_pipeline` — rewrite to assert chart fires inside `ConceptualSpace.forward`.
   - `test_compose_chart.py` — retarget chart-over-STM at C-tier.
4. **Update docs**: [doc/Architecture.md:289-329](../../doc/Architecture.md) "Per-word operational flow (planned)" → "Per-word operational flow"; remove "planned"/"deferred" tense. Remove [doc/Architecture.md:148-153](../../doc/Architecture.md) "Butterfly mode" section. Update [doc/Spaces.md](../../doc/Spaces.md) STM section to reflect sentence-length default. Update [doc/Language.md](../../doc/Language.md) chart-firing-site to C.

### Risks / open questions

- **Single-Space tests**: some unit tests build `Chart` in isolation against a `_FakeWS` ([test/test_chart_wordspace_wiring.py:54](../../test/test_chart_wordspace_wiring.py)). Confirm they still pass when per-word-stem is the default — `_FakeWS` doesn't have a `symbolicSpace`/`perceptualSpace` so the per-word stem can't run, but these tests call `Chart.compose` directly with synthetic data — they bypass `_forward_stem_per_word`.
- **AR/ARIR semantic loss**: today's AR microbatching feeds K progressive prefixes through the body in parallel. Under per-word stem with K=1, each forward processes one position. The training loop's loss target alignment changes — `outputSpace` sees one prediction per forward instead of K. Confirm `runEpoch` accepts this cadence change.
- **Reverse path**: today the reverse pass runs from `outputSpace.reverse` backward; with the chart at C, the reverse path needs to consume STM contents and produce the byte stream. Define the reverse contract explicitly.

### Verification

- Full ~700-test regression with the legacy path gone. The two pre-existing known-failing tests stay; nothing new fails.
- A representative non-trivial config (POS_smoke.xml with butterfly removed) trains for 50 steps and produces sentence-completed signals.
- MM_5M.xml trains for 10 steps end-to-end on CPU; gradient norm finite; no NaNs.

## Phase 3 — `embed.py` per-word integration

Depends on Phase 2.

### Scope

`embed.py` becomes a CLI on top of the model's forward pass. The CBOW/SBOW pretraining gets a per-word loss head; BPE codebook grows through the same model.

### Today's `embed.py` (legacy)

- Two phases ([bin/embed.py:1438](../../bin/embed.py) `discover_bpe` + [bin/embed.py:1763](../../bin/embed.py) `train_bpe`):
  - Phase A: stream text shards, run `ChunkLayer.train_step(words)` to grow the BPE vocab.
  - Phase B: with frozen BPE, train Embedding via CBOW/SBOW on (target, context) pairs.
- Output: `.kv` artifact. **No model loaded**; `ChunkLayer` and `Embedding` instantiated standalone.

### Target (after Phase 3)

- Single entry point: `embed_pretrain(config_path, shard_paths, num_epochs)`.
- Load the model via `BasicModel.from_config(config_path)` with `wordLearning=1` overriding the XML default of `0`.
- Stream text shards; for each text batch, feed bytes through `model.forward(text_bytes)` — the per-word stem runs as usual.
- New loss head: `embed_loss = CBOW_loss(stm_ideas, context_window)` computed over the STM contents.
- Backprop + optimizer.step.
- Save: `model.save_weights(ckpt_path)` — the bundle carries the lexicon, the BPE codebook, the well-known atoms, and the C-tier transform weights, co-trained.

### Implementation outline

1. **New helper in `embed.py`**: `embed_pretrain(model, shards, num_epochs)`. Replaces `discover_bpe` + `train_bpe` as the canonical entry. Old functions retired.
2. **Loss head**: implement `_embed_step(model, text_batch)` that:
   - Runs `model.forward(text_batch)` (per-word stem fills STM).
   - Pops the STM ideas for each row.
   - For each idea, computes CBOW context window (the surrounding ideas in STM).
   - Computes mean-squared error or contrastive loss between the predicted center and the actual STM-popped vector.
3. **`wordLearning` overlay**: `embed_pretrain` sets `TheXMLConfig.set("PerceptualSpace.wordLearning", 1)` *before* `from_config` so the ChunkLayer is built in active-growth mode.
4. **CLI**: replace the four-subcommand argparse setup ([bin/embed.py:1834-1939](../../bin/embed.py)) with `embed.py pretrain --config <xml> --shards <pattern> --epochs <N>`.
5. **Retire dead code**:
   - `WordVectors.train_step` standalone path ([bin/embed.py:669-757](../../bin/embed.py)) — replaced by the integrated forward loss.
   - `PretrainModel` class if it has no remaining consumers.
   - The `.kv` artifact save/load fallback paths.
6. **Save path**: `embed_pretrain` writes the same `.ckpt` bundle the AR loop would write. There is one artifact format across all training stages.

### Risks / open questions

- **Embedding-only training mode**: today `<trainEmbedding>CBOW</trainEmbedding>` toggles a "train embeddings, freeze the rest" mode. After Phase 3, this becomes a runtime knob inside `embed_pretrain` that gates which parameters the optimizer touches.
- **Context window semantics**: in classical CBOW, "context" is a fixed window of surrounding words. Under per-word STM, the context is "all ideas currently on STM except the target". Define the loss precisely.
- **Cold-start codebook**: the S codebook starts empty (or seeded with `well_known_atoms = {"words": 0}`). The first few words in embed pretraining have no codebook history; their quantization is essentially random. Either bootstrap with byte-level seed (256 initial atoms covering the byte range) or accept a warm-up period.
- **BPE codebook growth criterion**: `ChunkLayer.train_step` uses a frequency-based merge criterion. Under per-word stem, what triggers a new BPE merge? Either keep the existing frequency criterion (the per-word loop calls `train_step` on each new word) or replace with a "C-tier reconstruction error exceeds threshold → split atom" criterion. The user noted: "There may be some other method of learning a word."
- **Word boundary detection**: `embed.py` historically splits on whitespace. Under per-word stem with BPE, the chunker handles this. Verify the contract: does the BPE chunker emit one chunk per word (boundary at whitespace), or sub-word chunks? The current ChunkLayer's `BOUNDARY_BYTES` machinery suggests whitespace is honored.

### Verification

- `embed.py pretrain --config data/POS_smoke.xml --shards data/sample.txt --epochs 1` runs end-to-end and produces a `.ckpt` bundle that the model can load.
- The bundle's `wv._vectors` is non-trivial (CBOW actually trained).
- The bundle's BPE merge table has > 256 entries (vocabulary grew beyond the byte seed).
- The bundle's `well_known_atoms` contains `{"words": 0}`.
- After loading the bundle into a fresh model, the lexicon delegates work (`S.vocabulary` returns the trained Embedding) and the BPE codebook is frozen (`wordLearning=0`).

## Combined verification at end of all three phases

```bash
cd basicmodel
# Full regression
BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/

# embed.py + train round-trip smoke
BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python bin/embed.py pretrain \
    --config data/POS_smoke.xml --shards data/sample.txt --epochs 1
BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python bin/train.py \
    --model data/POS_smoke.xml --data text --log --num-epochs 1 --batches 10

# Per-word + chart-at-C invariants
BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest \
    test/test_per_word_stem.py test/test_chart_wordspace_wiring.py \
    test/test_grammar_binary_ops.py test/test_lexicon_ownership.py -v
```

Success criteria:
1. **All ~700 tests pass** (the two pre-existing failures stay, nothing new fails).
2. **`grep butterfly bin/` returns no Python hits** (only doc references for historical context, if any).
3. **`embed.py pretrain` produces a `.ckpt` that loads cleanly** and contains the trained lexicon + BPE codebook + well-known atoms.
4. **A full train+reverse round-trip** on a 4-byte text sample reproduces the input via the S→C→P→I→bytes path.
5. **The per-word path is the only forward path** (no `if per_word_stem:` branches remain).

## Files & line numbers to read in Phase 1 exploration (next-conversation)

| File | Lines | What's there |
|---|---|---|
| [bin/Layers.py](../../bin/Layers.py) | 1790-2012, 3330-3410 | PiLayer / SigmaLayer butterfly pack-unpack branches |
| [bin/Models.py](../../bin/Models.py) | 1935-1936, 4227-4271, 4393, 4431, 4471, 4478, 5276, 5922-5930 | `useButterflies` reads, validation, factory schedule |
| [bin/Language.py](../../bin/Language.py) | 2095-2109 | `perWordStem` reader, current opt-in machinery |
| [bin/Models.py](../../bin/Models.py) | 4778-4900 | `_forward_stem` / `_forward_stem_per_word` / `_forward_body` / `_chart_compose_at_C` |
| [bin/embed.py](../../bin/embed.py) | 1438-1939 | `discover_bpe`, `train_bpe`, CLI argparse |
| [bin/Spaces.py](../../bin/Spaces.py) | 6610-6650, 8184-8210, 8870-8920 | ChunkLayer construction, STM sizing, well-known atoms |

## Notes on ordering

Phase 1 → Phase 2 → Phase 3 is a hard dependency chain:

- Phase 2 can't ship while butterfly configs exist because flipping the default breaks them.
- Phase 3 can't ship before Phase 2 because the per-word stem must be the canonical path that embed.py drives through.

Each phase has its own gate (regression green before the next). The combined verification only runs after Phase 3.

Estimated landing surface: ~800 lines deleted (butterfly machinery), ~150 lines deleted (legacy chart-at-stem), ~400 lines net change in embed.py (replaces argparse maze with single CLI on top of `model.forward`). About 3-5 PRs, one per phase + a final cleanup pass.
