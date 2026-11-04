# Item 2.0, the order-zero loop: one order-0 reverse, the parallel conceptual field, output by the same reverse (Claude, 2026-11-04; spec in code names, with decisions for Alec)

**Status.** The loop was decided in direction on 2026-11-03 ([item 2.0 spec §8](2026-11-03-gradient-projection.md), the "Thinking is Optional" conversation) and refined by Alec the same evening (§1 below). This is its design in code names, the last part of item 2.0. Governed by step-2 §39 ([2026-10-21-step2-one-ltm.md](2026-10-21-step2-one-ltm.md)): within an authorized round Codex reviews the spec, fixes what the code contradicts and continues, documenting each decision as a dated section; only reversing a §1 decision comes back to Alec. The Codex text is held until §5 is decided and joins item 2.0's one Codex text. Code names and line numbers are from the 2026-11-04 working tree, which carries Codex's uncommitted item 2.5 completion round; Codex re-locates them. The derivation was read from the code and not run.

## 1. The record (Alec, 2026-11-03)

1. "Reconstruction should be initiated from order-zero concepts rather than explicit understanding (STM0 and STM1). Understanding is a thought process that transforms order-zero to a novel order-zero. Output, on that model, can be the same as reconstruction (except that it uses a transformed understanding)."
2. "The decoder is already implemented: this is just calling reverse, but we are separating out the higher-order reverse() involving symbols."
3. "Reconstruction does not move, we just have a different intermediate state (parallel conceptual activation)."
4. "Output does change, it turns order0 concepts into order0 concepts and then calls the reverse() method as above."
5. The generative grammar is still necessary for output; compose, think and generate are System 2 on System 1; System 1 learns from System 2 "mainly by system 2's activation of the conceptual space. That will train the expectation (which should still be separate from intuit())"; the grammar matters "as the grammar influences current symbolic activation state".
6. The two acceptance checks of item 2.0 §8: feeding an input's own order 0 through the decoder as if it were thought's output gives exactly the reconstruction; ablating STM and MTM leaves reconstruction unchanged.

## 2. What exists, in code names

- **The order-0 reverse exists.** `BasicModel.reverse` (`bin/Models.py` about 24421) is four steps: `_chart_generate_from_stm(x)`, `_reverse_body`, `_reverse_perceptual` (re-stamping `_concepts_recon` across the hand-off) and `inputSpace.reverse`. Its only callers are tests (`test/test_dual_input_contract.py`, `test/test_radix_recon_render.py`). The first step is the symbolic reverse: it reaches `SymbolSpace.reverse` (`bin/Language.py` about 14532), "grammar GENERATE over the STM snapshot", which fills `generate_rules` that only SyntacticLayer cursors not on the live chain pop.
- **Its field branch is the grammar-free decoder.** `_reverse_body(sub, *, field=None)` dispatches to `_reverse_field(sub, field)` only when `not self.word_brackets and self.concept_binding == 'aligned'` (about 24288), and returns `sub` unchanged when `field` is None. `_reverse_field` (about 24199) calls `cs.cs_percept_attribution(field.evidence, observed=field.evidence, ...)`; `_reverse_perceptual` then runs `PartSpace.reverse`'s radix-activity branch (`bin/Spaces.py` about 12024-12039), which decodes bytes by `RadixLayer.decode_activity` (detached) and stores the continuous, differentiable face `_reconstruction_percepts = activity @ radix.active_prototypes()`; `inputSpace.reverse` adopts it.
- **`_reverse_input_surface(carrier, *, field=None)`** (Models.py about 8659-8696) is the same chain, with an identity readback in front (`readback_scores` and `identity.read`) that runs first whenever `identity is not None and self.word_brackets` (about 8666). That readback is an exact lookup of the input-scoped bank, ending in long bytes: no gradient.
- **The parallel conceptual field exists.** `FamiliarTier.open_sentence` computes `a_0 = owner.cs_read_memberships(...)` (`bin/FamiliarTier.py` about 300-318), once per sentence at the open, after every word is perceived and before either compose trial: shape `[n0, B, E, 2]` (E the word slots of sentence `sid`, poles $(c^+, c^-)$), addresses `_cs_field_rows` and `_cs_field_concept_ids`, coordinates `_cs_field_where` and `_cs_field_when`, percepts `_percept_field`. It reads native percepts and definitions only, no STM or MTM; the returned tensor is live, and its gradient reaches the definition exponents (`fold_presence`, Layers.py about 5343-5384) and `PrimitiveProperties.members` (`bin/PerceptProperties.py`). Only the stored copies `_cs_last_a0` and `_definition_fold_field` are detached. The parallel path already publishes such a field to the carrier (`_concept_activations`, Models.py about 18002-18010) and reconstructs from it (`reverseReconstruct`'s field branch, about 8773-8776). `doc/Spaces.md` calls it "the parallel conceptual field".
- **Reconstruction today on the serial path** runs back through the understanding: the field objective along the compose tree (`DerivationReconstruction.reconstruct`), the free decoder (`_reconstruct_trial` to `_output_generate_walk`), and the terms recorded in [GradientFlow](../GradientFlow.md#terms-on-internals). The grammar-free terms `reconstruction_reverse.what/where/when` (Models.py about 14824-14863) are the field reverse scored against the observed native event (`_reverse_event_loss`, about 7348), a diagnostic in tied mode.
- **Output today** (`reverseOutput`, about 9690-9893): the generate walk, `cs.commit_event(walked)`, `_reverse_body`, `_reverse_perceptual`, `outputSpace.from_percepts` (an `LDUReadout` to the output shape); answer text comes from `_generated_word_text` on the walk bank; text targets are scored through `_answer_surface_from_percepts`, a second perceptual reverse. Reconstruction, output text and output tensor therefore use three different lexical decoders today.
- **Thought's order-0 writers.** `Intuition.execute` (`bin/Intuition.py` about 248-327) and `write_knowing` (about 343-367) deposit paired containments into the carrier's order-0 field, both under `@torch.no_grad()`, and `write_knowing` returns early on the serial path; `apply_thought_effect` appends occurrences (`bin/AccessibleMind.py` about 198-211, 286) with where and when of -1. Expectation's predictor is `ConceptSpread.open_prediction` with `concept_predictor = ActivationPredictor(...)`, separate from `IntuitionLayer`, and dark on the serial path (`_serial_concept_path`).

## 3. The design, in code names

### 3.1 The order-0 reverse

`BasicModel.reverse(self, x, *, field) -> SubSpace` is the decoder. `x` is a pure conceptual carrier (`conceptualSpace.subspace.carrier_like()`, its event unread). `field` is `Understanding.ConceptualField(concept_ids, evidence, where, when, percepts=None)`, with `evidence` `[S, B, E, 2]` whose rows at or above `n0` are zero. The body is the existing chain minus its first step: `_reverse_body(x, field=field)` to `_reverse_field`, then `_reverse_perceptual`, then `inputSpace.reverse`. It returns the input SubSpace with its two faces, the detached bytes and the differentiable `_reconstruction_percepts`.

Changes:

- **The symbolic step leaves `reverse`.** `_chart_generate_from_stm` is no longer called by `reverse` (§3.2).
- **The field branch runs on every aligned configuration.** `_reverse_body` dispatches `field is not None and self.concept_binding == 'aligned'` to `_reverse_field` whatever `word_brackets` is. A missing field raises; the pass-through when `field` is None goes.
- **The decode takes no perception.** `_reverse_field` lays out brackets from `field.where` alone and locates property support by `field.where`; its reads of `field.percepts.spans` go, and with them the cap `torch.minimum(output, events)` in `cs_percept_attribution`. Perception becomes the target only. Reconstruction and output then make one identical call, so acceptance check 1 holds by construction.
- **`observed=field.evidence` stays,** so the concept step is an exact identity on an observed field and the decode is a function of the order-0 field alone.
- **The identity readback becomes an audit.** In `_reverse_input_surface` it runs only when `field` is None, as an adapter for D-wide leaves and an audit; the function becomes `return self.reverse(carrier, field=field)`.
- **The activity layout is sparse.** `_reverse_field` allocates a dense `activity [B, max(inputShape[0], where.max()), len(radix)]` today, which on `data/BasicModel.xml` is 8,192 rows times the admitted percept rows per stream. The reverse reads only the brackets the field covers; `decode_activity`'s per-byte Python loop is vectorized. Preallocated at construction, never resized.

### 3.2 The symbolic reverse, separated

What involves symbols leaves `reverse` and belongs to System 2, used by output only:

1. **The generate walk** (`_output_generate_walk`, compiled by `_compiled_output_walk`), the generative grammar that output keeps and the source of word order, is lifted out of `reverseOutput` into `BasicModel.generate(idea, *, structure, occupied, infix_rows, walk_options, prepared)`, returning the emitted words with their trace and bank. It ends in a new writer that turns each emitted terminal into an order-0' occurrence: the row is the `readback_scores` argmax over `bank.terminal_valid` that `_generated_word_text` already selects, the poles the leaf's activation.
2. **The descent of thought-written higher rows**: `cs_reverse_presence(field, observed=None)` restricted to rows at or above `n0`, the step from symbols to order 0 applied before the order-0 reverse. On an observed field it changes nothing, so it matters only for fields thought writes.
3. **`_chart_generate_from_stm`** and its `SymbolSpace.reverse` retire, or stay as a report; nothing on the live chain reads what they fill.
4. **The compose-forest inverse** (`DerivationReconstruction.reconstruct`, `inverse_step`) leaves reconstruction and stays a System 2 tool.
5. **Reconstruction's free decoder** (`_reconstruct_trial` to `_reconstruct_sentences` to `_decode_conceptual_sentence`) is no longer called by reconstruction; `_complete_input_reconstruction` drops it.

### 3.3 The parallel conceptual field

The intermediate state reconstruction starts from is `a_0`, the live field `cs_read_memberships` returns in `FamiliarTier.open_sentence`. It is packaged as `ConceptualField(owner._cs_field_concept_ids, a_0 padded to the field caps, owner._cs_field_where, owner._cs_field_when, None)`, held per sentence slot on the sentence record (`_trial_understanding`), invariant across the two compose trials, and published to the carrier as the parallel pump already does (`_concept_activations`, `_concept_ids`, `_concept_inventory_rows`, `_concept_where`, `_concept_when`, `_thought_occurrence=None`). Thought then transforms the input's order 0 in place. A new serial writer does that publication; today the serial path leaves the carrier empty and thought starts from zeros.

Not this state: the per-word readout `concepts_from_percepts` (`IndexedSigmaConceptsFromPercepts`), which stays System 1's hand-off to compose, the leaves `stage_cs_lang` pushes; nor `_pushed_word_slab`, which takes interpretation's object rows, possibly MTM transients, and so would break acceptance check 2.

Reconstruction keeps its place: `_complete_input_reconstruction`, its owner `InputReconstruction`, `runBatch` and `reverseReconstruct` are where they are. Only the argument changes: `self.reverse(cs.subspace.carrier_like(), field=open_field)` replaces `_reverse_input_surface` over the recovered ideas, scored by `_reverse_event_loss(rev._reconstruction_percepts, observed_native_event)`. The serial path needs that per-byte observed native target supplied.

### 3.4 Reconstruction's terms

| Term | Under the loop |
|---|---|
| `reconstruction_reverse.what/where/when` | the reconstruction, on every configuration, from the parallel field |
| `reconstruction.what/where/when` (masked percept prediction) | unchanged |
| `<category>.definition_fold` | unchanged; it already reads the open field |
| `reconstruction.free_bytes`, `.coverage` | value-only audits of `reverse`'s decoded bytes against the input words |
| `reconstruction.free_percept` | keeps its residual form once its forecast context stops reading `_last_sentence_understanding.end_slots` |
| `reconstruction.part` (a test lesson) | reads back from `reverse` |
| `reconstruction.field`, `.iterations`, `.forward_discard` | leave reconstruction with the compose-forest inverse |
| `reconstruction.walk_residual` | the decoder lane goes; the answer lane is output's |
| `reconstruction.walk_credit` | the decoder lane goes; the output lane is output's |
| `reconstruction.compose_score_function` | output's, whole (item 2.0 §8) |
| `reconstruction.reference` | output's judge (D2) |
| `reconstruction.content` | leaves reconstruction: the field reverse has no meaning face (D1) |
| `reconstruction.decomposition`, `.walk_policy` | retired (item 2.0 §§3-4) |

`expectation.*` stay expectation's; `expectation.percept`'s context stops reading `end_slots`.

### 3.5 Output, and order 0'

Order 0' is the carrier field `conceptualSpace.subspace._concept_activations` after thought, its order-0 rows: the input's brackets followed by thought's occurrences. Output is `generate`, then the occurrence writer, then `self.reverse(cs.subspace.carrier_like(), field=ConceptualField(ids, order-0 rows of the occurrences System 2 selects, in output order, where', when', None))`. Answer text targets are scored as reconstruction is. `_answer_surface_from_percepts` retires; the conceptual and perceptual `synthesis_layer` retire (never applied on the walked path), with their checkpoint keys dropped as before; `outputSpace.from_percepts` stays for numeric heads only (D4).

Thought occurrences carry where and when of -1 today. Under the loop they get consecutive brackets of their literal's length plus a separator, in occurrence order (D5).

### 3.6 Thought's producers and expectation's target

All System 2:

- **intuit, within a step** (`ThoughtFaces.intuit` to `Intuition.execute`), whose `knowing` pairs `write_knowing` deposits into the field. Its weights are trained by output (thinking spec §22), and under the loop output reaches them only if the deposit carries gradient (D6).
- **Thought effects** (`apply_thought_effect`'s reference and frame seeds).
- **The generate grammar** (§3.2), which writes ordered terminals as occurrences: this is how the grammar influences the symbolic activation state and why it is still needed for output.
- **Expectation, across steps** (`ConceptSpread.open_prediction` with `ActivationPredictor`), separate from intuit. Its target becomes the detached order-0 slice of the carrier after thought, System 2's activation, so that System 1 learns from System 2 through it; it is enabled on the serial path (D3). Its prediction is pooled over occurrences, so it is rendered only after the grammar orders it.

### 3.7 The gradient chains

Under the loop the reconstruction chain is perception, the parallel field (definition exponents, `PrimitiveProperties.members`, the percept codebook) and the order-0 reverse; the output chain is thought, the generate grammar, and the same reverse. The projection of item 2.0 §1 acts on what both reach: perception, the field's parameters and the reverse. Every grammar parameter has one writer, output. The reconstruction-side bookings of item 2.0 §2 that were interim (the decoder walk's credit and residual) disappear with the decoder walk.

### 3.8 Verbs and modals over the state of affairs

The memory spec's causality links ([2026-11-03-item2-0-behavioral-chains.md](2026-11-03-item2-0-behavioral-chains.md)) give consecutive order-0 states of one course of events. The verb action is trained to map an identity's earlier order-0 state to its later one, and the modifier action to map the actual state to the alternative state at equal time; both states are readings of input. The re-read of item 2.0 §5 stays beside this, so later answers also credit the verb.

### 3.9 Fail loud

The stage loop in `_reverse_body` swallows a `RuntimeError` at every stage, and `runBatch` wraps `reverseReconstruct` in `except Exception`; both would hide a broken shared decoder and both go. `ConceptualSpace.reverse` names an undefined `_inputShape` on `ProjectionBasis` configurations, a latent `NameError`; it is repaired.

## 4. Gates (one frozen hash, sized to the change; certificates, no learning claim)

1. **Acceptance check 1.** `reverse` given the input's own parallel field as output's order 0' returns exactly reconstruction's bytes and `_reconstruction_percepts`: one call, compared byte for byte.
2. **Acceptance check 2.** Reconstruction is unchanged with STM and MTM emptied before the open.
3. **`reverse` is grammar-free.** With `_chart_generate_from_stm`, `SymbolSpace.reverse`, `_output_generate_walk` and `DerivationReconstruction.reconstruct` patched to raise, reconstruction runs.
4. **Gradients.** Reconstruction's loss reaches the definition exponents, `PrimitiveProperties.members` and the percept codebook, and no grammar parameter; output's loss reaches the generate grammar and, under D6, intuit's weights.
5. **Output.** `generate`, the occurrence writer and `reverse` produce the answer; `_answer_surface_from_percepts` and the `synthesis_layer`s are absent, with certificates; old checkpoints load with their dropped keys reported.
6. **Memory.** The reverse's allocation read shows no dense `[B, 8192, len(radix)]` activity; preallocated, never resized.
7. **Fail loud.** A decoder that raises inside a stage propagates; the swallows are absent.
8. **Terms.** The table of §3.4 holds in the registry; [GradientFlow](../GradientFlow.md) is updated by exact replacement.
9. **Tests ported** with before and after text: `test/test_dual_input_contract.py` and `test/test_radix_recon_render.py`, which call the old `reverse`; the reconstruction and output files whose terms move. The unchanged sets on the same hash, one native lesson smoke, `make test` at the end of item 2.0.

Rules, as in every round: no language named in code, fixtures or tests; no seed selected to pass; nothing retired silently; the item 1.5 lookups are ported onto this tree afterwards, keeping the `working[:n0]` contract that `cs_read_memberships` and `_compose_order0` share.

## 5. Decisions for Alec, with recommendations

- **D1 `reconstruction.content`** leaves reconstruction, since the field reverse has no meaning face. Recommended: yes.
- **D2 `reconstruction.reference`** moves to output's judge, since the bind is System 2. Recommended: yes.
- **D3 Expectation's target** becomes the detached order-0 slice of the carrier after thought, and expectation is enabled on the serial path. This changes Architecture's statement that concept expectation selects active concepts on the parallel pass. Recommended: yes, as Alec's "System 1 learns from System 2 ... that will train the expectation".
- **D4 Numeric heads** (`data/model.xml`, `answerSynthesis` false) keep the forward head; the loop applies to synthesis configurations this round. Recommended: yes.
- **D5 Brackets for thought occurrences**: consecutive brackets of the literal's length plus a separator, in occurrence order. Recommended: yes.
- **D6 The knowing deposit carries gradient**, so output reaches intuit's weights through the field as §22 intends. Recommended: yes; `Intuition.execute`'s choices stay discrete and credited by the comparison.

## 6. Where

After the review of the §41 return, inside item 2.0's one Codex text, on main, after the memory file and the gradient changes of item 2.0 §§1-5. Files: `bin/Models.py` (`reverse`, `_reverse_body`, `_reverse_field`, `_reverse_input_surface`, `_complete_input_reconstruction`, `reverseOutput`, the new `generate` and occurrence writer, `runBatch`'s wrapper), `bin/Spaces.py` (`cs_percept_attribution`'s cap, `ConceptualSpace.reverse`'s name, the serial publication of the open field, `PartSpace.reverse`'s activity layout), `bin/Layers.py` (`RadixLayer.decode_activity`), `bin/FamiliarTier.py` (`open_sentence` hands the live field to the record), `bin/Intuition.py` and `bin/AccessibleMind.py` (the deposit and the brackets, under D5 and D6), `bin/ConceptSpread.py` (expectation's target, under D3); `doc/GradientFlow.md`, `doc/Architecture.md` and `doc/Spaces.md` by exact replacement where they describe reconstruction through the understanding.
