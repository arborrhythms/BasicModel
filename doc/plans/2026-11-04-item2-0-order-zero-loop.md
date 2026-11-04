# Item 2.0, the order-zero loop: one order-0 reverse, the parallel conceptual field, output by the same reverse (Claude, 2026-11-04; spec in code names, with decisions for Alec)

**Status.** The loop was decided in direction on 2026-11-03 ([item 2.0 spec §8](2026-11-03-gradient-projection.md), the "Thinking is Optional" conversation) and refined by Alec the same evening and on 2026-11-04 (§1 below; §7 records the revision of 2026-11-04). This is its design in code names, the last part of item 2.0. Governed by step-2 §39 ([2026-10-21-step2-one-ltm.md](2026-10-21-step2-one-ltm.md)): within an authorized round Codex reviews the spec, fixes what the code contradicts and continues, documenting each decision as a dated section; only reversing a §1 decision comes back to Alec. The Codex text is held until §5 is decided and joins item 2.0's one Codex text. Code names and line numbers are from the 2026-11-04 working tree, which carries Codex's uncommitted item 2.5 completion round; Codex re-locates them. The derivation was read from the code and not run.

## 1. The record (Alec, 2026-11-03)

1. "Reconstruction should be initiated from order-zero concepts rather than explicit understanding (STM0 and STM1). Understanding is a thought process that transforms order-zero to a novel order-zero. Output, on that model, can be the same as reconstruction (except that it uses a transformed understanding)."
2. "The decoder is already implemented: this is just calling reverse, but we are separating out the higher-order reverse() involving symbols."
3. "Reconstruction does not move, we just have a different intermediate state (parallel conceptual activation)."
4. "Output does change, it turns order0 concepts into order0 concepts and then calls the reverse() method as above."
5. The generative grammar is still necessary for output; compose, think and generate are System 2 on System 1; System 1 learns from System 2 "mainly by system 2's activation of the conceptual space. That will train the expectation (which should still be separate from intuit())"; the grammar matters "as the grammar influences current symbolic activation state".
6. The two acceptance checks of item 2.0 §8: feeding an input's own order 0 through the decoder as if it were thought's output gives exactly the reconstruction; ablating STM and MTM leaves reconstruction unchanged.
7. (2026-11-04) "We used to have an internal state that was all symbols. Now, higher-order symbols are optional; the input can be reconstructed from zeroth order symbols (or their associated concepts), and higher order concepts participate by altering the zeroth order face, which is very meaningful ... So I guess I'm saying there is an input reconstruction, and for a human mind, an order-zero reconstruction; symbols do play a part in the latter. But we can keep calling these stages input, output, thinking, where reconstruction is 'input (thinking)* output', and knowing is order zero concepts and stm, which can be dense (subsymbolic) or sparse (symbolic)."
8. (2026-11-04) "MTM, when it links order-zero concepts (events), creates a single extended event (sequence). When it links order-one concepts (nouns), creates a single extended noun (identity). It exists as a precursor to concepts and (taxonomic) relations that store the same thing." The question was whether sequence learning in memory over order-0 concepts changes D2.
9. (2026-11-04) "Order 0 concepts are a part of System 1"; the last perceptual layer is basic percepts, which tie parts and wholes and are collected by order-0 concepts ([basic percepts](2026-11-04-item2-0-basic-percepts.md)).
10. (2026-11-04) "Thought occurrences is not a valid structure. Thoughts are, more formally, ideas and relations (e.g. stm0 and stm1) across the architecture. The occurrence of a thought (as a concept) is represented by its associated percept, or when in an idea or relation, simple its .when. I think ... please check this logic for fit." §3.5 records the check.

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
- **The decode runs through basic percepts and reads no perception spans.** An order-0 concept's evidence goes to its basic percepts, and a basic percept to its parts and bytes at the field's bracket ([basic percepts §3.4](2026-11-04-item2-0-basic-percepts.md)). `_reverse_field`'s reads of `field.percepts.spans` go, and with them the cap `torch.minimum(output, events)` in `cs_percept_attribution`; perception becomes the target only. (The first version of this spec said the decode takes no perception; that was wrong for today's WholeSpace step, which spreads whole columns over the covering perception spans, and basic percepts are what remove it.) Reconstruction and output then make one identical call, so acceptance check 1 holds by construction.
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

In Alec's terms (§1, 7) the stages stay input, thinking and output, and reconstruction is input (thinking)* output: the order-0 face is reversed after zero or more thinking steps. Knowing is the order-0 concepts and STM, dense (subsymbolic) or sparse (symbolic). With no thinking step this is the **input reconstruction**, which involves no symbol and no grammar and is what acceptance checks 1 and 2 test. After understanding, the composed symbols descend to order 0 (§3.2, item 2) and alter the order-0 face; reversing that face is the **order-zero reconstruction**, in which symbols take part (D1).

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
| `reconstruction.reference` | memory's link judge for an extended noun, beside the same judge for an extended event (D2) |
| `reconstruction.content` | replaced by the order-zero reconstruction `reconstruction_order_zero.what/where/when` (D1) |
| `reconstruction.decomposition`, `.walk_policy` | retired (item 2.0 §§3-4) |

`expectation.*` stay expectation's; `expectation.percept`'s context stops reading `end_slots`.

### 3.5 Output, and order 0'

Order 0' is knowing after thought: the carrier field `conceptualSpace.subspace._concept_activations` at the input's brackets, as thought has altered it, together with STM's ideas and relations. Output is `generate`, then the occurrence writer, then `self.reverse(cs.subspace.carrier_like(), field=ConceptualField(ids, the order-0 concepts System 2 selects, in output order, where', when', None))`. Answer text targets are scored as reconstruction is. `_answer_surface_from_percepts` retires; the conceptual and perceptual `synthesis_layer` retire (never applied on the walked path), with their checkpoint keys dropped as before; `outputSpace.from_percepts` stays for numeric heads only (D4).

**Thoughts as ideas and relations (the check of §1, 10).** The logic fits the code in all but two places.

- *Fits.* STM's ideas and relations carry `.when` as a chunk field (`STMChunks.FIELDS`), and the deposits from LTM recall, prepended symbols and the clause journal pass it (`ChunkJournal.py` about 166, `EntryRead.py` about 219, `PrependedSymbols.py` about 214). Compose's pushes (`Layers.py` about 14394) pass none, but each belongs to its sentence, whose entry is stamped. A concept rendered for output is represented by its percept: under the loop, the basic percepts of its order-0 concept, placed at the brackets the generate grammar's order gives.
- *Misfit 1: the appended field columns.* `apply_thought_effect` appends columns to the order-0 field with `_concept_where` and `_concept_when` of -1, marked by `_thought_occurrence`, and `Intuition.write_knowing` writes at those columns. These are the structure Alec calls invalid: neither an idea or relation with a `.when` nor a concept represented by its percept. They go. `write_knowing` writes into knowing instead (an order-0 concept's evidence at the input's brackets, or an STM idea with its `.when`); the reference and frame seeds become STM ideas; `_thought_occurrence` and its readers (`AccessibleMind`, `Intuition`, the `Models` checkpoint, the `WalkTrials` snapshot, the resets) go with it, with the checkpoint key dropped and reported.
- *Misfit 2: thought's own ideas have no time.* `ThoughtChunks.put` deposits with `where=address(identifier + 1)`, a synthetic four-digit key, and no `when`, and `ThoughtChunks` finds its ideas by `chunks.locate(state, where)`. A thought idea's `.where` is used as an address, not a location. Under the rule each thought idea is stamped with the `.when` of the thought step that made it (a model clock that counts thought steps), and is found by that `.when`; its `.where` stays empty (-1), since an idea from thought has no location (D5).

### 3.6 Thought's producers and expectation's target

All System 2:

- **intuit, within a step** (`ThoughtFaces.intuit` to `Intuition.execute`), whose `knowing` pairs `write_knowing` deposits into the field. Its weights are trained by output (thinking spec §22), and under the loop output reaches them only if the deposit carries gradient (D6).
- **Thought effects** (`apply_thought_effect`'s reference and frame seeds).
- **The generate grammar** (§3.2), which writes ordered terminals as occurrences: this is how the grammar influences the symbolic activation state and why it is still needed for output.
- **Expectation, across steps** (`ConceptSpread.open_prediction` with `ActivationPredictor`), separate from intuit. Its target becomes the detached order-0 slice of the carrier after thought, System 2's activation, so that System 1 learns from System 2 through it; it is enabled on the serial path (D3). Its prediction is pooled over occurrences, so it is rendered only after the grammar orders it.

### 3.7 The gradient chains

Under the loop the reconstruction chain is perception, the parallel field (definition exponents, `PrimitiveProperties.members`, the percept codebook) and the order-0 reverse; the output chain is thought, the generate grammar, and the same reverse. The projection of item 2.0 §1 acts on what both reach: perception, the field's parameters and the reverse. Every generate-grammar parameter has one writer, output; the compose rules have two, output and the order-zero reconstruction (D1). The reconstruction-side bookings of item 2.0 §2 that were interim (the decoder walk's credit and residual) disappear with the decoder walk.

### 3.8 Verbs and modals over the state of affairs

The memory spec's extended events ([2026-11-03-item2-0-behavioral-chains.md](2026-11-03-item2-0-behavioral-chains.md)) give consecutive order-0 states of one course of events. The verb action is trained to map an identity's earlier order-0 state to its later one, and the modifier action to map the actual state to the alternative state at equal time; both states are readings of input. The re-read of item 2.0 §5 stays beside this, so later answers also credit the verb.

### 3.9 Fail loud

The stage loop in `_reverse_body` swallows a `RuntimeError` at every stage, and `runBatch` wraps `reverseReconstruct` in `except Exception`; both would hide a broken shared decoder and both go. `ConceptualSpace.reverse` names an undefined `_inputShape` on `ProjectionBasis` configurations, a latent `NameError`; it is repaired.

## 4. Gates (one frozen hash, sized to the change; certificates, no learning claim)

1. **Acceptance check 1.** `reverse` given the input's own parallel field as output's order 0' returns exactly reconstruction's bytes and `_reconstruction_percepts`: one call, compared byte for byte.
2. **Acceptance check 2.** The input reconstruction is unchanged with STM and MTM emptied before the open. (The order-zero reconstruction reads the symbols of understanding, so it is exempt.)
3. **`reverse` is grammar-free.** With `_chart_generate_from_stm`, `SymbolSpace.reverse`, `_output_generate_walk` and `DerivationReconstruction.reconstruct` patched to raise, the input reconstruction and the order-zero reconstruction run.
4. **Gradients.** The input reconstruction's loss reaches the definition exponents, `PrimitiveProperties.members` (through the basic-percept read) and the percept codebook, and no grammar parameter; the order-zero reconstruction's loss also reaches the compose rules and the symbols' definitions through the descent, and no generate parameter; output's loss reaches the generate grammar and, under D6, intuit's weights.
5. **Output.** `generate`, the occurrence writer and `reverse` produce the answer; `_answer_surface_from_percepts` and the `synthesis_layer`s are absent, with certificates; old checkpoints load with their dropped keys reported.
6. **Memory.** The reverse's allocation read shows no dense `[B, 8192, len(radix)]` activity; preallocated, never resized.
7. **Fail loud.** A decoder that raises inside a stage propagates; the swallows are absent.
8. **Terms.** The table of §3.4 holds in the registry; [GradientFlow](../GradientFlow.md) is updated by exact replacement.
9. **Tests ported** with before and after text: `test/test_dual_input_contract.py` and `test/test_radix_recon_render.py`, which call the old `reverse`; the reconstruction and output files whose terms move. The unchanged sets on the same hash, one native lesson smoke, `make test` at the end of item 2.0.
10. **Thoughts.** No column is appended to the order-0 field by thought (`_thought_occurrence` absent, with a certificate); every idea `ThoughtChunks.put` deposits carries a `.when` and is found by it.

Rules, as in every round: no language named in code, fixtures or tests; no seed selected to pass; nothing retired silently; the item 1.5 lookups are ported onto this tree afterwards, keeping the `working[:n0]` contract that `cs_read_memberships` and `_compose_order0` share.

## 5. Decisions for Alec, with recommendations

- **D1 The order-zero reconstruction replaces `reconstruction.content`.** Today `reconstruction.content` (Models.py about 20455-20490) compares the meaning bands, the `[meaning_start:]` part, of the free decoder's read-back leaves with the clause's own word leaves, signed lanes summed separately; it exists so that "a surface can be spelled from its form even after its meaning was erased" cannot pass. The loop's decoder reads the order-0 field and has no leaves with meaning bands, which is all the first version's "the field decoder has no meaning face" meant. Proposed in its place, from §1, 7: `reconstruction_order_zero.what/where/when`, the order-0 face after understanding (compose's symbols descended to order 0 by `cs_reverse_presence` over rows at or above `n0`) reversed by the same `reverse` and scored against the same observed native event as the input reconstruction. Symbols may alter the order-0 face as meaningfully as they like among concepts that share basic percepts, and are charged only when the alteration loses the input. It is the content term's purpose in the new frame, and it gives compose a second writer beside output, which item 2.0 §1's projection handles (§3.7). Recommended: yes. The alternative retires `reconstruction.content` with nothing in its place this round.
- **D2 `reconstruction.reference`** is the coverage judge of a link: a closing mention's description should be covered by the record it was linked to (`ReferenceReconstruction.cost`). With Alec's point (§1, 8) this is memory's judge, not output's: linking order-1 concepts makes an extended noun and linking order-0 concepts makes an extended event, by the one `link` of the memory spec, and the same coverage judge applies to both, on input readings. So sequence learning over order 0 does change D2: the first version moved the term to output's judge because the bind is System 2; it moves instead to `ConceptualSpace.link` as `memory.link_coverage`, one term for both orders, and its gradient reaches what the coverage reads (the resolver's coordinates), the choice staying discrete and credited by the departure rule. Recommended: yes.
- **D3 Expectation's target** becomes the detached order-0 slice of the carrier after thought, and expectation is enabled on the serial path. This changes Architecture's statement that concept expectation selects active concepts on the parallel pass. Recommended: yes, as Alec's "System 1 learns from System 2 ... that will train the expectation".
- **D4 Numeric heads** (`data/model.xml`, `answerSynthesis` false) keep the forward head; the loop applies to synthesis configurations this round. Recommended: yes.
- **D5 Thoughts as ideas and relations** (§3.5, replacing the first version's brackets for thought occurrences): the appended field columns and `_thought_occurrence` go; `write_knowing` writes into knowing; each thought idea in `ThoughtChunks` gets the `.when` of its thought step and is found by it, with `.where` empty. Recommended: yes.
- **D6 The knowing deposit carries gradient**, so output reaches intuit's weights through the field as §22 intends. Recommended: yes; `Intuition.execute`'s choices stay discrete and credited by the comparison.

## 6. Where

After the review of the §41 return, inside item 2.0's one Codex text, on main, after the memory file and the gradient changes of item 2.0 §§1-5. Files: `bin/Models.py` (`reverse`, `_reverse_body`, `_reverse_field`, `_reverse_input_surface`, `_complete_input_reconstruction`, `reverseOutput`, the new `generate` and occurrence writer, `runBatch`'s wrapper), `bin/Spaces.py` (`cs_percept_attribution`'s cap, `ConceptualSpace.reverse`'s name, the serial publication of the open field, `PartSpace.reverse`'s activity layout), `bin/Layers.py` (`RadixLayer.decode_activity`), `bin/FamiliarTier.py` (`open_sentence` hands the live field to the record), `bin/Intuition.py` and `bin/AccessibleMind.py` (the deposit and the brackets, under D5 and D6), `bin/ConceptSpread.py` (expectation's target, under D3); `doc/GradientFlow.md`, `doc/Architecture.md` and `doc/Spaces.md` by exact replacement where they describe reconstruction through the understanding.

## 7. Revised (Claude, 2026-11-04, from Alec's answers of the same day)

Alec answered D4 and D6 "Good", asked about D1, D2 and D5, and answered D3 with "Order 0 concepts are a part of System 1" (§1, 7-10). D3 is consistent with that: expectation learns in System 1's space from System 2's activation of it; the same answer led to basic percepts. The revision: §3.1's decode runs through [basic percepts](2026-11-04-item2-0-basic-percepts.md), which corrects the first version's claim that the decode takes no perception; §3.3 records the stages and the two reconstructions; D1 becomes the order-zero reconstruction; D2 moves the reference term to memory's link; D5 becomes the check of thoughts as ideas and relations; gates 2-4 and 10 follow. D3, D4 and D6 stand as written; D1, D2 and D5 are open again in their revised form.

