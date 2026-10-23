## Verdict: **HOLD**

As instructed, I only read files and ran nothing. I reviewed frozen source `867452a79` against its receipt and ledgers, the whole `source-diff.patch`, the new tests, the §§58–64 plan text, and the retained captures. The gate summary reads 1,109 selected cases: 1,100 passed, 9 existing skips, 0 failures.

The B1–B3 repairs and the follow-ups are sound. One new problem blocks landing: the §64 word-form repair changes much more than the receipt says.

### Blocker

**N1. §64 makes every bound word row spellable, not only a word that is also another word's referent.** That changes which rows can end a word walk, the reconstruction byte targets, the derived form codes, and the reverse decoder's candidates. The receipt says "no association, conceptual reading, choice, identity audit, or existing assertion changes" (`README.md:19-24`, plan §64). That is not accurate.

- **What the old reader did.** A word that has an object leaves `_pending` (`Interpret.py:300`). So `lexical_form(word)` went to `definitions.words(word)` (`Interpret.py:71-73`). For an ordinary bound word that is empty, so the word's row had **no surface**. Only objects and objectless words could be spelled. The old docstring says the same: "Spell a witnessed object or a form still waiting for its object."
- **What the new reader does.** `description()` is keyed by the word side of every definition edge (`Definitions.py:61`). So `Spaces.py` `word_surface_for_row` now returns its own form for **every** bound word row. Your own capture shows this. In `held-content/rung0-replay/capture/failure.json`, the selected word rows 11, 13, 17 and 18 (words 12, 14, 18, 19) have `"expected": null`. Under §64 they all gain surfaces. The new regression's first assertion (word row reads `b'12'`) would also fail under the old reader without the `define` call. It certifies the broad change, not only the referent case.
- **Effects downstream:**
  - Word walks: `PrimedSymbols.terminal_valid` is "only a row with a lexical realization can end a word walk" (`SentenceUnderstanding.py:84-86`). Its bytes come from `surfaces` (`Models.py:6772-6790`), so the sentence's own word rows are now valid terminals next to their objects. Both emit the same bytes, so choices can now split between them.
  - Readback: readback selection uses those terminals (`Models.py:12166-12173`, `13470-13475`).
  - Rung-zero audit: it now checks many more rows (`Models.py:6776-6782`).
  - Derived codes: `derive` now pins every bound word row's form slice to `identity.form(raw)` instead of its part interval `lower` (`MereologicalCodes.py:231-236`).
  - Reverse decoding: `RadixLayer.reverse` adds every word row to its candidates (`Layers.py:12663`).
- **Gate coverage.** Direct consumers of this reader are not in any selected group: `test_radix_layer_reverse.py`, `test_active_decode_candidates.py`, `test_echoic_decoder.py` and `test_tied_operator_reconstruction.py`.
- **Fix (pick one; Alec's call):**
  - **(a) Narrow repair (recommended).** A bound word row never spells through a definition. If `definitions.objects(identity)` is non-empty, return `None`, which was the old value for bound words. Objects and objectless words keep their current paths. The capture suggests this also clears the failure: ordinary bound word rows already decode correctly from `lower` (e.g. row 18 → `"6"`). Bonus: who owns a row that is both a word and an object then stays open for the upcoming identity spec.
  - **(b) Keep the broad behaviour.** Declare it as a deliberate change to terminals, audit and derived codes. Add regressions for an ordinary bound word, a terminal ambiguity case and `reverse`. Bring the four files above into scope.

  Either way it needs a re-freeze and a rerun of the three groups.

### Rechecked and sound
- **B1:**
  - `ThoughtClosing.py:41` takes the mode from the live parse or clause onto the committed row.
  - `Models.py:20419` puts the clause mode on the retained meaning, and no mode is stored.
  - The held reader (`Models.py:5738-5758`) applies query intent only to LTM addresses and follows only a valid signature. Live thought records keep the mode guard and still raise loudly on errors.
  - The four write-and-close cases and the two empty-mask cases assert the right things, and the `kind='observation'` setup fix leaves the assertions unchanged.
- **B2:** I checked every changed test file in `source-diff.patch` against `frozen-source.json`; each is in a group, including all eight native files. The changes to the shared test helpers only add code.
- **B3:** Occurrence end states keep `(-1,-1,-1)`. Part returns keep `(left,-1,right)` (`Models.py:20419-20426`). Typed references stay in the meaning.
- **Follow-ups:** the predicate cache is gone (`ClauseRow.py:283-297`). The `where` entry and the §46 wording are in place.
- **§60 predicate ports:** both writers read the same (`predicate-arguments.json`). The link from each source occurrence to its argument is the real content check.
- **§61 joint-objective port:** it reads the actual bounded predictor context, and the gradient and update assertions are unchanged.
- **§41 notes:** forgetting, kind-based question refusal and the mode-consumer list are unchanged from the version I accepted.

### Non-blocking notes
1. **Part answer seeds.** On the Part path, `Models.py:20419` now also applies the clause mode. Before 2b, a Part meaning read back as `assertive`, so a completed *interrogative* relation was an answer seed; now it is skipped. That is probably the right behaviour, but it should be declared.
2. **Mode precedence.** `ThoughtClosing.py:41` prefers `parsed.mode`; before 2b the clause's mode was stored. These are the same in every fixture, but worth stating.
3. **Broad exception catch.** The `(TypeError, ValueError)` catch at `Models.py:5753` also silently gives an empty mask for malformed stored thoughts (wrong slots, negation without a truth value). This matches the idiom at `:6313` and `:20307`, but an explicit eligibility check would be clearer.
4. **Predicate port strength.** The `roles[0] == leaves[role]` check mostly confirms the slot was not changed. A uniqueness check (exactly one occurrence has root `('sym', argument)`) would make the identity link functional.
5. **Narrow-idea diagnostic.** `narrow-idea.json` confirms both writes land on row 0. The sharper check is correct, but it tests a second sighting of the same row, not a new row.
6. **Untested consumers.** Several files that use the changed helpers or production paths are outside every group: `test_ltm_consolidation`, `test_existence_ingestion`, `test_thought_binding`, `test_truth_ideas_routing`, `test_query_work_budget`. The step-2 full sweep should cover them.
7. **Observer key collision.** The copied observer keys priming by surface (`test-ports.patch:442-445`). Under the current §64 a word row and its object produce the same key, and one silently overwrites the other.

There is no learning claim to assess. Once N1 is fixed, a re-freeze and rerun of the three groups (plus the reader's consumer files, if you take option b) should be all that is needed to land.
