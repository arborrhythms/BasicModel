# Code review of `origin/item1-5` (Claude, 2026-11-27)

Reviewed: `origin/item1-5` at `4ac29137c5` against `main` at `ffbed4fd5b`
(10 commits, 44 files, about 2,960 lines added and 820 removed). The whole
diff was read. The branch was not checked out and none of its tests were
run, so these are findings from reading; where one was confirmed against the
branch's source it says so. Eight are to fix before the merge, two are minor.

A trial merge with `origin/item1-4` (`git merge-tree`) conflicts in
`bin/BasicPercepts.py`, `bin/Models.py` and `bin/Spaces.py`;
`ConceptSpread`, `FamiliarTier`, `FieldProjection`, `Salience` and
`SentenceStrata` merge by themselves.

## To fix before the merge

1. **The replayed prefix of an explored reading is one look short**
   (`bin/SentenceLooks.py:353`; confirmed). `_look_interpret_and_spread`
   compares `state.taken` with `draw['attention_round']`, but `run()` calls
   `state.record_look(...)` first (line 521), which adds one to `taken`
   (`bin/Attention.py:55`). The code it replaced used `turn`, read before
   that. With a departure at look 3, look 2 is no longer replayed: it draws
   afresh, and the paired cost difference is credited to a departure the
   paths had already left. The function is shared, so eager runs have it
   too. Fix: pass `turn` in.
2. **A new place's key can be lost** (`bin/Places.py:106`). `_meet_tensor`
   writes keys with `scatter_` at `safe_seat`. A second reading of the same
   new place targets the same seat, is not `novel`, and so writes the seat's
   old contents while the first writes the key; `scatter_` with one index
   and two values is not determined. The place is then met as new again.
   Fix: write `qkeys` wherever the seat is new.
3. **The host and the compiled path number places apart**
   (`bin/Places.py:160`, `:121`; confirmed). The host numbers seats from the
   dict `basic._place_cone_rows` and sets `_place_cone_used` to its length;
   `_meet_tensor` numbers from `_place_cone_used` and never adds to the
   dict. A place met compiled and the next met on the host take one seat.
   The keys are also rounded differently (`round(v, 6)` on doubles against
   `torch.round(c * 1e6)` in float32), and the compiled path returns `[]`,
   so its counts never nominate a place's row. Fix: one registry, one key.
4. **A failed host prologue leaves its staged input**
   (`bin/Models.py:8710`). `_invoke` sets `_staged_in_sub` before the
   `prepare_*` calls and enters the `try/finally` that clears it about a
   hundred lines later. If a preparation raises, the next forward finds
   `_staged_in_sub` set, skips the prologue, and reads the new input against
   the old one's staged words. Fix: open the `try` before the assignment.
5. **Staging for compiled regions runs on every eager forward**
   (`bin/Models.py:8716`). `FormCentroids.prepare_host` loops over every
   concept row in Python (`bin/FormCentroids.py:315`; 65,536 in
   `data/BasicModel.xml`); `DefinitionFold.prepare_host` walks every part
   witness with a device read each, and calls `_prepare_part_learning` for
   all of them (line 89), where `prepare` staged only those the field
   exemplified. Fix: only where `_compiled_sentence_regions` is set, and
   from the same records as `examples()`.
6. **Compiled regions read host tables taken once for a forward**
   (`bin/PrependedSymbols.py:425`). `prepare_host_stage`,
   `prepare_symbol_bank_host`, `_prepare_closure_plan_host`,
   `_form_percept_row_map`, `_def_fold_host` and `_compiled_ref_banks` are
   built in the prologue, but a sentence's commit adds MTM entries, promotes
   symbols and seats concepts before the next sentence of the same input.
   The new `if torch.compiler.is_compiling() and cached is not None`
   returns (`case_edges`, `seated_names`, `inventory_indices`,
   `spread_indices`, `concept_rows`, `spelling_seats`, `_occupied_names`,
   `DefinitionIndex._ensure`) skip their own stamps in the same way. Fix:
   refresh at each sentence's boundary on the host.
7. **Lookup caches keyed by the `id()` of a dict that is replaced**
   (`bin/Layers.py:6917`, `:6611`). `concept_rows` and `inventory_indices`
   kept the dict and compared by identity; they now keep `id(...)` and a
   length. A checkpoint restore replaces the map; a new dict of that length
   at the freed address reads as unchanged. Fix: keep the reference, or a
   revision counter.
8. **A change to when a sentence's root depth is valid, outside any compile
   gate** (`bin/Models.py:23665`). Where `interpret.mandatory` is false, a
   root holding an occurrence now makes the depth invalid, in eager runs
   too. No test in the branch covers it and no commit message names it.
   Fix: confirm it is meant; if so, its own commit and certificate.

## Minor

9. **Compiled and eager runs take different draws from one seed**
   (`bin/Language.py:6702`; `bin/BasicPercepts.py:1874`;
   `bin/Models.py:23236`). The compiled interpret bias draws for every
   offered word where eager draws for the fresh ones; the lookup uses the
   full width where its comment says the precise width keeps attention's
   draw; the compiled reservoir draws once a round. Either draw full width
   in both, or say that replay parity holds within a mode only.
10. **The whole-forward gate tests reading of one look**
    (`test/test_sentence_fullgraph_regions.py:85`). It sets
    `hardwire_reading` false, which makes `Looks.run` take one look
    (`bin/SentenceLooks.py:454`), because the default 32 take about 38
    minutes to unroll. `architecture.hardwireReading` is a new key read only
    in code (`bin/Models.py:16801`): it is in neither `data/model.xml`,
    `data/model.xsd` nor `doc/Params.md`. Fix: document it, and a slow gate
    over more than one look.

## Read and found sound

The field-name lists that replace `dataclasses.fields` (`SentenceUnderstanding`,
`InputReconstruction`, `ExpectedWords`) match their classes as they stand on
the branch. The reverse's draws are taken before the compiled region in the
order the eager code takes them.
