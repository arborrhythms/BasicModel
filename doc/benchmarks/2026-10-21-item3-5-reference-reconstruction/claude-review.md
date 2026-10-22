I recommend landing item 3.5 line 1 at `3a9db635e`. I found no blocking correctness or contract issue. One design question for Alec comes out of finding 1, and it can wait until after landing. This review was read-only: I edited no files and ran no gates.

## 1. Grammar-only contract

- **Five §10 sites:**
  - `ClauseJournal.finish_clause` now opens a leaf request only when the leaf's stamped order is 1 (`bin/ClauseJournal.py:17-20, 92-95`). It reads the stamp from `reference_orders` where set, else `leaf_orders`, and the request carries the leaf's own value as content and cue.
  - `close_member` also requires order 1 (`:363`) and still requires a participant position (`:365`).
  - `Language.resolve_lexical_references` and `reference_requests` no longer take `no_object`.
  - The marker test in `Models.py:20618-20621` now reads only the `surface_argin1` occupation.
  - `PartClosing.preflight` finds markers by their surface argument position in the plan.
- **Both `ReferenceClose.sites` recovery branches** (`bin/ReferenceClose.py:629-675, 688-697`): a site is recovered only from a leaf that is a participant term, stamped 1, and carries the forward name. Content and order come from that leaf. No hard-coded order 1 or zero point remains, and an open role whose leaf is not order 1 gets no site.
- **`PartRead.argument_reader`:** the definitions read and the object-type veto are gone (`bin/PartRead.py:102-137`). It reads only the symbol's own occupations. Its callers get the same reader, and `test_retained_argument_classification_uses_only_its_own_occupations` fails if the definitions are read at all.
- **What remains is allowed:**
  - The `no_object` calls left in `Spaces.py:14164, 16606, 16742`, `Models.py:13484` and `Interpret.py:362, 430` are all the "a word stands for itself" identity fallback that §10 keeps.
  - `'pronoun'` now appears only in sets of modes that a grammar rule declares (`ClauseJournal.py:142, 204, 297`, `ReferenceContext.py:278, 391, 491, 511`). Nothing creates it.
  - The only order condition is §12's site rule. It is not a grammar rule or a forced choice.

## 2. The reference term

`bin/ReferenceReconstruction.py:18-30` and `bin/Models.py:19823-19833` match §§8–9:

- **Eligibility comes from the current item** (`:18-20`): mints, own objects (which is where the dummy outcome lands) and gaps (no selection, or no word) are skipped.
- **Same-close aliases are charged**, and a fresh mint's record falls back to its `content` (`:26-27`).
- **Arrivals are excluded** because the sum runs over `plan['items']` only. Thought episodes never call this path.
- **Empty content costs zero explicitly** (`:22-24`).
- **The denominator is the full description**: coordinates the candidate lacks count as uncovered (`:28-29`).
- **No gradient and no memory read:** both operands go through `IdentityRows.coordinates`, which detaches and checks values are in [0, 1] (`IdentityRows.py:32-37`). The cost is therefore between 0 and 1, adds no new detach, and reads neither MTM nor the store.
- **Registration:** one `[batch]` vector, baseline 1, the active-row mask, the independent `reconstructionReferenceWeight` (validated at `Models.py:2412-2415`, default 0.1), all under the existing reconstruction switch.

## 3. Tests and receipts

- **The new tests** run the real `BasicModel._sentence_path_cost` and `Error` registry, plus SentenceCredit's `components`, `comparison` and `score_function`.
  - A half-uncovered bind costs 0.05, flips the keep to greedy with reconstruction deciding, and gives a nonzero resolver gradient.
  - Memory and bank are checked unchanged.
  - The alias case, the per-row sum with an inactive row, weights 0 / 0.1 / 0.35, a single registration, the disabled switch and all six zero-cost endings are covered.
- **Gate ledger:** 233 passed and 7 skipped, every run stamped with the frozen hash (15 + 12 + 113 + 4 + 2 + 1 + 1 + 85). Only the three lexical assertions §§12–13 authorized changed.
- **Fresh and replay runs:**
  - The twelve fresh first draws are confirmed distinct. They record 0 bound / 8 open (forward) and 0 bound / 16 open (reverse), and all 192 kept readings have x stamped 1 with no determiner.
  - The five replay inputs are checked by hash, with the original decomposition seeds.

## 4. Findings (none blocking)

1. **A same-close alias to a name's own first mention is priced as if the name had no content.** `ReferenceClose.py:791-797` still records `content*0` for an own object that has a forward name. Before the contract commit that changed nothing, because those sites always had zero content. Under §12 every order-1 leaf site has a forward name *and* its own content. So a later mention in the same sentence that binds to it sees an empty record and pays the full 0.1. The same token from an earlier close is read from its stored rows (`token_value`, `:116-138`) and would be covered. That breaks the symmetry §9.4 intended.
   - No gate reaches this case: x carries no conceptual coordinates, and the alias test binds to a mint.
   - **Question for Alec:** should a named own object's record carry the site's own content, or stay empty until an arrival fills it? I recommend its own content, as §12 makes it the site's content. Gate it with a same-sentence repeated name. Changing it also changes how the resolver scores that case, so it belongs in its own small change, not in line 1.
2. **The bind-departure series is null by construction, and the README doesn't say why.** All 11 binds of an existing referent in the fresh keep captures have all-zero query coordinates. Most of the 102 departures are arrival decisions, which the term excludes. I suggest adding one sentence to the README (a receipt edit, not a source change).
3. **Marker status is based on history, not the current position.** Once a symbol has any `surface_argin1` occupation, it is a literal in every implicit participant read (`Models.py:20618-20621`, `ReferenceContext.py:399-403`) and a marker in `PartRead.py:137`. §10 asked for exactly this, so it complies. It belongs with item 3's parts-of-speech work. The comment at `ReferenceContext.py:400` still says "objectless marker".
4. **A gap that binds pays nothing** (`ReferenceReconstruction.py:20`, via `has_word`). This matches "gaps pay nothing", but the eligibility wording in §8's answer to R3 doesn't name it. Worth one line in the spec.
5. **The two decomposition replay snapshots are in `output/`, which git ignores.** They are verified by hash but won't be published.

Findings 3 and 5 are separate from the deferred root-position work (FutureWork, "Root participant positions") and from re-identification (`mtm_reidentification`). Nothing here touches either one. None of the findings needs a learning claim or a second sweep. Step 2 is not affected, apart from the question in finding 1.
