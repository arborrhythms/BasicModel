# Review of the step 2b preflight questions (§18), 2026-10-22

This was a read-only review of published `0b030e3eb` against §§10–12 and §§16–18. I ran nothing and changed no files. Everything below is a recommendation for Alec, not a decision.

**Overall:** Q1–Q3 are sound with small corrections. Q4's two tuples are right, but §16 and §18 miss a second swap site in `write_clause`. Q5's conflict is real, and the proposed replacement needs a few more assertions and two definitions before it can be authorized.

---

## 1. Leaf postings: keep the standalone wire view?

**Confirmed:**
- The loader checks dimensions and dtypes (`MemoryIndex.py:161`), matching lengths (`:183`), code, role and row ranges (`:188`), and sorted, unique postings (`:192`).
- Every load sets `_leaf_needs_owner_reindex = True` (`:158`).
- A bound owner reindexes at its next `configure_model_index` (`:523–525`, `:586`). `reindex_meanings` clears the postings before rebuilding (`:262–263`), so a loaded cache is never merged with rebuilt terms.
- The stale-cache test (`test_part_read.py:26–33`) passes because of that path.
- The checkpoint test (`test_accessible_mind.py:31`) writes terms through a forced reader (`index_fixtures.py:40–46`). No stored row can produce those terms, so a standalone store genuinely needs the wire view.

**Missing from Q1:** `test_accessible_mind.py:381` is a third live contract. It deletes the posting columns and `index_stream` and expects an owner rebuild. The recommendation keeps it working.

**Assessment:** This adds no second authority. With an owner present, the cache is thrown away rather than reconciled. Without an owner, there is no symbol-to-code map, so the cache is the only possible source, as it is today. Two points need stating in the spec:
- **Retrieval before rebuild.** Rebuild only runs when a caller calls `configure_model_index`. Between load and that call, the stale cache is served. The new wording "before owner-backed retrieval" should say it relies on that same calling convention, not on the loader.
- **Order of restore.** `load_semantic_extras` (`Layers.py:9311–9318`) rebuilds missing postings from `meaning_of` without an owner. Once 2b moves `role_refs` into LTM1 rows, that LTM1 must be loaded before this runs, or this rebuild must be deferred.

**Tag gap:** §10 S6's list of tag-aware readers omits the two readers that actually do this rebuild:
- `MemoryIndex._meaning_leaf_terms` (`:71–110`);
- the `read` closure in `configure_model_index` (`:538–573`). It walks `store.parts.refs[symbol]` without tags (`:565–566`), so an occurrence handle would be read as a symbol.

Add both to the S6 gates.

**Recommendation:** Keep the wire view, as §18 proposes.

## 2. Adjacency witnesses

**Confirmed:**
- The writer is `Models.py:20368`, and re-witnessing replaces one address's pairs (`MemoryIndex.py:213–232`).
- Compaction keeps only live addresses (`Layers.py:10107` → `retain_adjacency`), and `reset` clears the witnesses.
- `begin_forward` takes a frozen snapshot (`MereologicalCodes.py:97–98`), and `place` weights words by witness count (`SymbolCentroid.py:16–20`).
- The keys are word rows checked with `word_surface_for_row`, not native symbol IDs.

**Correction:** "Codebook remapping updates code addresses" is wrong for adjacency. `remap_leaf_codes` (`MemoryIndex.py:279–288`) remaps only `_leaf_postings`; nothing remaps `_adjacent_postings`. If word rows can be compacted, the adjacency keys go stale. That is an existing gap, not a 2b change, but the spec shouldn't claim the remap exists.

**Positions:** `place` only uses positions to keep repeated pairs distinct. Exact positions matter for re-witness comparison and for the test at `test_operators_final.py:39–47`, which asserts the private `_adjacent_postings` dict directly.

**Assessment:** Retaining the four columns as justified witness data is the right call, and §18 correctly doesn't present them as derived. Each `(address, position)` points inside one sentence's entry. §12 forbids that for relation and MTM references, which is why an LTM1 version needs its own reviewed layout. One addition: say explicitly that adjacency witnesses do not protect rows under S7. Today they don't, and if they did they would form a second reference graph.

## 3. `unspecified` mode

**Confirmed live:** `ClauseRow.py:443–445` builds the estimate with `mode='unspecified'`. It is written through `append_estimate` → `append_meaning`, which stores the mode at `Layers.py:9538`, whenever an expectation has source occurrences. Migration is `Layers.py:9372–9376`.

**The question layout cannot derive the mode, in either direction:**
- **Interrogative without an open site.** The mode comes from the grammar's ask rule (`AskLayer.meaning_mode`, `Language.py:4203`, applied at `ClauseJournal.py:397`), not from open sites. `Models.py:171–172` admits an interrogative with `(0., 0.)` evidence as a question.
- **Open site that is not a question.** `test_part_read.py:59–61` and `:96–97` store an assertive row with an open referent slot and evidence `(1., 0.)`.
- **Record kind doesn't settle it either.** `unverified` rows can be assertive (`Layers.py:9473`) or legacy `unspecified`. `append_estimate` doesn't check mode.
- **Other readers of the mode:** `Layers.py:9491` (a fact must be assertive) and `ClauseRow.py:389` (no TruthSet interrogatives).

**Recommendation:** Keep the three-valued column with no extra flag, as §18 proposes. It is already the minimal lossless form.

## 4. The two implicit-pair tuples

**Arithmetic checks out** for `test_operators_round3a.py:151–152`:
- trust `.7`, negative: `(.7, 0)` was swapped to `(0, .7)`; without the swap it stays `(.7, 0.)`.
- trust `-.4`, negative: `(0, .4)` was swapped to `(.4, 0)`; without the swap it stays `(0., .4)`.

The assertion at `:162` (`store.trust == e[0] - e[1]`, with `trust` defined as `c_plus - c_minus` at `Layers.py:9881`) still holds verbatim. The other three cases are unaffected.

**Corrections to §16's list of swap paths:**
- **A second swap site.** `ClauseRow.write_clause` swaps again at `:468–472`. It fires for trust without evidence, and **always** under the TruthSet assertion context (`_clause_assertion`), even when explicit evidence is given. Neither §16 nor §18 names it.
- **`append_idea` can't swap.** It always builds `from_description` with the default positive polarity (`Layers.py:9821`). That swap path is inert.

**Layout question for Alec:** For a completed relation, `write_clause` writes the pair onto a shared LTM1 `Part` edge, and those edges have no polarity column. Removing the swap at `:471–472` would turn "not P, trust .7" into support *for* P.

Three tests pin the current folding, using explicit pairs:
- `test_lies_can_be_learned.py:9–11`
- `test_clause_storage.py:186–188`
- `test_relative_sentence_codebook_insertion.py:14–16`

Each writes P with `(.8, 0)`, then not-P with `(0, .7)`, and expects one edge holding `(.8, .7)`. They don't depend on the implicit-swap *code*, so they need no port under the narrow §16 rule. They do depend on the *convention* that relation-edge lanes are stated for P, which §16 doesn't obviously cover.

The reader `Queries.py:1104–1105` already flips poles when stored and queried polarity differ, which matches §16 for LTM0 rows.

**Recommendation:** Authorize the two tuples. Have Alec decide whether §16 applies to LTM1 relation edges and whether `ClauseRow.py:471–472` is in 2b's removal. I'd recommend leaving it out for now: no relation-edge polarity field is specified, and those three tests stay untouched.

## 5. Coarsening fixture

**The conflict is real**, and no reading of the current test preserves it under §§10–11:
- `deletion()` (`Forgetting.py:300–334`) rewrites the surviving parent's `role_refs` and `refs` (coarsening) or removes it in the same pass (cascade). The test asserts both outcomes (`test_forgetting.py:151–159`).
- The parent depends on the child **twice**: through the sidecar's occurrence reference (`role_refs[0]`, `:137–138`, read at `Forgetting.py:206–210` and `:315–316`) and through `refs` (`:139`). Porting only the `refs` write leaves the old policy running through the sidecar.

**The replacement behavior traces correctly.** Capacity is 4 and `low_water=.5`, so the target is 2. Under protection, the child (trust 0) is not a candidate, so rows 2 (.7) and 3 (.8) go. Parent and child survive, with no coarsening or cascade, in both variants. To make that non-vacuous, the replacement should also assert:
- rows 2 and 3 are gone;
- `report['target_reached']`;
- the existing content check (`roles == root.roles`).

**The next-pass case needs fixture conditions.** A second `forget` returns `None` without pressure (`:252–254`). The case needs refilling or `epoch_boundary` with `at_epoch`, and `min_interval=0` (already set in `config()`). If the released target leaves the first pass short of its goal, the test must expect the `RuntimeWarning` (`:415`).

**Layout questions to settle before the port:**
- **(a) Which tagged LTM1 row holds the parent's role-0 reference to the child?** Here the parent is an LTM0 occurrence, not a relation, and §10 S2 doesn't give an occurrence-to-occurrence role edge.
- **(b) Direction of protection between an occurrence and its reading or derivation rows.** Under the acyclic rule, the later reading row protects the earlier occurrence, but nothing protects the reading row. `forget_parts` (`Forgetting.py:115–148`) could then drop a surviving occurrence's derivation edge. The parent row isn't rewritten, but its read loses role 0, which is coarsening again by another route.
- **(c) What counts as one pass.** `model_boundary` runs `forget_parts` and then `forget` back to back (`Forgetting.py:460–461`). Recommendation: one boundary call is one pass, so release takes effect at the next boundary.

**Snapshot at `:126`:** The exact port, following the §17 precedent, is to drop `'refs'` from the key tuple. Adding an LTM1 snapshot there would be a new assertion that needs its own approval.

**Two more `refs` readers to port with 2b:**
- `test_live_parent_pins_its_child` (`:162–166`) reads `role_refs[0]`. Its behavior is unchanged but its reader moves.
- `ClauseJournal.py:405–406` reads `row()['refs']` (exported at `Layers.py:9850`) and uses columns 0 and 1 as the two operands. Forgetting treats the columns as roles 0–2, so check that layout when porting.

Agreed that this must not be labeled a column-name port.

---

## Choices for Alec

1. **Q1:** Keep the standalone wire view (recommended). Also add the two reindex readers to the S6 gates, and require LTM1 to restore before `load_semantic_extras`.
2. **Q2:** Keep the adjacency columns in 2b as non-protecting witness data (recommended). Decide whether the adjacency remap gap gets its own item.
3. **Q3:** Keep the three-valued `grammatical_mode` (recommended).
4. **Q4:** Authorize the two tuple changes. Decide whether `ClauseRow.py:471–472` and the relation-edge polarity convention are in scope for 2b. My recommendation is no, pending a specified relation-polarity layout.
5. **Q5:** Authorize changing the old-policy assertions, with the extra assertions and fixture conditions above. Before implementation, settle layout questions (a)–(c) and the snapshot port at `:126`.

Out of scope here: the two R10 arrival notes stay in 2c, and the `SentenceEndState.where` observation stays with the 5.5 spec.
