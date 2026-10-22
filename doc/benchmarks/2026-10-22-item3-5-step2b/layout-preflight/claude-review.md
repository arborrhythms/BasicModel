I read §§10–12, 20–23 and 27, plus `LTM1.py`, `PartRead.py`, `ClauseRows.part_row`/`write_clause`, `Definitions.py`, `ReferenceClose._derivation_symbol` and `forget`/`forget_parts`. I made no edits and ran nothing.

## 1. The decided data does not identify the reading uniquely

**Short answer:** no rule built only from `refs`, `ref_kinds`, the graph shape and grammar occupations picks out an occurrence's reading row in every case.

**What today's code does:** an occurrence names its root through `role_refs[0]` (`ClauseRow.py:540–544`). That field sits on the occurrence's side and points one way, so each occurrence has exactly one root. §10 S2 (spec lines 528–531) replaces it with `Part(OCC occ, SYM root)` and describes that row as "the same relation as 'Felix is a cat'". `LTM1.py:2–5` also says the row stores no reading. So the pointer that marked "this is my root" is gone, and the reading row is shaped like any other row.

Each candidate rule fails:

- **Tag pattern.** `rF = (OCC F, SYM felix)` and `e = (OCC F, SYM runs)` have the same tags. In both, F is in position 0 and the second argument is a word.
- **Unreferenced row with OCC first.** In §20's example this works: `rS` references `e`, so `e` is not F's reading. It breaks when an unreferenced `(OCC, SYM)` row exists that is not a reading:
  - A complete relation is still written as an LTM1 edge only, with no occurrence row (`ClauseRow.py:461–464, 509–513`). Nothing in 2b changes that path.
  - §10's equality layout is `equal(OCC x_occ, SYM three)`, next to x's reading `Part(OCC x_occ, SYM x)`. Neither row is referenced, both have tags (OCC, SYM), and their second arguments are a word and a singleton. A reading whose word is itself a singleton (a mention of "three") would also type as `equality`, so `LTM1.readings()` cannot tell them apart.
  - Even in the §20 case, the rule only holds if a reading and its derivation are always dropped together. §21/§23 call them "stages" but do not say they go in the same pass.
- **Earliest handle.** An LTM1 row's index is a handle from the allocator (`LTM1.py:88–99`), so "earliest" really means creation time. You have ruled that out as a role tag. It also fails once the true reading is dropped, which is your §23 gist case.
- **Occupations.** `PartsOfSpeech` is keyed by word and gives no exclusive word classes. A candidate's second argument having some occupation proves nothing about whether the row is a reading.
- **Coalescing.** `find` keys on the pair, and `_index` refuses duplicates (`LTM1.py:47–48, 61–63`). Tag-aware keys still put the reading `(OCC F, SYM w)` and any derivation or fact `(OCC F, SYM w)` into the same row. That one row would then fall under both the ownership lifetime rule and the own-value forgetting rule.
- **MTM** expires by design (§11.3, §12), so it cannot be the persistent source after restore.

The definitions-index exclusion is not affected. `Definitions.py:51–53` needs `'object'` on the left argument, and an OCC-tagged argument has no symbol types, so the tag alone keeps reading rows out.

## 2. This is a genuine spec question for Alec, not an implementation detail

Three decided rules depend on finding "the reading of occurrence x":
- §27's occupation read.
- §20/§21's ownership rule: a reading is never valued on its own and is dropped as a stage of its occurrence, while a fact row is forgotten by its own value. `forget_parts` has to sort rows into those two groups, and do it again after every restore.
- The P4/P7 ports that replace `role_refs[0]` reads.

None of these can be implemented without picking a discriminator.

**Smallest option I'd recommend Alec consider:** add a third value to the already-decided `ref_kinds` column, `READING`. It is an occurrence reference meaning "this row is that occurrence's reading". It is allowed only in position 0, with the other argument tagged SYM. So `rF = (READING F, SYM felix)`. The rule then becomes: the reading of x is the single live row keyed `(READING, x)`, and if there is none, x is a gist. This depends only on `refs` and `ref_kinds`, so it survives forgetting and restore. It also keys the reading apart from an OCC row with the same integers, which ends the coalescing problem.

**Costs:**
- It is a role tag after all. It partly reverses S2's "membership, same relation" framing and the "no stored reading" line in `LTM1.py:2–5`. Alec has to accept that explicitly.
- Every tag-aware boundary in the S6 list has to handle three values.
- Protection and integrity checks must treat READING as an LTM0 handle for validity but not as a protecting reference. That spells out §20's "ownership is not a reference".
- The writer and the loader must both enforce at most one READING row per occurrence.

**Test ports it would need:**
- P2 does not change (still seven columns).
- P5's tag-range validator accepts three values and still names `ref_kinds`.
- The P4/P7 reads go through READING rows.
- New gate cases:
  - §20's layout with a reading and a derivation that share a pair: they must be two rows.
  - The reading is found again after save and load.
  - After a reading is dropped, the occurrence has no reading, and a surviving `e` is never taken as one.
  - The same integer used as an occurrence handle and as a symbol.
  - Reading rows stay out of the definitions index.
  - Writing or loading a second reading for one occurrence is rejected.

**Alternatives, both larger:**
- **(b)** Put a root column back on LTM0. This reverses S2 and the P4 absence ports, and adds a protecting edge from LTM0 to LTM1.
- **(c)** Make the graph shape enforce it: complete relations also write occurrences, readings and derivations are always dropped together, and coalescing is banned. That changes P6's counts and hides the role in the shape of the graph. I don't recommend it.

## 3. Problems with the example and the circularity claim

- **The circularity is weaker than stated.** When `argument_reader` classifies `e`, it can leave `e` itself out of the candidates for F's reading. In §20's three-row example, that leaves only `rF`. The real problem is ambiguity once F has any other incident `(OCC, SYM)` row: another derivation, a fact, or a pair that coalesces with the reading. It is not a loop.
- **The forgetting worry is narrower under today's code.** `forget` refuses to coarsen a protected row (`Forgetting.py:327–333`). While `e` lives, F is protected, so F's reading would not be dropped before `e` is. That ordering is not written into §21/§23, though. If it ever happened, `e`'s operation could no longer be classified. I'm only flagging that consequence, not reopening the policy.
- **The example's layout is otherwise correct as decided.** No reading of the existing decisions resolves the gap without a discriminator. §26/§27's precedence for `_derivation_symbol` (token kind first) only helps while the MTM entry is alive.

**Bottom line:** this is a real gap in the decided layout. I recommend 2b stop until Alec chooses the discriminator (my suggestion is the READING tag value). P1–P7, minting, ownership, evidence and the forgetting policy are unchanged.
