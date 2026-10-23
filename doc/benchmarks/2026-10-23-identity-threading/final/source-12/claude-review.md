**Verdict: DO NOT LAND.**

The source-11 repair is correct as far as it goes: with `identity_state`, composition values now use the typed concept for orders other than 1. But the stored reference for some of those same operands still comes from a path that never does the typed lookup. So under the identity overlay, the composed value and the stored reference can disagree. That fails the criterion you set: capture references and composition values must agree under the existing typed-order contract.

I worked read-only. I traced the code and did not run anything, so no test, learning or GPU result comes from me.

## Necessary fix

**1. Implicit operands at orders other than 1 compose with the typed concept but store the word's raw reference.**

This affects bare words in implicit S-subject and relation (part/whole/equal) roles.

- **Composition side:** `bin/ReferenceContext.py:348-353` now replaces the value and the id with the typed concept (e.g. id 7). Those ids go into `operation_refs` (`Models.py:12856`).
- **Storage side:** for identity programs, `bin/ClauseJournal.py:120` skips the `operation_refs` block. Before this line, that block was how an implicit operand's typed id reached its stored reference.
- **Capture doesn't make up for it:** `reference_requests` (`ReferenceContext.py:155-157`) only looks at explicit `reference_roles`, so `resolve_lexical_references` (`Language.py:12632`) never resolves implicit roles. The stored reference stays the raw concept id.
- **Result:** for S subjects and relation operands, `operand()` takes its reference from `concept(item)`, which reads the raw `selected['ref']` (`ClauseJournal.py:522`, `:359`). Its point is the typed value from the journal. A row can therefore hold concept 7's point under the raw id.
- **What is fine:** VP operands (`ClauseJournal.py:587-601`) read the typed ids directly from the operation refs, so they agree. Explicit determiner roles agree too.
- **How it got here:** before this line, both sides were typed. At source-11, both were raw. Source-12 fixed the value but not the reference.
- **Reach:** this only happens when a word's stamped order differs from its source concept's order and a typed concept exists at that order. That is exactly the shape of the kind-point test fixture. Nothing in the gates exercises it end to end. The new seven-case test asserts `unary_refs == 7`, but under identity the capture path throws that value away.
- **Minimal fix:**
  - In the `identity_actions` branch at `ClauseJournal.py:120`, accept the operation ref as `selected['ref']` only when all of these hold:
    - `binding_spec(..., implicit=True)` gives an order that is set and not 1;
    - the mode isn't kind/generic;
    - the ref is a native id that differs from the leaf's own ref.
  - Leave out the open/mint/formation branches, because bind still owns order 1.
  - Add a `finish_clause` test with an identity program and a bare implicit S subject at stamp 2. It should assert that the stored reference is 7 and the point is the typed value.
  - Then freeze a new source and rerun all the scoped groups on it.

## Notes for a later round (not blocking)

- **The literal exemption is narrower than the receipt says.** In the ordinary path, `literal_ids` only stops reference binding (`:409-413`). Literals at orders other than 1 still get their typed value (`:381`). The identity path now keeps them raw. That is consistent with what gets stored (raw value and raw id), so I accept it. But spec §30 calls it restoring the existing contract, and that isn't accurate; it's a deliberate choice and should be recorded as one. My source-11 wording ("keep the existing literal_ids exemption") contributed to this.
- **Decomposition adapter:** the unary name ordering is only authenticated indirectly, through the binary list matching. Adding `[n for n in layer.unary_names if n != 'bind'] == unary_names` would check it directly. The ledger itself is consistent:
  - unary rows 0–6 map to themselves;
  - bind sits at chooser row 28, and the field actions move to rows 29–34;
  - category rows skip 7–8 and rule-predictor rows skip 35.
- **Event-history notes:** the two FutureWork entries (`doc/FutureWork.md:1667-1677`) match my source-11 notes. They aren't claimed fixed, and no new policy was added for them.

## Verified

- **Repair shape:** order 1 stays raw until bind. Exactly one option is returned and choice tags are -1, so no determiner reference variants come back. The kind-point test runs with and without identity, keeps its old assertions and adds the single-option check. The origin assertion is at `test-ports.patch:1237`.
- **Only the repair changed:**
  - The source-11 and source-12 `production.patch` files have the same structure. The only difference is the `ReferenceContext` hunk, which is 8 lines longer.
  - The rest of the line is the same code I reviewed at source-11, and I re-checked its key paths in the source-12 patch. These are the MTM state and checkpoint, the menu and its zero dummy, recorded binds at close, `_commit_events`, the stream guards, the c/e guards, the dummy alias in `PartClosing`, and words not being allowed as objects. They match my source-11 findings.
  - I could not recompute the source-audit hashes myself.
- **Gate counts:**
  - Passes add up: 87 + 123 + 707 + 698 + 2 + 2 = 1,619.
  - There are 12 skips: nine absent phase artifacts (seven native, plus fineweb and stage3) and three opt-in MPS/Inductor cases.
  - 1,631 cases ran across 132 distinct files, with no compile-cache retries.
- **Decomposition:** both arms use seeds 4199722147 and 2428440506 with the original snapshot hashes. All four records show surviving parameters equal to the originals.
- **Pending premise:** the forward certificate shows closure not forced, 0 bound and 16 open. I did not read the reverse certificate (spec says 0 bound / 32 open).
