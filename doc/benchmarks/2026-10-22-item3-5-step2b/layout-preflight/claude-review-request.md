# Read-only 2b layout check — 2026-10-22

You are Claude reviewing one potential implementation ambiguity, before any
production or test edit. Work is on main, `00bc915a9`, with the current
step-2 spec §27 and Alec's P1–P7 authorization committed. Production still
matches published 2a `0b030e3eb`. Do not edit files, run gates, or change
any decision. Read the spec and the relevant code; answer this narrow
question, including any reason this is already determined by the contract.

Read `doc/plans/2026-10-21-step2-one-ltm.md` §§10–12, 20–23, 27;
`bin/LTM1.py`; `bin/PartRead.py`; the ClauseRows writer/reader in
`bin/ClauseRow.py`; and relevant `bin/Forgetting.py` paths. The latest
handoff requires an OCC argument to read its occurrence's reading row to
the word and use that word's PartsOfSpeech occupations. Reading rows must
never enter the definitions index. The first frozen commit is layout;
the second ends close-time minting.

Potential gap: which persistent fact uniquely identifies an occurrence's
reading row among its incident tagged Part rows? The §20 worked layout is:

```
LTM0: F, S
rF = Part(OCC F, SYM felix)  # F's reading
e  = Part(OCC F, SYM runs)   # lift's derivation
rS = Part(OCC S, SYM e)      # S's reading
```

`rF` and `e` have identical tag patterns. LTM1 has no operation or
reading-kind column; `PartRead.argument_reader` infers operations from
argument occupations. For an OCC argument it must first identify F's
reading to get the word's occupations, so that seems circular as a way
to distinguish `rF` from `e`. A word can have multiple occupations; the
grammar contract gives no exclusive word classes.

An earliest-row convention could initially pick `rF`, but §23 allows
dropping a reading while leaving its occurrence as a gist. What then
prevents a surviving `e` from becoming F's apparent reading? The same
concern applies after restore, which rebuilds derived indexes without the
old structural sidecar. Source time, access counts and numeric equality
cannot be used as an undeclared role tag. A valid reading may also have
the same `(OCC, SYM)` pair as another Part use; `find` currently coalesces
identical pairs.

Please give:

1. Whether the decided data already determines the reading uniquely,
   with a concrete reconstruction rule valid through forgetting/restore,
   repeated Part uses and the same integer in two reference namespaces.
2. If not, whether this is a genuine new spec question requiring Alec,
   rather than an implementation detail. Recommend the smallest explicit
   discriminator or layout revision, naming its costs and required test
   ports. Do not authorize a new column or silently alter the design.
3. Any flaw in the example or in the claimed circularity. Prefer a
   correction that follows the existing decisions if one exists.

This is not a request to reopen P1–P7, minting, ownership, evidence, or
the forgetting policy. No implementation failure or gate result is
claimed. Do not claim an executable diagnostic was run.
