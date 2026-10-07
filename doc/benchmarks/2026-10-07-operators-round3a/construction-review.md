# Round 3a — construction check for Claude

Status: specification counterexamples found before runtime changes or gate
training. This is not a measured round-3a candidate. No commit or push.

The accepted 2e source matches all 705 files in its frozen manifest,
`3fd07f0cd516da3102867207614e11a9153638d37057bcc82a8c7ff821689106`.
Git still points to the round-1 landing. The uncommitted baseline is preserved
in [development/baseline.json](development/baseline.json) and
`development/round2-working-baseline.zip`; the original receipts are intact.

## 1. Cumulative hashed atoms preserve order but do not encode exact length

The hand-off gives every atom three randomly selected bits of 64, using a
private generator seeded from the atom's bytes. Taking the join of cumulative
length atoms is monotone, but the next atom can contribute no new bit. It is
therefore not an exact thermometer.

The [executable probe](development/construction_probe.py) fixes a simple,
declared implementation: tagged atom bytes, SHA-256, the first eight hash bytes
as a little-endian seed, and a private CPU Torch generator's first three
`randperm(64)` positions. It tries no alternative encoding or seed.

With that construction, `aaaaaa` and `aaaaaaa` both have sparse form
`0x1e03845556c402b3`. The `len≥7` atom selects bits 5, 42 and 58, all already
present in the shorter word's form. Both words have the same boundary pair
set and the same triple set `{#aa, aaa, aa#}`. No triple's presence in the
string distinguishes them. Their triple sequences do differ by position:
selecting `aa#` for the shorter word and `aaa` for the longer word separates
their forms in this implementation. That positional interpretation of the
mint can repair this particular length collision; it does not make the
original hashed length atoms an exact thermometer.

This example depends on the declared atom encoding; the underlying lack of
a guarantee does not. For a fixed pair set, successive binary forms can
increase strictly at most 64 times. A configuration with bounded word length
can use reserved cumulative length bits, but the current hand-off neither
reserves them nor exempts length atoms from the random three-bit rule.

## 2. A different triple is not necessarily a distinguishing code

The required `calaba`/`cabala` fixture starts with identical forms,
`0x1203954dd6c41bb5`. The first differing adjacent triples are `cal` and `cab`.
Their codes differ, but joining each into its word gives the same new form,
`0x1203974dd6e41bb5`. Their different bits are masked by the existing join.

The mint must therefore check the resulting forms. The next differing
positional pair, `ala`/`aba`, does separate this fixture. The hand-off needs
a rule for continuing when the first distinct triple fails, and clarification
of positional selection versus distinguishing triple-set membership. Adding
the very same triple atom to both colliding
words would necessarily preserve their collision; the probe uses each word's
own triple, the interpretation that can distinguish them.

On the identity toy's unchanged dictionary sample (19,986 unique sampled
words), the probe finds six initial collision groups. Five separate after
their first differing triple; `calaba`/`cabala` does not. These counts belong
to this construction diagnostic, not to any model configuration or training
gate. They are not a claim about every possible byte-seeded generator.

## 3. The requested containment witnesses need correcting

Boundary pairs give `an` the atom `n#`, absent from both `and` and `ant`.
Consequently neither proposed relation is a part-set inclusion. The intended
`bana`/`banana` inclusion is valid and its sparse forms satisfy the order.

The four native grammar-gate words (`hello`, `world`, `loving`, `there`) have
distinct forms and no nontrivial part-set inclusion. A nonvacuous audit can
run witness fixtures under each configuration, reported separately from its
native vocabulary. It cannot honestly report native comparable pairs where
none exist without changing the gate data.

## Requested clarification

Please clarify whether length must be exact before minting or can depend on
collision repair, and specify a mint fallback when the first distinct triple
does not change the sparse form enough to separate the words. A reserved
thermometer block up to a declared configuration limit is one possible length
rule. Checking successive positional triples, with a declared longer-atom
fallback if none works, is one possible mint rule. Both need clarification of the current
"three random bits for every atom / first triple and otherwise nothing"
contract; no such amendment has been made in the model.

Please also distinguish the native-vocabulary containment census from the
separate nontrivial witnesses run under each configuration.

## Evidence and boundary

[Raw results and bit-level proofs](development/construction_probe.json),
[command output](development/construction-probe.log), and
[probe source](development/construction_probe.py) are saved. The JSON records
source and dictionary hashes and Python, Torch and NumPy versions. Assertions
verify both counterexamples, the valid and invalid inclusion witnesses, and
monotonicity on the witness set. The global Torch RNG state is unchanged.
The private seed-zero NumPy generator reproduces only the pre-existing toy's
dictionary sample; no model is instantiated or seeded. Re-execution after
adding proof assertions reproduces the same diagnostic results.

No gate training, full sweep, acceptance count or runtime implementation has
been started. All previously frozen source files and earlier receipts remain
unchanged. Work that depends on the identity contract awaits clarification.
