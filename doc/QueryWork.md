# Shared selected-thought work accounting

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are built (accepted as implemented, Alec, 2026-11-01).


Implementation reference, September 20.

QueryWorkBudget is one transient integer allowance for a selected thought tree.
Its read-only spent, remaining, and per-kind counts report actual debits. It
owns no meaning, checkpoint state, child allowance, reset, learned parameter,
or objective. A debit is atomic: an overlarge request fails without partly
consuming the allowance. ThoughtGrammarContext.work carries the same object
through preparation, execution, and nested calls.
[Meter](../bin/QueryWork.py#L15),
[context](../bin/Queries.py#L117).

| Cost kind | Charged before |
| --- | --- |
| controller | Choosing an operation/conclusion or entering/returning a child context |
| operation | Calling a fully validated selected executor |
| reference | Reading a selected native VP or operand address |
| payload | Reading an addressed conceptual payload |
| node | Inspecting a taxonomy node or codebook candidate |
| record | Reading an LTM/thought row, taxonomy constituent, or prediction-context record |
| expansion | Using a captured taxonomy edge in a proof or neighbor result |

Also charged: `description` (entering a description resolution),
`context_record`, `unfold` (an inverse or forward check while unfolding or
evaluating an idea), `effect_node` (each committed thought effect), `index`,
`concept`, `edge` and `part` (`intuit`'s rule index, order-zero expansion,
literals and pervasion parts), `space-read` and `serial-word-loop` (the reading
field), and `bracket` (the input's bracket work).

These units are explicit bounded work, not time or FLOPs. Local node, record,
depth, and expansion limits can only tighten the allowance. Taxonomy capture
reserves a share for proof traversal, so a snapshot cannot always consume the
entire meter before a direct proof. Capture, traversal, and nested work still
debit that one object.
[Capture allocation](../bin/QueryWork.py#L63),
[taxonomy capture](../bin/Taxonomy.py#L127), and
[traversal](../bin/Taxonomy.py#L70).

## Validate, charge, then read

Phase permission plus declared argument, domain, and width validation occur
before an operation debit. Invalid calls therefore cannot spend work or invoke
an executor. The grammatical thought registry also charges selected VP validation and
native operand preparation. Description-valued operands resolve through their
existing occurrence owner, so preparation cannot read a full description before
it shares the allowance.
[Checked call](../bin/Queries.py#L1779),
[registry dispatch](../bin/Queries.py#L2659), and
[occurrence resolution](../bin/Queries.py#L1324).

Each complete path of a sentence reads its answer on its own copy of the
stream's attention meter: before its answer reads, the path reserves on that
copy the bracket work it has already incurred, and only the kept path's charges
are transferred to the live meter after the comparison. A path therefore cannot
spend work its composition already used, and a rejected path spends nothing.

The standalone fact reader records actual inspected facts and keeps partial
signed evidence when work ends. Lookup rows, native vectors, thought
occurrences, and taxonomy records use the same meter. A nested `ask`
continuation receives the original meter rather than start a new allowance.
[Fact evidence](../bin/reasoning.py#L137).

Exhaustion reports explicit work_budget incompleteness; it neither proves
falsity nor permits a later uncharged read. Evidence acquired within the
allowance remains available. A partial failed occurrence resolution is still
recorded in the meter even where an older result field cannot expose its
intermediate scan count.

## Gradients, ownership, and controller integration

The meter uses host integers and creates no parameter, loss, or ordinary
derivative. Charging does not detach a live input payload or a live
thought-occurrence meaning; durable LTM descriptions retain their existing
detached boundary. Hard reference matching and taxonomy traversal remain
nondifferentiable, so accounting does not supply policy credit or make query
utility learned. See [Gradient flow](GradientFlow.md).

Standalone audited readers can omit work and retain their explicit local
limits. The normal selected-meaning controller creates one meter from
`attentionBudget` (`selectedThoughtBudget` fails at load), debits its own hard
choices as `controller`, forwards the meter through every selected reader and
nested `ask` continuation, and commits each actual meter delta to the ordinary
record. A child neither gets a new
allowance nor begins another optimizer episode. The existing thought owner
still enforces the cutoff-drain bound; neither the meter nor that accounting
creates residual policy credit or learned utility. The chooser's credit is the
whole path's reconstruction and supplied-answer cost; the work an episode spent
is not part of it. See [SelectedMeaning](SelectedMeaning.md).
