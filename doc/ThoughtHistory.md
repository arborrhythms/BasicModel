# Ordinary thought history

Current contract: [item 6.2, October 7](specs/2026-10-07-thinking.md).
The [held 6.2 receipt](benchmarks/2026-10-07-item6-2/README.md) remains intact. The [repair receipt](benchmarks/2026-10-07-item6-2-repair/README.md) records constituent ownership, ported contracts and new measurements. Alec accepted it as the mechanism landing under spec §11; learned chaining remains for §10's MM_math_chain.

`SymbolSpace.what_memory` owns the chronological history. `begin`, `thought`,
`descend`, `return`, `cutoff` and `finish` records determine dependency order;
there is no separate planner stack. The owner's historical attribute name is
not a callable thought operation: that operation is now `ask`.

A closing with a free referent or relation opens one episode. An interrogative
region with no evidence also opens one; `(0,0)` is ignorance, not a slot.
Each nested `ask` shares the same `attentionBudget`; no callback replenishes
work. `conclude` requires bound variables and evidence while work remains,
unless the configured number of complete queries has exhausted the search.
An exhausted declarative provisionally mints its forward referent; an
exhausted question remains open. Cutoff drains at most the existing depth of
returns and one finish. Stored open columns have stable row-and-role addresses
that later binds can co-refer with and copulas can fill in place.

Every checked result is a serial slot carrying content, both evidence poles,
witnessing rows and the operation that produced it. Later candidates can bind
those slots. Two queries can therefore retrieve `a < b` and `b < c`, then
conclude `a < c` with both references. Implication additionally requires its
antecedent evidence. A nested answer returns the filled row.

The kept walk writes conclusions as `inference` rows, with occurrence addresses
from document/turn, per-turn ordinal and content. Witness references remain
provenance. The closing stores unresolved questions. Candidate formation does
not append temporary descriptions to LTM.

The existing compose scorer also scores thought requests. Greedy and one
uniform departure are credited by `K · R · p(a_dep) · ΔC`: reconstruction
and supplied answer, plus the thought walk's own metered work. Work does not
enter the enclosing sentence comparison. Expectation trains its predictors
through their owner-step registry. Exact ties contribute nothing.
The REINFORCE/EMA path and `selectedThoughtPolicyWeight` are retired. There is
one configured work name, `attentionBudget`; older budget names fail at load.

Checked results detach reader tensors. Policy probabilities retain only the
chooser graph until the owning cost arrives; no target enters a request.
Exploration forks the detached greedy episode at a reservoir-sampled departure
and completes its suffix. Checkpoint history preserves full meanings, pairs
and provenance; restoring it never replenishes work. The existing v3 result tags continue to carry nested
meanings and typed results; old v1/v2 history remains readable.

At a fully bound declarative, an uncancelled expectation image can yield
`not X`, with its confidence in the against pole of an inference. Ordinary
fully bound XOR closings do not open thinking episodes.

## Evidence

The reviewer probes first failed against the pre-installation state in
`output/tests/20260917-101448-ea1b90`. The focused contract set then passed
9/9 in `20260917-101736-c6a3eb`, its expanded lifecycle/checkpoint set passed
33/33 in `20260917-102018-1d12d3`, and the affected integration selection
passed 69/69 in `20260917-110047-33f23a`. These prove the storage, replay,
credit and occurrence-read foundation only; they do not prove normal thought
selection, learned utility, or end-to-end answer quality.

The later typed-result retention probe first failed because a `ThoughtRecord`
had no result field (`20260918-174718-55accf`), and its prediction-shaped
boundary probe then caught live `MeaningExpectation` tensors
(`20260918-175218-2ea26b`). The detached restore, prediction, integrated
checkpoint, and v1-compatibility cases passed 1/1 in
`20260918-175049-035967`, `20260918-175317-45450d`,
`20260918-175425-d9c1c`, and `20260918-175647-d350e6`; the final bounded
controller/history/query selection passed 134/134 in
`20260918-180026-3af0eb`. This is a typed-boundary and replay result only, not
evidence of learned controller utility.

The nested-result reviewer probe then failed because v2 serialized a complete
meaning as an ordinary mapping (`20260918-191638-1dc00d`). The v3 tag restores
the typed detached meaning while accepting v1/v2 sidecars; the focused repair
passed 1/1 in `20260918-191815-72d0c3` and the full history-boundary file
passed 14/14 in `20260918-191957-9fefa6`. The affected
controller/history/query selection then passed 136/136 in
`20260918-192121-5f6dc4`.

## LTM roots retained by ordinary history

The existing owner also walks typed result evidence, including retrieved frames
and their nested meanings, so a retained result keeps its referenced occurrences
alive. No separate frame cache owns them.

The existing owner derives LTM roots from every retained ordinary record's role
references, bindings, scope and recorded sources. It reads that data on demand,
without a second reference index. Those roots protect required content during
request-origin replacement even after the content's evidential authority is
withdrawn. See [nested retention](NestedRetention.md).

The completed item 1 integration and its learning/adapter evidence are recorded
in [SelectedMeaning](SelectedMeaning.md) and [Testing](Testing.md#selected-meaning-and-one-controller-september-20).
