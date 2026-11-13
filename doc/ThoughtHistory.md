# Ordinary thought history

`SymbolSpace.what_memory` owns the chronological history. `begin`, `thought`,
`descend`, `return`, `cutoff` and `finish` records determine dependency order;
there is no separate planner stack. The owner's historical attribute name is
not a callable thought operation: that operation is now `ask`.

Every completed field is offered one root episode at its close. For a field
with no open slot, not thinking is the chooser's free first choice: it spends
no work and writes no `begin` or `conclude` record; an episode the chooser
opens there records `opened_by_chooser` on its `begin`. A field with a free
referent or relation always opens its episode. `(0,0)` is ignorance, not a
slot. Each nested `ask` shares the same `attentionBudget`; no callback
replenishes work. `conclude` requires every free variable bound while work
remains, unless the configured number of complete queries has exhausted the
search or the menu holds only the self-inverse `not`. An exhausted declarative
provisionally mints its forward referent; an exhausted question remains open.
Cutoff drains at most the existing depth of returns and one finish. Stored open
columns have stable row-and-role addresses that later binds can co-refer with
and copulas can fill in place.

Every checked result is a serial slot carrying content, both evidence poles,
witnessing rows and the operation that produced it. Later candidates can bind
those slots. Two queries can therefore retrieve `a < b` and `b < c`, then
conclude `a < c` with both references. Implication additionally requires its
antecedent evidence. A nested answer returns the filled row.

The kept walk writes conclusions as `inference` rows, with occurrence addresses
from document/turn, per-turn ordinal and content. Witness references remain
provenance. A conclusion that leaves its source unchanged (the same content,
references, scope, polarity, evidence, open slots and sentence kind) writes
nothing new: it reuses the source's entry and original stamp, whichever
operation inspected it, and an unresolved working read does not copy its
question. Only a changed result, including a newly bound provisional identity,
takes the ordinary writer (`ThoughtStream.commit`). The closing stores
unresolved questions. Candidate formation does not append temporary
descriptions to LTM.

The existing compose scorer also scores thought requests. Thinking is part of
each whole path: a departure at a thought decision is credited by the
proposal-corrected score function on its complete path's change in
reconstruction plus supplied-answer cost (R + A). The walk's metered work
bounds it but does not enter that comparison. Expectation trains only its own
predictor. Exact ties contribute nothing. There is one configured work name,
`attentionBudget`; older budget names fail at load.

Checked results detach reader tensors. Policy probabilities retain only the
chooser graph until the owning cost arrives; no target enters a request.
An explore path departs at a uniformly drawn eligible decision of the exploit
episode, restores the state recorded there, takes another action and completes
its suffix. Checkpoint history preserves full meanings, pairs and provenance;
restoring it never replenishes work. The existing v3 result tags continue to
carry nested meanings and typed results; checkpoint history is version 6 and
older versions are rejected at load (`load_thought_extras`).

Before a field with no open slot is offered its episode, an expectation image
whose omission of a predicted symbol exceeds `expectationOmissionThreshold` can
yield `not X`, with its confidence in the against pole of an inference. A fully
bound closing, an XOR closing included, opens an episode only if the chooser
chooses an operation over not thinking.

## LTM roots retained by ordinary history

The existing owner also walks typed result evidence, including retrieved frames
and their nested meanings, so a retained result keeps its referenced occurrences
alive. No separate frame cache owns them.

The existing owner derives LTM roots from every retained ordinary record's role
references, bindings, scope and recorded sources. It reads that data on demand,
without a second reference index. Those roots protect required content during
request-origin replacement even after the content's evidential authority is
withdrawn. See [nested retention](NestedRetention.md).

The controller that writes these records is described in
[SelectedMeaning](SelectedMeaning.md).

## Addressed records

`ThoughtRecord.address` is the written chunk's `.when` stamp (four floats), not
a physical index. A new thought idea takes one tick of the model clock and has
an empty `.where`; `ThoughtChunks.put` refuses an occupied stamp, and an update
names its chunk explicitly. The episode's focus chunk is pinned on the STM
stack from `begin_thought_episode` until its finish. `replay_thoughts` resolves
an address through `STMChunks.locate(..., band=WHEN)` when a chunk state is
supplied, recomputes each derivation's pair from the recorded supports (minimum
along the path, maximum across alternatives, lane exchange for negation) and
rejects a record whose stored pair differs. The worked line replays with its
pair and ordered witnesses: for a question about a token's property, `query`
brings a kind-level rule, backward `infer` asks the token's membership, `query`
fills it, and forward `infer` writes the conclusion with the minimum pair.
