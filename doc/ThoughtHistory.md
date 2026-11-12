# Ordinary thought history

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23; recorded 2026-10-30):** an LTM1 row is two tagged references (each to an LTM0 occurrence, a symbol or another LTM1 row, its tag saying which) with its kind read from the referents, not three. The unused `LTM0` columns are removed and the row has this layout (Alec, 2026-11-01).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are built (accepted as implemented, Alec, 2026-11-01).


Item 6.2 closed on 2026-10-09 with its mechanism complete; whether the model
learns to chain is measured under item 0 (`MM_math_chain`).

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
predictor. Exact ties contribute nothing. The REINFORCE/EMA path and
`selectedThoughtPolicyWeight` are retired. There is one configured work name,
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

*Moved to [DevelopmentHistory](DevelopmentHistory.md#evidence) (2026-10-30): "Evidence".*
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
in [SelectedMeaning](SelectedMeaning.md) and [DevelopmentHistory](DevelopmentHistory.md#selected-meaning-and-one-controller-september-20).

## Addressed records (2026-10-29)

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
