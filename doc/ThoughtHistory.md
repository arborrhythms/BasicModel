# Ordinary thought history

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23, [step 2 §71](plans/2026-10-21-step2-one-ltm.md); recorded 2026-10-30):** an LTM1 row is two tagged references with its kind read from the referents, not three. As built at `3718d9d0dc`, `LTM0` still carries `c_plus`/`c_minus`, `source_trust`, `order`, `row_ids`, `origin`, `record_kind`, `polarity` and `witness_count` columns; only `refs`, `grammatical_mode` and the index columns are retired (audit 2026-10-30, Alec to decide).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Expectation is a **gain field** `c = m⊙a⊙o` with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec; [thinking spec §§16, 22](plans/2026-10-27-item2-5-thinking.md), [2.7 record §42](plans/2026-10-27-item2-7-corpus.md)): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are not built.


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
meanings and typed results; checkpoint history is version 6 and older
versions are rejected at load (`load_thought_extras`).

At a fully bound declarative, an uncancelled expectation image can yield
`not X`, with its confidence in the against pole of an inference. Ordinary
fully bound XOR closings do not open thinking episodes.

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
in [SelectedMeaning](SelectedMeaning.md) and [Testing](DevelopmentHistory.md#selected-meaning-and-one-controller-september-20).

## Addressed records (item 2.5 round A, 2026-10-29)

`ThoughtRecord.address` is the written chunk's `.where` (four floats), not a
physical index; `replay_thoughts` resolves it through `STMChunks.locate` when
a chunk state is supplied, recomputes each derivation's pair from the recorded
supports (minimum along the path, maximum across alternatives, lane exchange
for negation) and rejects a record whose stored pair differs. Checkpoints
write history version 6; older versions are rejected at load. The worked line of
[thinking spec §4](plans/2026-10-27-item2-5-thinking.md) item 6 replays with its pair and ordered witnesses.
