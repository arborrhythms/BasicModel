# Existence evidence and grammatical descriptions

An idea is conceived and does not exist (decided 2026-10-20). A noun phrase's
default verb phrase is read as *occurs*: presence to the mind in a situation
and nothing more, which holds of an idea, a percept and a relation alike.
Existence proper is a further relation asserted by evidence, and that evidence
is what the readers below weigh.

The grammatical `exist` is `isTrue`'s conceptual face: it reports
the presence of a concept's content as two independent lane magnitudes
(`ThoughtConceptualCapability.extent`) and reads no LTM row. Whether long-term
memory supports a description is `isTrue`'s question. Given an ended
proposition by reference, it reads that row's two lanes, trust and witness
(`ThoughtLTMCapability.end_evidence`). Given a description with no open
operand, it reads the description's region
(`ThoughtLTMCapability.truth_evidence`): the lane for takes pervasion, a
trusted relation that covers the region from a higher order, or a row whose
content matches the description exactly; the lane against takes pervasion
against or any row in the region held against it. A row that is only like the
description is analogy, reported beside the answer and never taken as proof,
and a row of zero trust never answers. The two readings of a region are in
[Logic](Logic.md#existence-and-pervasion-alec-2026-11-13). Neither face brings
a frame into STM; `query` does that. See
[AccessibleMind](AccessibleMind.md) and [QueryContracts](QueryContracts.md).

The standalone `TruthGroundedReasoner` reader below is the public face of the
same region read. It is not the normal thought executor; the storage rules are
shared.

## Standalone reader contract

The reader's `existence_evidence` asks whether a trusted row of long-term
memory lies in the region of a description. The region is read through the
description's extension, the active concepts each of its operands gives
([Extend](Extend.md)), and is matched order by order against the entries of
the rows in the read window (`ExtendedRecall.matches`). A row's brightness is
its relevance to the region times the magnitude of its trust, on the lane of
its trust's sign. Each lane takes its brightest row, never a sum: repeating a
source cannot increase its supported degree.
[Full lookup](../bin/reasoning.py).

The result includes both degrees, the witnessing row on each lane, and each
candidate's occurrence, trust and match. No match means unknown. Low support
remains visible even below the posture threshold; strong positive and
negative support yields `BOTH`. The normal controller retains the checked
typed result and its sources through child returns, history and final answer
preparation. [Posture](../bin/reasoning.py), [controller](SelectedMeaning.md).

The reasoner's `exist()` leaves conceptual content and its pair, and its
`is_true()` an ended proposition's pair and witnesses, or a description's
region read as above; neither returns a scalar. Checked evaluation and the
normal controller use the rich evidence result. Conceptual presence is never a
substitute for a fact, and existence alone is analogy: it does not make a
claim deductively true. [Compatibility view](../bin/reasoning.py).

## Meaning ownership and storage

`ConceptualMeaning` is an owned value, not another memory or planner. Roles are
`[3, D]`, with a separate Boolean presence mask, polarity and an optional
sentence kind (`idea` or `relation`); bindings, scope and typed references are
immutable metadata. It carries no grammatical mode: whether a sentence asserts
is a fact of its kept program. When the `ask` rule lies on the kept head path
(an `ask` inside an operand does not count), the clause's evidence pair and
source evidence are set to $(0,0)$, so an asked clause asserts nothing. A live
clone keeps its current-step gradient. The explicit adapter converts STM order
to infix NP1/VP/NP2 order; depth three uses `[1, 2, 0]`, and depth two uses
`[1, 0]`. The predictor and observation writers share that adapter. A row of
`LTM0` keeps one idea, the meaning's one occupied role; a relation between two
is kept as two references in `LTM1` (`write_clause`).
[Value](../bin/Meaning.py),
[adapter](../bin/Meaning.py),
[prediction adapter](../bin/Layers.py),
[observation writer](../bin/Models.py).

`LTM0`, the table of ideas, owns each row's idea vector (`slots`, with the
row's time written into four of its coordinates), one signed `trust` from -1
to 1, an optional
`referent` (the concept, or the earlier row, that the idea is of), its slots
(`episodic_concepts`, `episodic_values`: a few concepts of the field the
sentence left, of every order, each with its value) and an `activity`. `LTM1`, the table of
relations, owns two tagged references, a pair of trust lanes (`truth_plus`,
`truth_minus`) and an `activity`. A row has no origin and no kind. A row is a
truth when its trust is not zero; every reader treats a row of zero trust as
making no truth claim. A question is a closing whose meaning still has open
references, held in MTM while they are pending. An idea with evidence on both
lanes is written as two rows, one on each lane. A root whose evidence is
$(0,0)$ commits no event to any identity, since an unasserted sentence changes
no state. A truth-store entry may assert only a closed clause: asserting one
with open references raises. A question cannot certify its own referent.
Where a row's trust comes from is in
[Philosophy](Philosophy.md#the-three-sources-of-trust-in-the-model-alec-2026-11-13).
[Store](../bin/Memory.py),
[admission](../bin/ClauseRow.py),
[write boundary](../bin/Models.py).

A durable reference to an idea is the store's namespace and the row's time
(`LTM0.row_name`). Times rise strictly, so a row keeps its name when
compaction moves it. A relation is named by its handle in `LTM1`. The bounded
nested-meaning reader and retention graph are described in
[NestedRetention](NestedRetention.md); selected composition and all three
observation writers preserve nested occurrences
([SelectedMeaning](SelectedMeaning.md)).
[Occurrence identity](../bin/Memory.py),
[compaction](../bin/Memory.py).

## Checkpoints

Every column of both tables is tensor state. No semantic sidecar is saved
beside them, and the tables keep no dictionary-valued PyTorch `_extra_state`
entry. [Save](../bin/Models.py), [columns](../bin/Memory.py).

A checkpoint in the preceding layout, where each idea held two evidence lanes
and a separate sidecar held its bindings and source, is imported once: the
pair becomes the signed trust, each row's activity starts at its default, its
episodic record is kept as its slots, and selected referents and pending
bindings move to the `referent` column and to MTM; the rest of the old
provenance and every retired field are dropped with a warning. An idea that
held both lanes cannot be kept in one signed row, and that checkpoint is
refused. [Import](../bin/MemoryCheckpoint.py), [layout check](../bin/Memory.py).

A checkpoint in a layout older than that is not loaded: loading one fails with
the unsupported layout version.

## Predictions and trust

What expectation gives is never stored as a row: it primes, and it is read
detached. A row is a truth only by its trust, which its source sets when it
is written. No confidence of a reading and no scalar conversion changes it;
an idea's trust is never rewritten, and a relation's pair only gains, by the
per-lane maximum, when a source states the relation again. A contradiction is
measured and reported, never repaired by the store. A kept thought is written
at what its premises give, the least on each lane. Ordinary thought history
records checked support and typed results; it does not promote generated
content into truth. See
[Queries](../bin/Queries.py), [thought history](ThoughtHistory.md) and
[SelectedMeaning](SelectedMeaning.md).

## Gradients

The meaning-value clone preserves live representation gradients, while durable
store writes detach them. Region reads, thresholding and support aggregation
are hard operations with scalar evidence; they provide no ordinary
derivative through their choices, and they add no learned parameter
or loss. The architecture-wide gradient boundaries are documented in
[GradientFlow](GradientFlow.md).
[Owned value](../bin/Meaning.py),
[detached write](../bin/Memory.py),
[hard lookup](../bin/reasoning.py).
