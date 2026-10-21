# Eight-seat gate on the §10.2 source

Source `f9faae610f35d5ca660d0ac49249637cb23e7adf`. Original
`data/MM_grammar.xml`, eight seats, 900 epochs and .20 threshold. Neither
fixture, menu nor closing boundary changed. These are the requested gate
measurements, not a new learning claim.

## Saved rng-119

| Question | Evidence |
|---|---|
| Does the original saved gate pass? | `mm-saved-02`: one pass |
| What happens to the old capacity-02 lower plan? | `mm-saved-02-state/plans.jsonl`, call 3: marker 4, description reference -1, inline lower; every entry has `seat: false`, no edge; allocator stays 5 → 5 |
| Does lower ever seat or write an edge? | 80 lower nodes, zero lower edges across 84 preflights |
| Does the entire fixture allocate nothing? | No. A later verb has one unidentified argument: call 13 seats symbol 6 and writes edge 7. The gate passes with that genuine relation-argument seat |

## The one fresh initialization

| Question | Evidence |
|---|---|
| Does the fresh gate pass? | `mm-fresh-02`: fails at the second preflight, before completing the budget |
| Is this another lower cut? | No: zero lower nodes. First chosen operation is `part`, then `verb` |
| Why are the arguments unidentified? | `mm-fresh-diagnosis-03-state/journals.jsonl`: both input concept IDs are -1, no word IDs, no lexical reference IDs/orders. The embedding input has no native identity carrier before the chooser; no kind reference was cut by lowering |
| First close | Two fresh order-0 arguments get seats 5 and 6; Part(5,6) gets edge 7. Four bootstrap rows plus these two seats fill the six order-0 positions |
| Second close | Verb's first argument is native 1; its second is unidentified and needs one fresh order-0 seat. Preflight raises with occupied=6, planned=1 |
| Exact replay | `mm-fresh-diagnosis-03` restores `mm-fresh-02-state/initial-rng.pt`; the same two plans and same capacity exception recur |
| Initial interpretation | Failed gate retained. The plan alone appeared to require genuine fresh relation-argument seats. The exact-code audit below disproves that interpretation: these values already have admitted identities |

The first failed run and its exact diagnostic replay are distinct retained
receipts. The latter adds only journal serialization; it does not change
the model, call an additional model read or draw randomness. The saved
rng-119 pass does not replace the fresh failure.

## Exact identity audit and repair

`mm-fresh-identity-04` replays that same failed initialization, additionally
comparing each planned argument with the existing native bank. It produces
the same failure. This read-only diagnostic finds **distance exactly zero**:

| Close | Planned new seat | Existing native identity | Distance |
|---|---:|---:|---:|
| First, part argument 1 | 5 | 1 | 0 |
| First, part argument 2 | 6 | 2 | 0 |
| Second, verb argument 2 | 8 | 3 | 0 |

Cause: `_stage_reading_grammar_references` already supplies
`_ar_grammar_object_ids` beside the grammar rows and atoms.
`_pushed_word_slab` consumes those atoms, but `_capture_reading_programs`
consults only `_word_symbol_concept_ids`, which correctly returns `None`
for this mixing path's absent WORD lane. Capture replaces the existing
OBJECT IDs with -1. The unchanged Part boundary then correctly refuses
another *apparently* fresh argument. This is a lost source-identity carrier,
not real exhaustion by new numerical content, and not a lower edge leak.

Repair: capture carries the admitted grammar OBJECT IDs paired with its
valid grammar rows. No code search or nearest-code inference is added;
the exact distances are diagnostic evidence only. The aligned WORD owner
and the absence of WORD IDs on the mixing path remain unchanged. A new
regression forces a native part read at eight seats and requires preserved
OBJECT IDs, no fresh seats and no fabricated lexical definitions.

The failed gate, both diagnostic replays, and the initial mistaken capacity
interpretation are retained. The fresh draw is replayed after the repair;
it is never replaced by another initialization. Results follow in this
receipt after completion.

The first carrier-only attempt, `mm-fresh-repair-05`, still fails and is
retained. Its journal now has `concept_ids=[1,2]` but
`reference_ids=[-1,-1]`: `Language.resolve_lexical_references` still cut
every order-1 non-individual source to -1. Removing this second cut keeps
the kind reference while stamping the token's order, consistent with
§10.2. `mm-fresh-repair-06` then passes; no new initialization was drawn.

## Confirmation on the follow-up source

Source **`67c5b203f34e4299703d4c7801b34e06c1045eb6`** (also fixes
occurrence participant reads; no closing-boundary or fixture change).

| Gate / receipt | Result | Preflights | Seats | Lower nodes | Lower edges |
|---|---|---:|---:|---:|---:|
| rng-119, `mm-saved-repair-07` | Pass | 32 | 0 | 22 | 0 |
| Same captured fresh draw, `mm-fresh-confirm-08` | Pass | 24 | 0 | 0 | 0 |

Original eight-seat inventory, 900-epoch cap and .20 threshold throughout.
No budget increase or replacement draw. This corrects the initial inference
that the missing IDs represented genuinely new content. The exact native
codes and their already-staged IDs establish the carrier defect instead.
