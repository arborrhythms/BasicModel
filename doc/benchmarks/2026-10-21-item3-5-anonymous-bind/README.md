# Step 1b §10.4: bind opens and refines anonymous tokens

Alec's decision in bind plan §10.4 supersedes the earlier §18 evidence
carry paragraph: **all landing gates run on the next frozen source**.
`eac2fe87b` remains the accepted MTM source; its incomplete gates are not
continued or promoted. No existing certificate assertion is changed.

## Mechanism

Equality rows no longer select or revoke a forward binding in
`ClauseRows.read_reference_bindings`. That reader interprets only a chosen
token or an explicitly retained derivation. A word site's first unresolved
bind opens a zero-content, anonymous MTM token at its occurrence. A gap
without a word remains open. The journal keeps an objectless word's name
through a unary bind; explicit open formations still own exactly one site.

Arrival binds are planned with the existing `ReferenceClose.score` and
`choice_cost`, and join the existing reference credit in `SentenceCredit`.
The keep law, cost weights, answer targets and objective are unchanged.
Each pending occurrence is visited once, with its containing row and
anonymous child naming the same site. Candidate tokens come from the
current frames and the arrival's actual constituents. Both a selected
operand and its identified lexical source remain available if they differ.
Only selected extra constituents are admitted, through the ordinary writer.

Preview and prepare do not mutate MTM. Only the kept commit opens the token
or appends the chosen candidate occurrence. `bind` assigns a kind only when
the entry has none. A bound use of an unrefined token cannot supply a new
fold kind before that arrival. Tokens store no working vectors or labels;
their content and retained derivation are interpreted on read. PartClosing,
PartRead and individuals use that interpretation. Native individual binds
still make no pool entry. No pool read enters the compiled body.

A raw parent keeps its structural address to the anonymous child and its
pending annotation. That address alone does not close the role. Its working
read clears the forward marker only after refinement; the closing query
therefore retains the original predicate and occurrence. Raw rows, Part
records and bank values are never rewritten by MTM commit or these reads.

`ForcedGrammar` extends its existing open/name choice rule to the resolver
choice seam. It recognizes candidates by their actual derivation (including
retained semantic heads), not a compound's surface word hint. The live keep,
explore suffix, departure count, zero budget and assertions are untouched.

## Development evidence, before the freeze

These runs diagnose implementation errors; none is final-source gate evidence.
Every failure, log, source manifest and captured state remains beside this receipt.

| Run | Result | Diagnosis / scope |
|---|---|---|
| `mechanism-01` | 100 passed, 2 failed | An initial journal edit created a second NP request. Reverted; virtual site materialization moved to prepare. Both old assertions preserved. |
| `anonymous-02` | 7 passed | First five new cases plus both restored site-ownership assertions. |
| `anonymous-03` | 5 passed | Added parent working-read checks. |
| `reverse-5-dev-04` | Failed D2 | Unary bind dropped its word name; fixture failed to recognize a composed token's semantic head. Both trials and keep probe retained. |
| `reverse-5-dev-05` | Failed D2 | Arrival enumeration excluded a lexical constituent wrapped by a bind. Removed that exclusion. Both-trial probe retained. |
| `reverse-5-dev-06` | Passed | Saved reverse draw 5 on the repaired development tree. Not a fresh draw. |
| `mechanism-07` | Collection error | Misspelled occurrence-view filename; no cases ran. |
| `mechanism-08` | 107 passed, 1 failed | MTM, reference close/mentions, G1–G3/episode, all 14 PartClosing and four occurrence views passed. c/e queried a zero-content child without the parent predicate. |
| `forward-dev-09` | Failed strict forward assertion | Wrong antecedent: the fixture accepted a compound's word hint. Both trials and keep probe retained before repair. |
| `c-e-dev-10` | Failed c/e | Capture identifies the unrelated one-slot query answer. Initialization and native query traces retained. |
| `forward-dev-11` | Both failed | Regular bind assigned a fold kind before refinement; the lexical source of a changed operand was missing. c/e exposed the raw parent losing its open-role semantics. Both trials/keep and c/e traces retained. |
| `forward-dev-12` | Forward passed; c/e failed | Moving query ownership to the child caused a redundant self-query. That change was removed; raw structural references remain pending until the working read clears their forward marker. |
| `c-e-dev-13` | 50 passed | Unchanged c/e, all 44 bind-thought cases and five initial anonymous-token cases. |
| `anonymous-14` | 10 passed, 2 failed | New regression exposed root-node alias loss when a selected lexical constituent was attached. The other failure was an incomplete test frame constructor. |
| `anonymous-15` | 12 passed | Root alias repaired; named native-individual binding reads closed with no pool entry. Eager/compiled trial-isolation checks pass. |
| `anonymous-16` | 13 passed | Each distinct pending occurrence is refined once; named binds to retained, unpooled occurrences keep the selected kind without reviving an old token. |

## First freeze: `51cce2005be3860f4da5ef15eebfac96812f437b`

Six forward first draws passed. Reverse draws 1 and 2 passed; draw 3 failed
the unchanged line 438 assertion (zero second-sentence roots carried three,
at least two required). Reverse draws 4–6 and the remaining gates were not
run. All nine first draws have full arrival captures from initialization.
The [source audit](source-audit-51cce2005.json) matches all 825 validated
files in every run to the frozen Git blobs. None of these passes is promoted
to the repaired source.

The [reverse-3 probe](fresh-final-reverse-3-tables.md) retains both trials'
readings, R, E, A and keep decisions for all sixteen closes. The failure
[replays exactly](frozen-reverse-3-replay-validation.json), including MTM,
words and reference audits. Six pending x tokens already have kind three.
A first-sentence explore mint in another stream caches a lexical x object;
the next sentence consequently has no no-object request for its new x use.
Its Part derivation keeps that cached object instead of binding the use to
the accessible token. This is a mechanism failure, with the keep untouched.

A separate [native-arrival audit](native-arrival-audit.json) fails on the
same frozen source: selecting native individual 4 leaves the old anonymous
token's kind unset even though the frame contains its actual occurrence.
Commit had discarded the native candidate's occurrence.

The repairs bind cached mentions of accessible pending names before Part
admission, and carry a selected native candidate's occurrence into arrival
commit. A symbol-only frame materializes that selected native constituent
through the ordinary writer. Neither path adds a pool entry for a native
individual. These binds use the existing score and reference credit.
Arrival refinements precede attachment of this close's ordinary uses: if
refinement reassigns a candidate's introducing occurrence, the new use
follows that chosen antecedent's current token instead of being detached.
Model timestamps follow that commit order. MTM's accepted detachment and
eviction rules are unchanged.

| Subsequent development run | Result | Scope |
|---|---|---|
| `frozen-reverse-3-repair-17` | 14 passed | Captured reverse-3 replay and the thirteen anonymous-token tests. |
| `anonymous-18` | Collection error | Incorrect c/e selector; no tests ran. |
| `anonymous-19` | 16 passed, 1 failed | Both native-arrival regressions and unchanged c/e passed; the new cached-word fixture left its lexical x vector outside native [0, 1] coordinates. |
| `anonymous-20` | 15 passed, 1 failed | The cached-word case exposed a bare row ID passed to MTM for a standalone pending leaf. Converted it to an occurrence address. |
| `anonymous-21` | 15 passed, 1 failed | The root has the correct three derivation; the new test called the answer matcher on the literal root instead of the bound site retained as its companion. The regression checks that site's answer and the root's derivation separately. |
| `anonymous-22` | 15 passed, 1 failed | Refining another pending token detached the selected introduction after the new use had been appended. The use lost its token link. |
| `anonymous-23` | 30 passed | Append kept uses after arrival refinement; all sixteen anonymous-token and fourteen MTM cases pass, including eager/compiled isolation. The new cached-use regression keeps the expected three antecedent and follows the selected introduction through an explicit arrival reassignment. |
| `frozen-reverse-3-repair-24` | 17 passed | Captured reverse-3 replay plus all sixteen anonymous-token cases after timestamps follow the new commit order. |

The [second freeze receipt](../2026-10-21-item3-5-anonymous-bind-v2/README.md)
owns the next complete evidence round. Every failure and its original source
manifest remains here. No second full sweep or learning claim.
