# Item 6 R2: forward discarded-meaning charge

Rejected in plan §9: the charge was mis-scoped and accepted certificates
were changed. The [inverse-domain repair](../identity-r2-inverse-domain/README.md)
restores those certificates; this receipt preserves the rejected measurement.

R/S/T landed as BasicModel `e38a19fbd`, WikiOracle `64a06d3`, with no
learning claim. R2 mechanism and driver: `0cd221d74` on
`codex/item6-r2-forward-discard`. No rule is retired.

Alec's October 11 scope: four retained clauses and factorial/confounded
recovery with the selected-mint gate on. Both branches restore the accepted
R/S/T stages-1–4 checkpoint, seed 20261010, revision-3 corpus, learning rate
.003, 144 training and 48 held-out documents each. Closed meaning is observed
within the existing held-out pass. Identity and the gate-off contrast wait;
there is no repeated warm-up or held-out replay.

## Does the old objective favor an empty reading?

Yes: coverage favors it by .2 and the inverse field term by .0823–.0836.
The supplied all-`sum`/STOP reading is a mechanism witness, not a learned parse.
[Diagnostic](diagnostic/objective/result.json), measured before the change on
the retained R/S/T factorial-off checkpoint:

| Term / norm | Greedy (each) | black cat runs | white cat moves | small cat waits | large cat sleeps |
|---|---:|---:|---:|---:|---:|
| content | 1 | 1 | 1 | 1 | 1 |
| coverage | .8 | 1 | 1 | 1 | 1 |
| field | 0 | .082280 | .082789 | .082626 | .083580 |
| iterations | .05 | .05 | .05 | .05 | .05 |
| Keep cost | 1.85 | 2.132280 | 2.132789 | 2.132626 | 2.133580 |
| Closed meaning norm | 0 | .059073 | .057282 | .059073 | .103833 |

## Is the first-word `non` preference learned?

Yes, on all four retained inputs. [Logits](diagnostic/warmup-logits/result.json):

| Chooser | `non` logit | Other legal scores | `non` wins? |
|---|---:|---:|---|
| Initial, same input | .040492 | .040914–.041296 | No |
| Accepted warm-up | .156446 | about −.100 | Yes, unique maximum |

## Does skipping one lossy step win?

**4/4 keep explore.** The charge sums relative L1 meaning mass removed at
each selected forward step; binary overlap counts once. Inverse κ = 0 and
`reconstruction.content` remain. No operator is forbidden.
[Native trial diagnostic](validation/skip-non/result.json):

| Retained clause | Greedy discard / total | Explore discard / total | Advantage | Closed norm after suffix |
|---|---:|---:|---:|---:|
| the black cat runs . | 3 / 4.85 | 2.4 / 4.25 | −.6 | 0 |
| the white cat moves . | 3 / 4.85 | 2.5 / 4.35 | −.5 | 0 |
| the small cat waits . | 3 / 4.85 | 2.4 / 4.25 | −.6 | 0 |
| the large cat sleeps . | 3 / 4.85 | 2.5 / 4.35 | −.5 | 0 |

This is one legal STOP departure with the native suffix, at learning rate
zero. The other keep terms tie at 1.85. It establishes local credit, not
retention. The mass charge can miss equal-mass substitutions; content remains
the certificate. One regression retains these four native paired programs.

## Does recovery retain meaning and identify the present atoms?

**No held-out recovery gain.** Both branches have **0/48** nonzero closed
meanings and **0/240** identification, as in R/S/T. All norm minima, means
and maxima are exactly zero, including the four retained clauses.
Identification requires an **active** column with |cos| ≥ .9; confounded's
one atom-aligned column is inactive and earns no identification credit.
[Factorial readings](recovery/factorial.json),
[confounded readings](recovery/confounded.json):

| Gate-on branch | Nonzero meanings | Identification | Minted / total columns | Merged pairs | Selected mint / commits | Kept operations (rows out of 48) |
|---|---:|---:|---:|---:|---:|---|
| Factorial | 0/48 | 0/240 | 0 / 58 | 0 | 192/192 | `non` 48, `lower` 48, `verb` 12 |
| Confounded | 0/48 | 0/240 | 4 / 62 | 0 | 136/192 | `non` 48, `adverb` 48, `lower` 1, `lift` 1 |

The programs change (19 factorial signatures, 5 confounded), but **all 96
held-out readings still use `non` and end with zero meaning**. Complete action
sequences are in the linked files. The 192 observed commits include 48 frozen
held-out calls. Mint is selected on every factorial call and
on 136 confounded calls; it is not universally selected in the latter.

The on/off gate contrast is unmeasured under this scope; its rule stays. Carried, not remeasured:
relation trace **2/4**, **43** unadmitted leaves, two-word order CE
**.031999–.032067** and four-word CE **.016526–.016706** (both gates **.01**),
free readback **0/4**. Identity lessons and rule retirement remain deferred.
The parts-of-speech pass in plan §6 follows R2.

Validation: 91 affected checks passed before the scope correction; five
extra new unit tests have since been removed. The one retained-clause
regression passes (1 test, 2.67 seconds). The already completed full sweep had **5,501 passed, 287 skipped,
2 failed, 1 XFAIL** ([numbers](full/summary.json),
[baseline comparison](full/baseline-comparison.json)). It is not repeated at
this step. The math-chain `five` failure persists; the review17 observer's
`KeyError: '1'` is **new against accepted 6.1 and R/S/T**. One unchanged
isolated rerun passed; its cause remains unresolved
([diagnostic](validation/observer-failure/result.json)). Cleared-cache overlap
remains XFAIL versus 6.1's XPASS. Existing operator certificates now account
for real OR discard and check shared contexts directly
([diagnostic](development/additive-input/certificate-diagnostic.json)).

Diagnostic context and reproduction details are in [notes](diagnostic/notes.md).
The superseded eight-branch run was intentionally stopped during warm-up,
before any branch began; its partial output stays in
`output/item6-identity-r2`. Candidate commits are the source archive.
