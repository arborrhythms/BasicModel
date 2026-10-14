# Item 6: admission-gate measurement (2026-10-14)

Accepted R2 `90ef40a42` is on BasicModel main; WikiOracle is bumped to
`066fb5d`. The gate-off measurements use that production code and the same
accepted R/S/T warm-up, seed 20261010, revision-3 corpus, learning rate .003,
144 training and 48 held-out documents per branch. Only
`require_selected_mint=False` changes the learning experiment. No new warm-up.

The search-inverse exemption is applied separately after measurement so
the gate-on/off comparison has only one changed factor. Its production
change receives one final sweep. The commit is the source archive.

## Does removing the selected-mint gate pass its retirement lesson?

**No.** Factorial identification is 0 overall and for every kind, unchanged
from gate-on; confounded training produces no measured merged pairs.
Required: factorial ≥ .9 overall and per kind, better than gate-on, with
confounded merges. None of these conditions passes; no rule is retired.

| Branch / gate | Overall | cat | dog | bird | horse | Minted / total columns | Merged pairs | Nonzero closed meanings | Mean norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Factorial on](../2026-10-10-item6/identity-r2-inverse-domain/recovery/factorial.json) | 0/240 | 0/24 | 0/24 | 0/24 | 0/24 | 0 / 58 | 0/4 | 44/48 | .643407 |
| [Factorial off](recovery/factorial-off.json) | 0/240 | 0/24 | 0/24 | 0/24 | 0/24 | 2 / 60 | 0/4 | 44/48 | .643407 |
| [Confounded on](../2026-10-10-item6/identity-r2-inverse-domain/recovery/confounded.json) | 0/240 | 0/24 | 0/24 | 0/24 | 0/24 | 0 / 58 | 0/4 | 4/48 | .016667 |
| [Confounded off](recovery/confounded-off.json) | 0/240 | 0/24 | 0/24 | 0/24 | 0/24 | 4 / 62 | 0/4 | 4/48 | .016667 |

The linked off reports give **every column × every atom absolute cosine**
(60 × 12 factorial, 62 × 12 confounded), including all 58 warm-up columns,
and every kept derivation. Identification requires an **active** column
with |cos| ≥ .9; the alignment matrix alone is not the identification score.
No column reaches .9; the off maxima are .638625 and .722386. All four
kind/property merge tests (`cat`/`black`, `dog`/`white`, `bird`/`small`,
`horse`/`large`) have empty column sets. A merge requires |cos| ≥ .9 to
the normalized pair sum without a clean atom match.

All four branches keep `verb` × 4, × 8 or × 12, **16 rows each**; no
`non`, `lower` or NP reading appears. Their closed norms also match exactly
between on/off. The gate contrast is no longer vacuous: off admits two/four
new columns versus zero on, without improved identification. Mint is
selected on 37/192 observed commits in each branch, including the 48 frozen
held-out calls. The gate-on controls are reused, not rerun.

## Do singleton meanings and warm-up columns align with their atoms?

The frozen warm-up contains all **24** stage-1/2 singleton observations.
Their labels come from native concept references (source text is absent),
and repeated presentations share the stored row and witness count. Eight
have cosine ≈ 1; **16 have zero closed meaning** although every word meaning
has norm .2. This reads stored observations, with no new singleton presentations.
Row numbers below are physical store indices; the diagnostic includes each
native row address and witness count.

All **58** columns fall below |cos| .9 against both the 12 recovery atoms
and the full 24 stage-1/2 atoms. The largest nearest cosine is .684137
over 24 atoms and .638419 over the 12 recovery atoms. Column cosines use
native dictionary event coordinates; singleton cosines use raw meaning
bands. [Full values and references](diagnostics/singletons.json).

| Stored observation / warm-up column | Word or nearest atom (24) | Cosine | Nearest recovery atom (12) | Absolute cosine | Closed / word norm |
|---|---|---:|---|---:|---:|
| Singleton row 4 | luma | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 5 | nemi | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 6 | tavo | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 7 | suri | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 21 | pako | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 23 | feni | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 25 | balo | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 27 | runi | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 53 | dax | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 55 | wug | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 57 | nep | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 59 | zot | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 77 | cat | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 79 | dog | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 81 | bird | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 83 | horse | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 93 | black | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 95 | white | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 97 | small | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 99 | large | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 109 | runs | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 111 | sleeps | undefined (zero row) | — | — | 0.0 / 0.2 |
| Singleton row 113 | waits | 1.000000 | — | — | 0.2 / 0.2 |
| Singleton row 115 | moves | undefined (zero row) | — | — | 0.0 / 0.2 |
| Column 67 | cat | 0.264273 | cat | 0.264273 | — |
| Column 71 | moves | 0.388024 | moves | 0.388024 | — |
| Column 73 | white | 0.451638 | white | 0.451638 | — |
| Column 74 | sleeps | 0.288698 | sleeps | 0.288698 | — |
| Column 78 | pako | 0.267564 | horse | 0.248898 | — |
| Column 79 | luma | 0.523273 | white | 0.246868 | — |
| Column 81 | moves | 0.316030 | moves | 0.316030 | — |
| Column 83 | suri | 0.395607 | large | 0.355748 | — |
| Column 84 | large | 0.537858 | large | 0.537858 | — |
| Column 86 | nemi | 0.360377 | horse | 0.283351 | — |
| Column 87 | sleeps | 0.463240 | sleeps | 0.463240 | — |
| Column 90 | waits | 0.444597 | waits | 0.444597 | — |
| Column 91 | dax | 0.684137 | sleeps | 0.638419 | — |
| Column 93 | horse | 0.378066 | horse | 0.378066 | — |
| Column 94 | cat | 0.283139 | cat | 0.283139 | — |
| Column 96 | bird | 0.284914 | bird | 0.284914 | — |
| Column 100 | luma | 0.306661 | cat | 0.305634 | — |
| Column 102 | feni | 0.358590 | waits | 0.339705 | — |
| Column 104 | pako | 0.386466 | cat | 0.368630 | — |
| Column 105 | dax | 0.323285 | waits | 0.228123 | — |
| Column 107 | feni | 0.290308 | small | 0.276768 | — |
| Column 121 | cat | 0.451032 | cat | 0.451032 | — |
| Column 125 | black | 0.307324 | black | 0.307324 | — |
| Column 133 | sleeps | 0.421185 | sleeps | 0.421185 | — |
| Column 136 | dax | 0.579490 | large | 0.563585 | — |
| Column 140 | waits | 0.360526 | waits | 0.360526 | — |
| Column 141 | small | 0.383806 | small | 0.383806 | — |
| Column 143 | large | 0.452064 | large | 0.452064 | — |
| Column 144 | black | 0.429663 | black | 0.429663 | — |
| Column 145 | suri | 0.555444 | black | 0.444956 | — |
| Column 146 | waits | 0.428537 | waits | 0.428537 | — |
| Column 148 | cat | 0.271769 | cat | 0.271769 | — |
| Column 151 | luma | 0.433311 | runs | 0.261963 | — |
| Column 152 | bird | 0.422393 | bird | 0.422393 | — |
| Column 153 | bird | 0.418199 | bird | 0.418199 | — |
| Column 155 | pako | 0.199018 | horse | 0.163257 | — |
| Column 157 | suri | 0.425088 | sleeps | 0.310787 | — |
| Column 159 | bird | 0.486365 | bird | 0.486365 | — |
| Column 161 | bird | 0.430064 | bird | 0.430064 | — |
| Column 162 | cat | 0.307235 | cat | 0.307235 | — |
| Column 163 | cat | 0.312538 | cat | 0.312538 | — |
| Column 164 | cat | 0.272840 | cat | 0.272840 | — |
| Column 176 | waits | 0.352859 | waits | 0.352859 | — |
| Column 177 | dog | 0.339345 | dog | 0.339345 | — |
| Column 178 | pako | 0.301526 | cat | 0.290485 | — |
| Column 180 | tavo | 0.298512 | bird | 0.298161 | — |
| Column 181 | pako | 0.208375 | horse | 0.165864 | — |
| Column 183 | dog | 0.303898 | dog | 0.303898 | — |
| Column 185 | cat | 0.531003 | cat | 0.531003 | — |
| Column 187 | dog | 0.340869 | dog | 0.340869 | — |
| Column 188 | dog | 0.332592 | dog | 0.332592 | — |
| Column 190 | dog | 0.343361 | dog | 0.343361 | — |
| Column 193 | dog | 0.383582 | dog | 0.383582 | — |
| Column 195 | dog | 0.309036 | dog | 0.309036 | — |
| Column 198 | dog | 0.299013 | dog | 0.299013 | — |
| Column 199 | luma | 0.306112 | dog | 0.241126 | — |
| Column 200 | cat | 0.487737 | cat | 0.487737 | — |
| Column 201 | dog | 0.393055 | dog | 0.393055 | — |

## Where does the confounded verb reading lose meaning?

`the black cat runs .`, from the accepted confounded gate-on checkpoint.
The original first batch is replayed frozen; only this row is traced. Its
native read order is **the → black → cat → . → runs**. [Full trace](diagnostics/trace.json).

| Step | Selected leaf / operation | Left norm | Right norm | Output / leaf norm |
|---:|---|---:|---:|---:|
| 0 | the | — | — | 0.0 |
| 1 | black | — | — | 0.2 |
| 2 | verb | 0.0 | 0.2 | 0.0 |
| 3 | cat | — | — | 0.2 |
| 4 | verb | 0.0 | 0.2 | 0.0 |
| 5 | . | — | — | 0.0 |
| 6 | verb | 0.0 | 0.0 | 0.0 |
| 7 | runs | — | — | 0.2 |
| 8 | verb | 0.0 | 0.2 | 0.0 |

The content words arrive with nonzero meaning. The first fold removes
`black`'s meaning; subsequent folds remove `cat` and `runs` the same way.
The zero accumulator starts with `the`. The trace locates the loss at the
composition boundary, not in those three word rows; it does not by itself
attribute the loss to a particular learned weight. Closed meaning norm is 0.
Leaf journal slots are padding, so leaf norms use the actual pushed rows.

## Are search-kind inverses exempt from the forward charge?

The charge now exempts the registered `inverse_kind='search'`, including
OR. There is no operator-name rule. The field reader still leaves those
coordinates undefined (κ = 0); search does not invent an expected operand.
Other local inverse domains and accepted certificates are unchanged.
[Before regression](diagnostics/search-before.log),
[after regression](diagnostics/search-after.log).

| Check | Accepted R2 | Search exemption |
|---|---:|---:|
| Field OR, exact bank-search recomposition: forward charge | .629629672 | 0 |
| Field OR inverse coordinates | Undefined | Undefined |
| Retained clauses keep the trial skipping one `non` | 4/4 | 4/4 |
| Exact reconstruction certificates | R = 0 | Pending final sweep |
| Full sweep | 5,502 passed / 287 skipped / 1 XFAIL / 0 failed | Pending |

One final sweep follows this production correction. The learning results
above predate the correction, preserving the single-factor gate contrast;
there is no learning claim about the corrected objective. No POS change
is made; that pass waits for an NP reading.

Carried, not remeasured: relation trace **2/4**, **43** unadmitted leaves,
two-word CE **.031999–.032067**, four-word CE **.016526–.016706** (both gates
**.01**), free readback **0/4**. Identity lessons remain outside this pass.
