# Item 6: corrected-objective warm-up (2026-10-15)

The accepted gate-off measurement and search-inverse exemption are on
BasicModel main at `f1dbd12a2`; WikiOracle main is bumped to `27a60da`.
Both are pushed. This next pass changes measurement code and documentation
only. The production objective is the accepted R2 charge plus the search
exemption. No rule is retired and no learning claim accompanies the landing.

The amended scope follows [Orders.md](../../Orders.md) and plan §12:
dictionary columns are order-1 individuals and recurrence remains their
admission evidence. Singleton cosines check content retention. If at least
20 of 24 pass, the next measurement is the balanced order-1 identity branch,
gate on. Factorial/confounded kind recovery is an order-2 question and is
deferred; neither proposed order-2 design is implemented here.

## Did the exemption move the departure choice?

**No in the matched source-boundary replays.** The only production difference
between accepted R2 `90ef40a42` and `f1dbd12a2` is the search exemption in
`SentenceCredit.forward_discard`. Each unchanged certificate was run with
either exact implementation, holding all other code and initialization
fixed (diagnostic seed 20261015, Python/NumPy/Torch, CPU/eager fixture).
Each call also evaluates the counterfactual charge on the same journal.

Across the three certificates, all ten paired charge calls are zero on
both implementations; departures, reservoir replacements and keep
comparisons agree exactly. The shared `MM_math_chain` fixture has actual
meaning slice **[104:104]**, despite the XML's nominal meaning width.
There is no meaning mass for this charge to change.

| Certificate / October 14 failure | R2 replay | Exemption replay | Maximum charge difference | Cause found |
|---|---|---|---:|---|
| Bound declaratives: only departure 6; new against both R2 and 6.1 | Pass | Pass | 0 | Independent reservoir and walk sampling does not guarantee two distinct compose rounds. Conditional failure probability 2.15%; native frozen-menu replay 87/4096. |
| Eight-sentence fork: only departure 24; new against R2, recurring from 6.1 | Pass | Pass | 0 | Same diversity contract mismatch. Conditional failure probability 26.21%; native frozen-menu replay 1063/4096. |
| c's supplied answer: does not identify `five`; new against R2, recurring from 6.1 | Fails `matches(..., 'five')` | Same failure | 0 | Filled reference ends at an order-one child associated with `the`, with no lexical reference to `five`. Both batches have distinct compose rounds; failure precedes the diversity assertion. |

The two diversity failures share a cause. c's observed failure is a
different immediate defect; calling all three departure-diversity failures
would misstate its trace. Its question has no open slot, but the selected
child has native references `[-1, -1, -1]`, no semantic role reference,
and `subject_word_id=21`, whose inherited form is `the`. The row is absent
from `five`'s lexical association set, so the unchanged strict verifier
cannot reach `five`. Every paired keep comparison for c is a greedy tie.

The diversity probabilities condition on the recorded eligible menus,
not on the full training test. `SentenceFork.record` independently samples
an eligible round for each row; `SentenceCredit.departure` then independently
chooses among three available walks. Non-compose walks contribute -1 and
are omitted by the assertions. The native sampler reproduces the historical
single-round outcomes (6 and 24) without involving keep costs. No assertion
is relaxed and no production repair is included.

Diagnostics: [bound before](diagnostics/bound-old.json) /
[after](diagnostics/bound-new.json), [fork before](diagnostics/fork-old.json) /
[after](diagnostics/fork-new.json), [c before](diagnostics/answer-old.json) /
[after](diagnostics/answer-new.json), [native sampler](diagnostics/sampler.json),
[terminal association](diagnostics/answer-associations.json),
[meaning-slice provenance](diagnostics/answer-provenance.json).
Replay driver: [bisect_departures.py](bisect_departures.py);
sampler: [replay_sampler.py](replay_sampler.py).

## Does the new warm-up carry at least 20 singleton atoms?

The fresh run uses seed **20261010** and the original stages 1–4 protocol:
76 singleton presentations, 64 learning presentations for each two-/four-word
order stage, then eight reconstruction and eight supplied-answer learning
presentations for each of four stage-4 lessons. Original initial/final
observations are retained. No checkpoint from the old warm-up is reused.
The source archive for the seeded driver is `111d121c3`.

Stages 1–2 in that protocol call `present` with `train=False`: they write
the greedy reading without a paired explore or optimizer update. The
corrected keep objective is present, but these presentations do not use a
keep comparison to select a different reading. The corpus's actual singleton
fields are bare word forms (`cat`, `black`), preserved exactly.

**Singleton gate: 8/24 carry their atom; 20/24 required — fail.**
The other 16 stored meanings are zero. All 24 word vectors have meaning
norm approximately .2; no singleton is missing. This is the same retention
count as the old R/S/T warm-up, despite the corrected training objective.
The native meaning slice here is [104:232].

**Columns minted: 0.** The saved checkpoint agrees: all four component
owners have no noun direction parameters, no witnessed noun columns and
no pending noun prototypes. There are therefore no nearest-atom column
cosines to report.

| Record | Word / nearest atom | Cosine | Closed meaning norm |
|---|---|---:|---:|
| Singleton | `luma` | 1.000000 | 0.200000 |
| Singleton | `nemi` | undefined (zero row) | 0.000000 |
| Singleton | `tavo` | 1.000000 | 0.200000 |
| Singleton | `suri` | undefined (zero row) | 0.000000 |
| Singleton | `pako` | undefined (zero row) | 0.000000 |
| Singleton | `feni` | undefined (zero row) | 0.000000 |
| Singleton | `balo` | undefined (zero row) | 0.000000 |
| Singleton | `runi` | undefined (zero row) | 0.000000 |
| Singleton | `dax` | undefined (zero row) | 0.000000 |
| Singleton | `wug` | undefined (zero row) | 0.000000 |
| Singleton | `nep` | undefined (zero row) | 0.000000 |
| Singleton | `zot` | 1.000000 | 0.200000 |
| Singleton | `cat` | 1.000000 | 0.200000 |
| Singleton | `dog` | 1.000000 | 0.200000 |
| Singleton | `bird` | undefined (zero row) | 0.000000 |
| Singleton | `horse` | 1.000000 | 0.200000 |
| Singleton | `black` | undefined (zero row) | 0.000000 |
| Singleton | `white` | undefined (zero row) | 0.000000 |
| Singleton | `small` | undefined (zero row) | 0.000000 |
| Singleton | `large` | undefined (zero row) | 0.000000 |
| Singleton | `runs` | 1.000000 | 0.200000 |
| Singleton | `sleeps` | undefined (zero row) | 0.000000 |
| Singleton | `waits` | 1.000000 | 0.200000 |
| Singleton | `moves` | undefined (zero row) | 0.000000 |
| A columns: **0** | none minted | not applicable | — |

Full precision and native row addresses: [singletons.json](diagnostics/singletons.json).
Budget completion and checkpoint dictionary counts:
[warmup-summary.json](diagnostics/warmup-summary.json).

The carried order CE gates also fail: maximum CE on presentation 64 is
**.032051** for two words and **.016668** for four words, against **.01**.
Both stage-3 final free readbacks are **0/4**. All four stage-4 final
supplied-answer readbacks are **0/4**. These are warm-up observations;
no identity-branch score is inferred from them.

Singleton cosine uses the stored row and the word's final meaning bands.
The gate counts signed cosine ≥ .9; zero rows have undefined cosine and
do not carry an atom. Column alignment uses absolute cosine in the native
dictionary event coordinates, over all 24 stage-1/2 atoms. Column alignment
does not establish identity.

## Does the gate permit identity measurement, and where does a zero singleton lose meaning?

**Balanced order-1 identity, gate on: not run**, because 8/24 is below
20/24. Its bindings, row pairs and minted-column witnesses are unmeasured
in this pass. Factorial/confounded recovery remains deferred by the
order amendment. Recurrence admission remains unchanged.

The last native `black` singleton presentation (index 68 of 76, zero-based)
shows where its stored zero originates. This is its original stage-2 write,
not an additional exposure under the final policy. The word supplies .2
meaning norm; `non` removes it immediately. The following operations keep
zero, and the closed row remains zero.

| Step | Native action | Input meaning norm | Output meaning norm |
|---:|---|---:|---:|
| 0 | word `black`, field position 0 | — | 0.200000 |
| 1 | `non` | 0.200000 | 0.000000 |
| 2 | `generic` | 0.000000 | 0.000000 |
| 3 | `generic` | 0.000000 | 0.000000 |
| 4 | `generic` | 0.000000 | 0.000000 |
| 5 | `generic` | 0.000000 | 0.000000 |
| 6 | `generic` | 0.000000 | 0.000000 |
| 7 | `generic` | 0.000000 | 0.000000 |
| 8 | `generic` | 0.000000 | 0.000000 |
| 9 | `generic` | 0.000000 | 0.000000 |
| 10 | `generic` | 0.000000 | 0.000000 |
| 11 | `generic` | 0.000000 | 0.000000 |
| 12 | `generic` | 0.000000 | 0.000000 |
| 13 | `generic` | 0.000000 | 0.000000 |
| 14 | `generic` | 0.000000 | 0.000000 |
| 15 | `generic` | 0.000000 | 0.000000 |
| 16 | `generic` | 0.000000 | 0.000000 |
| 17 | `generic` | 0.000000 | 0.000000 |
| 18 | `generic` | 0.000000 | 0.000000 |
| 19 | `generic` | 0.000000 | 0.000000 |

[Native trace and retained row](diagnostics/zero-singleton.json).
All 76 singleton presentations were observation-only; each of their 19
batches had one greedy trial and no explore comparison. Thus the new keep
charge never chooses an alternative on these singleton writes. The loss
shown here occurs at `non`, before closing, rather than in a zero word
vector. Changing that teaching protocol is outside this pass.

## Reproduction and scope

Run from the BasicModel repository with a fresh output directory:

```sh
PYTHONPATH=bin:test .venv/bin/python doc/benchmarks/2026-10-15-item6-corrected-warmup/run_warmup.py output/item6-corrected-warmup-seeded-20261015
```

[run_warmup.py](run_warmup.py) reuses `identity_experiment.warmup` and observes
singleton steps during their original presentations. It suppresses only
the old hash/archive writers. [Setup notes](diagnostics/setup-notes.json)
identify the interrupted unseeded start and discarded diagnostic-observer
attempts; none is treated as a measurement. The commit is the archive.

No full sweep is run: production source is unchanged from the accepted
landing. The unchanged native assertions are exercised only in the scoped
bisection above. The prior sweep remains 5,503 passed, 287 skipped, one
XFAIL and the three failures analyzed here. The parts-of-speech pass waits
for a reading with NP structure. Identity/pronoun/determiner lessons and
rule retirement remain open.
