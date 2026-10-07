# Operators round 3a — identity by construction

Status: **accepted by Alec as the round-3a landing, 2026-10-07**, under plan §29. Commit, push and the WikiOracle submodule bump are authorized. [Acceptance record](acceptance.json).

The accepted candidate is manifest `84edc57e0ef1c95478a0e17a78a44b23fbafb8bc441031591c5f70bb27d580dc`, **707 source files**. Finalizing this receipt and recording acceptance change no runtime, configuration or test source. No training, seed selection, retry or replacement is added.

Start: the accepted round-2e candidate, manifest `3fd07f0cd516da3102867207614e11a9153638d37057bcc82a8c7ff821689106`. Its separate round-2 landing is committed first, then this candidate, with each commit checked against its own receipt manifest. The original archive remains in `before/`.

## Gates

| Measurement | 6.8 landing | Round 2e | Round 3a |
| --- | ---: | ---: | ---: |
| Class, four correct labels and MSE < .05 | 7/10 | 9/10 | **10/10, all in the “at zero” band** |
| Four-multiset reconstruction | 9/10 | 10/10 | **10/10** |
| Joint class and reconstruction | 6/10 | 9/10 | **10/10** |
| Sum controls at ¼ | 10/10 | 10/10 | **10/10** |
| MM_xor, raw convergence count | 10/10 | 10/10 | **10/10** |

“At zero” is the unchanged §20.5 band, MSE < .05, not a claim that every floating-point MSE equals zero. The class read uses the presented reader on the final committed root. All forty labels are correct and all forty final roots are conjunction. **R is exactly zero on both trials of every sum/XOR training sentence**; the keep therefore remains greedy. Every identified-word reconstruction audit reports zero errors.

The single delivered-source sweep completed **5,289/5,289** selected cases: **5,003 passed, 285 skipped, one xpassed**, in 394.849 seconds. It preceded exactly thirty unseeded trainings: ten 400-epoch sum controls, ten shared 400-epoch class/reconstruction trainings, then ten MM_xor trainings with the unchanged 200-epoch maximum. The standing gate passes. Sentence-path perception gradients, native-code displacement and ownership conflicts are zero.

[Validation](results-validation.json), [full sweep](full-sweep/result.json), [campaign completion](measurements/complete.json), [full per-run tables](measurement-results.md), and [reader trajectories](reader-trajectories.md) preserve all outcomes.

## Per-run results

Each XOR row is one training shared by the two gates. Stable conjunction is the epoch after the last greedy disjunction, including reversals; epoch 1 means conjunction throughout.

| Run | Presented class MSE | Labels | Multisets | Stable conjunction from epoch |
| --- | ---: | ---: | ---: | ---: |
| [1](measurements/xor-01/run.log) | 4.392483589e-06 | 4/4 | 4/4 | 20 |
| [2](measurements/xor-02/run.log) | 0.0207201172 | 4/4 | 4/4 | 27 |
| [3](measurements/xor-03/run.log) | 0.0160247575 | 4/4 | 4/4 | 33 |
| [4](measurements/xor-04/run.log) | 7.507328093e-13 | 4/4 | 4/4 | 39 |
| [5](measurements/xor-05/run.log) | 8.260059303e-14 | 4/4 | 4/4 | 21 |
| [6](measurements/xor-06/run.log) | 3.649935653e-06 | 4/4 | 4/4 | 8 |
| [7](measurements/xor-07/run.log) | 4.332534331e-12 | 4/4 | 4/4 | 41 |
| [8](measurements/xor-08/run.log) | 2.025046797e-13 | 4/4 | 4/4 | 29 |
| [9](measurements/xor-09/run.log) | 0.003708353323 | 4/4 | 4/4 | 13 |
| [10](measurements/xor-10/run.log) | 8.759345533e-07 | 4/4 | 4/4 | 1 |

| Run | Sum MSE | Sum contrast | MM best MSE | MM epochs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.25 | 8.9407e-08 | 0.19575128 | 51 |
| 2 | 0.2500000596 | 1.7881e-07 | 0.192050308 | 18 |
| 3 | 0.2500001192 | 5.0664e-07 | 0.193357274 | 26 |
| 4 | 0.2499999404 | -1.7881e-07 | 0.177393943 | 52 |
| 5 | 0.2500001192 | 3.5763e-07 | 0.194779262 | 75 |
| 6 | 0.2499999553 | -1.7881e-07 | 0.195753545 | 63 |
| 7 | 0.2500000298 | 1.1921e-07 | 0.179967463 | 89 |
| 8 | 0.2500001192 | 4.4703e-07 | 0.195012897 | 36 |
| 9 | 0.2500000596 | 2.0862e-07 | 0.191194475 | 117 |
| 10 | 0.2499999553 | -2.3842e-07 | 0.198101357 | 52 |

All sum controls stay within the declared quarter-floor band. The MM values are raw gate results; the repeated seed-zero bisection confirms the historical RNG-only path under amended §5. Round 3a matches round 2e in construction, first output and after-forward parameter/RNG digests. The bisection and separate paired replays remain in [their results](mm-first-forward/result.json) and [paired record](paired-mm/result.json).

## Form audit, separate from training

The frozen construction is 64 pair coordinates, three initial bits per pair/mint, a separate 32-coordinate exact thermometer, a fixed dense projection to 64 binding coordinates, and checked positional mints. The [construction review](construction-review.md) and plan §26 amendment remain intact. Sparse forms key identity; projected directions enter the unchanged binding kernels.

| Configuration / sample | Distinct native words | Collision groups before → after | Native containment pairs | Violations | Indexed byte errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| XOR_grammar.xml | 4 | 0 → 0 | 0 | 0 | 0 |
| MM_grammar.xml | 4 | 0 → 0 | 0 | 0 | 0 |
| BasicModel.xml | 67,391 | 2 → 0 | 18,309 | 0 | 0 |
| Dictionary sample | 19,986 | 1 → 0 | 682 | 0 | 0 |
| Numeric MM_xor | 0 | Not applicable | Not applicable | Not applicable | Not applicable |

Each configuration has a **separate witness bank**, never appended to the gate inputs. `bana ≤ banana`, `cat ≤ concat`, and `aba ≤ ababa` hold in both parts and forms. `an/and` and `an/ant` are not comparable because `n#` is missing from the longer word. All lengths 1–32 are exact, and `aaaaaa/aaaaaaa` differ before minting. `circus/cursic/cirrus` remain distinct. `calaba/cabala` are separated by `cal@1/cab@1`, three bits each. Zero native containment pairs in the two grammar vocabularies is reported as vacuous; the witnesses supply nontrivial comparisons.

BasicModel’s two native mints are `SPSS/SSPS` using `#SP@0/#SS@0`, and `enteriditis/enteritidis` using `rid@5/rit@5`, three bits per atom. The dictionary sample is 19,986 unique words from the identity toy’s 20,000 draws; its one mint is `calaba/cabala`. All configurations share projection SHA-256 `1119d9d4f413fc861da7f5ecc5de78110a888c1aeaae8fdc9f60b3e0747d65fe`; atom admission does not consume the global RNG. [Full form audit](form-audit.json), [identity report](identity-report.md).

### Storage finding, not a gate result (§27 item 2)

The static census of the configured first 2,000 local FineWeb documents contains **67,391 distinct words** and **69,566 atom/word rows in the separate audit bank**, against the **32,768-row production bank**. It covers 90,676 accepted sentence splits; three are excluded. The corpus shard and lexer are identified and hashed in `form-audit.json`.

This is the all-distinct-word construction census, not a measurement of simultaneously resident, recurrence-admitted production vocabulary and not a full-corpus training pass. Admission still uses the recurrence threshold; hapax words can be assembled from pairs and length. The production capacity has not been raised. A resident recurring vocabulary beyond capacity still requires an explicit capacity decision, with forgetting remaining sequence item 5. This storage constraint is kept separate from the four gates above.

## Operator result and conditioning (§27 item 1)

Every run selects conjunction by cost. Across XOR there are **8,040 compose departures with nonzero advantage** and **7,960 narrowing departures, all exact ties**. R and E contribute zero differences; compose credit comes entirely from the comparison reader’s answer term. Reconstruction keeps greedy on all 16,000 XOR rows. All 16,000 sum departures also tie. The final selecting costs and every contrary comparison remain in [the aggregate](aggregate-audit.json) and each raw run audit.

The review’s “disjunction reads at .26–.58” describes the **relative answer cost A**, not raw MSE. On the compose rows sampled at epoch 400, its per-run means are .2590–.5689. For a window less sensitive to that epoch’s sampled rows, the table also gives comparison-reader raw MSE over paired compose rows in epochs 381–400. These are pre-update comparison reads, distinct from the presented class MSE above.

| Run | Epoch-400 disjunction A | Late conjunction raw MSE | Late disjunction raw MSE | Mean late absolute ΔC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.4030874 | 0.01446108 | 0.3138825 | 0.41919 |
| 2 | 0.3183894 | 0.03053176 | 0.3007528 | 0.3853377 |
| 3 | 0.3465196 | 0.007856709 | 0.3098171 | 0.4227445 |
| 4 | 0.2590297 | 0.01589197 | 0.3213638 | 0.4295967 |
| 5 | 0.4242785 | 0.1080593 | 0.3667961 | 0.4700129 |
| 6 | 0.4663274 | 0.04917549 | 0.3810321 | 0.4859233 |
| 7 | 0.51332 | 0.02002301 | 0.3040773 | 0.397676 |
| 8 | 0.4870716 | 0.06304847 | 0.3766167 | 0.4748961 |
| 9 | 0.3874155 | 0.05028748 | 0.3358576 | 0.41069 |
| 10 | 0.5688587 | 0.03708369 | 0.3725267 | 0.473664 |

Both kernels’ four roots are affinely separable, as §27 states. The landing check reconstructs the fixed projected roots from the saved float32 forms and unchanged kernels, then centers the four rows and computes SVD in float64. It reproduces all ten saved final conjunction root matrices within **1.19209290e-7**, consumes no global Torch RNG, and runs no training or optimizer. All runs share these fixed roots.

| Kernel | Centered singular values (four values) | σ₁/σ₃ | Minimum affine-fit weight norm |
| --- | --- | ---: | ---: |
| conjunction | 0.877855382, 0.800606924, 0.482921466, 1.27634946e-16 | 1.817802 | 2.093925 |
| disjunction | 0.691669863, 0.523711215, 0.0588574228, 1.62930857e-16 | 11.75162 | 20.45068 |

The fourth value is the necessary near-zero value after centering four rows. Disjunction’s third direction is about **8.2 times smaller** than conjunction’s, and its minimum affine-fit coefficient norm is about **9.8 times larger**. Both retain centered rank three at the stored float32 precision; the disjunction direction is well above the precision cutoff, so this is not numerical rank loss. The exact affine fits remain near 1e-30.

These measurements support a conditioning difference and make finite-step convergence a plausible explanation for the comparison reader’s operator preference. They do not prove that conditioning alone causes it: the comparison reader is trained jointly on both root families, whereas the static affine fits optimize each family separately. The class gate certifies identity, presented-reader convergence, and operator discrimination in this measured training; it does not distinguish the operators in principle. Round 4’s semantic membership gate remains the proper operator certificate. [Root matrices, singular values, costs and method](landing/root-conditioning.json), [report script](landing/geometry_report.py.txt), [process log](landing/root-conditioning.log).

## Ownership, acceptance and preserved evidence

Each reader takes 400 updates per sum/XOR run, with the presented reader on the reconstruction keep and the comparison reader on both compose roots or the kept narrowing root. Ownership, sentence-path perception gradients and native-code displacement remain zero. The form, polarity, exact reconstruction, and trust-to-pair ports are covered by the accepted sweep; no source is changed for acceptance.

- [Frozen source manifest](delivered-source/source.json), [archive](delivered-source/source.zip), [diff](delivered-source/changes.patch), [30 complete test ports](delivered-source/test-ports.json), [188 helper hashes](delivered-source/measurement-helpers.json), and [seed-call audit](delivered-source/seed-port-audit.json).
- [Measurement protocol](measurement-protocol.json), [summary](measurements/summary.json), [audit](measurements/audit-summary.json), [reader diagnostics](reader-diagnostics.json), and [precision report](precision-geometry.md). All process logs and every run remain in this directory.
- [Acceptance record](acceptance.json) identifies the source manifest, acceptance date, documentation-only finalization and separate packaging metadata. Large receipt archives retain their exact bytes through Git LFS.

The round-2 receipts, both toys, the complete plan, and Claude’s document edits are included in the two ordered landings. The plan itself is not edited by this acceptance. Round 3b is folded into round 4 under §28; no round-4 implementation is part of this landing.

The final document-link check passes **312 tests** (`landing/doc-links-final.log`).
The initial check found two broken relative-link sets inside partial temporary
diagnostic checkouts; its log remains in `landing/doc-links.log`. All **2,110**
non-cache files from those three copies are preserved byte-for-byte in a
[verified archive](landing/temporary-source-copies.zip), with the full file
inventory and hashes in the [packaging record](landing/temporary-source-cleanup.json).
The duplicate extracted trees were removed; the frozen source, tests, inputs
and measurement results remain unchanged.
