# Parts of speech from operand participation: a toy of plan §6.1 (2026-10-13)

**Setup.** `sim.py` (numpy, `../../../.venv/bin/python sim.py`, about 3 minutes; full tables in `results.md`/`results.json`) simulates the learning rule of [plan §6.1](../../plans/2026-10-10-item-6-stored-idea-generativity.md). A hidden grammar with hidden classes (determiners a/the; 4 adjectives; 8 nouns = 4 kinds + 4 proper names; 4 intransitive and 4 transitive verbs; pronouns it/they; in corpus C the ambiguous run/walk, noun and verb) generates sentences such as *the black cat sleep*, *a dog see tom*, *it purr*, and their true parse over nine operand positions: `det(argin1, argin2)`, `adj(argin1, argin2)`, `clause(argin1=subject NP, argin2=VP)`, `vp(argin1=verb, argin2=object NP)`, `bind(argin1=context, argin2=pronoun)`. As in `record_category_observations` (bin/Language.py) a use is a word that is the *direct* operand of an operation; a composed phrase is not (the `heads` variant also counts the head word of a phrase). Corpora: **A** ordinary; **B** 40% of NPs are bare common nouns, so `adj.argin2` and `vp.argin2` share their words (nouns; the rare names at `vp.argin2` are the "almost"); **C** = A plus run/walk at 10% of noun and of intransitive-verb picks. The learner stores one class pointer per word and a class matrix in position space (rows = online k-means centroids, EMA rate `eta`). Column share at p = `C[k,p] / (eps0 + sum_j C[j,p])` (softmax over classes of log typicality with a null class of mass `eps0`); E step switches the pointer when the best share beats the current class's by `margin`; M step moves the chosen class (rate `eta`); a switch pulls old and new class together (rate `rho`); growth when no class has typicality `>= grow` at p; merge when two class vectors are within `merge` (L2), smaller class remapped. Chooser: the true parse competes with every single-word relocation; score = `m·[true] + w·Σ log(share+.02) + Gumbel(tau)`; credit = the true parse was chosen, and only credited sentences are observed. No oracle label is read by the learner or chooser. Defaults: `eta=.02, rho=.05, margin=.2, grow=.05, eps0=.05, merge=.35, w=.5, m=1, tau=.5`, order `EM` (plan: E step, then the chosen class takes the M step); `ME` is the one-line variant in which the word's current class also takes the M step before the switch. Seeds 0–4, mean ± sd; purity/completeness are word-weighted against the six hidden classes (names scored as nouns; run/walk excluded).

## Q1 Class count, purity, completeness (order EM = the plan)

| corpus | N | classes | purity | completeness | names separate | pronoun separate | vi/vt merged |
|---|---|---|---|---|---|---|---|
| A | 200 | 7.6 ± 0.5 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| A | 1000 | 7.0 ± 0.0 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| A | 5000 | 7.0 ± 0.0 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| B | 200 | 7.6 ± 1.2 | 1.00 | 0.93 ± 0.08 | 2/5 | 5/5 | 0/5 |
| B | 1000 | 6.0 ± 0.0 | 1.00 | 1.00 | 0/5 | 5/5 | 0/5 |
| B | 5000 | 6.0 ± 0.0 | 1.00 | 1.00 | 0/5 | 5/5 | 0/5 |
| C | 200 | 7.8 ± 0.4 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| C | 1000 | 7.0 ± 0.0 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| C | 5000 | 7.0 ± 0.0 | 1.00 | 0.83 | 5/5 | 5/5 | 0/5 |
| A+heads | 5000 | 4.4 ± 0.5 | 0.75 | 1.00 | 0/5 | 0/5 | 5/5 |

Class count over presentations (A): 8.6 → 8.6 → 7.6 → 7.2 → 7.0 at 50/100/200/500/1000; 9 growths (one per position), 2 merges. Purity is 1.00 everywhere. Completeness 0.83 in A and C is the proper names: under direct-operand observation names never share a position with common nouns (names fill `clause.argin1`/`vp.argin2`, nouns `det.argin2`/`adj.argin2`), so they are a seventh class — the "proper noun" of plan §6 decision 3, not an error. vi and vt never merge (they share no position). With head propagation (A+heads) names merge into nouns (NP), vi/vt merge (vt heads the VP at `clause.argin2`), and **pronouns merge into NP in 5/5 seeds** — the `bind.argin2` position alone does not hold them apart from the pull; the ME order keeps them separate in 4/5. Order ME gives the same final classes as EM in A/B/C and converges faster (B: 6.0 classes at N=200 vs 7.6).

## Q2 Collapse: do `adj.argin2` and `vp.argin2` read as one class?

| corpus | collapsed | collapse at (per seed) | noun+name one class | merges | final classes |
|---|---|---|---|---|---|
| A (EM) | 0/5 | — | 0/5 | 2.0 | 7.0 |
| B (EM) | 5/5 | 157 222 274 203 239 | 5/5 | 3.0 | 6.0 |
| B (ME) | 5/5 | 151 175 166 174 161 | 5/5 | 3.0 | 6.0 |
| C (EM) | 0/5 | — | 0/5 | 2.0 | 7.0 |
| A+heads (EM) | 5/5 | 3882–4934 | 5/5 | 16.0 | 4.4 |

In B the noun class and the name class merge by presentation 157–274 and the two positions read as one class from then on; in A and C they never do. The A+heads time is late because `vp.argin2` is shared by NP and pronoun heads in similar proportion and reads as one class only once pronouns have merged into NP.

## Q3 Ambiguity (corpus C)

| corpus | noun/verb separate | noun–verb distance | kept uses of run+walk | flips per kept use | final class (10 word-seeds) | credit on novel uses |
|---|---|---|---|---|---|---|
| C (EM) | 5/5 | 1.08 ± 0.14 | 375 ± 20 | 0.18 ± 0.02 | noun 8 / verb 2 | 0.15 ± 0.02 |
| C (ME) | 5/5 | 1.06 ± 0.16 | 381 ± 23 | 0.19 ± 0.04 | noun 8 / verb 2 | 0.14 ± 0.04 |
| A (EM) | 5/5 | 1.23 ± 0.01 | 0 | — | — | 0.20 ± 0.13 |

Noun and verb stay separate (distance 1.08 against 1.23 without ambiguity, max √2); run/walk keep flipping — 18% of their kept uses move the pointer — and end in whichever class their last kept use fitted (noun 8/10, since noun uses are the majority and the chooser keeps a verb use of a noun-class word only 15% of the time).

## Q4 Stability and reuse across document streams (EM)

| corpus | words with id change (4000–5000) | switches | stream 2: new classes | stream 2: ids changed | stream 3: new classes | stream 3: novel words reused | final classes |
|---|---|---|---|---|---|---|---|
| A | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 1.00 | 7.0 |
| B | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 1.00 | 6.0 |
| C | 0.06 ± 0.03 | 10.0 ± 5.7 | 0.0 | 0.03 ± 0.02 | 0.0 | 1.00 | 7.0 |
| A+heads | 0.19 ± 0.07 | 51.4 ± 9.9 | 2.2 ± 0.7 | 0.00 | 1.4 ± 0.8 | 1.00 | 4.4 |

Stream 2 = 1000 fresh sentences of the same grammar after N=5000; stream 3 = 1000 sentences with 11 unseen words (2 adjectives, 2 nouns, 2 names, 2 vi, 2 vt, 1 pronoun). No new class is grown for a known vocabulary, and every novel word joins the class its hidden class already holds. The only moving words in C are run/walk. Under heads the ids churn (growth and re-merge, see Q1).

## Q5 Soft participation vs hard gate (N=5000)

| corpus | chooser | classes | completeness | growths | switches | merges | credit rate | credit on novel uses |
|---|---|---|---|---|---|---|---|---|
| A | soft w=0 (no bias) | 7.0 | 0.83 | 9 | 41 ± 4 | 2 | 0.22 | 0.27 ± 0.09 |
| A | soft w=.25 | 7.0 | 0.83 | 9 | 39 ± 2 | 2 | 0.59 | 0.46 ± 0.32 |
| A | soft w=.5 (default) | 7.0 | 0.83 | 9 | 40 ± 3 | 2 | 0.79 | 0.20 ± 0.13 |
| A | soft w=1 | 7.4 ± 0.5 | 0.83 | 9 | 30 ± 9 | 1 | 0.67 | 0.00 |
| A | hard gate (dominant position) | 8.0 ± 1.3 | 0.79 ± 0.02 | 8 ± 1 | **0** | **0** | 0.49 | 0.00 |
| C | soft w=.5 (default) | 7.0 | 0.83 | 9 | 111 ± 14 | 2 | 0.75 | 0.15 ± 0.02 |
| C | soft w=1 | 7.2 ± 0.4 | 0.83 | 9 | 38 ± 7 | 2 | 0.68 | 0.01 |
| C | hard gate (dominant position) | 8.6 ± 0.5 | 0.81 ± 0.03 | 9 | **0** | **0** | 0.46 | 0.00 |

Under a hard gate no word is ever credited in a position its class does not hold: zero switches, zero merges, one frozen class per first position (gating on the class's support instead of its dominant position gives identical numbers, since a gated class never acquires a second position). The soft bias must be weak relative to the structural evidence: at `w=1` (penalty `log .02 ≈ -3.9` against a margin of 1) novel uses get credit 0–1% and the learner is effectively gated; at `w=.5` 15–20% of novel uses are kept and learning completes; with no bias at all the chooser picks the true parse only 22% of the time (the bias is what makes parsing work), so the weight is a real trade-off, not a free parameter.

## Q6 Sticky margin (N=5000)

| corpus | margin | classes | completeness | switches | switches 4000–5000 | words with id change | B-collapse | noun/verb separate |
|---|---|---|---|---|---|---|---|---|
| A | 0 / .2 / .4 | 7.0 / 7.0 / 7.0 | 0.83 | 40 / 40 / 22 | 0 / 0 / 0 | 0.00 | 0/5 | 5/5 |
| B | 0 / .2 / .4 | 6.0 / 6.0 / 6.0 | 1.00 | 56 / 58 / 39 | 0 / 0 / 0 | 0.00 | 5/5 | 5/5 |
| C | 0 / .2 / .4 | 7.0 / 7.0 / 7.0 | 0.83 | 111 / 111 / 91 | 10 / 10 / 10 | 0.06 | 0/5 | 5/5 |

The margin changes nothing but the number of early switches: a word's first kept use in a position held by another class always moves it (share gap ≈ 1, no margin below 1 stops it), so "a single odd use does not move a noun" is not delivered by the margin. What delivers it in practice is the chooser's credit: an odd use is kept only 15–20% of the time, so most odd uses are never observed.

## Parameter sweep: pull rate × merge distance × order (N=5000; cell = A classes / B collapsed / C noun-verb separate)

| order | rho | merge .2 | merge .35 | merge .5 |
|---|---|---|---|---|
| EM | 0 | 9.0 / 0/5 / 5/5 | 9.0 / 0/5 / 5/5 | 9.0 / 0/5 / 5/5 |
| EM | .02 | 9.0 / 0/5 / 5/5 | 9.0 / 0/5 / 5/5 | 7.0 / 5/5 / 5/5 |
| EM | .03 | 7.0 / 1/5 / 5/5 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 |
| EM | .05 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 |
| EM | .08 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 0/5 |
| EM | .1 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 2/5 | 7.0 / 5/5 / 0/5 |
| ME | 0 | 9.0 / 0/5 / 5/5 | 9.0 / 0/5 / 5/5 | 7.0 / 4/5 / 5/5 |
| ME | .02–.05 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 5/5 |
| ME | .08 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 3/5 | 7.0 / 5/5 / 0/5 |
| ME | .1 | 7.0 / 5/5 / 5/5 | 7.0 / 5/5 / 0/5 | 7.0 / 5/5 / 0/5 |

## What this means for plan §6

- **Supported.** With a pull rate in the right band the rule as written does everything §6.1 asks: pure classes (1.00) from the first few hundred presentations; growth makes one class per position and merging reduces them to one per hidden class; the NP collapse in B happens by ~200 presentations and does not happen in A; 10% ambiguous words do not merge noun and verb (the fraction decides, as in k-means); ids are stable (0 changes in A/B over the last 1000 presentations) and a second document and even unseen words reuse the existing classes (1.00). vi/vt do not merge under direct-operand observation; they do under heads, which is acceptable.
- **The pull is load-bearing and its band is narrow.** Under the plan's E-then-M order the class a word switches *to* takes the M step, so a class only ever sees its own position; two one-position classes that share all their words are held apart by their own M steps and only the pull brings them together. `rho = 0` leaves one class per position for the whole run (sweep: 9.0 classes in A, B never collapses). At `merge = .35` the rule works for `rho/eta` between 1.5 and 4 (below: no collapse; above: noun and verb merge in C). Letting the word's current class also take the M step (`ME`) moves the band to 1–3 (0.5 in a finer check not in the table) and converges faster; it is a one-line change worth considering.
- **The sticky margin does nothing** (Q6). Stickiness lives in the chooser's credit gate, not in the E step. If "a single odd use does not move a noun" matters, the E step needs either a stochastic switch (switch with probability < 1) or a count, neither of which §6.1 allows; the toy did not test them.
- **Growth must be far below a class's smallest legitimate typicality.** With `grow = .1` and `eta = .05` the merged NP class (four positions, one at ~.14) kept spawning spurious classes from EMA noise (27 switches per 1000 presentations, 3 new classes per document stream). `grow = .05 = eps0` with `eta = .02` removed it entirely. A "share below threshold" test (as the hand-off phrased it) is wrong on shared positions: two classes at .3 each give a best share of .46 and spurious growth; growth must read typicality (the plan's distance), not share.
- **Soft participation works only with a weak weight** (Q5): the log-share bias must stay below about half the structural margin (`w ≤ .5` here), or exploration dies and the result is the hard gate. If the weight is learned by credit it will climb (the bias raises credit on 80% of sentences), so the learned weight needs a cap or an exploration floor; the toy fixed it.
- **Head propagation is a risk for pronouns.** If participation were ever credited to phrase heads, nouns, names and pronouns share the NP positions and the pull merges pronouns into NP (5/5 seeds under the plan's order), which would defeat decision 4 (identity behaviour learned per class). The code's direct-operand observation keeps pronouns separate (5/5).
- **Not tested:** real parses (the identity corpus), learning the bias weight, the stochastic sticky E step, relative clauses and copulas, count-based (1/n) centroid rates.
- **Parameters that worked:** `eta = .02`, `rho = .05` (2.5·eta; band 1.5–4·eta for EM, 0.5–3·eta for ME), `grow = .05 = eps0`, `merge = .35` (.2–.35 safe; .5 merges noun/verb at `rho ≥ .1`), `margin = .2` (inert), chooser `w = .5` with `m = 1`, `tau = .5`.
