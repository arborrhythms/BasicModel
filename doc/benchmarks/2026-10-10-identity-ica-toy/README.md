# Identity by unmixing: does ICA-shaped teaching data teach identity? (toy, 2026-10-10)

This toy describes the pre-repair learner and its labelled variants. The
[part-1 repair](../2026-10-10-item6/part1/README.md) subsequently adopts residual
minting and greedy support selection. The frozen toy results below are not a
measurement of that production implementation; part 2 still builds the corpus
and its measurement before retiring identity rules.

**Setup.** `sim.py` (numpy; sklearn only for two references) mirrors `bin/IndependentComponents.SparseDictionary`. It uses unit columns with a relevance scale. Encoding is a one-shot top-k by |projection·relevance| at the declared ceiling k, least squares on the selected columns, then softshrink(λ=.05)·relevance. `observe` starts a pending prototype when the residual norm is above τ=.2 (matched at |cos| ≥ .8) and mints after 4 distinct witnesses; the minted direction is the normalized *observation*, as in the code. Training is one SGD step per new row on the code's loss (reconstruction + λ|codes| + λ·coherence + λ·mean relevance), over a 64-row sample of every row seen so far.

Variants that are *not* the code are labelled as such:
- `mint=residual`: mint the normalized residual instead of the observation.
- `greedy`: greedy pursuit instead of one-shot top-k.
- `k=4`: one slack source above the true support of 3.

The world: D=64 random codes (8 kinds, 8 properties, 6 verbs; Q3/Q4 have their own small worlds). A row is a sum of codes with magnitudes U[.7,1.3] plus noise of norm ≈.08.

Metrics:
- **Recovery**: the max |cos| from each true atom to any column.
- **Identification**: on 500 held-out rows, the share of present atoms carried by an *active* column with |cos| ≥ .9 to that atom. A column that memorizes the whole frame scores 0.

Every number is the mean ± sd over seeds 0–4. The full tables, including sparse codes and the k=3 residual variant, are in `results.md` / `results.json`. Rerun with `../../../.venv/bin/python sim.py` (about 4 minutes).

## Q1 Factorial vs confounded data (dense codes, 3,000 rows, rows = kind + property + verb)

Corpora: **A** is factorial. **B1**: o0 and p0 are always present together (their magnitudes still vary independently). **B2**: p0 occurs only with o0, and o0 also occurs with other properties.

| corpus | learner | recovery | columns | one-atom cols | identification | o0 / p0 | o0+p0 col | merged |
|---|---|---|---|---|---|---|---|---|
| A | code: mint=value, top-k, k=3 | .88 ± .01 | 456 | .02 | **.02 ± .01** | .83/.86 | .90 | 2/5 |
| A | mint=residual, greedy, k=3 | .98 ± .00 | 66 | .40 | .95 ± .03 | .98/.98 | .71 | 0/5 |
| A | mint=residual, top-k, k=4 | .98 ± .00 | 33 | .76 | .97 ± .04 | .98/.98 | .69 | 0/5 |
| A | ref: DictLearning (22 given, OMP 3) | .92 ± .02 | 22 | .87 | .84 ± .08 | .85/.93 | .71 | 0/5 |
| A | ref: FastICA (22 given) | .70 ± .09 | 22 | .36 | .31 ± .20 | .67/.76 | .57 | 0/5 |
| B1 | code | .86 ± .01 | 419 | .02 | .02 ± .01 | .77/.74 | .96 | 3/5 |
| B1 | mint=residual, greedy, k=3 | .96 ± .00 | 45 | .48 | .90 ± .03 | .75/.84 | .99 | 3/5 |
| B1 | mint=residual, top-k, k=4 | .96 ± .01 | 55 | .48 | .80 ± .13 | .84/.84 | .98 | 4/5 |
| B2 | code | .88 ± .01 | 430 | .03 | .01 ± .00 | .89/.81 | .95 | 3/5 |
| B2 | mint=residual, greedy, k=3 | .97 ± .00 | 59 | .41 | .92 ± .05 | .98/.70 | .99 | 0/5 |
| B2 | mint=residual, top-k, k=4 | .97 ± .01 | 29 | .75 | .96 ± .02 | .98/.77 | .98 | 0/5 |

"Merged" means an o0+p0 column at ≥ .9 and neither atom at ≥ .9 on its own. In B1 the best possible identification is ≈ .92, because a merged pair cannot count as two atoms.

- The code's mint form memorizes frames on factorial data: about 450 columns, of which 2% are single atoms. Recovery (.88) hides this; identification (.02) shows it.
- The cause is two compounding effects:
  - **Minting the whole observation** makes every recurring frame a column.
  - **One-shot top-k at the exact ceiling:** once any redundant column exists, it takes a slot, a present atom is left in the residual, and that leftover is minted again (a cascade).
- Minting the residual and selecting greedily (or with one slack source) gives 33–66 columns and identification .95–.97.
- B1: every learner, references included, builds an o0+p0 column (.93–.99) and mostly fails to recover either member (merged 3–5/5).
- B2: in the online learners the p0 column absorbs o0 (p0 .70–.90; o0+p0 column .95–.99).
- FastICA reaches .70 when each role holds exactly one source, and 1.00 when sources are present independently. Exactly-one-per-slot data breaks classical ICA's independence assumption; the sparse coder does not need it.

## Q2 Support curriculum (22 atoms, 3,000 rows; cell = identification on 3-source rows (columns))

| curriculum | learner | n=250 | n=1000 | n=3000 |
|---|---|---|---|---|
| staged 1→2→3 | code | .99 (22) | .98 (23) | .94 ± .11 (27) |
| staged 1→2→3 | mint=residual, greedy, k=3 | .99 (22) | 1.00 (22) | 1.00 (22) |
| staged 1→2→3 | mint=residual, top-k, k=4 | 1.00 (22) | 1.00 (22) | 1.00 (22) |
| mixed 1–3 | code | .62 (19) | .71 (49) | **.27 ± .05 (201)** |
| mixed 1–3 | mint=residual, greedy, k=3 | .81 (20) | 1.00 (24) | 1.00 (24) |
| mixed 1–3 | mint=residual, top-k, k=4 | .71 (20) | 1.00 (24) | 1.00 (24) |
| three only | code | .00 (0) | .70 (39) | **.07 ± .01 (328)** |
| three only | mint=residual, greedy, k=3 | .00 (0) | .11 (9) | .97 ± .01 (34) |
| three only | mint=residual, top-k, k=4 | .00 (0) | .05 (9) | 1.00 (26) |

Residual floor under the **true** atoms, by sources per row (1, 2, 3, 4, 5): .09, .11, .14, .22, .35. The share of rows above τ is 0, .001, .03, .14, .31.

Stage gap: the singletons stage, then "X P"/"X V" rows (verbs either absent from this stage or kept), then "the X V the Y" rows. Identification:

| corpus | code | mint=residual, greedy, k=3 | mint=residual, top-k, k=4 |
|---|---|---|---|
| verbs absent from stage 2 | .04 ± .01 (187 cols) | .99 ± .01 (32) | .80 ± .14 (54) |
| verbs kept in stage 2 | .44 ± .40 (26, 144, 90, 30, 132) | 1.00 ± .00 (26) | .96 ± .05 (32) |

Single sources first make the code's mint form learn the 22 atoms almost immediately. Without singletons, the code's form rises and then collapses as frames are minted. The variants recover from three-source rows alone, but need about 8× the rows.

The code's form stays fragile on three-source stages even with staging. Three-source rows sit 3% above τ under perfect atoms, so recurring frames get minted whole. A class left out of a stage also loses relevance: it decays at λ/n per step, and I measured it at about .85–.9 in the Q3 corpus.

## Q3 Pronoun binding credited by prediction ("the X V the Y . it P .")

The dictionary was learned by the code's form from a staged corpus (26 columns, .99 single atoms). Each kind has a characteristic predicate in 70% of its "X P" rows.

- **Content feature:** the cosine between the observed P and the follow-up predicted from the candidate's column co-occurrence. Its margin is .79 for a strong follow-up ("it purrs") and .26 for a weak one (a generic predicate plus a faint characteristic one).
- **Scorer:** softmax over [content, recency, subject]. Trained by REINFORCE with credit = that prediction quality; no binding labels.
- **Training data:** 30% of the training follow-ups are generic.

| credit | training | held-out CB (strong/weak) | reversed recency | reversed subject | control: picks recent | control: picks subject | w content | w recent | w subject |
|---|---|---|---|---|---|---|---|---|---|
| prediction | counterbalanced | 1.00 / .88 | 1.00 / .90 | 1.00 / .88 | .49 ± .14 | .44 ± .27 | 9.9 ± .1 | −.04 ± .73 | −.17 ± 1.36 |
| prediction | recency-biased | .79 / .58 | **.59 / .19** | .78 / .66 | .92 ± .14 | .40 ± .18 | 5.4 ± 1.5 | 3.30 ± 1.48 | −.72 ± 1.29 |
| prediction | subject-biased | .86 / .65 | .81 / .54 | **.59 / .18** | .62 ± .08 | .98 ± .02 | 4.6 ± .6 | .92 ± .55 | 3.23 ± 1.31 |
| prediction, lr .1 | counterbalanced | 1.00 / .93 | 1.00 / .93 | 1.00 / .96 | .55 ± .10 | .44 ± .08 | 6.1 ± .2 | .09 ± .23 | −.14 ± .24 |
| reconstruction, learned dict. | counterbalanced | 1.00 / .92 | 1.00 / .93 | 1.00 / .97 | .51 ± .13 | .41 ± .09 | 2.7 ± .4 | .00 ± .17 | −.11 ± .12 |
| reconstruction, true atoms | counterbalanced | .57 / .51 | .40 / .37 | .80 / .66 | .66 ± .42 | .39 ± .20 | .1 ± .1 | .13 ± .21 | −.05 ± .08 |

In this toy, the content feature and the prediction credit are the same quantity. So Q3 tests whether position features absorb credit, not whether content can be discovered.

- **Counterbalanced data** learns content binding (1.00 strong / .88 weak). The control is at chance only on average: per learner, the position weights wander (sd up to 1.36). Generic follow-ups give updates with zero mean but nonzero noise. A smaller step shrinks the wander but does not remove it.
- **Biased data** teaches the shortcut. It beats weak content (.19 / .18) and dominates the control (.92 / .98).
- **Reconstruction of the bound row "it P"** as credit works only because the learned predicate column absorbed part of its kind (Q1-B2). It favours the antecedent in 87% of items with the learned dictionary, and in 50% with the true atoms, where binding falls to chance.

## Q4 Same-kind individuals and determiners

Individuals are (kind, colour, size); each property is named with p = .5. In the training discourses, "a"/"the" mark first and repeat mention correctly 90% of the time. The chooser decides between binding to the STM candidate and minting a new individual.

- **Credit:** the error in predicting a third sentence, "it is <colour> <size>", from the chosen file. No identity labels.
- **Exclusivity:** property pairs that never share a noun phrase in a separate 2,000-phrase corpus (colour excludes colour, size excludes size). This uses atom identities.
- **Ceiling:** about .92 on ordinary items.

| chooser | ordinary | a dog . a dog (2) | a dog . the dog (1) | a black dog . the white dog (2) | a dog . the black dog (1) | a black dog . a black dog (2) | w the | w residual | w excl. |
|---|---|---|---|---|---|---|---|---|---|
| content only (bind iff residual ≤ τ) | .64 | **.00** | 1.00 | 1.00 | **.00** | .00 | – | – | – |
| credit, det + residual | .89 ± .03 | 1.00 | 1.00 | **.09 ± .18** | .91 ± .17 | 1.00 | 7.0 ± .7 | −3.0 ± .6 | – |
| labels (ref.), det + residual | .91 ± .01 | 1.00 | 1.00 | **.00** | 1.00 | 1.00 | 4.0 | −1.1 | – |
| credit, det + residual + exclusivity | .92 ± .01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.1 ± .5 | −2.0 ± .7 | −4.2 ± .5 |
| labels (ref.), + exclusivity | .92 ± .01 | 1.00 | 1.00 | .87 ± .12 | 1.00 | 1.00 | 3.8 | −.8 | −2.1 |

Content-only unmixing cannot separate "a dog . a dog" from "a dog . the dog". The dog column explains the second row in 100% of items, so no second column can be minted.

With two dogs in STM ("a black dog … a white dog …") and then "the … dog V .", the right dog is bound:
- 1.00 when the colour is named again;
- .50 when it is not;
- .48 when colours were never named.

The last two are a coin flip, which is correct.

## What this means for item 6

**Teaching-data rules the toy supports.**
1. **Present single sources first, then pairs, then triples.** With the code's present mint form this is the difference between 22 clean columns and memorizing frames (Q2: .99 vs .27 / .07).
2. **Keep every column class recurring in every stage.** A stage that drops a class costs .04 identification for the code's form (Q2 gap).
3. **Keep rows to ≤ 3 sources** at this noise level, or the mint threshold has to scale with support (the floor table).
4. **Use factorial co-occurrence for what should be separate columns.** An always-co-present pair becomes one column (B1), and a property exclusive to one kind is pulled into it (B2). This is also the lever for teaching a whole or an individual on purpose.
5. **For pronouns, counterbalance antecedent recency and role, and make the follow-up predict the antecedent's kind.** Biased data teaches a shortcut that overrides weak content (.19).
6. **Determiners at about 90% reliability, plus a later sentence that reveals a persistent property,** are enough to learn "a"/"the" by credit, with no identity labels.

**What the toy does not support, and its limits.**
- **Same content, same kind:** unmixing cannot individuate. The mint gate never fires, because the existing column explains the row. Individuation has to come from the determiner cue or from continuity (.where/.when), and continuity was not tested.
- **Residual is not contradiction.** With [determiner, residual], no amount of this data teaches the "the white dog" conflict override (.09 by credit, .00 by labels). Fixing it needed a contradiction feature (exclusivity), which the current mechanism does not have.
- **Prediction credit is silent when the follow-up is uninformative.** Counterbalancing removes the position bias only on average across learners. A default for ambiguous pronouns cannot be taught this way.
- **Credit must be prediction from the column, not reconstruction of the bound row.** Reconstruction is at chance with a factorial dictionary.
- **The largest single limit is the code's mint form, not the data:** minting the whole observation, plus one-shot top-k at an exact ceiling and an absolute τ.
  - It memorizes frames on factorial data (Q1 .02).
  - It degrades on three-source stages even with staged data (.94 ± .11; gap .44 ± .40).
  - Minting the residual with greedy selection removed every such failure here. That is a mechanism change, not a data rule.

**Hand-written identity rules.**
- **Declared determiner mint/bind modes and the a/the mode masks: replaceable** by a learned determiner weight (1.00 on determiner-only items, credit-trained). This holds only if the chooser also gets a contradiction/exclusivity feature; without one, a property conflict cannot override "the". "every" and other determiners were not tested.
- **Positional nearest-left bind: replaceable** where the follow-up predicts the antecedent (1.00 / .88).
- **Positional nearest-left bind for content-free follow-ups: not replaceable by prediction credit.** The data-taught version is a coin flip, or whatever bias the data carried. A positional default would have to be declared, or taught with deliberately biased data, and that then overrides weak content.
- **Two same-kind, same-content mentions:** nothing data-taught in the toy separates them except the determiner.

**Caveats.**
- The toy uses linear mixing, random codes, and given STM candidates.
- The scorer features were hand-chosen.
- Exclusivity was computed from atom identities.
- SGD step .5 was fixed before any results; lr .1 is shown for Q3.
- The DictLearning reference uses the declared ceiling (OMP, 3 nonzeros). With sklearn's defaults it scored .05.
