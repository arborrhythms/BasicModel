# Reference at the close of the sentence: a toy of membership by evidence at order 1 (2026-10-16)

**Setup.** `sim.py` (numpy, `../../../.venv/bin/python sim.py`, 53 s; full tables in `results.md`/`results.json`) runs the [identity corpus](../../../data/identity_from_data/README.md) through a label-free resolver that decides every NP mention **at the close of its sentence** among {individuals already in the document's STM, NEW}, as [plan §2](../../plans/2026-10-10-item-6-stored-idea-generativity.md) and [Orders.md §3](../../Orders.md) describe (an individual = a fuzzy set of occurrences; membership by evidence). The learner sees the text, the mention heads with their grammatical roles and the sentence boundaries (what a kept derivation supplies), the determiner forms `a`/`the`, and that `it` is a word with no content. Entity labels, candidate lists, targets and row-pair answers reach only the grader. Each content word has a random unit code in R^64; a mention's content is the sum of its content words' codes (a pronoun's is zero); an individual's signature is a saturating union of the atoms of its bound mentions (weight 1) and of each sentence's predicate head (`is white` → `white`, weight ρ_p = .5), with atom weights read back through the dictionary's pseudo-inverse (projection alone has ±.3 crosstalk at D = 64, which broke the reading in 2/5 seeds). Score(candidate) = w_c·explanation (mean over the mention's atoms of their weight in the signature, minus the weight of any atom the signature holds that *excludes* it; exclusivity is learned from the text as atoms that never share a mention — 12 pairs, all kind×kind and property×property) + w_p·prediction (log p(predicate | signature) − base rate, from a count table over (atom, predicate) that predicts the base rate for unseen pairs) + w_rec·(−recency rank) + w_lr·last role + w_cr·current role; score(NEW) = log α + cue[a/the/pronoun] + w_p·prediction(from the mention's own content). Hard argmax, ties to the most recent; a content-less mention cannot mint when candidates exist (no residual). Learning without labels: for every alternative the credit R = κ·(explanation − 1) [NEW: κ·(ν − 1)] + log p(this predicate | post-binding signature) + log of the CRP-mixture likelihood of the next sentence's mentions and predicate under the resulting STM; q = softmax(R); the policy weights take one step toward q (w += η Σ(q − π)φ) and the association counts the (atom, predicate) pairs of every alternative weighted by q³ (soft EM; hard assignment is equivalent in the final configuration, Q6). Baselines: **B1** the code's rule (`a` mints, `the`/pronoun must bind, cosine argmax of content vs candidate signatures, EMA signatures, recency tie-break); **B2** `a` mints, otherwise the most recent individual; **B3** the proposal deciding during compose (no prediction term in the score; same credit). Train = the 494-document `train` split, 2 epochs, ε = .05 exploration; eval on `eval` (150) and `reversed_eval` (16) with learning off; seeds 0–4, mean ± sd (no ± = identical in all seeds). Defaults: `rho_p .5, kappa 6, nu .5, lam_assoc 10, atom_floor .3, excl_min 8, eta_w .2, beta 1, lam_next 1, alpha_credit 1, beta_assoc 3, forget .9993, decay 1e-4`.

## Q1 Binding accuracy by family (eval; probe mention, or the final `it` for the determiner families and the two `the colour kind` mentions for same_kind)

| family | n | proposal | B1 code rule | B2 recency | B3 during compose |
|---|---|---|---|---|---|
| pronoun_one | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| pronoun (2 candidates) | 32 | 1.00 | 0.50 | 0.25 | 0.50 |
| pronoun_three | 54 | 1.00 | 0.33 | 0.11 | 0.33 |
| document_context | 8 | 1.00 | 0.50 | 0.50 | 0.50 |
| neither_candidate (control, chance .50) | 32 | 0.50 | 0.50 | 0.25 | 0.50 |
| determiner (mentions) | 16 | 1.00 | 1.00 | 1.00 | 0.95 ± 0.10 |
| determiner_conflict (mentions) | 8 | 1.00 | 0.50 | 0.50 | 0.90 ± 0.20 |
| same_kind (mentions) | 24 | 0.67 | 0.67 | 0.83 | 0.67 |
| all probes | 130 | 0.88 | 0.45 | 0.23 | 0.45 |
| recency rank 0 / 1 / 2 (informative probes) | 42/38/18 | 1.00 / 1.00 / 1.00 | 1.00 / 0.00 / 0.00 | 0.52 / 0.00 / 0.00 | 0.82 / 0.14 / 0.13 |
| role subject / object | 64/34 | 1.00 / 1.00 | 0.44 / 0.41 | 0.22 / 0.24 | 0.42 / 0.45 |

The neither-candidate control is at chance by construction: the policy is deterministic and the paired items carry both answers; it stays at .50 only because a pronoun cannot mint (with minting allowed the unseen `glittered` makes every candidate tie and NEW wins: .20 ± .10, Q6). Collisions are 0 and coverage 1.00 for the proposal, B1 and B3; B2 merges `the cat` into the most recent individual (52% of probes collide, hence its .23).

## Q2 Row pairs (one vs two individuals)

| family / expected rows | n | proposal | B1 | B2 | B3 | proposal ν = 1 (flat prior) | proposal, no exclusivity | proposal, 20% determiners wrong |
|---|---|---|---|---|---|---|---|---|
| determiner / 1 (`a black cat … the black cat`) | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 ± 0.20 | 1.00 | 1.00 |
| determiner / 2 (`a black cat … a white cat`) | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| determiner_conflict / 2 (`a black cat … the white cat`) | 4 | **1.00** | **0.00** | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| same_kind / 1 (`a cat … the cat`) | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| same_kind / 2 (`a cat … a cat`) | 4 | **0.00** | **1.00** | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| all | 20 | 0.80 | 0.80 | 0.80 | 0.80 | 0.98 ± 0.04 | 0.60 | 0.80 |

## Q3 Reversed recency (matched branches: warm-up on train without the two-candidate pronoun family, then 32 balanced presentations or 16 biased × 2 (× 8 for "x4"))

| model | reversed_eval | eval pronoun | learned recency weight |
|---|---|---|---|
| proposal, full train | 1.00 | 1.00 | −0.00 ± 0.01 |
| branch balanced / biased / biased x4 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −0.02 / −0.05 / −0.09 (± .02) |
| branch balanced (B3) / biased (B3) | 0.10 ± 0.20 / 0.00 | 0.50 / 0.50 | 0.16 ± 0.10 / **1.54 ± 0.08** |
| B1 / B2 / B3 (full train) | 0.00 / 0.00 / 0.20 ± 0.40 | 0.50 / 0.25 / 0.50 | — / — / 0.32 ± 0.20 |

The biased set does not install a recency shortcut in the proposal: prediction already explains every biased item, so q ≈ π and the recency weight gets no gradient (it drifts slightly negative). Without the prediction term (B3) the same 16 documents push the recency weight to 1.5 and reversed_eval to 0.

## Q4 Learned weights and the recurrence gate

| run | compat | pred | recency | last role | cur role | log α | cue a | cue the | cue pronoun |
|---|---|---|---|---|---|---|---|---|---|
| proposal (2 epochs) | 3.94 ± 0.06 | 1.00 | −0.00 | −0.09 | −0.36 ± 0.04 | 1.17 ± 0.05 | 1.25 ± 0.03 | −0.08 ± 0.06 | 0 (masked) |
| 1 epoch / 3 epochs | 3.46 / 4.18 | 1.03 / 1.01 | .01 / −.01 | −.13 / −.09 | −.28 / −.41 | 1.00 / 1.27 | 1.31 / 1.20 | −.31 / .07 | 0 |
| ν = 1 (flat prior) | 2.98 | 1.01 | 0.04 | −0.09 | −0.68 | 1.89 | 1.25 | 0.63 | 0 |
| pronoun may mint | 3.86 | 1.02 | −0.02 | −0.05 | −0.20 | 0.88 | 1.58 | 0.24 | −0.94 ± 0.02 |
| 20% determiners wrong | 4.42 | 1.01 | 0.01 | −0.14 | −0.29 | 1.26 | 0.85 ± 0.03 | 0.41 ± 0.04 | 0 |

Only compat, pred and the a-cue are identified; log α, the cues and the roles are collinear in this corpus (first mention ⇔ `a` ⇔ subject), so their individual values are not interpretable and the `the` cue never gets a gradient that favours binding (nothing a `the`-mention does later is predicted better by one individual than by two, Q2). The pronoun cue learns −0.94 when minting is allowed, not enough against log α for an unseen predicate.

| recurrence gate | individuals/doc | rows (durable)/doc | probes strict / lenient | row pairs strict / lenient |
|---|---|---|---|---|
| off (proposal) | 2.28 | 2.28 | 0.88 / 0.88 | 0.80 / 0.80 |
| r = 2 | 2.28 | 1.54 ± 0.04 | 0.26 ± 0.02 / 0.88 | 0.40 / 0.80 |
| r = 3 | 2.28 | 0.33 ± 0.04 | 0.00 / 0.88 | 0.40 / 0.80 |
| B1, r = 2 | 2.25 | 1.03 | 0.03 / 0.45 | 0.60 / 0.80 |

The gate as mirrored here (a NEW decision opens a pending individual that is a candidate like any other and becomes a row at its r-th mention) changes no decision; it only withholds rows. Strict grading (a pending individual names no row, as the code's un-admitted residual) loses every probe whose non-target was mentioned once (`a cat saw a dog .` — the dog never recurs) and every two-row pair; lenient grading is the gate-off result.

## Q5 Robustness

| condition | probes | pronoun | pronoun_three | document_context | row pairs |
|---|---|---|---|---|---|
| proposal, noise σ = 0 / .3 / .6 / 1.0 per atom (atom codes have norm 1) | .88 / .87 / .87 / .84 ± .02 | 1.00 / 1.00 / 1.00 / .94 | 1.00 / .98 / .98 / .96 | 1.00 / 1.00 / 1.00 / .95 ± .10 | .80 / .80 / .76 ± .08 / .81 |
| B1, any σ | .45 | .50 | .33 | .50 | .80 |
| proposal, 20% determiners wrong (train and eval) | .88 | 1.00 | 1.00 | 1.00 | .80 |
| B1, 20% determiners wrong | .36 ± .02 | .46 | .20 | .33 | .64 ± .07 |
| ν = 1 variant, σ = 1.0 / 20% wrong | .73 ± .04 / .88 | .78 / 1.00 | .87 / 1.00 | .93 / 1.00 | .66 ± .07 / .78 |

## Q6 Ablations (probes / pronoun / three / document_context / row pairs / reversed_eval)

| run | probes | pronoun | three | doc ctx | rows | reversed |
|---|---|---|---|---|---|---|
| proposal | 0.88 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |
| no prediction term (score and credit) | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 1.00 |
| no learned salience / no determiner cue / no exploration / hard association | 0.88 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |
| no next-sentence credit | 0.88 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |
| no predicate absorption (ρ_p = 0) | 0.85 | 1.00 | 1.00 | **0.50** | 0.80 | 1.00 |
| no exclusivity | 0.88 | 1.00 | 1.00 | 1.00 | **0.60** | 1.00 |
| pronoun may mint | 0.80 ± 0.02 | 1.00 | 1.00 | 0.90 ± 0.09 | 0.80 | 1.00 |
| ν = 1: default / no det cue / no next-sentence credit / 20% wrong | 0.88 | 1.00 | 1.00 | 1.00 | .98 / .76 / .80 / .78 | 1.00 |

## What this means for the proposal

- **Deciding at the close holds, and it is the whole effect.** Every informative probe family is 1.00 against B1's .45 and B3's .45 (chance per candidate count), uniformly over recency rank and role; the prediction term carries it (ablating it from score and credit gives exactly B3). Deciding during compose with the same learning leaves the pronoun at chance and, trained on the biased set, learns the positional shortcut (recency weight 1.5, reversed 0); at the close the biased set installs nothing (Q3), because prediction explains it first.
- **Membership by evidence beats the cosine argmax where content is partial**: `the white cat` after `a black cat` is a second individual (1.00 vs B1 0.00), by learned exclusivity (black/white never share a mention) under the default prior, or by the flat prior alone. Document context (`a cat is black … it is black`) needs the predicate's head absorbed into the signature (ρ_p = .5; 0 → .50) and the copula excluded (with `is` as a predicate word the association learned that black things are the subjects of `is`, and credited `it is white` to the black one in 3/5 seeds).
- **What fails.** `a cat runs . a cat sleeps .` is one individual under the default (same_kind/2 = 0.00; B1 gets it by the declared `a` = mint): identical content is 3 nats of evidence for binding (κ(1 − ν)), the one-step credit never sees the fourth sentence where the two cats diverge, and the `a` cue is learned only from the half-explained cases (`a white cat`), where it needs to be ~1.2, not ~4. With a flat prior (ν = 1: a fresh individual explains its own mention perfectly) the cue decides and same_kind/2 passes 5/5, but `the`-binding then rests on the verb-sequence association alone (one determiner/1 miss in 1/5 seeds; .76–.80 rows without the cue or the next-sentence term; probes .73 at σ = 1). The two determiner conventions need opposite priors and prediction does not arbitrate between them in this corpus. Row pairs are therefore .80 for the proposal and .80 for B1, with complementary failures.
- **The recurrence gate** is neutral on decisions and purely destructive on measurement (strict probes .26 at r = 2, .00 at r = 3; row pairs .40): in these lessons the non-target is a singleton. A recurrence count across documents, as in the code, would be counting kinds, not individuals (Orders.md §3).
- **Not identified**: the salience weights (recency, roles, log α, cues) are collinear here and contribute nothing measurable (ablation = proposal); the `the` cue has no label-free training signal. The pronoun cannot be allowed to mint without a residual rule (otherwise an unseen predicate mints a content-less individual). The next-sentence credit term is inert under the default (needed only under ν = 1 for `the`-binding).
- **Limits.** One predicate word per sentence (its head); exclusivity learned as never-co-occurring atoms is a corpus artefact for black/small (one property per mention); the soft-EM association is a choice, not a necessity here (hard assignment locked in a wrong kind→predicate in 1/5 seeds only before the predicate head and the pseudo-inverse read were in place); D = 64 random codes require the pseudo-inverse read; nothing here is ICA — the "dictionary" is the atom codes themselves.
- **Parameters that worked:** `rho_p .5` (.3 loses document context), `kappa 6`, `nu .5` (see above for 1), `lam_assoc 10`, `beta_assoc 3`, `eta_w .2`, 1–3 epochs identical; results are seed-stable (sd 0 on most cells).
