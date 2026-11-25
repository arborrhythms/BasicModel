# Regularizing the Logical Forest: Eliminating Extraneous Logic

The signal-router's chart commits to a Boolean derivation per input
row, and on the XOR fixture it learns to do so with 100% accuracy. But
the **forest** of derivations the chart selects across inputs is
strictly larger than necessary. With three grammar rules (NOT,
conjunction, disjunction) over four leaves, the canonical XOR DNF
`(X1 ∧ ¬X2) ∨ (¬X1 ∧ X2)` requires exactly 2 NOTs + 2 ANDs + 1 OR — five
operations, one parse tree, the same for every input. What we observe
instead is two distinct derivation classes: a **tight** sum-of-products
that *is* the canonical DNF, and a **loose** all-disjunction tautology
`X̃1 ∨ X̃2 ∨ X̃3 ∨ X̃4` that subsumes the DNF as a wider Boolean cover.
Both fit the four training rows because XOR's truth table happens to
sit inside the looser cover for the chosen NOT placements; the
downstream classification head absorbs the slack. The chart has the
expressive capacity to commit to the precise derivation — and on rows
1, 3 it does — but the optimizer has no incentive to converge there
universally because the loss signal saturates at zero on the training
set regardless. This is the textbook **loss-driven under-specification**
failure mode for symbolic / structural learning: outcome-based losses
don't pressure the model toward minimal or canonical structural
commitments, they just reward "any correct enough" derivation.

The task of regularizing the forest is therefore a search for the right
**inductive bias toward minimality**. The literature offers several
levers, all variants of "prefer the smallest description that fits":
the **Minimum Description Length** principle (Rissanen 1978; Voita &
Titov 2020 for an information-theoretic probe) charges the optimizer
for emitted route entropy, so two routes covering the same data lose
to one route; **batch-variance regularizers** that penalize per-row
variation in the routing emission collapse a section of constant
functions into a single constant function, surfacing whichever
derivation generalizes best; **temperature annealing** on the anchor
softmax sharpens late-training commitments so degenerate basins
(all-OR, all-COPY) get squeezed out as the gradient signal tightens;
**OOD validation** on Boolean inputs the model hasn't seen exposes the
gap between Class A (looser cover, fails to generalize) and Class B
(tight DNF, generalizes) and lets early stopping or model selection
pick the right one. The right combination depends on the goal: if the
chart is meant to **be** the explanation (xAI emphasis), MDL plus
batch-variance regularization is the cleanest path because it produces
a single committed derivation per grammar that *is* the canonical
function under the rules. If the chart is just a structural prior
under a richer downstream model, OOD validation alone is usually
sufficient. Either way, the work isn't building a new optimizer — it's
adding a few well-chosen scalar terms to the existing loss so the
single optimizer's gradient already in place can prefer the smaller
derivation over the larger one.
