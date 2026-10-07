# The soft attention filter: the fixed point before round 5 (Claude, 2026-10-07)

Gate: the XOR content pairs embedded among 2–4 filler words (8-word filler
vocabulary, random positions); 64 training sentences, 256 held out. A
per-word filter `m = σ(θ·code + b)` on detached codes; an unattended word drops
out of the conjunction by interpolating toward the product's identity (the
all-ones direction); an affine answer reader on the normalized root; budget
`λ Σ m`.

| variant | soft acc | hard m>½ | hard top-2 | top-2 = content pair | mean m content / fillers |
| --- | ---: | ---: | ---: | ---: | --- |
| no filter (all attended) | .859 | — | — | — | 1 / 1 |
| filter from m = ½, λ = .01 | .453 | — | — | — | .00 / .00 — collapses to attending nothing |
| filter from "attend all" (m ≈ .95), λ = .01 | 1.000 | 1.000 | 1.000 | .773 | .48 / .00 |
| filter from "attend all", λ = .001 | 1.000 | 1.000 | 1.000 | **1.000** | .76 / .13 |
| filter from "attend all", λ = .001, budget after 200 epochs | 1.000 | 1.000 | 1.000 | 1.000 | .72 / .13 |

The filter has a degenerate fixed point: started undecided with the budget
on, it switches everything off before the reader can read anything, and
nothing pushes back. Started from the hard mask's present behaviour (attend
everything) with a small budget, the answer keeps the content words and the
budget removes the fillers; the hard choice at test agrees with the soft one.
With identity by construction an unattended word costs perception nothing,
so the filter's signal is the answer (and expectation) through the root,
plus the budget.
