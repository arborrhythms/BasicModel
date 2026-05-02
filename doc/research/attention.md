# Attention, Routing, and Scoring Mechanisms

Notes on how the SignalRouter folds its placement scorer into the same
gradient path as the rules it routes — and the wider research context
of "routing as a derived property of the data" vs. "routing as a
separate policy network."

The motivating observation: when routing decisions are produced by a
dedicated MLP that takes the slab as input and emits scalar logits, the
MLP's parameters constitute a **second optimizer riding the same graph**.
Its loss surface is decoupled from the actual computation it routes —
gradient flows back to it only through whatever transformation the
routing imposes on the slab — and it competes for capacity with the rest
of the network. Research has converged on several patterns for
collapsing the policy network into a derived quantity of the data
itself, so a single optimizer trains both.

---

## Approaches 1–5: detailed comparison

| # | Mechanism | Score is computed by | Added params | Forward cost (per rule, per position) | Backward cost | Gradient coupling | Discrete commitment | Best when … | Key reference |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **Anchor inner product** (Stern-style span scoring; lightweight Q-style attention) | `<rule_output(x), anchor_rule>` — one dot product against a per-rule learnable vector. Anchor lives next to the rule's own `nn.Parameter` set. | `D` per rule (one anchor vector). Negligible. | `O(D)` per (position, rule). One einsum across the batch. | Same path as the rule itself — gradient flows through the anchor and through the rule's output simultaneously. | Tight: same gradient path trains rule and score. | Cheap argmax + Viterbi; can pair with straight-through estimator. | The placement structure is essentially type-driven and doesn't need iterative negotiation. Default starting point. | Stern, Andreas, Klein. *A Minimal Span-Based Neural Constituency Parser*. ACL 2017. |
| **2** | **Routing-by-agreement** (capsule networks, modern Hopfield) | Iterative: every rule produces a candidate; routing weight at position `i` for rule `r` is the cosine / dot agreement between rule `r`'s vote and the slot's running consensus. Refined over `K` rounds (K=2–3 typical). | None on the routing side — the votes come from the rule's own params; the consensus state is a tensor, not a parameter. | `O(K · R · D)` per position; K times more forward passes than (1). | Backprop through the K-step iteration (or implicit-diff at the fixed point). | Routing is a fixed point of the rule outputs — fully derivative, no separate scorer at all. | Soft per round; converges to one-hot in practice as agreements compound. | Multiple rules need to **negotiate** for shared inputs (e.g. when ops can plausibly fire at the same site). | Sabour, Frosst, Hinton. *Dynamic Routing Between Capsules*. NeurIPS 2017. Ramsauer et al. *Hopfield Networks Is All You Need*. ICLR 2021. |
| **3** | **Energy-based scoring** | Score is a statistic of the rule's output: `‖rule(x)‖²`, `−E(x, route; θ)`, or any energy with shared `θ`. Argmin over routes is the Viterbi path. | None or one scalar per rule (energy temperature). | `O(D)` per (position, rule). | Backprop through the energy and through the rule. | Tight, but the energy must be carefully shaped or magnitude-only routing wins (well-known failure mode). | Hard via argmin; energy gradient flows back through structured-prediction loss. | A principled scalar fitness exists for each rule independently of position context (e.g., committee of frozen experts ranking themselves). | LeCun, Chopra, Hadsell, Ranzato, Huang. *A Tutorial on Energy-Based Learning*. 2006. Belanger & McCallum. *Structured Prediction Energy Networks*. ICML 2016. |
| **4** | **Self-attention as routing (Q/K integrated)** | Score = `<Q_r(x_i), K_r>` (or `<Q(x_i), K(x_j)>` for span-style routing). The Q-projection is the same `nn.Linear` whose output also feeds the next layer's data flow. | Q-projection per rule (or shared Q across rules with rule-specific K). Larger than (1) by a factor of `D × D`. | `O(D²)` per (position, rule) due to the projection. Standard transformer cost. | Standard backprop. Same path as the data flow. | Tightest possible: routing **is** the data flow, no decoupling. | Soft by default; sharpened with low-temperature softmax or top-k. | Routing should produce a contextualized representation that downstream layers consume — i.e., the routed output is itself the next layer's input. | Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015. |
| **5** | **Implicit / fixed-point routing** (Deep Equilibrium Models) | Slab and route are jointly the fixed point of `(slab*, route*) = f(slab*, route*; θ)`. Solve via root-finding; backprop via implicit differentiation. | None on the routing side — `θ` is the same set of params used in `f`. | One forward pass = one root-find (typically 5–20 inner iterations). Heavier than (1)–(4). | One backward pass = one implicit-Jacobian solve. `O(1)` memory regardless of inner iterations. | Tightest: routing is mathematically inseparable from the data. | Continuous fixed point; can be sharpened by penalizing entropy at the equilibrium. | Routing and data are deeply mutually dependent; the loss surface benefits from constant-memory backprop. | Bai, Kolter, Koltun. *Deep Equilibrium Models*. NeurIPS 2019. |

### Side-by-side at a glance

| Property | (1) Anchor IP | (2) Capsule | (3) Energy | (4) Self-attn | (5) DEQ |
|---|---|---|---|---|---|
| Extra parameters | `O(R · D)` | None | `O(R)` or none | `O(D²)` per layer | None |
| Iterations to score | 1 | K (2–4) | 1 | 1 | many (root-find) |
| Routing is a separate module? | No | No | No | No (it **is** the data flow) | No |
| Discrete commitment available? | Argmax over IPs | Yes via consensus | Argmin over energies | Top-k or sparse softmax | Sharpen at fixed point |
| Easy to combine with structured DP? | Yes | Yes | Yes | Yes | Awkward |
| Implementation complexity | Lowest | Medium | Low | Medium (but well-trodden) | Highest |
| Compute cost vs. dedicated scorer MLP | **Lower** (`O(D)` vs `O(D²)`) | Higher (K×) | Lower | Higher | Much higher |

### Recommendation

For the SignalRouter we adopt **(1) anchor inner product** as the
default. It is the smallest move, has the strongest empirical track
record on similar parsing problems (constituency parsers, dependency
parsers, grammar-induction work), and the implementation path is a
two-line change: replace each scorer MLP with one `nn.Parameter` per
rule, score via einsum. (2) is layered on top by iterating the same
anchors with consensus updates, paid for in extra forward passes if
routing decisions stay underconfident in practice.

---

## Standard connectionist vs. Hopfield vs. QKV attention

These are the three foundational ways a network can ask "given this
input, which stored thing is most relevant?" They share more structure
than is usually appreciated; in fact (per Ramsauer et al. 2021)
softmax-based attention is provably equivalent to one update step of a
modern continuous-state Hopfield network.

### Comparison table

| Property | Standard connectionist (e.g., MLP / feedforward classifier) | Hopfield network (classical, then "modern") | QKV attention (Vaswani-style) |
|---|---|---|---|
| **Storage of "memories"** | Implicit, distributed across all weights. No explicit slot per fact. | Explicit: each pattern is a row in a weight matrix `Ξ`. Modern variants: rows are stored in a key matrix that is itself learnable or directly populated with data. | Explicit: a key matrix `K = X W_K` and a value matrix `V = X W_V`, derived from the input set on every forward call. |
| **Lookup primitive** | Forward pass through a chain of nonlinearities. No notion of "addressing." | Energy minimization: `E(ξ) = -½ ξᵀ W ξ` (classical) or `E(ξ) = -lse(βξᵀΞᵀ) + ½ξᵀξ` (modern). The argmin is the retrieved memory. | Softmax over inner products: `attention(Q, K, V) = softmax(QKᵀ/√d) V`. |
| **Capacity** | Bounded by parameter count; degrades gracefully under interference. | Classical: `~0.14 N` patterns for `N` neurons. Modern (Ramsauer et al.): exponential in `D` — effectively unbounded for practical purposes. | Bounded only by sequence length and value dimension. New keys/values per forward call — the network is "amnesiac" by default but can be made associative via cross-attention to a learned codebook. |
| **Convergence** | One feedforward pass — no notion of dynamics. | Iterative: state evolves under `ξ_{t+1} = ξ_t - η ∇E(ξ_t)` until fixed point. Multiple steps. | One pass per attention layer; "dynamics" only via stacking layers (each layer is one Hopfield update step in the modern correspondence). |
| **Learned vs. data-derived storage** | Storage IS the weights. | Classical: weights = outer products of patterns (Hebb rule), no gradient learning. Modern: `Ξ` is either weights (learned via backprop) or stacked input features. | `K` and `V` are computed afresh from the input on every forward call via `W_K`, `W_V` projections. The "memory" is the current input; persistence comes from stacking layers or KV-caches. |
| **Routing emergence** | None — there is no routing. The MLP is the computation. | Routing **is** the energy gradient flow. The system "decides" which pattern to retrieve by following the energy. | Routing is the softmax weights. It is determined entirely by the Q/K geometry of the current input. |
| **Parameter sharing with downstream computation** | All params participate equally; no separate "routing" subset. | The pattern matrix `Ξ` is both the memory and the computation; lookup IS the operation. | Q, K, V projections produce both the routing weights AND the routed signal — the scorer and the computer share `W_Q`, `W_K`. |
| **Discrete commitment** | None. Output is fully distributed. | Softens to one-hot at low temperature; the modern Hopfield update is provably the same as one softmax-attention step. | Argmax / top-k / sparse softmax variants. |
| **Where it shines** | Universal-function approximation; no need for explicit memory addressing. | Recall under noise; structured retrieval; auto-associative memory. Modern variants underpin transformer-style routing theoretically. | Variable-length context, dynamic attention, contextualized encoding. The default of every contemporary sequence model. |
| **Failure mode when used as a router** | Has to be added as a separate network — exactly the "second optimizer" pathology that motivates this document. | Retrieval is parameter-free given the codebook, but choosing the codebook is its own problem (chicken-and-egg). | If used naively as a discrete router, the softmax stays soft and commitment doesn't sharpen — needs explicit annealing or top-k. |
| **Foundational reference** | Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*. Nature 1986. | Hopfield. *Neural networks and physical systems with emergent collective computational abilities*. PNAS 1982. Ramsauer et al. *Hopfield Networks Is All You Need*. ICLR 2021. | Bahdanau, Cho, Bengio. ICLR 2015. Vaswani et al. NeurIPS 2017. |

### What this means for our SignalRouter

Anchor-based scoring (approach 1 above) is exactly a **single-step
Hopfield retrieval** where the "memory" is the per-rule anchor and the
"query" is the rule's own output: `<rule(x), anchor_rule>`. By the
Ramsauer correspondence this is also one step of softmax attention
restricted to a single key per rule. So the choice between (1) and
explicit Hopfield/QKV is really a choice along a continuum:

- **Single-step Hopfield retrieval / anchor IP** = approach 1.
- **K-step Hopfield retrieval with rule-vs-rule consensus** = approach 2 (capsules).
- **K-step QKV with K, V re-projected from the input set** = approach 4 (self-attention as routing).
- **Energy minimization to a fixed point** = approach 5 (DEQ), which
  collapses to one of the above when the energy is restricted appropriately.

The unifying claim is that routing-as-a-derived-quantity sits on the
single optimizer's gradient path; routing-as-a-separate-MLP does not.
The choice between (1)–(5) is a tradeoff between expressivity, compute
cost, and how much iterative negotiation the routing needs to perform.

---

## References (PDFs in this folder)

- `2006-lecun-energy-based-learning-tutorial.pdf` — LeCun, Chopra, Hadsell, Ranzato, Huang. *A Tutorial on Energy-Based Learning*. 2006.
- `2017-vaswani-attention-is-all-you-need.pdf` — Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
- `2017-sabour-dynamic-routing-capsules.pdf` — Sabour, Frosst, Hinton. *Dynamic Routing Between Capsules*. NeurIPS 2017.

Other works cited (not stored locally):

- Stern, Andreas, Klein. *A Minimal Span-Based Neural Constituency Parser*. ACL 2017. arXiv:1705.03919.
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015. arXiv:1409.0473.
- Bai, Kolter, Koltun. *Deep Equilibrium Models*. NeurIPS 2019. arXiv:1909.01377.
- Belanger, McCallum. *Structured Prediction Energy Networks*. ICML 2016. arXiv:1511.06350.
- Ramsauer et al. *Hopfield Networks Is All You Need*. ICLR 2021. arXiv:2008.02217.
- Hopfield. *Neural networks and physical systems with emergent collective computational abilities*. PNAS 1982.
- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*. Nature 1986.
