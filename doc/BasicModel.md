# Basic Model

![BasicModel reference architecture](diagrams/mm5m_architecture.svg)

***The Basic Model of Cognition is a neural network of four spaces: real space, perceptual space, conceptual space, and symbolic space.***

**Model Entities:**

| Entity | Meaning |
| --- | --- |
| $x$, $\widehat{x}$, $X$ | Actual and predicted input (experiences), input space |
| $y$, $\widehat{y}$, $Y$ | Actual and predicted output (actions), output space |
| $p$, $P$ | Percepts, perceptual space |
| $c$, $C$ | Concepts, conceptual space |
| $s$, $S$ | Symbols, symbolic space |
| $W_e$ | Experiencing, experiential weights |
| $W_a$ | Acting, action weights |
| $W_k$ | Knowing, knowledge weights |
| $W_t$ | Thinking, thought weights |

***Background***

Based on the Basic Model of Cognition at
[cognitivesciencesociety.org/framework-cognitive-science](https://cognitivesciencesociety.org/framework-cognitive-science/);
see also [The Whole Part](https://thewholepart.com).

## Inputs and Outputs

Inputs and outputs derive from a real space that is ineffable. The original
theory framed learning as tabula-rasa perturbation at temperature; the current
implementation expresses that as initialized weights plus ergodic `bias`/`var`
buffers driven by gradient energy. See [Ergodic.md](Ergodic.md).

Inputs scale to `[-1, 1]` via global data min/max. The signed range
represents presence and absence symmetrically. Paraphrasing Augustine,
*negative is the privation of the positive*.

Action is a function of perceptual space and our expectation, filtered by
conceptual projections. Performance errors are defined relative to action,
desired effect, and exhaustion. Expectation errors are defined relative to
perception and perception-as-reconstructed.

To accommodate a changing input context window, input is augmented with
relative spatiotemporal encoding by extending its dimensionality. The
attention mechanism treats what and where pathways separately.

## Perceptual Space

Perceptual spaces are composed of perceptual prototype vectors, or
***percepts***.[^1] Percepts that do not correspond to objects are
*hallucinations*.

Percepts form a Voronoi tessellation using prototype and/or exemplar vectors,
related via Self Organizing Map connections, operating in parallel as
prototype-theory categories.

Perceptual accuracy is the distance between inputs and prototype vectors ---
similar to cosine similarity, becoming a simple distance measure for poorly
known percepts.

Attraction and aversion warp perceptual space: attraction is a pole at the
percept's location, aversion is a zero. Perfectly recognized percepts lie on
the unit hypersphere within conceptual space. Two percepts can be joined into
one concept without loss.

## Conceptual Space

***Concepts*** create boundaries in conceptual space, grounded in the support
vectors of perception. Concepts support logical negation; they may refer to
negative entities. Conceptual spaces are capable of representing negative
entities. Concepts that do not correspond to percepts are *false*.

Concepts collect percepts into fuzzy sets --- linear separating hyperplanes
partitioning conceptual space, or sets defined as linear combinations of
percept activations.

Concepts are relative: meaningful only insofar as they partition a meaningful
space. As concepts approach certainty, decision boundary slope becomes
infinite. A single unknowing state is common to all knowledge vectors
(activation slope zero).

Conceptual spaces are similarity spaces partitioned by decision boundaries
imposing positivity and negativity on a hypersphere of knowing.

Within conceptual space, attraction/aversion to concepts expresses as
hyperplane bias. Certainty is the slope of the activation function (tuned
after other parameters stop changing).

A concept can be split into two percepts without loss.

### Order, extension, and intension (2026-07-02)

Concepts occupy a **ramsified order**. An **order-0 concept** is the output of
mixing a part and a whole at the corpus callosum — it is defined
**extensionally**, by its perceptual support (the percepts that fall under it),
and it is **subsymbolic**: a conceptual-space citizen with no relational
definition yet. It is grounded directly. A **higher-order concept** is defined
in terms of other symbols (its sparse constituent edges) and so acquires an
**intensional** definition (its *sense* — the relations that fix it), while
still retaining extensional grounding, inherited transitively through its
constituents down to order-0's support and refinable by direct experience.
Ascending the orders is therefore a **graded extensional $\to$ intensional
shift, never a hard switch**, and the grounding becomes increasingly *mediated*
(reached through the definitional graph rather than directly from percepts) —
which is why superordinate concepts feel more theoretical and less perceptual
(Rosch et al. 1976; the basic $\to$ superordinate trajectory). This is the
reconciliation of the prototype-vs-definition tension: a concept is *both* its
region and its definition, because intension and extension are carried by two
different structures over the same object. Concretely — **intension is the
sparse edge-graph; extension is the atom's position** (Frege's sense /
reference; Carnap's intension / extension, realized as graph-over-geometry).

Because every concept, at any order, has an extensional footprint (a grounded
atom), specific and general objects share **one conceptual space**: a single
dictionary holds each concept as a *point*, and the relational graph is
structure laid *over* that space, not a second space. The single-space claim is
not a new assumption — it follows from grounding (symbols are a subtype of
perceptual objects; see [Symbolic Space](#symbolic-space) below). One
refinement matters: a concept's home in the space is its **point/atom**, *not*
necessarily a convex region. Gärdenfors's convexity holds for *natural*
concepts and is the **order-0 / extensional limit**, cleanest at the grounded
base; **discontiguous, high-order** objects still live in the same space as
points, but their extension is **graph-defined** (the things reachable through
their definition), not a convex region — which is exactly why the
word$\leftrightarrow$object bridge needs a MetaSymbol ("no convex set is
specific enough," see [doc/Mereology.md](Mereology.md)). So: **one space of
points plus a relational graph, with convexity as the basal/extensional
reading.** The cognitive grounding for the dense-perceptual / sparse-symbolic
split is in [doc/Architecture.md](Architecture.md) "Cognitive grounding."

The 2026-07-02 two-phase rework makes the extensional/intensional seam
OPERATIONAL: the forward is a purely continuous PS/WS$\leftrightarrow$CS pump
(extension settling) followed by ONE late snap to the order-0 codebook block
-- quantization exactly at the bandwidth seam -- and then the symbolic
composition over the relational graph (intension). Order-0 concepts ARE
codebook rows (extension = the atom's position, no edges); order $\ge 1$
concepts are edge-defined (intension = the sparse graph). Three relational
primitives carry the graph (Alec 2026-07-02): the sec-4c ORDERED PAIR
$[whole, part]$ (whole $\Rightarrow$ part / if $\to$ then); the SINGLETON --
a whole containing exactly one symbolic part, the unit-set $\{x\}$ Lewis
resisted, here the constructive primitive; and the recursion VINE (ordered
chains from nested pairs -- order from nesting, since each row is set-like).
Where the relation does NOT need order (the word/object meta), the read-out
is typed intersection, not slot order. See doc/Architecture.md sec A and
doc/plans/2026-07-02-two-phase-loops-sparse-relation.md; the un-ramsified
successor (iteration over one untyped square layer, with Kripke groundedness
as the cycle diagnostic) is sketched in
doc/plans/2026-07-02-iterated-symbolic-loop.md.

## Symbolic Space

*Symbolic Spaces* are composed of percepts that reference concepts --- a
subtype of perceptual spaces mapping onto conceptual spaces.

Symbols have no spatial context: zero-dimensional points within perceptual
space. In humans, perceptual space projects onto a 3D cerebellar subspace
before symbolic encoding.

Symbols are independent, permanent, and integral (as references). The
top-down serial activation differs from bottom-up perceptual activation ---
it casts shadows.

The formation of symbols increases the dimensionality of conceptual space
(outer product); the formation of concepts from percepts decreases the
dimensionality of perceptual space (inner product).

Symbols are erroneous if their associated concept is not true. Participation
in multiple sets creates structural relations that replace positional
information.[^2]

## Reasoning

Reasoning in symbolic spaces involves computing with parts and wholes ---
Venn-diagram computations enabling dynamic Bayesian-type statistics. Learned
probabilities can be stored as unidirectional connection weights between
concepts, possibly with part-whole structure imposed by corresponding
symbols.

## Symmetric Perception: the Single Layer Linear Case

Perception is bidirectional: we see the world and the world sees us. The
egocentric/cultural view treats perception as passive; in neural networks
this manifests as forgetting the influence of outputs on inputs.

Symmetric Perception finds $f(x)$ minimizing MSE with respect to:

$$
\begin{pmatrix} in \\ out \end{pmatrix} \cdot \begin{pmatrix} f(x) & 0 \\ 0 & f^{-1}(x) \end{pmatrix} = \begin{pmatrix} out \\ in \end{pmatrix}
$$

In the linear case, $f(x)$ becomes a single invertible matrix $A$. When the
mapping is nonlinear and an inverse may not exist:

$$
\begin{pmatrix} in \\ out \end{pmatrix} \cdot \begin{pmatrix} f(x) & 0 \\ 0 & g(x) \end{pmatrix} = \begin{pmatrix} out \\ in \end{pmatrix}
$$

This resembles $Ax = b$, except $x$ is known and we want weights $A$ mapping
one space to another. The standard LLS solution $x = (A^\top A)^{-1} A^\top b$
becomes finding $A$ such that $Ax = b$ and $A^{-1} b = x$.

Retrocausality suggests an additional objective: $A$ should be optimal for
both forward and inverse transforms. Approximately: $A$ as a product of two
matrices, each performing a perfect surjection: $A \cdot A^{-1} = I$.

$$(\frac{b}{2} * x^{-1}) \cdot (\frac{x}{2} * b^{-1}) = I$$

Best $A$ requires iterative solution (NN learning, ping-pong style) or
Gram-Schmidt orthogonalization with respect to $A$ and $A^{-1}$.

## Symmetric Perception: the Single Layer Nonlinear Case

If $f(x)$ and $g(x)$ share weight matrix $W$:

Forward computation and gradient:

$$\widehat{y} = f(g^{-1}(x) W)$$
$$e_{y} = \frac{1}{2}(\widehat{y} - y)^{2}$$
$$\Delta W_{y} = \eta(\widehat{y} - y)\delta_{f}(W^\top \cdot g^{-1}(x)) g^{-1}(x)$$

Reverse computation and gradient:

$$\widehat{x} = g(W^\top f^{-1}(y))$$
$$e_{x} = \frac{1}{2}(\widehat{x} - x)^{2}$$
$$\Delta W_{x} = \eta(\widehat{x} - x)\delta_{g}(W \cdot f^{-1}(y)) f^{-1}(y)$$

Since $W$ is shared, summing the partial derivatives gives:

$$\frac{\partial e_{y}}{\partial W} + \frac{\partial e_{x}}{\partial W} = \eta(\widehat{y} - y)\delta_{f}(W^\top \cdot g^{-1}(x)) g^{-1}(x) + \eta(\widehat{x} - x)\delta_{g}(W \cdot f^{-1}(y)) f^{-1}(y)$$

Setting gradient to zero, both errors going to zero when $x = i$ and $y = o$.
But $xy - iy = ox - yx$ may hold --- simultaneous gradient descent
ping-pongs. A better cost function multiplies the two error partials ---
intersection of the spaces satisfying both:

$$\frac{\partial e_{y}}{\partial W}\frac{\partial e_{x}}{\partial W} = \eta(\widehat{x} - x)(\widehat{y} - y)\delta_{f}(W^\top \cdot g^{-1}(x))\delta_{g}(W \cdot f^{-1}(y)) f^{-1}(y) g^{-1}(x)$$

## Distance Metrics

$$d(x_{1},x_{2}) = \frac{\lVert x_{1} \rVert^{n} \cdot \lVert x_{2} \rVert^{n}}{\lVert x_{1} - x_{2} \rVert^{2n}}$$

Z-plane form:

$$d(x_{1},x_{2}) = \lVert x_{1} - x_{2} \rVert \cdot \frac{(x - z_{1})}{(x - z_{2})}$$

where $z_1$ is a zero and $z_2$ is a pole.

## Exploration/Exploitation as a function of temperature

$$y = x \cdot \frac{(W + rt)}{\lVert W \rVert}$$

where $r$ is a random vector and $t$ is temperature.

## Transfer Functions

Current invertible transfer functions use factorized linear maps rather than
the earlier SVD-rotation sketch. `InvertibleLinearLayer` parameterizes the map
with triangular factors and a diagonal scale so forward and reverse share the
same learned transform without forming a dense inverse on every call. `SigmaLayer`
and `PiLayer` wrap those maps for additive and multiplicative composition. See
[Architecture.md](Architecture.md) for the implementation contract.

[^1]: Perceptual space is known as "embedding space" in transformer literature.

[^2]: For example, {0,1} and {0,1}, if orthogonal, pose no constraint on the
location of the 1 in their 2x2 grid. However, {0,1,1} and {0,1} partition a
3x2 space and require two 1's on the same row --- a fact present in neither
set alone.
