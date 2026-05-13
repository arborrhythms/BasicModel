
## Immediate

### 1. Fix broken chart rule dispatch
`SymbolicSpace._forward_with_rule_dispatch()` (`bin/Spaces.py:~9395`) calls
`layer.project(layer.grammar, rule_id, left, right=right, subspace=...)` on
`WordSpace.syntacticLayer`. `SyntacticLayer` has no `project` method — the
call raises `AttributeError` and silently falls back to returning the left
operand unchanged. Chart-parsed rule firing is a no-op. Fix or replace the
dispatch so rules actually execute their semantics.

### 2. Implement subsymbolic / symbolic split in chart rule dispatch
Once dispatch is working, route operations by tier:
- **Subsymbolic** (SigmaLayer / PiLayer): `lift`, `lower`, `union`, `intersection`
- **Symbolic** (SyntacticLayer `_RULE_METHODS`): all other rules

Currently all rules route through the unified `SyntacticLayer.host_layer`
registry with no split. The architecture in `doc/Language.md:263–303`
describes the intended routing; the code does not implement it.

### 3. Fix resolve() — one line
`bin/Spaces.py`, `SymbolicSpace.resolve()`, the `setW` call:
```python
# wrong
subspace.activation.setW(pos + neg)
# correct — signed Degree of Truth
subspace.activation.setW(pos - neg)
```
See `doc/plans/2026-05-04-resolve-luminosity-handoff.md` for full context.

### 4. Replace TruthLayer.luminosity()
Replace the existing positive-pole conjunction norm with the area-overlap
formula: `luminosity = area − overlapArea × |t_A − t_B|` ∈ [−1, 1].
Full spec and replacement code in `doc/plans/2026-05-04-resolve-luminosity-handoff.md`.

### 5. Conceptual introspection — area(), luminosity(), directPartOf()
Add introspective operations as grammar rules available inside the
conceptual order loop. At each loop level, a learned `question_head`
projects the current activation to a query; the introspective function
answers it; the result is injected as a sidecar channel into the next
level's input. Expose `directPartOf()` only — not generic `part()`. The
transitive closure of parthood emerges from iteration across conceptual
orders, matching the cognitive science reaction-time data on hierarchical
mereological processing.
External non-differentiable lookups use the STE wrapper:
`a_external + (q − q.detach())`.
Full design in `doc/plans/2026-05-04-conceptual-introspection-handoff.md`.

### 6. MM_xor.xml — fix XOR non-convergence
Set `<ramsified>true</ramsified>` in `data/MM_xor.xml`. The current
`false` causes `MentalModel` to run `ConceptualSyntacticLayer` which
collapses both words to position 0, making "hello world" and "hello there"
identical (cosine sim = 1.0). Setting `ramsified=true` uses the butterfly
path (`_CSLevelView`) which preserves both words.

---

* Enforce use of Mereonomy (per-symbol DoT graph).

* Ensure correct sentence-prediction

* Remove all random seeds from the tests, they are a crutch that defends against learning robustness

Sentences are sometimes composites of ideas.
* For example, questions relate two ideas:
  * Is subject predicate ?
  * part( x, y ) ?
* The IS of definition: equals ( x,y )
  * Store explicit parthood on WordSpace's Mereonomy when encountering a definiton (please rename from MereologicalTree to mereonomy) 
  * This should only happen when some measure of sentence confidence is high.

Memory of previous sentences requires prediction relating one to the next
* Store the sentences explicitly 
* predict from one sentence to the next

* Process truth statements
  * truth statements should become conceptual bivectors (ideas) and/or meronymic relationships
  * score the user's query in terms of the change in luminousity() vs the Truth

* Currently nWhere on percepts is unnecessary, because percepts are dense on input space (the network wiring densely covers the input).
  If perception is guided by attention, it can roam on input space, in which case the .where is particularly useful.

================================== April 24 ==================================


### Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

### Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

### Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

### Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

### Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

