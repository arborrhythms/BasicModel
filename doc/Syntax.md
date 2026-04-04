# Syntax

This note summarizes the syntax system implemented in:

- `basicmodel/bin/Model.py`
- `basicmodel/bin/Space.py`
- `basicmodel/bin/BasicModel.py`
- `basicmodel/data/MentalModel.xml`

It also corrects one common confusion in the code comments and discussion:
the main model pipeline is not itself a shift/reduce parser. The shift/reduce
logic is an optional syntactic layer that sits on top of percept, concept, and
symbol states.

## Hard-Coded Grammar Catalog

`Grammar` in `basicmodel/bin/Model.py` defines a fixed catalog of rule IDs:

| ID | Rule | Role |
|---|---|---|
| 0 | `S` | start symbol |
| 1 | `S -> S EQUALS S` | symbolic binary |
| 2 | `S -> S AND S` | symbolic binary |
| 3 | `S -> S OR S` | symbolic binary |
| 4 | `S -> NOT S` | symbolic unary |
| 5 | `S -> NON S` | symbolic unary |
| 6 | `S -> C` | symbolic-to-conceptual transition |
| 7 | `S -> C VERB C` | sentence-level verb composition |
| 8 | `S -> C VERB` | sentence-level intransitive verb |
| 9 | `C -> C PART C` | conceptual binary |
| 10 | `C -> C UNION C` | conceptual binary |
| 11 | `C -> C INTERSECTION C` | conceptual binary |
| 12 | `C -> P` | conceptual-to-perceptual transition |
| 13 | `P -> W` | perceptual terminal |
| 14 | `S -> S REWRITE S` | symbolic rewrite / where swap |
| 15 | `C -> C REWRITE C` | conceptual rewrite / where swap |

Two rule IDs are treated specially as cross-space transitions:

- `6`: `S -> C`
- `12`: `C -> P`

`Grammar.configure()` loads an XML `<grammar>` section and restricts the active
rule set by left-hand side:

- `<S>...</S>` entries become the active symbolic rule set
- `<C>...</C>` entries become the active conceptual rule set
- `<P>...</P>` entries become the active perceptual rule set

If no XML grammar were present, the code would fall back to broader defaults:

- symbolic: `1, 2, 3, 4, 5, 14`
- conceptual: `7, 8, 9, 10, 11, 15`
- perceptual: `13`

## Grammar Active Under `MentalModel.xml`

`basicmodel/data/MentalModel.xml` enables syntax and defines this grammar:

```xml
<grammar>
  <S>S equals S</S>
  <S>not S</S>
  <S>C verb C</S>
  <S>C</S>
  <C>C part C</C>
  <C>P</C>
  <P>W</P>
</grammar>
```

That maps to the following rule IDs:

- symbolic `<S>` rules: `1`, `4`, `7`, `6`
- conceptual `<C>` rules: `9`, `12`
- perceptual `<P>` rules: `13`

So the active syntax under `MentalModel.xml` is:

- symbolic: `EQUALS`, `NOT`, `C VERB C`, and transition `S -> C`
- conceptual: `PART` and transition `C -> P`
- perceptual: terminal `P -> W`

Important consequence:

- `MentalModel.xml` does not activate `AND`, `OR`, `NON`, `UNION`,
  `INTERSECTION`, or `REWRITE`.
- `C VERB C` is active because it appears under `<S>`, so the symbolic
  syntactic layer may predict rule `7`.
- Explicit verb semantics are implemented in `ConceptualSpace.projectConcepts()`,
  not in `SyntacticSpace.projectSymbols()`.
- Since `MentalModel.xml` does not activate rule `7` or `8` under `<C>`,
  conceptual verb composition is not currently exercised by the conceptual
  syntax stage.

`MentalModel.xml` also enables syntax at the architecture level:

- `<type>mental</type>`
- `<syntax>true</syntax>`
- `<conceptualOrder>1</conceptualOrder>`
- `<symbolicOrder>1</symbolicOrder>`

`<shiftReduce>true</shiftReduce>` is inherited from `basicmodel/data/model.xml`.

## Pipeline Versus Syntax

The main model pipeline is a chain of geometric projections between spaces.
These are not shift/reduce operations.

### `BasicModel`

Forward:

`InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace`

Reverse:

`OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace`

### `MentalModel`

The iterative `MentalModel` loop is:

1. `InputSpace -> PerceptualSpace`
2. Repeat `conceptualOrder` times:
   - concatenate `[percepts, previous_symbols]`
   - `ConceptualSpace.forward(...)`
   - optional `SyntacticSpace.forward(symbols)`
   - `SymbolicSpace.forward(concepts)`
3. `OutputSpace.forward(final_concepts)`

Reverse is the corresponding output-to-input unwinding:

1. `OutputSpace.reverse(...)` gives concepts
2. unwind the iterative concept/symbol loop
3. `PerceptualSpace.reverse(...)`
4. `InputSpace.reverse(...)`

The key distinction is:

- `PerceptualSpace.forward()` accumulates input into percepts
- `ConceptualSpace.forward()` accumulates percepts, and in `MentalModel`
  also previous symbols, into concepts
- `SymbolicSpace.forward()` projects concept activation into symbol activation
- syntax is a separate derivation mechanism layered on top of those states

So the statement "chunking happens when the input stream is accumulated into
percepts, and when concepts are accumulated into symbols, and both are
shift/reduce operations from perceptual space" is not correct.

More accurate wording:

- input-to-percept, percept-to-concept, and concept-to-symbol are ordinary
  space transforms
- syntax-level chunking happens in the `writePercepts`, `writeConcepts`, and
  `writeSymbols` methods
- reverse understanding is likewise not all "in perceptual space":
  `SymbolicSpace.reverse()` maps symbols to concepts, `ConceptualSpace.reverse()`
  maps concepts to percepts, and `PerceptualSpace.reverse()` maps percepts to input

## `SyntacticLayer`

`SyntacticLayer` in `basicmodel/bin/Model.py` is the shared rule predictor.
It does not itself execute semantics.

It learns:

- `input_proj`
- `derivation_layer`
- `rule_head`
- `depth_embed`

Its job is:

1. take an activation vector
2. predict rule logits over a space-specific active rule set
3. produce `rule_probs`
4. generate `(batch, vector, rule)` tuples called "words"

During training it uses Gumbel-softmax. During evaluation it uses softmax and
argmax-style decoding.

The actual rule semantics are owned by the spaces:

- `PerceptualSpace.projectPercepts()`
- `ConceptualSpace.projectConcepts()`
- `SyntacticSpace.projectSymbols()`

## `write*` and `read*` Methods

There are no singular methods named `writePercept` or `readPercept`. The code
uses plural names:

- `writePercepts` / `readPercepts`
- `writeConcepts` / `readConcepts`
- `writeSymbols` / `readSymbols`

### `writePercepts` / `readPercepts`

These live in `PerceptualSpace`.

`writePercepts(activation, vectors)`:

- shifts one percept activation/vector
- calls the perceptual `syntactic_layer` to generate percept-level word tuples
- applies the terminal rule `P -> W`
- uses the reverse `PiLayer` path to recover an input-level word embedding

This is a degenerate terminal case rather than a rich reduce stack.

`readPercepts(words, batch_size)`:

- rebuilds a percept activation mask from word tuples
- does not reconstruct percept vectors directly

### `writeConcepts` / `readConcepts`

These live in `ConceptualSpace`.

`writeConcepts(activation, vectors)`:

- shifts one concept activation and one concept vector set onto a stack
- predicts a rule on the stack head
- computes a soft reduce using `projectConcepts()`
- records word tuples
- marks whether a transition rule `C -> P` fired

Conceptual reductions operate on vectors, not scalar activations.

`readConcepts(words, batch_size)`:

- rebuilds concept activation from word tuples
- does not reconstruct concept vectors directly
- leaves vector reconstruction to `ConceptualSpace.reverse()`

### `writeSymbols` / `readSymbols`

These live in `SyntacticSpace`, not `SymbolicSpace`.

That distinction matters:

- `SymbolicSpace` is the concept-to-symbol projection layer
- `SyntacticSpace` is the symbol-grammar layer

`writeSymbols(symbol_act, where=None)`:

- shifts one symbol activation onto the stack
- predicts a symbolic rule
- computes a soft reduce using `projectSymbols()`
- optionally carries and rewrites `where` encodings
- records word tuples
- marks whether a transition rule `S -> C` fired

`readSymbols(words, batch_size)`:

- reconstructs symbol activation from word tuples
- delegates to `SyntacticLayer.reverse()`

### `composeSyntax` Versus `write*`

Each syntax-capable space has two styles of syntax execution:

- `composeSyntax(...)`: batch/tree-style soft composition
- `write*` methods: incremental shift/reduce

When `<shiftReduce>true</shiftReduce>`, the incremental `write*` path is used.

## Rule Semantics by Space

### Perceptual

`PerceptualSpace.projectPercepts()` only implements the terminal rule:

- `P -> W`: reverse the percept toward input-word space using the reverse `PiLayer`

### Conceptual

`ConceptualSpace.projectConcepts()` implements:

- `VERB`: via `VerbLayer`
- `PART`: `torch.min(left, right)`
- `UNION`: `torch.max(left, right)`
- `INTERSECTION`: `torch.min(left, right)`
- `REWRITE`: pass through "what"; `where` is handled separately
- `C -> P`: transition pass-through

Under `MentalModel.xml`, only `PART` and `C -> P` are active in conceptual syntax.

### Symbolic

`SyntacticSpace.projectSymbols()` implements:

- `EQUALS`: association layer on `left`, then multiply by `right`
- `AND`: elementwise `min`
- `OR`: elementwise `max`
- `NOT`: negation
- `NON`: attenuated positive presence
- `REWRITE`: pass through "what"; `where` is handled separately
- `S -> C`: transition pass-through

Rule `7` (`S -> C VERB C`) is not given special symbolic semantics here.
If predicted by `SyntacticSpace`, it falls through the default pass-through path.

## How Syntax Is Trained

### The Intended Differentiable Mechanism

The syntax code is written to be differentiable.

In particular:

- `SyntacticLayer.forward()` produces soft `rule_probs`
- `writeConcepts()` and `writeSymbols()` use soft reduce/shift interpolation
- `composeSyntax()` also uses soft superposition over all candidate rules

This means a downstream loss can, in principle, backpropagate into:

- rule probabilities
- rule-prediction weights
- depth embeddings
- rule-execution layers such as `equals_layer` and `VerbLayer`

### The Actual Losses in `runBatch()`

`BasicModel.runBatch()` computes only three losses:

1. output loss
2. optional reconstruction loss
3. optional embedding loss

There is no explicit syntax loss, parse-tree loss, or rule-label supervision.

The combined loss is:

```text
total = (1 - reverse_scale) * lossOut + reverse_scale * lossIn + embedding_scale * sbow
```

### What This Means for the Current Code

Under the stock `MentalModel.xml` configuration:

- syntax is enabled
- syntax modules are instantiated
- syntax rules are predicted
- syntax states and word tuples are produced

But there is an important limitation:

- `MentalModel.forward()` calls `self.syntacticSpace.forward(symbols)` and stores
  the result in `self.syntax_states`
- the returned syntax state is not then fed into concept formation or output
- `MentalModel.reverse()` calls `self.syntacticSpace.reverse(...)`, but the result
  is not used to update the reconstruction path

So in the stock `MentalModel` training loop, syntax is active as computation and
analysis, but it is not strongly coupled to the task loss.

The same caveat applies to `BasicModel.SyntacticDerivation()`:

- it predicts rules and collects parse words
- but its composed syntax outputs are not part of the default output or
  reconstruction loss path

One exception exists in `BasicModel` when `symbolicOrder > 1`:

- `SymbolicThought()` runs an additional `SyntacticSpace3 -> SymbolicSpace3` cycle
- that cycle is concatenated into the symbol tensor sent to `OutputSpace`
- in that configuration, the extra symbolic syntax path can receive indirect
  task and reconstruction gradient

That exception is not active in `MentalModel.xml`, which uses:

- `<symbolicOrder>1</symbolicOrder>`

## Bottom Line

The syntax system is currently best understood as:

- a real, parameterized, differentiable grammar module
- configured by XML rule subsets
- implemented with shift/reduce helpers and batch composition helpers
- partially integrated with the model pipeline
- not yet trained with explicit syntax supervision
- not yet placed on a strong loss path in the stock `MentalModel.xml` loop

So the code already contains the machinery for learned syntax, but the stock
configuration still treats syntax more as structured analysis/scaffolding than
as a fully trained parser.
