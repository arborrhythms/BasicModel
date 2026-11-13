# Reasoning and thinking

A question, like any sentence, specifies a region in conceptual space: the
roles it fixes and its free variables. A free variable is an open referent or
relation. A declarative carries its source's evidence; a question carries none
and stands at `(0,0)`, which is ignorance, not an evidence slot, so an
episode's goal is to bind the free variables and find or derive evidence.
Truth comes two ways. Direct truth is a content match: `query` returns the best
row in the region with its own pair and trust, and the match binds the free
variable. Derived truth applies a premise to a rule (an implication, an
equality or a parthood), and the conclusion's pair is the minimum along the
path in each lane. The two lanes are never netted, so a region reached both
ways keeps both and prompts refinement. Rows live in LTM0, the occurrence
store, and LTM1, its relation stratum.

Every completed field is offered one root thought episode at its close
(`ThoughtClosing.close`). A field with a free variable is always thought about
while work remains; for a field with no open slot, not thinking is the
chooser's free first choice, so whether to think about a closed sentence is
learned from the parse's form
([QueryContracts](QueryContracts.md#phase-work-and-lifecycle)). Nested `ask`
shares the root's `attentionBudget`.

Openness is read from the row's shape as well as its annotation
(`ThoughtReferences.open_slots`): a relational meaning with an unfilled operand
role is open without an `_open_references` entry, and a pending row returned by
`query` keeps its open reference. A meaning with an unfilled role never reaches
the answer readers as a bound view. An elided participant also opens a role,
detected from the verb's history rather than from a word class: when a sentence
closes with a predicate whose subject or object role is empty, and MTM's event
histories show that predicate with a participant in that role at least the
verbs' recurrence threshold of times (four), the journal opens a referent there
(`ReferenceClose.expected_roles`). Its expected content is the coordinatewise
maximum of the descriptions of the participants that filled the role before
(`expected_content`). The open role then enters thought like a question's;
`bind` may fill it from the situation or leave it open.

While the episode's root has an open slot, the thought menu offers only `ask`,
`query`, `intuit`, `infer`, `not`, `isTrue`, `bind` and `exist`, and an `equal`
whose two operands a query has already supplied (`ThoughtStream.candidates`).
Every path to an answer is therefore a sequence of sound steps, and the chooser
decides which premises and rules to use. `conclude` requires every free
variable bound, unless work is spent, `thoughtSearchExhaustion` completed
queries have found no candidate, or the menu holds only the self-inverse `not`.
At an exhausted search, a source-supported forward name is provisionally
minted; a question's free variable remains open on its stored row. A later
copula can bind directly to that row's addressed column and fill it.

| Symbolic face | Conceptual face |
| --- | --- |
| `isTrue(P)`: ended row's pair, trust and witnesses | `exist(P)`: conceptual presence and its pair |
| `isPart(a,b)`: LTM and taxonomy parthood | `part(a,b)`: containment content and pair |
| `isEqual(a,b)`: LTM1 equality relations (membership in a singleton) and references | `equal(a,b)`: identity of codes and pair |
| `infer(premise, rule)`: the rule chunk's consequent from the premise chunk by identity, membership, equality or instantiation, with the path-minimum pair, or its antecedent opened as a subgoal (2026-10-29) | `implies(P,Q)`: containment of regions and pair |

`bind` (a selected reference site, evidence `reference`) and `inferBackward`
(`infer`'s converse spelling, the subgoal face) complete the catalogue with
`intuit`, the order-zero proposal described under
[Intuition, understanding and thought](#intuition-understanding-and-thought-alec-2026-10-30):
`intuit` proposes and `infer` selects.

`infer(premise, rule)` reads two held chunks: a premise and a recalled rule (an
implication, a part or kind row, or an equality). Forward, the premise matches
the rule's antecedent by identity, by membership through a bounded taxonomy
proof, by equality through an `isEqual` proof, or by instantiation of one role
of a premise with several; the match kind is recorded as the record's `reason`.
The instantiated consequent takes the pair
$(\min(p^+, m^+, r^+), \min(p^+, m^+, r^-))$, where $p$ is the premise's pair,
$r$ the rule's lanes and $m^+$ the positive support of the membership or
equality proof ($m^+ = 1$ for a direct identity), so a weak edge caps both
lanes. If a chunk with the same references is already held, its pair is raised
by the per-lane maximum and no chunk is added. Backward, the goal matches the
consequent and the antecedent is written as a question chunk with an open
referent, pair $(0, 0)$ and evidence kind `subgoal`; `query` or `ask` fills it,
and a forward `infer` over the filled chunk concludes with its pair. A question
claims nothing. `ThoughtInference.replay_pair` recomputes the pair from the
recorded supports.

No operation duplicates what calling an existing one gives. `not` applied to
the premise before `infer` gives the contrapositive (the rule is read from
consequent to antecedent and the lanes exchange), so there is no `refute`; an
equality row used as the rule substitutes one term for the other, so there is
no `substitute`; and an `isImplied` would be `infer` forward with the rule
fixed, so there is none. Transitivity over part or kind rows is two forward
steps. A rule over relations matches by kind and by its two references'
identity or membership; there is no further unification.

Time adds no relation kind: a relation's readings come only from the types of
its two referents (conditional, attribution, definition, subsumption, equality,
part; `LTM1.readings`). Before and after are read from the events themselves,
an identity's timeline or the rows' stamps, and cause is a conditional between
two situations whose stamps run from antecedent to consequent:
`relation_evidence` skips an implication whose antecedent was stamped after its
consequent. Tense adds no time either. `.when` is model time, the moment a
sentence was read; a phase question is answered by content entailment over the
event's recorded case, never by a stamp (`TimelineRead.case_coverage`), and a
fragment of the requested content proves neither completion nor its negation.
An event whose phase no marker selected is unknown about its phase, never read
as present.

**Two routes to a truth.** Whether a property pervades an object needs no
operator of its own. A part or property question (`part`, or the taxonomy's
`isPart`) can be answered implicitly in order-zero space or explicitly by the
chain, and both may run within the budget. The implicit route is pervasion
(`Intuition.pervasion`, which `intuit` tries before its rules): the knowing is
narrowed to the subject's parts, read from its Part edges and recorded
constituents and kept only where the subject's content contains the part's,
each weighted by its edge's evidence and joined with the wholes that contain the
subject and with recalled rules fired at unit exponents, so learned proposal
strength never becomes truth. For each part the property's coverage is read in
each lane, and the answer's pair is the minimum over the parts in each lane
(`Intuition.pervade`): pervasion is complete only when every part carries the
property, and a mixed object (furry outside, not inside) keeps both lanes
rather than averaging them. The result carries evidence kind `pervasion`. The
explicit route is the chain: `query` recalls rows from LTM1 and the taxonomy,
`infer` follows them, and the pair is the minimum along the path. At the close
`Intuition.judge` joins an explicit proof and a pervasion proof of the same
proposition by the per-lane maximum: agreement fills one lane, and disagreement
fills both, the corner that prompts refinement.

**What a deduction may take (Alec, 2026-11-13).** Both routes are pervasion,
which "is just coverage from a higher order": the subject's region lies wholly
within the property's by a relation between the two kinds. A remembered
statement that is only like the claim gives an instance, and "analogy is not
deductively true": it is reported with an answer and is no premise. A
conclusion is trusted at what its premises give, the least on each lane, since
"object concept truth is established by reasoning from existing trusted
premises (in LTM or conceptual space)"
([Logic](Logic.md#existence-and-pervasion-alec-2026-11-13),
[Philosophy](Philosophy.md#the-three-sources-of-trust-in-the-model-alec-2026-11-13)).

`query(pattern)` returns the best matching row. `ask(row)` attempts to fill
an open reference, using queries and nested questions. `not` exchanges poles
of a pair or meaning; the compose operator `non` withholds both (Alec,
2026-11-12: "non- removes both positive and negative evidence, while not swaps
the two types of evidence."). Gain changes the next sentence's global
expectation. No thought operation returns a scalar. Neither trust nor lack of
support is silently substituted for an evidence pole.

Every result becomes a serial slot with its content, pair, witnesses and
producing operation. Chains bind those slots: the `a < b`, `b < c` fixture
performs two queries and concludes `a < c` with both source references.
Modus ponens requires an antecedent witness as well as the implication.
Conclusions are LTM `inference` rows, addressed by document/turn, ordinal and
content. No second semantic store or planner stack is introduced.

Before a field with no open slot is offered its episode, the close may conceive
one absence: when the closing expectation image's omission of a predicted
symbol exceeds `expectationOmissionThreshold` (0.5), it writes `not X`, an
absence inference with that confidence against (`ThoughtClosing.absence`).

The grammar's chooser makes the choices. Thinking is part of each whole path
([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)):
a departure at a thought decision is credited by the proposal-corrected score
function $p(a)\,\Delta C$ on its complete path's change in reconstruction plus
supplied-answer cost (R + A); exact ties move no policy weight. Work bounds the
walk but is not a cost in the comparison. The presented answer is the filled
row; there is no separate comparison reader.

Boundary admission, typed capability views, work limits and the retired
operator names are documented in [QueryContracts](QueryContracts.md). Replay,
nesting and occurrence retention are in [ThoughtHistory](ThoughtHistory.md).
Numerical proof utilities in `TruthGroundedReasoner` remain diagnostics;
production reasoning uses the ordinary grammar controller and native row
evidence.

## Retention and reasoning

Forgetting does not reason. It values each row by its own worth (the larger
evidence pole as trust, the recorded surprise, dissonance or coverage, and
age), and no pass checks whether a row is deducible from others (Alec,
2026-11-01: "Forgetting had been checking if rows were easily deductible; let's
delete that requirement."). No pass changes stored evidence. Live thought
results and pending expectation sources protect their addressed evidence
throughout a pass. The pass itself is described in [STM](STM.md).

## Intuition, understanding and thought (Alec, 2026-10-30)

In the dual-process terms of 2026-11-09, expectation and `intuit` are System 1,
and `infer`'s selection in the serial workspace, with `query`'s recall, is
System 2
([Architecture](Architecture.md#dual-process-theory-as-a-design-consideration)).
A `query` by time alone reinstates the order-zero concepts kept for that moment
as the attended field, so thought can reason over a past sentence as an
experience lived before
([Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11)).

Alec: "Intuition is beyond understanding, it is conditioned by desired
output, and thoughts are a distillation of intuition."

- *Understanding* is the conceptual definitions, trained by
  reconstruction: they explain the input.
- *Expectation* is bottom-up. `ActivationPredictor` predicts the next
  concept state from the current field and STM, and that prediction is
  also the spread of activation that primes attention, the chooser and
  retrieval. It learns only from its own error against the detached next
  state.
- *Intuition* is top-down. `intuit()` is a Sigma over Pi with negation at
  the literals and its own learned weights. It reads the knowing but is
  trained only by its use in output, so it is biased toward doing
  something with what is understood (problem solving). The relations and
  ideas that fired are pulled into an STM slot as candidates.
- *Thought* is `infer`'s selection from those candidates in the serial
  workspace (the eight STM chunks), its trust pair computed from evidence,
  never learned; an intuition is provisional until `infer` selects it.

In one line: understanding explains the input, intuition is understanding
put to use, and thought is intuition that has been checked.

*The calculus.* At thought time `intuit` reads the knowing $K$: the question's
order-zero content, the working values MTM prepends for the stream and, within
a third of the remaining work, the order-zero concepts active in the last
observed knowing field. Its rules are the LTM1 rows that reference the concepts
the predicted field holds active, taken in descending order of predicted
activation, with nonzero evidence and within the work meter
(`ThoughtLTMCapability.intuition_rules`). For a rule with lanes $(t^+, t^-)$,
antecedent literals $\ell_i$ (a negated literal exchanges its lanes before the
meet) and consequent $c$, the coverage is
$\kappa = \sum \max_i \min(K, \ell_i)^{\pi_i} / \sum \max_i \ell_i$, the firing
is $f = \min(\kappa, (t^+, t^-))$, and the term's field is
$\sigma(\max(\min(c, f^+), \min(\bar c, f^-)))$, where $\bar c$ exchanges the
lanes and $\sigma(x) = 1 - (1 - x)^{s}$. The slot's field is the maximum over
the fired terms. The exponents $\pi_i$ (one per rule literal) and $s$ (one per
rule) are output's parameters in two sparse banks of `intuitionCapacity`
addresses, initialized to one, where the map is exactly the max-min disjunctive
normal form; an overrun fails before a partial term is reserved. Each fired rule
becomes one `infer` candidate carrying its firing pair. Each consequent concept
receives the knowing deposit $\sum \min(\text{field}, \text{code}) / \sum \text{code}$
per lane, and `write_knowing` re-evaluates the selected proof live and joins it
into the field by maximum at the existing brackets. When `infer` selects a
firing, the pair is recomputed at unit exponents, $\min(\kappa_1, (t^+, t^-))$,
so learned weights shape what is proposed and never what is accepted. An
`intuit` cut off by the budget returns no content and leaves the goal and STM
unchanged.

`intuit` maps order zero to order zero. Its knowing is the order-zero field, a
rule's two sides enter as their order-zero content, and the consequent's
content is joined back into that field. Higher orders enter only through their
order-zero constituents: a higher-order concept as the join of its
constituents' content, an individual as its working value. So `intuit` cannot
tell two individuals apart except by their order-zero differences; individuation
is the serial grammar's work.

*Proposal and pick.* The breadth of the parallel pass and the narrowness of the
serial pick are two learners. Which fired candidate matters most for the focus
is not computed inside `intuit`: the grammar's operation scorer, the one compose
uses (`OperationSelectionLayer.thought_logits`), scores each `infer` candidate
from the focus and the candidate together, the candidate's firing pair among its
inputs, to one logit. The Sigma and Pi charts belong to combination, which must
stay monotone and bounded in the cube; a choice needs an unbounded logit, which
the softmax turns into the probability that the score function credits.

*What learns.* Memory is evidence, not a parameter. A row's lanes are what its
closes wrote: no answer loss, gradient or chooser credit changes a row or its
lanes, and a contrary statement observed later adds its own witness through an
ordinary close. An STM chunk has no parameters either; it holds references and
a detached working value, so what learns is what the references point at.
`intuit` improves as the things around it improve, each by its own owner:
memory (which rules exist and how true they are, written at kept closes),
meaning (the conceptual definitions and the taxonomy's memberships, which decide
how well an antecedent matches the knowing), place (which rules the predicted
field brings into reach), choice (the chooser, which learns when to call
`intuit`, which candidate `infer` takes, which premises to keep and when to
conclude: the path, never the truth of a step), and `intuit`'s own
output-owned weights. A wrong rule is not corrected in memory; it stops
mattering as the concepts it is stated over learn, as the chooser learns not to
take it, and as forgetting removes it when it goes unused.

*Where it fits.*

- *Bottom-up expectation as priming.* `ActivationPredictor`'s spread is
  spreading activation (Collins & Loftus 1975) read as prediction, as in
  predictive processing (Clark 2013; Friston 2010): priming as anticipation
  of the next state.
- *Intuition beyond understanding, shaped by goals.* Expert intuition is
  recognition of what to do (Simon 1992; Klein 1998, recognition-primed
  decisions),
  learned from feedback on outcomes rather than from explaining the input;
  Polanyi's tacit knowing ("we know more than we can tell") is the
  "beyond understanding" part.
- *Two kinds of fast thinking.* Glöckner & Witteman (2010) distinguish four
  kinds of intuition, among them an associative and a constructive,
  goal-driven one; that is the split between the predictor and `intuit`.
- *Thought distilled from intuition.* Poincaré and Hadamard describe
  discovery as unconscious combination offering candidates, a sense of fit
  picking one, and conscious work verifying it; global workspace theory
  (Baars, Dehaene) has the same shape, many parallel processes and a
  narrow serial workspace. The eight STM chunks and `infer`'s pick from
  the candidate menu are that workspace.

*Where it departs.*

- *Rationalization.* Intuition shaped by desired output is also how
  motivated reasoning works: in Haidt's social intuitionist model
  intuition decides and reasoning follows as justification. Here `intuit`
  proposes but an accepted answer's trust comes only from `infer`, recall
  or pervasion, computed from evidence. The design is
  normative rather than descriptive: thought distills intuition and then
  checks it, the step people tend to skip. Keep that.
- *Not every thought comes from intuition.* Reasoning also proceeds step
  by step from recall with no hunch first; the menu allows it through
  `query` and `infer` over recalled rules without `intuit`. "Thoughts
  distill intuition" is one route, not all of them.
- *Intuition is only as good as its feedback.* Intuition is reliable
  where feedback is accurate and prompt, a kind environment, and
  misleading where it is not (Hogarth 2001). Because only output trains
  `intuit`, its quality tracks the supplied answers': the supplied-answer
  lessons are its kind environment, and wrong or noisy answers would
  train confident bad hunches.

## The two networks of System 1 (Alec, 2026-11-12)

"So we have separate system1 inference networks, one that learns the input
based on missing information, and one that learns from conceptualizing or
thinking about the input."

Both take a conceptual state to a conceptual state without the grammar, and
both are of sigma and pi. They differ in their teacher: "The expectation()
and intuit() operations that operate from concepts to concepts are shaped by
expectation over the input itself and the system2 content, respectively."

### Symbols, the field and the one hypersphere

The two networks work over a vertical structure and a horizontal one,
which Alec set out as follows.

- *Vertical.* "A sigma-pi graph that collects symbols (percepts are not
  0-d, so cant be collected)." The parts and the wholes of perception
  "unite at the basic percept, and basic percepts are collected as parts
  only from then on as order increases". A symbol is a point of the graph:
  "the dimensionality of symbols should be zero (they are epistemic points
  which we do not feel, but which serve to represent collections)". It has
  no location, and it does not take the centroid of what it collects: "The
  higher-order percepts are currently centroids of lower order percepts,
  which is meaningless."
- *The field.* The field is the symbolic activation: what is present, and
  what the folds of the graph make of it, going up to types and to types
  of types, and coming down again. It holds every order, "just like we had
  multiple layers of parts and wholes", and no grammatical operation;
  [extend](Extend.md) is what applies a sentence's operators to it. It was
  described before as the order-zero concepts only.
- *Horizontal.* "Its just one hypersphere, but concepts are distributed
  according to ramsified algorithms (types live in token space, wholes
  live in part space)." A concept is placed by its context, and its
  context is the field: "the symbolic activation determines placement
  within conceptual space, but again, only relative to other concepts
  within the same order". A class is placed in this way like any other
  concept. A type acts on its tokens through the field, up and then down
  to the related tokens, and not by a pull across orders: "this is system2
  influencing system1".
- *What the context may not include.* Expectation weights the members it
  hands down by the likeness of their places (below). The context of
  placement is the activation the graph gives before that weighting, so
  that no place is moved by a value computed from places
  ([Architecture](Architecture.md), the guard against a feedback loop).

This differs from the placement that [Architecture](Architecture.md)
describes under "Placement" in three ways: the context is the field and
not the words observed alone; a place is moved relative to concepts of its
own order only; and a class is placed by context, not at the centroid of
its members. A symbol above the basic percept also carries no perceptual
form of its own.

### The split of sigma and pi (Alec, 2026-11-13)

"So we need to split the sigmaPi. based on the discussion, I see boxes
applying to percepts (since they live in a positive hypercube) and cones
applying to concepts (since they live in a signed hypersphere). That means
symbols are PI(concepts), and concepts are SIGMA(symbols). concepts are also
SIGMA(basic percepts)."

- **Sigma is a decision boundary and gives two values**, the two lanes of a
  concept. **Pi is a measure of likeness and gives one**, a presence: "sigmas
  are decision boundaries and yield two values, whereas pis are vector
  similarity measures and have only one result".
- **System 1.** Percepts become basic percepts by pi, whose region is a box,
  and basic percepts become the concepts of order zero by sigma. "basic
  percepts are part of system 1, which has order-zero concepts over those."
- **System 2.** The concepts of an order become symbols by pi, whose region
  is a cone at the origin, and those symbols become the concepts of the next
  order by sigma. "system 2 creates a symbolic vertical structure on top of
  that, which is used by all higher-order concepts."
- **One sigma for a concept, by its order**: from basic percepts at order
  zero and from symbols above.
- **A cone in place of a nearest neighbour.** "in places where we now ask
  "what is close to this vector", we are effectively asking about the
  hypercone centered on that vector"; a hierarchy of cones lets a read "look
  for part inclusion instead, which is a much faster operation". Inclusion
  is read from the rows a symbol and a concept are made of, and no geometry
  is added for it: "Member rows with no new geometry". That one kind lies
  within another is the pervasion of
  [Logic](Logic.md#existence-and-pervasion-alec-2026-11-13).
- **Each lane is a symbol.** A sigma gives two values and a symbol holds one,
  so a concept's two lanes are two symbols: "two symbols being required to
  store (without loss) the conceptual information of a single concept over a
  possibly heterogenous input".

### Implicit and explicit: what each system holds (Alec, 2026-11-13)

"I guess we still need both, as implicit and explicit: system 1 and system 2,
system 1 still knows in virtue of spatial reasoning, but system 2 has explicit
representation. The edges as promotion of LTM relations is good. But the
intuit() and expect() operations should not be explicit: THEY should be maps
from one space to the other, and we already have a description of how they are
taught."

- **Two ways of holding inclusion.** "there is a conflation between classical
  sets, which have discrete membership for every element, and a space that
  permits multiple hyperplanes. the interconnectivity of the first is N*M, and
  of the second there is no interconnectivity per say: there are two spaces
  and a mapping". System 2 holds the first, as edges; System 1 holds the
  second, as maps.
- **`expect` and `intuit` are maps.** They run between the perceptual and the
  conceptual space and follow no edge. For expectation "we only need a large
  sigma layer", over what the perceptual mereology supplies. A sigma over
  single elements cannot give an expectation that depends on a combination
  (below), so its inputs are conjunctions: the basic percept is one, the least
  of its part and its whole.
- **One traversal, forwards and back.** "Once basic percepts run as a pi over
  parts and wholes (nicely covering that area in mereological space),
  concepts over those basic percepts will form concepts =
  sigma(pi(percepts)." And "Expect and intuit can make the same traversal
  forwards and backward (they do not have recursion)", by "the inverse path
  because it is already present in the architecture"; "I think that intuit is
  the same operation." The maps are not the forward path's: "For expect, both
  sigma and pi matrices are matrices of their own." What is already there is
  the kind of operation: "the neural pathways are there, in the sense that we
  already have those operations between those spaces."
- **What teaches the maps.** Expectation by omission and intuition by what
  thinking left, as below. And the explicit side teaches the implicit one:
  "explicit membership can be used to teach implicit maps, since we have input
  (token) and output (type) provided as the teaching example for order."
- **The explicit side.** Edges are the promotion of relations of long-term
  memory. Thought reads them, as the next section describes.
- **Symbols have no space.** "symbols are not related as siblings, as they do
  not live in a space of their own; they exist as participants in the vertical
  structure. concepts live in a conceptual space with a distance metric, in
  which any pair has a distance, so that is where we get similarity." A sigma
  orders its members by its weights, and "sorting just happens by gradient
  descent like all weights".
- **The two lanes are projections.** "Evidence lanes for a concept are
  basically positive and negative projections onto it in space, and in the
  heterogenous case there is a lack of pervasion and a mixed sign projection
  onto the concept vector."
- **A likeness is not a pi.** "vector quantization as a similarity metric in
  conceptual space is not the inter-vector multiplicative nonlinearity I
  associate with the PI operator". A pi multiplies the activations of
  particular members, which an edge names.

A combination of two different words is not available to System 1 alone. The
perceptual towers stop at the word: every whole is pinned to a maximal run of
letters, and a run of parts is promoted only inside one word. A conjunction of
two words first exists on the conceptual side, in what the grammar composes,
in a candidate of MTM and in a row of memory. That is left to System 2.

### The explicit structure, read up and down

"My feeling is that order-0 expectation is a nonlinear operation over the
conceptual field, which is not possible using the hypersphere. But a sparse
sigmaPi with ten million concepts is still enormous parameterwise and
computationally, and my general feeling is that type relations are in fact
strict (felix is a cat) but neighbor relations are not (Felix is like Ciao).
So perhaps expectation is nonlinear in virtue of vertical propagation."

This is how the explicit structure is traversed. It is System 2's, and the
maps of System 1 are taught from it.

- *Containers.* What holds a concept can be gone up to: a concept whose
  definition has it as a member, a candidate of MTM whose support has it,
  and a row of long-term memory whose entry refers to it.
- *Up, by one graded match.* The degree to which what is active entails a
  container is the share of the container that is present, lane with lane,
  raised to the container's `pi`, a power that starts at one. A degree of
  one is entailment and a degree below one is expectation: "we can make it
  fuzzy and achieve both: entailment is not a binary".
- *Down, to what is missing.* A container hands its degree to its members
  that are not present; a member takes the greatest of what it is handed,
  raised to its `sigma`.
- *Likeness ranks.* Among the members handed down, each is weighted by the
  likeness of its place to the places of the present members. "Felix is
  like Ciao" is read among cats; no product is taken over all the concepts
  of order zero.
- *One pass.* Up once and down once.

What reads an expectation, attention's relevance, the compose chooser,
retrieval and the order of `intuit`'s rules, reads it detached, and the guard
of [Architecture](Architecture.md) stands: codes never read a value computed
from codes. What teaches it is omission ([Training](Training.md), "Adaptive
omission"): the cost is the difference between the expectation from the lossy
input and the field of the actual input, which is a teacher and never a
differentiable target.

Likeness alone cannot give an expectation that depends on a combination.
Take four sentences in which the word to expect depends on the pair before
it: A and B give X, C and D give X, A and D give Y, C and B give Y. With
likeness pooled by a sum, X must beat Y in the first two and lose in the
last two, and adding each pair of conditions makes one quantity both
greater and less than zero. Going up to each sentence as a container and
down to its missing member gives the right word in all four.

### Intuition, taught by thought

`intuit` is a map between the spaces and not an explicit rule. Its teacher is
what thinking left: the order-zero field of the kept path after its thought,
compared with what `intuit` gives from the observed field. The gradient of
the output cost stays beside it. What recurs in relation rows and in the
steps of thought and of [extend](Extend.md) is what it learns from. It is
evaluated once when a sentence is
observed, before any thought, so that what it has learned is known
directly: "insights that trigger knowing as a result of observation
directly".

### Where they fit

*The first network.*

- *Priming.* A pair of associated words is recognized faster than an
  unassociated pair (Meyer & Schvaneveldt 1971), which Collins & Loftus
  (1975) explain as activation spreading from a concept along the links of
  a network. An automatic component is fast and takes no capacity, and a
  strategic one is slower, limited and inhibiting (Posner & Snyder 1975;
  Neely 1977).
- *The route with the best support is vertical.* A category name speeds
  its members (Rosch 1975; Neely 1977), in part by expectancy (Neely,
  Keefe & Ross 1989). Among abstract words, associated pairs prime one
  another (Bleasdale 1987), and priming has been found with association
  and not with likeness of meaning alone (Ferré et al. 2015). Priming
  between words related in meaning and not associated is small (Lucas
  2000) and disputed for members of one category (Hutchison 2003). No
  study was found of one category-level word priming another.
- *Levels, and units of sigma and pi.* The interactive activation model
  has feature, letter and word levels with excitation and inhibition
  between adjacent levels and, inside a level, inhibition only (McClelland
  & Rumelhart 1981): a precedent for propagation between levels, not for a
  unit raising its neighbour. A sigma-pi unit takes a weighted sum of
  products of other units' outputs (Rumelhart, Hinton & McClelland 1986);
  Feldman & Ballard (1982) have a conjunctive connection of a nearby kind.
- *Taught by what is missing.* A speech sound replaced by a cough is heard
  as present (Warren 1970). The cloze procedure deletes words for readers
  to restore (Taylor 1953), and the brain's N400 response to a sentence's
  last word is larger the less that word is expected by its cloze
  probability (Kutas & Hillyard 1984). A network trained only to predict
  the next word of a small artificial language groups its words by
  grammatical class and by meaning (Elman 1990). In predictive coding the
  connections downward carry predictions and those upward the remaining
  error (Rao & Ballard 1999). Reconstructing an input from a corrupted
  copy (Vincent et al. 2008) and predicting hidden words from their
  context, "inspired by the Cloze task" (Devlin et al. 2019), are the same
  rule in machine learning.
- *Memory as a container.* Recalling a stored event from a small part of
  it begins with Marr (1971). A fast system that stores single events
  beside a slow one that finds the structure across many is McClelland,
  McNaughton & O'Reilly (1995): here the rows and candidates are the fast
  one and the promoted concepts the slow one.

*The second network.*

- *Deliberate work becomes direct.* Controlled search becomes automatic
  detection with consistent practice (Schneider & Shiffrin 1977; Shiffrin
  & Schneider 1977). Knowledge held as facts is compiled with practice
  into procedures that apply directly (Anderson 1982). Performance is
  automatic when it rests on retrieving a stored past solution in place of
  an algorithm (Logan 1988). In Soar the result of a solved subgoal is
  stored as a rule that later applies in its place (Laird, Rosenbloom &
  Newell 1986). People reuse earlier inferences to make later ones faster
  (Gershman & Goodman 2014), and a network trained on the results of its
  own tree search is framed by its authors as the two systems (Anthony,
  Tian & Barber 2017).
- *Intuition as recognition.* "Intuition is nothing more and nothing less
  than recognition" (Simon 1992). It develops only where the environment
  is regular and has been learned with prompt, clear feedback (Kahneman &
  Klein 2009).

*That there are two.* The fast side of dual-process theory is not one
system: Stanovich (2004) names it the autonomous set of systems, with
implicit learning, conditioning and rules practised until automatic all in
it, and Evans & Stanovich (2013) call "System 1" a misnomer that should be
plural. Learning from regularities needs no deliberation (Reber 1967;
Saffran, Aslin & Newport 1996); compilation needs the deliberate work it
compiles.

### Where they depart

- *Strict types.* In people the membership of a category is graded.
  Here membership is strict and what is graded is the match
  ([Philosophy](Philosophy.md), "Strict inheritance").
- *Least and greatest.* The literature's spread adds activations, with
  inhibition beside them. Ours takes the share present on two lanes and
  the greatest on the way down, which is where its nonlinearity lies.
- *Prediction is graded.* The pre-activation of a specific word's sound
  was not reproduced in nine laboratories, while the effect of a word's
  predictability at the word itself was (Nieuwland et al. 2018).
- *Two, by their teachers.* The literature supports a difference of
  teacher between the two networks, not a difference of mechanism, and
  that anything is partitioned into exactly two systems is itself disputed
  (Keren & Schul 2009; Melnikoff & Bargh 2018).
- *Symbols.* That the first network is taught while language is processed
  is this design's; in the literature its learning needs no symbols.

Neither the pass nor this placement is built yet; until they are, the
predictor described above and the placement of
[Architecture](Architecture.md) are what run.

**References.** Anderson (1982), *Acquisition of cognitive skill*,
Psychological Review 89(4). Anthony, Tian & Barber (2017), *Thinking fast
and slow with deep learning and tree search*, Advances in Neural Information
Processing Systems 30. Bleasdale (1987), *Concreteness-dependent associative
priming: separate lexical organization for concrete and abstract words*,
Journal of Experimental Psychology: Learning, Memory, and Cognition 13(4).
Clark (2013), *Whatever next? Predictive brains, situated agents, and the
future of cognitive science*, Behavioral and Brain Sciences 36(3). Collins &
Loftus (1975), *A spreading-activation theory of semantic processing*,
Psychological Review 82(6). Devlin, Chang, Lee & Toutanova (2019), *BERT:
pre-training of deep bidirectional transformers for language understanding*,
Proceedings of NAACL-HLT 2019. Elman (1990), *Finding structure in time*,
Cognitive Science 14(2). Evans & Stanovich (2013), *Dual-process theories of
higher cognition: advancing the debate*, Perspectives on Psychological
Science 8(3). Feldman & Ballard (1982), *Connectionist models and their
properties*, Cognitive Science 6(3). Ferré, Guasch, García-Chico &
Sánchez-Casas (2015), *Are there qualitative differences in the
representation of abstract and concrete words? Within-language and
cross-language evidence from the semantic priming paradigm*, Quarterly
Journal of Experimental Psychology 68(12). Friston (2010), *The free-energy
principle: a unified brain theory?*, Nature Reviews Neuroscience 11(2).
Gershman & Goodman (2014), *Amortized inference in probabilistic reasoning*,
Proceedings of the Annual Meeting of the Cognitive Science Society 36.
Glöckner & Witteman (2010), *Beyond dual-process models: a categorisation of
processes underlying intuitive judgement and decision making*, Thinking &
Reasoning 16(1). Hogarth (2001), *Educating Intuition*, University of Chicago
Press. Hutchison (2003), *Is semantic priming due to association strength or
feature overlap? A microanalytic review*, Psychonomic Bulletin & Review
10(4). Kahneman & Klein (2009), *Conditions for intuitive expertise: a
failure to disagree*, American Psychologist 64(6). Keren & Schul (2009), *Two
is not always better than one: a critical evaluation of two-system
theories*, Perspectives on Psychological Science 4(6). Klein (1998), *Sources
of Power: How People Make Decisions*, MIT Press. Kutas & Hillyard (1984),
*Brain potentials during reading reflect word expectancy and semantic
association*, Nature 307. Laird, Rosenbloom & Newell (1986), *Chunking in
Soar: the anatomy of a general learning mechanism*, Machine Learning 1(1).
Logan (1988), *Toward an instance theory of automatization*, Psychological
Review 95(4). Lucas (2000), *Semantic priming without association: a
meta-analytic review*, Psychonomic Bulletin & Review 7(4). Marr (1971),
*Simple memory: a theory for archicortex*, Philosophical Transactions of the
Royal Society of London B 262(841). McClelland, McNaughton & O'Reilly
(1995), *Why there are complementary learning systems in the hippocampus and
neocortex: insights from the successes and failures of connectionist models
of learning and memory*, Psychological Review 102(3). McClelland & Rumelhart
(1981), *An interactive activation model of context effects in letter
perception: Part 1. An account of basic findings*, Psychological Review
88(5). Melnikoff & Bargh (2018), *The mythical number two*, Trends in
Cognitive Sciences 22(4). Meyer & Schvaneveldt (1971), *Facilitation in
recognizing pairs of words: evidence of a dependence between retrieval
operations*, Journal of Experimental Psychology 90(2). Neely (1977),
*Semantic priming and retrieval from lexical memory: roles of inhibitionless
spreading activation and limited-capacity attention*, Journal of
Experimental Psychology: General 106(3). Neely, Keefe & Ross (1989),
*Semantic priming in the lexical decision task: roles of prospective
prime-generated expectancies and retrospective semantic matching*, Journal
of Experimental Psychology: Learning, Memory, and Cognition 15(6). Nieuwland
et al. (2018), *Large-scale replication study reveals a limit on
probabilistic prediction in language comprehension*, eLife 7. Posner &
Snyder (1975), *Attention and cognitive control*, in Solso (ed.), Information
Processing and Cognition: The Loyola Symposium, Erlbaum. Rao & Ballard
(1999), *Predictive coding in the visual cortex: a functional interpretation
of some extra-classical receptive-field effects*, Nature Neuroscience 2(1).
Reber (1967), *Implicit learning of artificial grammars*, Journal of Verbal
Learning and Verbal Behavior 6(6). Rosch (1975), *Cognitive representations
of semantic categories*, Journal of Experimental Psychology: General 104(3).
Rumelhart, Hinton & McClelland (1986), *A general framework for parallel
distributed processing*, in Rumelhart, McClelland & the PDP Research Group,
Parallel Distributed Processing, vol. 1, MIT Press. Saffran, Aslin & Newport
(1996), *Statistical learning by 8-month-old infants*, Science 274(5294).
Schneider & Shiffrin (1977), *Controlled and automatic human information
processing: I. Detection, search, and attention*, Psychological Review
84(1). Shiffrin & Schneider (1977), *Controlled and automatic human
information processing: II. Perceptual learning, automatic attending, and a
general theory*, Psychological Review 84(2). Simon (1992), *What is an
"explanation" of behavior?*, Psychological Science 3(3). Stanovich (2004),
*The Robot's Rebellion: Finding Meaning in the Age of Darwin*, University of
Chicago Press. Taylor (1953), *"Cloze procedure": a new tool for measuring
readability*, Journalism Quarterly 30(4). Vincent, Larochelle, Bengio &
Manzagol (2008), *Extracting and composing robust features with denoising
autoencoders*, Proceedings of the 25th International Conference on Machine
Learning. Warren (1970), *Perceptual restoration of missing speech sounds*,
Science 167(3917).
