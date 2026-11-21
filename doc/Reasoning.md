# Reasoning and thinking

A question, like any sentence, specifies a region in conceptual space: the
roles it fixes and its free variables. A free variable is an open referent or
relation. A declarative carries its source's evidence; a question carries none
and stands at `(0,0)`, which is ignorance, not an evidence slot, so an
episode's goal is to bind the free variables and find or derive evidence.
Truth comes two ways. Direct truth is a content match: `query` returns the best
row in the region with its own trust, and the match binds the free
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
`query`, `intuit`, `infer`, `not`, `isTrue`, `bind` and `exist`
(`ThoughtStream.candidates`).
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
| `isWhole(a,b)`: `isPart(b,a)` | `whole(a,b)`: `part(b,a)` |
| `isEqual(a,b)`: LTM1 equality relations (membership in a singleton) and references | none: an identity is `part` each way, which `define(a,b)` writes as two relations |
| `infer(premise, rule)`: the rule chunk's consequent from the premise chunk by identity, membership, equality or instantiation, with the path-minimum pair, or its antecedent opened as a subgoal (2026-10-29) | `implies(P,Q)`: containment of regions and pair |

`bind` (a selected reference site, evidence `reference`), `inferBackward`
(`infer`'s converse spelling, the subgoal face) and `non` (both lanes
withheld) complete the catalogue with `intuit`, which writes the intuited
field of its own map into knowing and carries it up the orders
([Intuition, understanding and thought](#intuition-understanding-and-thought-alec-2026-10-30)),
with the evidence kind `intuition`.

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
recorded supports. A stored implication is applied the same way: its firing
over the knowing ([below](#intuition-understanding-and-thought-alec-2026-10-30))
is one firing of a rule, which `infer` takes (`ThoughtInference.derive`) with
the evidence kind `inference`.

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
(`Intuition.pervasion`, tried before stored rules are fired): the knowing is
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

A question with no open operand is read on its region, its field of
understanding over the rows of memory (`ExtendedRecall.matches`), in two
ways kept apart. Existence is the brightest row in the region: its relevance
times the size of its trust, on the lane of the trust's sign, never summed
with another row. Pervasion is a trusted relation in LTM1 between the
subject's kind and the property's class, the least along the path
(`ExtendedRecall.pervasion`, through `Taxonomy.capture_taxonomy`), never
gathered from the kind's known members. A deduction takes the claim's
pervasion, or the claim's own statement, a trusted row of its content at a
relevance of one, with the other form's existence against it; existence at a
lower relevance is carried with the answer as its analogy.

`query(pattern)` returns the best matching row, recalled by the pattern's
extension ([Extend](Extend.md#what-the-field-of-understanding-is-for)).
`ask(row)` attempts to fill an open reference, using queries and nested
questions. `not` exchanges poles of a pair or meaning; `non`, in compose and in
thought, withholds both (Alec, 2026-11-12: "non- removes both positive and
negative evidence, while not swaps the two types of evidence."). The strength
of expectation is a setting of the model, not a thought operation (Alec,
2026-11-13: "drop gain, we can set that as expectationGain on the model"). No
thought operation returns a scalar. Neither trust nor lack of support is
silently substituted for an evidence pole.

Every result becomes a serial slot with its content, pair, witnesses and
producing operation. Chains bind those slots: the `a < b`, `b < c` fixture
performs two queries and concludes `a < c` with both source references.
Modus ponens requires an antecedent witness as well as the implication.
A kept conclusion is an LTM0 row named by its time, at the trust its premises
give, the least on each lane; a conclusion with evidence on both lanes is
written as two rows, the one held true first. A row has no kind and no origin,
and no trust is changed after it is written. No second semantic store or
planner stack is introduced.

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

Forgetting does not reason. It drops the rows of least activity, a number
each row of either store carries: set at its write from the sentence's
surprise, raised each time the row is read, decayed by each pass, and never
lowered or dropped at one. No pass checks whether a row is deducible from
others (Alec, 2026-11-01: "Forgetting had been checking if rows were easily
deductible; let's delete that requirement."). No pass changes a stored trust.
Live thought results and pending expectation sources protect the rows they
address throughout a pass. The pass itself is described in [STM](STM.md).

## Intuition, understanding and thought (Alec, 2026-10-30)

In the dual-process terms of 2026-11-09, expectation and intuition's map are System 1,
and `infer`'s selection in the serial workspace, with `query`'s recall, is
System 2
([Architecture](Architecture.md#dual-process-theory-as-a-design-consideration)).
A `query` matches the top of the STM stack against the rows' slots and pushes
the best row onto the stack (`EntryRead.push_recalled`), so thought can reason
over a past sentence from the top of its own stack
([Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11)).

Alec: "Intuition is beyond understanding, it is conditioned by desired
output, and thoughts are a distillation of intuition."

- *Understanding* is the conceptual definitions, trained by
  reconstruction: they explain the input.
- *Expectation* is bottom-up. It is made at each look on the
  structure's own Pi, Not and Sigma
  ([below](#expectation-in-stages-on-the-structure)). The expected field is the
  spread of activation that primes attention, the chooser and retrieval.
  It is taught stage by stage, each order's expected values against what
  the look brings there, and it is never evidence.
- *Intuition* is top-down. `intuit` is an operator of thought with a Pi,
  Not and Sigma of its own outside the architecture: when thought calls it,
  its map gives the intuited field, which the operator writes into the
  sentence's knowing and carries up the orders
  ([below](#intuition-on-a-map-of-its-own)). Its map is taught by the field
  that thought leaves, with the output's cost; what is conditioned by
  desired output is also the choice to call it, which the chooser learns
  from the outcome. An intuition may answer, under its own evidence kind
  `intuition`.
- *Thought* is the work of the serial workspace (the eight STM chunks):
  `query` recalls, and `infer` applies a held or stored rule, its trust
  pair computed from evidence, never learned.

In one line: understanding explains the input, intuition is understanding
put to use, and thought is intuition that has been checked.

*Stored rules, fired.* A stored implication is applied to the knowing by
firing it, and `infer` takes each firing as a candidate (Alec, 2026-11-14:
"the existing logic can be introduced as implication"). The knowing $K$ is
the question's order-zero content, the working values MTM prepends for the
stream and, within a third of the remaining work, the order-zero concepts
active in the last observed knowing field. The rules are the LTM1 rows that
reference the concepts the expected field holds active, taken in descending
order of expected activation (`ConceptSpread.rule_order`), with nonzero
evidence and within the work meter
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
into the field by maximum at the places it already holds. When `infer` selects a
firing, the pair is recomputed at unit exponents, $\min(\kappa_1, (t^+, t^-))$,
so learned weights shape what is proposed and never what is accepted. A
firing cut off by the budget returns no content and leaves the goal and STM
unchanged.

The firing maps order zero to order zero. Its knowing is the order-zero field,
a rule's two sides enter as their order-zero content, and the consequent's
content is joined back into that field. Higher orders enter only through their
order-zero constituents: a higher-order concept as the join of its
constituents' content, an individual as its working value. So the firing
cannot tell two individuals apart except by their order-zero differences;
individuation is the serial grammar's work.

*Proposal and pick.* The breadth of the parallel pass and the narrowness of the
serial pick are two learners. Which fired candidate matters most for the focus
is not computed by the firing: the grammar's operation scorer, the one compose
uses (`OperationSelectionLayer.thought_logits`), scores each `infer` candidate
from the focus and the candidate together, the candidate's firing pair among its
inputs, to one logit. The Sigma and Pi layers belong to combination, which must
stay monotone and bounded; a choice needs an unbounded logit, which
the softmax turns into the probability that the score function credits.

*What learns.* Memory is evidence, not a parameter. A row's trust is what its
close wrote: no answer loss, gradient or chooser credit changes a row or its
trust, and a contrary statement observed later adds its own row through an
ordinary close. An STM chunk has no parameters either; it holds references and
a detached working value, so what learns is what the references point at.
The firing of stored rules improves as the things around it improve, each by
its own owner: memory (which rules exist and how true they are, written at kept
closes), meaning (the conceptual definitions and the taxonomy's memberships,
which decide how well an antecedent matches the knowing), place (which rules the
expected field brings into reach), choice (the chooser, which learns when to
call `intuit`, which candidate `infer` takes, which premises to keep and when to
conclude: the path, never the truth of a step), and the firing's own
output-owned weights. A wrong rule is not corrected in memory; it stops
mattering as the concepts it is stated over learn, as the chooser learns not to
take it, and as forgetting removes it when it goes unused.

*Where it fits.*

- *Bottom-up expectation as priming.* The expected field is
  spreading activation (Collins & Loftus 1975). It is not prediction in the
  sense of predictive processing (Clark 2013; Friston 2010): the model
  learns by reconstructing its input, and expectation fills in what is
  missing and answers nothing about the next state (Alec, 2026-11-17).
- *Intuition beyond understanding, shaped by goals.* Expert intuition is
  recognition of what to do (Simon 1992; Klein 1998, recognition-primed
  decisions),
  learned from feedback on outcomes rather than from explaining the input;
  Polanyi's tacit knowing ("we know more than we can tell") is the
  "beyond understanding" part.
- *Two kinds of fast thinking.* Glöckner & Witteman (2010) distinguish four
  kinds of intuition, among them an associative and a constructive,
  goal-driven one. Here the associative kind is expectation, and the
  goal-driven part is intuition's map, taught by what thought leaves and by
  the output, with the chooser's choice of when to call `intuit` and what
  `infer` takes.
- *Thought distilled from intuition.* Poincaré and Hadamard describe
  discovery as unconscious combination offering candidates, a sense of fit
  picking one, and conscious work verifying it; global workspace theory
  (Baars, Dehaene) has the same shape, many parallel processes and a
  narrow serial workspace. The eight STM chunks and `infer`'s pick from
  the candidate menu are that workspace.

*Where it departs.*

- *Rationalization.* Intuition shaped by desired output is also how
  motivated reasoning works: in Haidt's social intuitionist model
  intuition decides and reasoning follows as justification. Here an
  intuition may answer, but under its own evidence kind, and what is
  trusted as proved comes only from `infer`, recall or pervasion, computed
  from evidence. The design is
  normative rather than descriptive: thought distills intuition and then
  checks it, the step people tend to skip. Keep that.
- *Not every thought comes from intuition.* Reasoning also proceeds step
  by step from recall with no hunch first; the menu allows it through
  `query` and `infer` over recalled rules without `intuit`. "Thoughts
  distill intuition" is one route, not all of them.
- *Intuition is only as good as its feedback.* Intuition is reliable
  where feedback is accurate and prompt, a kind environment, and
  misleading where it is not (Hogarth 2001). The choice to call `intuit`
  is credited by the outcome, so its quality tracks the outcome's: the
  supplied-answer lessons are its kind environment, and wrong or noisy
  answers would train a confident bad habit of trusting hunches.

## System 1: two networks and a read to a point

On 2026-11-12 Alec set out two networks of System 1, "one that learns the
input based on missing information, and one that learns from conceptualizing
or thinking about the input". They are expectation and intuition's map, with
expectation now the architecture's (Alec, 2026-11-19: "if expectation comes
from the arch, please let intuit have the pi-not-sigma that exists outside of
the architecture"). Beside them is a read from a field to a point.

- *Expectation*, `expect`, is made at each look on the structure's own Pi,
  Not and Sigma. Its stages are the basic percepts, the concepts and the
  symbols, and not the output. It is built in, and taught stage by stage: at
  each order what was expected against what arrives there.
- *Intuition's map* is a Pi, Not and Sigma of its own, from the observed
  concepts of order zero to the same concepts, which the operator `intuit`
  calls. It is taught by the field that thought leaves, with the output's
  cost (Alec, 2026-11-20).
- *The read to a point* takes a field to one point, many concepts to one:
  "The N:1 read can be taught with System 2's idea as its target". Where
  sentences are comprehended it is a stacked sigma-pi of its own, taught by
  the idea that comprehension leaves; with no System 2, "The N:1 read can be
  centroid if we are doing animal cognition".

All three work without the grammar. They differ in their teacher: the
input, for expectation; thought and the output, for intuition's map; and the
System 2 content, for the read to a point. The mechanism is in
[Expectation](ExpectationRetention.md#intuitions-map-and-the-read-to-a-point).

### Symbols, the field and the spaces of the orders

The two maps work over a vertical structure and a horizontal one,
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
- *Horizontal.* Each order's concepts are placed in a space of their own
  (Alec, 2026-11-20: "it is not clear that the space of nouns, for example,
  should be constrained by the space of events"). A concept is placed by
  its context, and its
  context is the field: "the symbolic activation determines placement
  within conceptual space, but again, only relative to other concepts
  within the same order". A class is placed in this way like any other
  concept. The concepts of each order lie in a geometry of their own (Alec,
  2026-11-20: "the concepts at every order live in spaces with different
  geometries"), so a symbol's cone is tested against one order's concepts
  alone. A type acts on its tokens through the field, up and then down
  to the related tokens, and not by a pull across orders: "this is system2
  influencing system1".
- *What the context may not include.* The context of placement is the
  activation the graph gives before any weighting by the likeness of
  places, so that no place is moved by a value computed from places
  ([Architecture](Architecture.md), the guard against a feedback loop).

So placement works by the field. Its context is the field at every order,
what is present and what the folds give from it, a token's types among them
(`FieldProjection.close`), and not the words observed alone. `place_rows`
moves a row relative to rows of its own order only. A class is placed by its
context like any concept, not at the centroid of its members. A row above the
basic percept carries no perceptual form of its own: a reader that needs a
form takes the word's, through the mapping. Alec, 2026-11-12: "symbols should
not take the centroid of the percepts that they collect; they are epistemic
points."

### The split of sigma and pi (Alec, 2026-11-13)

"So we need to split the sigmaPi. based on the discussion, I see boxes
applying to percepts (since they live in a positive hypercube) and cones
applying to concepts (since they live in a signed hypersphere). That means
symbols are PI(concepts), and concepts are SIGMA(symbols). concepts are also
SIGMA(basic percepts)."

- **A Sigma gives one signed value for each concept**, $\tanh(\sum_i
  w_{ci} s_i)$ in $[-1, 1]$, over the signed values $s_i$ the Not gives, whose size is its certainty, stored as two unsigned lanes: its
  positive part on the lane for and its negative part on the lane against
  (Alec, 2026-11-17: "Signed sigma; good, this is knowing in the signed
  hyperball"). A **Pi gives unsigned rows**, each a product of its members'
  memberships, between zero and one; a percept's opposite is zero, not
  minus one.
- **System 1.** Percepts become basic percepts by a Pi, whose region is a
  box, and basic percepts become the concepts of order zero by a Sigma.
  "basic percepts are part of system 1, which has order-zero concepts over
  those."
- **System 2.** The concepts of an order become symbols by a Pi, whose
  region is a cone at the origin, and those symbols become the concepts of
  the next order by a Sigma. A symbol's constituents are coordinates of
  conceptual space, and its cone lies in "a subspace of that space (thus
  irrelevant details of members are removed by their variation in that
  subspace)" (Alec, 2026-11-20). "system 2 creates a symbolic vertical structure
  on top of that, which is used by all higher-order concepts."
- **One structure at every order** (Alec, 2026-11-17): a Pi, a Not and a
  Sigma. The Not is a layer of options, the value or the not-value of each
  row, and the Sigma's weights are at zero or above
  ([Architecture](Architecture.md#sigma-and-pi-layers)). A Sigma at order
  zero reads basic percepts and above it reads symbols.
- **A cone in place of a nearest neighbour.** "in places where we now ask
  "what is close to this vector", we are effectively asking about the
  hypercone centered on that vector"; a hierarchy of cones lets a read "look
  for part inclusion instead, which is a much faster operation". What falls
  in a symbol's cone is that thing; that one kind lies within another is
  read from the rows a kind is made of, and is the pervasion of
  [Logic](Logic.md#existence-and-pervasion-alec-2026-11-13).
- **A symbol is a pair of rows.** A symbol is two unsigned rows of the Pi,
  one for each lane, since a Pi has unsigned weights and cannot change a
  sign. It is a region of sameness and may hold one concept or several:
  "my dog" and "Voyager" are two concepts and one symbol (Alec, 2026-11-20),
  and there is no symbol of a concept. The Not offers the pair as one signed
  value, the value or the not-value, and a concept above reads the one its
  entry chooses with a weight at zero or above (Alec, 2026-11-19: "its two lanes get passed
  to Not, and those are combined into on signed lane for processing by
  sigma"). A concept stays one signed value; "both" is two concepts, one for
  presence and one for absence, the second reading through an entry of the
  Not.

### The layers (Alec, 2026-11-14 to 2026-11-19)

"We are basically just updating the algorithms used for the existing
SigmaLayer and PiLayer. We are also creating a fast SigmaPi layer that
combines both and that skips the invertibility requirement ( which is the
alternate to the MLP layer, and that works as fast)." And: "I think the
order0 basic percept building and zero-order concept building should use pi
and sigma, respectively."

- **`PiLayer3` is a product of memberships**, on unsigned values on two
  lanes: $z = \exp(W \log \mathrm{clamp}(m, \epsilon, 1) + b)$, a row of
  each lane having weight only on that lane. It is square and sparse, its
  diagonal at one or more, the rest at zero or more and its bias at zero or
  less, so a row requires the members it has weight on and never exceeds its
  own reading, and its exact way back is the triangular solve of its
  `L D U` matrix.
- **The Sigma is a weighted sum with one exit**, $y_c = \tanh(\sum_i
  w_{ci} s_i)$, over the signed values $s_i$ that the Not gives for its
  rows, with weights at zero or above and no bias, so it is monotone. The
  size of an entry is the Sigma's weight and its sign is the Not's: a Not
  entry reads the row's not-value, the pair exchanged, and an entry is never
  both. The gradient makes a Not entry where it carries an entry through
  zero, and a declared Not member makes one explicitly. Its way back gives the input by least squares where there are
  enough concepts, and a signed value on the way back is split into its two
  unsigned parts before a percept or a symbol holds it: -.5 is .5 on the lane
  against and nothing on the lane for.
- **`SigmaPiLayer` fuses the two without an inverse**, as the model's
  alternative to an ordinary hidden layer: a free `PiLayer3` (any weights at
  zero or more, no diagonal law, no bias) and a `SigmaLayer3` over its pairs,
  in one expression, so that a stack of them is captured as one graph.
  Intuition's map, the read to a point and `marker` are such stacks.
- **A basic percept is a row of the first Pi.** "we have basic percepts,
  formed as a PI, which is a (sparse) hypercone on hypercube parts and wholes
  (thus an approximation of the discrete collection that we were creating
  with the mereology)." A row exists only when MTM nominates it; it selects
  for, its parts and wholes being required, and against, the parts that the
  nearest forms holding it add, any one of which tells against it.
- **The concepts of order zero are a sparse Sigma** over the signed values
  that the Not gives for the nominated rows, one concept for each row to
  start with, and nothing one to one after. "Then
  those percepts are collected via a (sparse) hyperplane , via the SIGMA,
  which become zero-order concepts (or basic concepts)." A concept starts
  reading only its own row, at one, reads no code, and is then learned.
- **Both and neither.** A ball of black and white patches is, for the
  concept white, both: "We must, for those concepts, say "yes and no" (at
  different places). but concepts don't work like that; there is no
  pervasion. so we get a conceptual 0 value, which means either neither is
  true or both are true." Both stands at the basic percepts, one carrying
  the property for and another against, and above them it is two concepts,
  one for presence and one for absence.
- **The matrices above order zero are sparse**, stored as compressed rows
  that grow into room allocated beforehand, with an index of columns beside
  them. Membership is learned under a penalty on the entries of both the Pi
  and the Sigma, so that what is learned can be read from the rows; the
  Sigma of order zero is dense and is not penalized.
- **The way back.** From a concept alone the percept returns as its signed
  value, with no overlap of its lanes: "its grey. Until we remember that the
  soccer ball is composed of little hexagons (either conceptually or
  perceptually). then we can render the "both" condition". Both is rendered
  where the parts are had again.
- **Every order is made this way.** The structure repeats above order zero
  for a mind that has symbols; what tells a human mind from an animal's is
  this recursion, which symbols enable, and not the Not (Alec, 2026-11-17).

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
- **Expectation and intuition are maps.** They follow no edge. Expectation
  is the spread on the structure's own Pi, Not and Sigma, and intuition's
  map is a stacked sigma-pi of its own (below). A sigma over single elements
  cannot give an expectation that depends on a combination (below), so each
  starts with a pi, whose units are conjunctions of their inputs.
- **One traversal, forwards and back.** "Once basic percepts run as a pi over
  parts and wholes (nicely covering that area in mereological space),
  concepts over those basic percepts will form concepts =
  sigma(pi(percepts)." And "Expect and intuit can make the same traversal
  forwards and backward (they do not have recursion)", by "the inverse path
  because it is already present in the architecture". Expectation runs on
  the forward path's own matrices; intuition's map has matrices of its own.
  What is already there is the kind of operation: "the neural pathways are there, in the sense that we
  already have those operations between those spaces." In a stack the way up
  is the pi stage to the hidden layer and the way down the sigma stage.
- **What teaches the maps.** Expectation stage by stage at each look,
  intuition's map by the field thought leaves and the output's cost, and the
  read to a point by the idea comprehension leaves, as below. So the explicit side
  teaches the implicit one:
  "explicit membership can be used to teach implicit maps, since we have input
  (token) and output (type) provided as the teaching example for order."
- **The explicit side.** Edges are the promotion of relations of long-term
  memory. Thought reads them: `query` recalls them and `infer` applies them
  ([above](#intuition-understanding-and-thought-alec-2026-10-30)).
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

A combination of two different words reaches System 1 in two ways. A taught
form may hold a space (`ice cream`), and then it is a percept like any word.
And the basic level may read two or more words at one time, from one look
over them, into one idea that System 2 teaches by the idea it composed from
the same words one by one; the grammar may take that idea as one
constituent ([Language](Language.md#the-looks), "A fragment read at one
time"). No form, row or symbol is made for such a fragment. Otherwise a
conjunction of two words first exists on the conceptual side, in what the
grammar composes, in a candidate of MTM and in a row of memory.

### Expectation, in stages on the structure

"My feeling is that order-0 expectation is a nonlinear operation over the
conceptual field, which is not possible using the hypersphere. But a sparse
sigmaPi with ten million concepts is still enormous parameterwise and
computationally, and my general feeling is that type relations are in fact
strict (felix is a cat) but neighbor relations are not (Felix is like Ciao).
So perhaps expectation is nonlinear in virtue of vertical propagation."

Expectation has no map beside the structure. It is made at each look on the
structure's own Pi, Not and Sigma (Alec, 2026-11-19: "using the existing
architecture will allow us to to sparse in interconnection, which should
help amplify the salience"; 2026-11-20: "at each attend(), we want to expect
perceptual and symbolic activation and then learn that stage by stage as we
move over the orders"). What is expected is what the vertical
spread gives, once up and then down, on the matrices of every order
(Alec, 2026-11-17):

- *Up, by fan.* A concept's or a symbol's value is split among the rows that
  read it, each by its weight's share of all the weights it is read with.
- *Down, by share.* A row's value is split among its inputs, each by its
  weight's share of the row. What a row receives from its parents and
  children is joined as the product on complements (one less the product of
  one less each), which never reaches one, and a Not entry passes to the
  other lane.
- *Siblings get less than the concept that caused them:* a concept present
  at one gives every other concept the spread reaches less than one.
- *Across orders in one step.* A kind collects from every epistemic level
  below its own, so a meta concept primes its word and its object with no
  symbol between.
- *Into the first layer.* At order zero the way down goes on through the
  mereology's own entries of the first Pi, so a part that is missing
  receives from the rows its neighbours raised; this writes expected
  percepts, and a row still requires its members.

Its stages are the basic percepts, the concepts and the symbols, and not
the output (Alec, 2026-11-20). At each look what was expected at a stage is
compared with what the look brings there, and that cost teaches its own stage
alone; nothing of it is carried through the way up and the way down again.
The top-down reconstruction is a different thing: what an input is missing
shows as the reconstruction's own error, and no function fills it in. The
field is written at the sentence's open (`ConceptSpread.open_prediction`)
and anew at each look, which is forecast from everything outside it
([Expectation](ExpectationRetention.md#the-forecast-of-a-look)). It also runs
at the end of the read, before comprehension takes its leaves' objects, so
that the object a sentence's words primed is preferred. A model with no
symbols never runs it over the orders.

What reads an expectation, attention's relevance, the compose chooser,
retrieval and the order of stored rules (`ConceptSpread.rule_order`), reads
it detached, and the guard of [Architecture](Architecture.md) stands: codes
never read a value computed from codes. It "is not counted as evidence, it
is just removing some of the burden from later layers. it should be used
only to guess at network activation" (Alec, 2026-11-19); it joins the signal
as a gain with its gradient stopped and as a prime of selection. What
teaches it is the reconstruction's error, omission among it
([Training](Training.md), "Adaptive omission"): where units of the sentence
were hidden, the concepts' stage (`expectation.concepts`) is the difference
between what is expected from the lossy field and the field with the hidden
units present, which is a teacher and never a differentiable target. A
sentence with nothing hidden gives that stage no cost. The error at a hidden
word teaches the rows that hold things together, the symbols above, and no
entry from one word's row to another word's concept.

Likeness alone cannot give an expectation that depends on a combination.
Take four sentences in which the word to expect depends on the pair before
it: A and B give X, C and D give X, A and D give Y, C and B give Y. With
likeness pooled by a sum, X must beat Y in the first two and lose in the
last two, and adding each pair of conditions makes one quantity both
greater and less than zero. A row of a Pi that holds each pair and a Sigma
that gives the word expect the right word in all four, with each word on
coordinates of its own; one linear map does not.

### The stacked sigma-pi

The learned maps of System 1 beside the structure are one class, a stacked
sigma-pi (`SigmaPiStack`), built for intuition's map, for the read to a
point and for each kernel of `marker`.

- *A pair.* A stack is a row of fused pairs (`SigmaPiLayer`), each a Pi and
  then a Sigma. The Pi gives, for each symbol, a row for over the input's
  lanes for and a row against over its lanes against, each a product of
  memberships raised to weights of zero or more. The Sigma reads each
  symbol's pair and gives one signed number in $[-1, 1]$ for each output,
  its entries for or Not. A pair has no inverse and costs what an ordinary
  layer of the same weights costs.
- *Several pairs.* Each pair's signed output is read by the next pair as its
  two lanes, its positive part for and its negative part against, and
  nothing is clipped between pairs.
- *Lanes.* An input enters on its two lanes, for and against, and the
  stack reads concepts, each one signed value held as its two lanes. A
  symbol's two rows are joined by the Not into one signed value, for less
  against, before the Sigma reads it
  ([above](#the-split-of-sigma-and-pi-alec-2026-11-13)).
- *Nothing at the start.* The last sigma's weights start at zero, so a new
  stack gives zero and adds nothing. Its first weights come from a
  generator of its own, so building a stack changes no other random number
  of the model.

### Intuition, on a map of its own

`intuit` is an operator of thought, and it has a Pi, Not and Sigma of its
own that stands outside the architecture (Alec, 2026-11-19: "We should not
call it "the expected field", since it is called from an operator, whereas
expect is built in to a model parameter"). Its map is a stack with one
moderate hidden layer (Alec, 2026-11-14: "stack a sigmaPi with only a
moderate hidden layer (full input, full output, so it resembles only few
percepts)"), from the observed concepts of order zero, both lanes, to the
same concepts. When thought calls `intuit`, the map gives the *intuited
field*; the operator joins it into the stream's knowing at the places the
sentence observed, marked as intuition, and carries it up the orders, so
that thought, the answer and the output read it: "insights that trigger
knowing as a result of observation directly". Nothing reads it as expected.
Its teacher is the field that thought leaves, with the cost of the output
(Alec, 2026-11-20: "I approve your choice: restore what you approved on
2026-11-12, the field that thought leaves, with the output's cost");
omission teaches expectation alone. The firing of stored rules is not
`intuit`'s: it is how `infer` applies a stored implication
([above](#intuition-understanding-and-thought-alec-2026-10-30)).

### The read to a point, taught by System 2

The read takes a field to one point, of a code's width. Where sentences are
comprehended it is a stack of its own on the conceptual space that owns the
concepts (`idea_stack`; `ideaHidden` 64, `ideaLayers` 1). Its input is the
observed order-zero field on both lanes at the sentence's open, and its
teacher the idea that the kept reading leaves (`record.meaning_root`), both
detached; its cost is their squared error (`expectation.idea`). A model that
comprehends no sentence builds none, and reads a field to a point by its
centroid (`FieldProjection.key`). The expected point is the read of the
expected field; nothing reads it yet. Why a point is kept, as a key and not
as what is reasoned on, is in
[Philosophy](Philosophy.md#expectation-as-a-gain-attention-as-exclusion).

### The tools of thought (Alec, 2026-11-13)

"For thought, we are giving tools to the model so that it can change its
structures". The operators of [Language](Language.md) compose and
generate; these act at a completed boundary.

| Tool | Alec's note |
|---|---|
| `ask` | "as before" |
| `query` | "over memory" |
| `isPart` | "this is inferred after a difference" |
| `isWhole` | "probably keep, but equivalent to a swapped part" |
| `isEqual` | the test of identity; it "can unify over two different orders by doing lower or raise" |
| `isTrue`, `inferBackward` | kept |
| `not`, `non` | "as before" |
| `bind`, `exist`, `infer`, `intuit` | |
| `part/whole/implies` | "*creates* an STM1" |
| `define` | "creates two STM1" |

`define` is how an identity is written: two relations, `part` each way.

**Thought reads the structure** (Alec, 2026-11-17). `parts(c)` gives the
entries of c's row of the Pi, the symbols below c at every lower order,
each with its weight; `wholes(c)` gives the rows of the Pi above c that
have c as an entry, each with its weight. Of the Sigma, thought can read
what a concept reads, which concepts read a given row, and which of those
entries are Not entries. A result is a set of references with their
weights, not a yes or no; a reference's words are those whose meta concepts
join them to its concept, and a reference with no word, such as a percept,
is given as itself. An entry is read as present above a low threshold, a
key measured under the penalty on the entries, since a literal that
survives the penalty has a small weight. A read is serial, is charged to
the work budget as the queries are, and moves nothing. `isPart`'s proofs
keep `LTM1`'s part table as their evidence.
The strength of expectation is not a tool: "drop gain, we can set that as
expectationGain on the model."

### Where they fit

*The field map.*

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

*The read to a point.* It learns to give at once what comprehension
computed by the grammar.

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
- *Products, on two lanes.* The literature's spread adds activations, with
  inhibition beside them. Ours multiplies on the way up, in a pi stage, and
  joins on the way down, in a sigma stage, on two lanes kept apart; the
  products are where its nonlinearity lies.
- *Prediction is graded.* The pre-activation of a specific word's sound
  was not reproduced in nine laboratories, while the effect of a word's
  predictability at the word itself was (Nieuwland et al. 2018).
- *Two, by their teachers.* The literature supports a difference of
  teacher between the two maps, not a difference of mechanism, and
  that anything is partitioned into exactly two systems is itself disputed
  (Keren & Schul 2009; Melnikoff & Bargh 2018).
- *Symbols.* That the field map is taught while language is processed
  is this design's; in the literature its learning needs no symbols.

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
