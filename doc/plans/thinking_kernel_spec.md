# Thinking Kernel Specification

## 0. Purpose

This document specifies a minimal cognitive/reasoning kernel for a language-capable mind that learns by illuminating conceptual locations.

The architecture assumes that the model can already parse and generate language. The remaining task is to provide a compact mechanism for internal reasoning, subgoal formation, memory lookup, structural traversal, external inquiry, and answer generation.

The central claim is:

> Learning is luminosity: the movement of conceptual locations away from unknownness toward known truth or known falsity.

Prediction is treated as a special case of question-answering:

> “What comes next?” is a question whose referent is an unilluminated location in conceptual space.

---

## 1. Core Concepts

### 1.1 Conceptual Space

The mind has a conceptual space containing locations, regions, and relations.

A conceptual location may correspond to:

- an entity,
- a property,
- a proposition,
- a relation,
- a question-referent,
- a predicted object such as “the next sentence.”

Each location has an associated truth or value interval.

### 1.2 Luminosity

Luminosity is the degree to which a location is no longer unknown.

Truth values are modeled on an interval such as:

```text
-1 = known false
 0 = unknown
+1 = known true
```

Luminosity is the absolute magnitude of truth determination:

```text
luminosity(P) = |truth(P)|
```

Therefore, proving a proposition false is as luminous as proving it true.

### 1.3 Truth Intervals

Instead of returning a single truth point, the system should generally return an interval:

```text
lookup(P) -> [lower, upper], trust, provenance
```

Examples:

```text
[0.98, 1.00]      strongly true
[-1.00, -0.95]    strongly false
[-0.05, 0.05]     effectively unknown
[-0.4, 0.8]       uncertain / conflicting / broad interval
```

The interval is evidence-sensitive and may narrow as more information is gathered.

### 1.4 Trust

Trust measures the reliability of a truth interval or derived result.

Trust may derive from:

- memory reliability,
- source reliability,
- rule reliability,
- proof validity,
- empirical measurement reliability,
- user-provided assumptions,
- provenance chains.

Truth and trust are distinct:

```text
truth interval: what value is indicated
trust: how reliable the indication is
```

---

## 2. Memory Model

### 2.1 LTM: Long-Term Memory

LTM stores durable knowledge:

```text
concepts
relations
truth intervals
trust values
taxonomy
meronomy
provenance
conditionals
learned rules
source records
```

LTM is not directly modified by generated sentences.

Only grounded evidence, trusted derivation, external testimony, or validated learning can update LTM.

### 2.2 STM: Short-Term Memory

STM stores temporary reasoning state.

STM is organized as a stack of frames.

Each frame contains:

```text
target
purpose
local bindings
local query results
local part results
temporary symbols
candidate paths
trace
budget
status
```

A frame is pushed by `think()` and popped by `answer()`.

### 2.3 STM Stack

Nested subgoals are represented as nested STM frames.

```text
Main frame: answer Q0
    Child frame: answer Q1
        Grandchild frame: answer Q2
```

When a child frame closes, only its certified result, trust, and trace are returned to the parent frame. Speculative scratch is discarded unless explicitly committed by a grounded update.

---

## 3. Grammar and Cognitive Operations

### 3.1 NP

NP is not a special tool. NP already exists in grammar.

An NP is a grammatical phrase that denotes a conceptual location.

Examples:

```text
“whale”                 -> location: whale
“a mammal”              -> location: mammal
“the next sentence”     -> location: next_sentence(current_context)
“whether X is Y”        -> location: is_a(X, Y)
```

The NP specifies a location. It does not itself inspect truth.

---

## 4. Core Operations

The minimal operation set is:

```text
lookup(location)
part(location)
think(location/question)
query(addressee, location/question)
answer(value, trust, trace)
```

---

## 5. `lookup()`

### 5.1 Definition

`lookup()` reads the mind’s own LTM for the current truth interval at a location.

```text
lookup(location) -> TruthInterval
```

It does not open a new reasoning frame. It does not ask another agent. It does not generate new truth.

### 5.2 Return Object

```text
TruthInterval:
    lower: float
    upper: float
    luminosity: float
    trust: float
    provenance: list
    status: true | false | unknown | mixed | conflicting
```

### 5.3 Examples

```text
lookup(is_a(whale, mammal))
-> interval: [0.98, 1.00]
-> trust: 0.99
-> status: true
```

```text
lookup(is_a(whale, reptile))
-> interval: [-1.00, -0.95]
-> trust: 0.98
-> status: false
```

```text
lookup(causes(X, Y))
-> interval: [-0.05, 0.05]
-> trust: 0.10
-> status: unknown
```

### 5.4 Invariant

```text
lookup() inspects internal memory.
lookup() does not create truth.
```

---

## 6. `part()`

### 6.1 Definition

`part()` traverses conceptual structure.

It queries meronomy and taxonomy.

```text
part(location, mode?, direction?) -> StructuralRelations
```

### 6.2 Meronomy

Meronomy concerns part-whole structure.

English marker:

```text
has
```

Examples:

```text
part(hand, meronomy-down) -> fingers, palm, thumb
```

renders:

```text
A hand has fingers.
```

```text
part(finger, meronomy-up) -> hand
```

renders:

```text
A hand has fingers.
```

### 6.3 Taxonomy

Taxonomy concerns kind/type/class structure.

English marker:

```text
is a
```

Examples:

```text
part(whale, taxonomy-up) -> mammal
```

renders:

```text
A whale is a mammal.
```

```text
part(mammal, taxonomy-down) -> whale, dog, human
```

renders:

```text
A whale is a mammal.
```

### 6.4 Invariant

```text
part() exposes structure.
part() does not by itself establish truth unless the returned relation is already grounded in LTM.
```

---

## 7. `think()`

### 7.1 Definition

`think()` is internal speech or internal inquiry.

It pushes a fresh STM frame whose purpose is to illuminate a target location.

```text
think(target):
    push STM frame
    bind frame.target = target
    use lookup(), part(), nested think(), and possibly query()
    close with answer()
    pop STM frame
```

### 7.2 Purpose

`think()` is the mechanism for subgoals.

A subgoal is simply a nested `think()` call.

```text
Main goal:
    think(Q0)

Subgoal:
    think(Q1) inside think(Q0)
```

### 7.3 Example

Question:

```text
Is a whale a mammal?
```

Internal trace:

```text
think(is_a(whale, mammal)):
    lookup(is_a(whale, mammal)) -> unknown
    part(whale, taxonomy-up) -> mammal
    lookup(is_a(whale, mammal)) -> true
    answer(true, trust=.99)
```

### 7.4 Nested Example

```text
think(is_a(X, Y)):
    lookup(is_a(X, Y)) -> unknown
    part(X, taxonomy-up) -> A

    think(is_a(A, Y)):
        lookup(is_a(A, Y)) -> true
        answer(true, trust=.95)

    answer(true, trust=.95)
```

### 7.5 Invariant

```text
think() may create STM symbols.
think() may derive temporary conclusions.
think() may not write truth to LTM unless the result is grounded.
```

---

## 8. `query()`

### 8.1 Definition

`query()` is used for asking another agent, tool, database, source, or external system.

```text
query(addressee, target) -> testimony/evidence
```

Unlike `lookup()`, `query()` is outward-facing.

Unlike `think()`, `query()` does not open a purely internal reasoning frame.

### 8.2 Examples

```text
query(user, open_time(cafe))
query(expert, legal_status(R))
query(database, population(Vienna))
query(sensor, temperature(room))
query(web, current_weather(Geneva))
```

### 8.3 Result

A query result becomes testimony or evidence:

```text
ExternalResult:
    proposition
    source
    source_trust
    channel_trust
    timestamp
    provenance
```

The proposition is not automatically true.

It can later influence `lookup()` after being incorporated with appropriate trust.

### 8.4 Invariant

```text
query() asks outside the mind.
query() produces testimony or evidence.
query() does not create truth merely because an answer was received.
```

---

## 9. `answer()`

### 9.1 Definition

`answer()` closes the active `think()` frame.

```text
answer(value, trust, trace)
```

Values:

```text
true
false
unknown
mixed
conflicting
```

### 9.2 Closure Rule

An answer may close a frame only if one of the following holds:

```text
1. The target is sufficiently luminous.
2. The target is contradicted with sufficient trust.
3. A bounded search has failed, licensing unknown.
4. The frame budget is exhausted and returns bounded_unknown.
```

### 9.3 Parent Return

When a child frame closes, it returns:

```text
ChildResult:
    target
    value
    interval
    trust
    trace
    provenance
    relevance_to_parent
```

The parent frame may use this result as a premise only according to its trust.

---

## 10. Execution Model

### 10.1 Main Loop

A top-level user question is converted into a target location.

```text
user question -> target location Q
think(Q)
```

The runtime repeatedly calls the mind over the current frame state.

```text
while not frame.closed:
    op = mind.next_operation(frame_state)
    result = execute(op)
    frame.update(result)
```

The mind drives the loop by choosing the next operation.

The runtime enforces the semantics of each operation.

### 10.2 Operation Selection

At each step, the mind may choose:

```text
lookup(target)
part(location)
think(subgoal)
query(addressee, target)
answer(value)
```

### 10.3 Runtime Authority

The model may propose operations.

The runtime/tool system executes and validates them.

The model does not directly alter LTM truth values.

---

## 11. Subgoals

### 11.1 Definition

A subgoal is a `think()` frame opened inside another `think()` frame.

```text
subgoal = nested think(target)
```

### 11.2 Subgoal Contract

Each subgoal frame has:

```text
target
purpose
local STM
allowed operations
success condition
failure condition
reward template
budget
```

### 11.3 Local Success

A subgoal succeeds when its own target becomes luminous enough.

```text
success = luminosity(target) >= threshold and trust >= threshold
```

### 11.4 Parent Usefulness

A subgoal is useful only if its result helps illuminate the parent target.

```text
parent_credit = relevance(child_result, parent_target)
```

A subgoal can succeed locally but still receive low parent credit if irrelevant.

---

## 12. Reward and Training

### 12.1 Fundamental Objective

The basic objective is:

```text
maximize grounded, trust-weighted luminosity of the active target
```

### 12.2 Frame-Local Reward

Each `think()` frame automatically compiles a local reward from its target.

For a proposition `p` illuminated during a frame with target `Q`:

```text
reward =
    relevance_Q(p)
  * trust(p)
  * delta_luminosity(p)
  - step_cost
```

Where:

```text
delta_luminosity(p) = |truth_new(p)| - |truth_old(p)|
```

### 12.3 Terminal Reward

A large terminal reward is given when the active frame’s target is resolved.

```text
target true with trust      -> terminal success
target false with trust     -> terminal success
bounded unknown             -> valid terminal unknown
unsupported assertion       -> failure
```

### 12.4 No Arbitrary Reward Writing

The mind does not write arbitrary reward functions.

The mind may choose a subgoal:

```text
think(Q1)
```

The system compiles the reward:

```text
reward = illuminate Q1 with trust, relevance, and cost constraints
```

Thus the mind chooses what to investigate, but the architecture determines what counts as success.

### 12.5 Curriculum

Training proceeds from simple to difficult questions.

Examples:

```text
Depth 0:
    lookup(target) -> answer

Depth 1:
    part(X) -> lookup(neighbor) -> answer

Depth 2:
    part(X) -> think(subgoal) -> answer

Depth N:
    nested think() frames with query/part/lookup operations
```

### 12.6 Successful Trace Training

If final reward is sparse, successful traces can become supervised training data.

A successful trace:

```text
state_0 -> op_1
state_1 -> op_2
state_2 -> op_3
...
state_n -> answer
```

becomes training data for next-operation prediction.

### 12.7 Viterbi / Hard-EM Training

When traces are latent:

1. Given question, memory, and correct answer, search for the best valid trace.
2. Select the shortest or highest-trust successful trace.
3. Train the model to reproduce the operations in that trace.

```text
E-step:
    infer best trace

M-step:
    train next-operation prediction on that trace
```

No actor-critic architecture is required.

---

## 13. Next-Sentence Prediction

### 13.1 Prediction as Question Answering

“What is the next sentence?” is treated as a question.

```text
target = next_sentence(context)
think(target)
```

The phrase “the next sentence” denotes a referent in conceptual/output space.

### 13.2 Attested Continuation

For corpus training:

```text
next_sentence(context, observed_sentence) = true
```

But this establishes discourse truth, not necessarily world truth.

Example:

```text
The sentence “The moon is made of cheese” may be the attested next sentence.
```

This does not make the proposition `made_of(moon, cheese)` true.

### 13.3 Open-Ended Continuation

For open-ended generation, the target is future-valued and cannot be solved by memory lookup alone.

The system uses:

```text
context
style
grammar
conceptual presences
conceptual absences
trust constraints
world model
discourse purpose
```

to illuminate the likely continuation.

---

## 14. Safety and Anti-Hallucination Invariants

### 14.1 Generated Claims Do Not Create Truth

```text
generated(P) does not imply true(P)
```

### 14.2 Only Grounding Moves Truth

Truth intervals move away from zero only through:

```text
lookup of durable memory
trusted derivation
validated evidence
external query with source trust
empirical observation
accepted user assumption
```

### 14.3 Query Results Are Testimony

External `query()` returns testimony/evidence, not truth.

### 14.4 Think Frames Are Temporary

Speculation remains in STM unless grounded.

### 14.5 Assertion Requires Trace

A high-trust answer should have:

```text
target
truth interval
trust
trace
provenance
```

### 14.6 Unknown Is Valid

If no grounded path is found within budget, the correct answer may be:

```text
unknown
```

Unknown is preferable to unsupported assertion.

---

## 15. Minimal API Sketch

```python
class TruthInterval:
    lower: float
    upper: float
    trust: float
    provenance: list

    @property
    def luminosity(self) -> float:
        return max(abs(self.lower), abs(self.upper))


class Frame:
    target: object
    purpose: str
    stm: dict
    trace: list
    budget: int
    status: str


def lookup(location) -> TruthInterval:
    """Read internal LTM truth interval."""
    ...


def part(location, mode=None, direction=None):
    """Return meronomic/taxonomic structural relations."""
    ...


def think(target):
    """Push STM frame and internally resolve target."""
    ...


def query(addressee, target):
    """Ask an external agent/source/tool for testimony or evidence."""
    ...


def answer(value, trust, trace):
    """Close active think frame."""
    ...
```

---

## 16. Summary

The final simplified kernel is:

```text
NP      = grammatical form that denotes a location
lookup  = inspect internal truth interval
part    = traverse meronomy/taxonomy
think   = open internal STM frame for a subgoal
query   = ask another agent/source/tool
answer  = close a think frame
```

Thinking is nested internal speech over STM frames.

A question becomes a target location.

`think(target)` creates a local frame whose purpose is to illuminate that target.

`lookup()` checks existing internal truth.

`part()` exposes structural paths.

`query()` seeks external testimony or evidence.

`answer()` returns a certified result.

Learning trains the mind to choose operations that increase grounded, trust-weighted luminosity of active targets while preserving the distinction between internal memory, internal reasoning, and external inquiry.
