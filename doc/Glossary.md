# Glossary

*The terms of the model, each with one meaning. Where a term names something
in the source, the source's name follows in backticks. A term used in another
definition is in italics the first time it appears there. Where a definition
rests on a decision, the decision is quoted (Alec, with its date).*

## The layers

**Pi.** A matrix whose rows are products. A row's value in a reading is the
product of its members' values, each a number from 0 to 1, so a row is present
only as far as every member is. Its entries are zero or above; it has no
negative entry. A Pi *selects for and against* (Alec, 2026-11-19: "Pi should
select both strong for and strong against"): a row has members whose presence
tells for it, and members whose presence tells against it, and any one of the
second kind is enough. A row is a hypercone of the space beneath it: the Pi is
"identity/generalization on a hypercone" (Alec, 2026-11-19), and what falls
inside the cone is the same thing. A Pi makes what needs all of its members: the *basic
percepts* at order zero, and the *symbols* above it. Nothing is one to one: a
row may be read by any number of concepts, and a concept may read any number
of rows. It builds upward, from
*parts*. In the source: `PiLayer3`, and the first one `PerceptPi`.

**Not.** The layer between a Pi and a Sigma. It "provides options, so that
concepts can choose value or not-value" (Alec, 2026-11-19). From the two
values of each row of the Pi, evidence for and evidence against, it offers two
signed values from -1 to 1: the *value*, for less against, and the
*not-value*, the pair exchanged, against less for. A concept's entry chooses
one of the two; an entry that reads the not-value is a *Not entry*. A concept
of presence reads a row's value and a concept of absence its not-value. The
Not is the only place where a sign is changed (Alec, 2026-11-19: "that would
give it the ability to change sign, which should be restricted to not"). In
the source: `NotLayer3`.

**Sigma.** A matrix whose columns are weighted sums, squashed into the range
from -1 to 1 (`tanh`). It reads the signed values that the Not gives, any
number of them for one column. Its weights are zero or above and it has no
bias, so it is monotone: raising an input never lowers a column. A value below
zero comes to it from evidence against or from an entry of the Not, and it
makes none itself. A Sigma makes *concepts*: what any one of its members can
be enough for. A column divides the space beneath it by a hyperplane: the
Sigma is "set creation / bifurcation on a hyperplane" (Alec, 2026-11-19), and
what lies on one side is the set. In the source: `SparseSigmaLayer3`.

**Thing, kind.** What the two epistemic levels of an order make. A Pi makes a
*thing*, a basic percept or a symbol: which are the same, by a region. A Sigma
makes a *kind*, a concept: which belong together, by a list. A kind collects
any number of things, from every lower order and from none at or above its
own: `green things` collects Felix and it collects cats (Alec, 2026-11-20).

**Epistemic level.** One Pi or one Sigma (Alec, 2026-11-19: "My nomenclature
is *epistemic levels* as opposed to orders (which are half as many)"). Every
*order* has two epistemic levels, its Pi and its Sigma, so there are half as
many orders as epistemic levels. The source calls the matrix of a level a
layer (`PiLayer3`, `SigmaLayer3`); a layer is never an order.

**Pi, Not, Sigma.** The one structure that is repeated at every order (Alec,
2026-11-17), order zero among them: "Pi reads perceptual space, its two lanes
get passed to Not, and those are combined into on signed lane for processing
by sigma (which may process an arbitrary number of those signed values)"
(Alec, 2026-11-19). The Pi and the Sigma are its two *epistemic levels*.

**Entry.** One place of a matrix: a member of a row of a Pi, a weight of a
Sigma on one input, or an entry of the Not.

## What the layers make

**Percept.** What perception finds in the input: a form, with its place and
its time. A percept is a discriminant, a thing that is found or not found. It
bears no meaning. Its *code* lies in *perceptual space*. Percepts are "the
perceived positive and negative aspects" of concepts, "or more precisely,
hyperconic regions of them" (Alec, 2026-11-19).

**Basic.** Of order zero. A *basic percept* is one row of the first Pi over
its *parts* and its *properties*. It becomes a row only when MTM nominates it
on recurrence, never on a first reading. The *basic level* is the basic
percepts and the concepts of order zero. A mind with the basic level alone has
no symbols.

**Concept.** What bears meaning. In a reading a concept is one signed value:
positive for present, negative for absent, and its size is its certainty
(Alec, 2026-11-19: "The concept remains a signed fuzzy keen on certainty").
Its meaning is a point of *conceptual space*. A concept is a kind of objects:
it "is created implicitly by Sigma or when nominated by MTM when recognizing
that the objects form a kind" (Alec, 2026-11-20). What its Sigma reads are
rows of a Pi, basic percepts and symbols. From a statement such as "Felix is
a cat" MTM makes a link of part to whole or of member to kind, depending on
the orders of the two.

**Symbol.** A row of a Pi above order zero: a thing, as a region of sameness
over concepts of lower orders. "My dog" and "Voyager" are two concepts and one
symbol: "we don't want to represent both with one vector, we want a region in
semantic space. So the region specification for similarity is defined with
the pi" (Alec, 2026-11-20). It is "a neighborhood of identity", and it is
what lets a concept be reasoned over with other concepts. Its fold is the fold
of a basic percept, the plain product over its constituents, and its
constituents are coordinates of conceptual space, as a basic percept's are
coordinates of perceptual space. The concepts that are the same thing are what
falls in its region; a coordinate on which they differ has no weight, so the
region is a cone in the subspace of what they share (Alec, 2026-11-20). A symbol has no dimensions and "does not live in any space,
which is what makes its collection by sigma so set-like" (Alec, 2026-11-19).
It is its constituents occurring together: "a basic percept or symbol is
created implicitly by PI or when nominated by MTM when recognizing that their
constituents always occur together" (Alec, 2026-11-20). There is no symbol
*of* a concept: nothing makes a symbol for a concept, one for one. MTM also
makes a symbol from a statement such as "Felix is my cat", which says that
two are the same. "Since implicit generally precedes explicit, MTM should
check for an existing fit first."

**Order.** How far a concept stands above the basic percepts. A concept of
order zero is read from basic percepts. A concept formed above symbols is one
order above the concepts those symbols read. An order is raised implicitly,
by any concept formed above symbols, and explicitly, by statements that
share a side: "Felix is a cat" and "Ciao is a cat" call for a concept of cat
one order higher that ranges over both (Alec, 2026-10-21 and 2026-11-19).
Order zero has room for as many concepts as there are basic percepts, and
each order above has half the room of the one below.

## Evidence and logic

**Lane.** Every value in perception is stored as two numbers from 0
to 1: evidence for, and evidence against. "We are on a positive unit
hypercube with two lanes; evidence for and evidence against. When I am
talking about a negative number in the perceptual context, that is just a
shorthand for positive evidence against" (Alec, 2026-11-19). Zero on both
lanes is no evidence. The two lanes go no further than the Not, which joins
them into one signed value; a Sigma reads no pair.

**Negation.** Three different things, which the model keeps apart.

- *Evidence against* is a value on the against lane. It is something
  perceived: the `y` of `archery` is evidence against `archer`.
- *A Not entry* is an entry that reads a row's not-value. It is made
  implicitly, by the gradient, where teaching takes a concept's weight on a
  row down to zero and further (Alec, 2026-11-19: "the gradient should be able
  to make a not entry, that is the implicit definition"); and explicitly, by a
  member declared as a Not member. An omission, a thing expected that did not
  come, teaches through the gradient. The Pi has none, and no weight of a
  Sigma is below zero: the size of an entry is the Sigma's and the choice of
  option is the Not's.
- *The operator `not`* of a sentence or of a thought exchanges the two lanes
  of what it is put on (`FieldLogic.reduce`).

**Ternary logic.** The logic of one signed value: true, false, and neither.
A concept in a reading is ternary. On values between the three, the layers
take products and the squashed sum.

**Intersection, union.** The least and the greatest of graded values. They
are operations of the basic level, below symbols (Alec, 2026-11-19: "I like
the intersection/union operations as subsymbolic (basic level) and
conjunction/disjunction at the symbolic level").

**Conjunction, disjunction.** The symbolic level's two: a conjunction is a row
of a Pi, which needs all of its members, and a disjunction is a column of a
Sigma, which any member may be enough for. There is no third pair named `and`
and `or`.

**Quaternary logic.** The logic of two lanes: true, false, neither, and both.
Evidence is quaternary. "Both" is never a state of one concept: it stands
across the basic percepts of one reading, one row carrying a property for and
another against, or as two concepts, one for presence and one for absence
(Alec, 2026-11-19).

## The spaces

**Perceptual space.** The space of forms. A point of it is a percept's code:
64 coordinates for its parts and 64 for its properties, each on the two
lanes, with a place and a time. It is the positive unit hypercube. No meaning
is stored in it. In the source it is held by `PartSpace` and `WholeSpace`.

**Conceptual space.** The space of meanings. A concept's meaning is a point
of it, and so is an idea, the meaning of a whole sentence. Its values are
signed. In the source: `ConceptualSpace`.

**Symbolic space.** The symbols, taken together. It "is a set and a space of
points; each symbol has no space (i.e. dimensionality)" (Alec, 2026-11-19): a
symbol is a point with no coordinates and no address. What is known of a
symbol is which concepts its row reads, and which concepts read it.

**Perceptual towers.** The two orders in which percepts are built. The *part
tower* builds upward: characters at the bottom, then runs of characters, then
forms. The *property tower* builds downward: the class of all characters at
the top, then narrower classes. A basic percept is the meeting of the two: a
row over its parts and its properties. Only a strict whole in the part tower
is evidence against a form (Alec, 2026-11-19).

**Parts.** The members of the part tower. A part is a run of adjacent
characters; a form's parts are its characters and the longer parts that lie
inside it. In a code the parts are a lower bound: they start at all zeros and
are joined by the greatest. The parts that are stored are the *catalog*. See
[Mereology](Mereology.md#percept-code).

**Properties.** The members of the property tower, also called wholes. A
property is defined by what belongs to it, a set of characters or a number
for a feeling, or by other properties. In a code the properties are an upper
bound: they start at all ones, under the class of all characters, and are
joined by the least.

**Code.** The vector of a percept in perceptual space. No code lies inside
another's unless the one form is a part of the other (Alec, 2026-11-19).

## Parts, kinds and the order between them

**Mereology.** The relation of part and whole, and the order it gives: a thing
is at or above each of its parts and at or below each of its wholes. Parthood
is shared size, and it is not symmetric: `is` is part of `this`, and `this` is
not part of `is` (Alec, 2026-11-18). It is held in two ways that are not to be
read as one another ([Mereology](Mereology.md)). *Geometric* parthood is read
from codes and answers "looks like": one code lies inside another. *Asserted*
parthood is a relation kept in LTM and answers "is": "cats are animals" is one
row of `LTM1`.

**Taxonomy.** The order of kinds: a member and its kind stand one *order*
apart. "Felix is a cat" and "Ciao is a cat" make cat a kind one order above
Felix and Ciao. A statement that one thing is another makes "a mereological or
taxonomic link (depending on order)" (Alec, 2026-11-19): between two concepts
of one order it is a link of part to whole, and between concepts of two orders
it is a link of member to kind. The taxonomy and the relations of `LTM1` are
one structure, read by `Taxonomy.capture_taxonomy`.

## The grammars

**Grammar.** A set of operators, each taking meanings and giving a meaning,
with a chooser that picks the next operator. The model has three, and all
three belong to the mind that has symbols: "compose, think, and generate
grammars all operate as system 2, on top of system 1" (Alec, 2026-11-03).

**Compose grammar.** The grammar of reading. Its operators take the meanings
of a sentence's words and join them, step by step, into one idea, the meaning
of the sentence. Its chooser is `OperationSelectionLayer`.

**Think grammar.** The grammar of thought, also called the thought grammar.
Its operators work on what is already known: the ideas held in STM and what a
query brings from LTM. They change knowing into knowing between the reading of
a sentence and any answer. Among them are `query`, `ask` and `isTrue`.

**Generate grammar.** The grammar of speaking, the inverse of compose. Its
operators take an idea apart, step by step, into words in their order
(`BasicModel.generate`, with the chooser `GeneratePolicy`). The order of the
words of an output comes from it.

## Memory

**Time.** The model has one time: "one integer clock and one sinusoidal
encoding of it" (Alec, 2026-11-19). The clock is an integer on the model, and
its code is made in one place and held on the model beside it, where every
reader reads it. A concept's `.when` is the stamp of that clock at the
concept's moment, and it stays with the concept, which is opaque. The chooser
and the operators are given the present in the same code, with a *comparator*
that tells "how far back in history the concept they have recalled is".

**STM.** Short-term memory: the chunks held while a sentence is read and
thought about, some of them on a stack. The top of the stack is the most
recent. In the source: `ShortTermMemory`.

**MTM.** Medium-term memory: where things are tried before they are kept. It
counts what recurs and nominates rows of the Pi; from statements it makes
symbols and concepts, looking first for one that already fits. What it holds
is easily promoted and easily forgotten. In the source: `MediumTermMemory`.

**Slot.** A place of episodic memory: one concept of a moment, with its value.
A row of LTM keeps an idea, which is semantic memory, and a few slots, which
are episodic memory (Alec, 2026-11-19: "Entry should be slot in episodic
memory (the idea is semantic memory)"). A slot holds a concept and not a
percept (Alec, 2026-11-11). Which concepts of the moment fill the slots is not
settled: not the most active; "perhaps least expected, or sentence dominant
such as np, vp". A query on LTM matches against the rows' slots. In the
source: `episodic_concepts`, `episodic_values`; four to a row, by
`episodicConcepts`.

**LTM.** Long-term memory: what is kept. Its rows hold ideas, each with the
slots of its moment (`LTM0`), and relations between concepts (`LTM1`). It is read by content: a query matches the top of the STM stack
against the rows, and the best row is pushed onto the stack (Alec,
2026-11-19).

## Attention

**Look.** "One act of attention while a sentence is read. It filters the
perceptual field by a saliency map from attended concepts, which will filter
on both the what/where of conceptual space, and carries those selected
percepts into the activation space on all existing concepts" (Alec,
2026-11-19). In the model the filter is a number from 0 to 1 at each percept
of the sentence, and the first look of a sentence takes all of it at once.

**Expectation.** A guess at the network's activation, computed in the
architecture, on the structure's own Pi, Not and Sigma. It is built in, turned
on by a parameter of the model, and taught by the reconstruction in stages:
percepts, concepts, output. It "is not counted as evidence, it is just
removing some of the burden from later layers" (Alec, 2026-11-19). It joins
the signal as a gain on what arrives, with its gradient stopped, and as a
prime of selection. What it gives is the *expected field*.

**Intuition.** What the operator `intuit` gives when thought calls it: the
*intuited field*, from a Pi, Not, Sigma of its own that stands outside the
architecture (Alec, 2026-11-19: "it is called from an operator, whereas expect
is built in to a model parameter"). The operator writes it into knowing,
marked as intuition. It is not the expected field.

**Analysis, synthesis.** Synthesis is the way up: the Pi builds rows from
parts and properties, and the Sigma builds concepts from rows. Analysis is
the way down, and what drives it is conceptual attention: a request over
concepts carried down onto the what and the where, "over both particles and
properties" (Alec, 2026-11-19). It reads the same matrices as the way up, by
their transposes, and has no weights of its own.
