# Glossary

*The terms of the model, each with one meaning. Where a term names something
in the source, the source's name follows in backticks. A term used in another
entry is in italics the first time it appears there. Where an entry rests on a
decision, the decision is quoted (Alec, with its date).*

## The layers

**Pi.** A matrix whose rows are products. A row's value in a reading is the
product of its members' values, each a number from 0 to 1, so a row is present
only as far as every member is. Its entries are zero or above; it has no
negative entry. A Pi *selects for and against* (Alec, 2026-11-19: "Pi should
select both strong for and strong against"): a row has members whose presence
tells for it, and members whose presence tells against it, and any one of the
second kind is enough. A Pi makes what needs all of its members: the *basic
percepts* at order zero, and the *symbols* above it. It builds upward, from
*parts*. In the source: `PiLayer3`, and the first one `PerceptPi`.

**Sigma.** A matrix whose columns are weighted sums, squashed into the range
from -1 to 1 (`tanh`). Its weights are signed and it has no bias. It reads two
channels, evidence for and evidence against, each with weights of its own
(Alec, 2026-11-19: "after the Not layer we still have two channels, those need
not be combined"). A Sigma makes *concepts*: what any one of its members can
be enough for. In the source: `SigmaLayer3`, and at order zero
`OrderZeroSigma`.

**Epistemic level.** One Pi or one Sigma (Alec, 2026-11-19: "My nomenclature
is *epistemic levels* as opposed to orders (which are half as many)"). Every
*order* has two epistemic levels, its Pi and its Sigma, so there are half as
many orders as epistemic levels. The source calls the matrix of a level a
layer (`PiLayer3`, `SigmaLayer3`); a layer is never an order.

**Pi, Not, Sigma.** The one structure that is repeated at every order (Alec,
2026-11-17): a Pi, then the entries of the Sigma that read the opposite
channel (see *negation*), then the Sigma. The Pi and the Sigma are its two
*epistemic levels*.

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
Its meaning is a point of *conceptual space*. A concept is made in one of two
ways: implicitly, by a Sigma over the rows beneath it; or explicitly, by MTM,
from a statement such as "Felix is a cat", which makes a link of part to
whole or of member to kind, depending on the orders of the two.

**Symbol.** A row of a Pi above order zero: a conjunction of concepts of
lower orders. A symbol has no dimensions and "does not live in any space,
which is what makes its collection by sigma so set-like" (Alec, 2026-11-19).
It is made in one of two ways: implicitly, by a Pi, when MTM nominates what
recurs; or explicitly, by MTM, from a statement such as "Felix is my cat",
which says that two are the same. "Since implicit generally precedes
explicit, MTM should check for an existing fit first."

**Order.** How far a concept stands above the basic percepts. A concept of
order zero is read from basic percepts. A concept formed above symbols is one
order above the concepts those symbols read. An order is raised implicitly,
by any concept formed above symbols, and explicitly, by statements that
share a side: "Felix is a cat" and "Ciao is a cat" call for a concept of cat
one order higher that ranges over both (Alec, 2026-10-21 and 2026-11-19).
Order zero has room for as many concepts as there are basic percepts, and
each order above has half the room of the one below.

## Evidence and logic

**Lane, channel.** Every value in perception is stored as two numbers from 0
to 1: evidence for, and evidence against. "We are on a positive unit
hypercube with two lanes; evidence for and evidence against. When I am
talking about a negative number in the perceptual context, that is just a
shorthand for positive evidence against" (Alec, 2026-11-19). Zero on both
lanes is no evidence. *Channel* is the same pair as a Sigma reads it.

**Negation.** Three different things, which the model keeps apart.

- *Evidence against* is a value on the against lane. It is something
  perceived: the `y` of `archery` is evidence against `archer`.
- *A Not entry* is a weight of a Sigma that reads the opposite channel: a
  weight below zero on the for channel, or above zero on the against channel.
  At the basic level it is learned from omission, a thing expected that did
  not come. The Pi has none.
- *The operator `not`* of a sentence or of a thought exchanges the two lanes
  of what it is put on (`FieldLogic.reduce`).

**Ternary logic.** The logic of one signed value: true, false, and neither.
A concept in a reading is ternary. On values between the three, the layers
take products and the squashed sum.

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

**STM.** Short-term memory: the chunks held while a sentence is read and
thought about, some of them on a stack. The top of the stack is the most
recent. In the source: `ShortTermMemory`.

**MTM.** Medium-term memory: where things are tried before they are kept. It
counts what recurs and nominates rows of the Pi; from statements it makes
symbols and concepts, looking first for one that already fits. What it holds
is easily promoted and easily forgotten. In the source: `MediumTermMemory`.

**LTM.** Long-term memory: what is kept. Its rows hold ideas, each with an
entry of the concepts of its moment (`LTM0`), and relations between concepts
(`LTM1`). It is read by content: a query matches the top of the STM stack
against the rows, and the best row is pushed onto the stack (Alec,
2026-11-19).

## Attention

**Look.** "One act of attention while a sentence is read. It filters the
perceptual field by a saliency map from attended concepts, which will filter
on both the what/where of conceptual space, and carries those selected
percepts into the activation space on all existing concepts" (Alec,
2026-11-19). In the model the filter is a number from 0 to 1 at each percept
of the sentence, and the first look of a sentence takes all of it at once.
