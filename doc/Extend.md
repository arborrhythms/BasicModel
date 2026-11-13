# Extend

How a sentence's operators turn the concepts its words activated into the
field of understanding. Stated by Alec on 2026-11-12; quotations are his.

## Terms

- **Token and type, whole and part.** "For kinds, a token implies a type.
  For mereology, a whole implies a part."
- **The sentence-concept** is what a sentence's words compose to. "The
  sentence words can be transformed into a single vector in conceptual
  space, which does not (necessarily) exist as a single concept in
  conceptual space." Having no row of its own in the concept inventory, it
  cannot be referred to; it is held as a vector.
- **The idea** "is the intension of the sentence-concept": that vector.
- **The field** "is the extension of the sentence-concept": a set of known
  concepts, each with its trust on the two lanes, for and against.
  It is the symbolic activation at every order, tokens, their types and
  the types of those, and holds no grammatical operation
  ([Reasoning](Reasoning.md#symbols-the-field-and-the-one-hypersphere)).
- **The field of understanding** is the field a sentence is left with when
  every operator of its tree has been extended. "It is non-conceptual
  understanding in that not all order-zero concepts are necessarily
  symbolized (although in the current architecture all of them are)": a
  member of the field need not have a word.

## The four uses of an operator

"compose, think, generate, extend".

In the first three an operator takes intensions and gives an intension: "In
a linguistic context operators select the best match, they compose and
generate intentions." Compose builds the idea from the words, thought works
on ideas, and generate walks an idea back to words.

In the fourth it takes a set of active concepts and gives a set: "We need
those operators to generate an extension from an intension, which is easy
for known concepts, and then we can perform a selection. So the result of
the operation is the selected field, not the sentence idea or its
constituent word ideas."

## The operation

"So *every* operator needs an extend(): we start with the set of concepts
in the extension of the noun phrase, and then extend() it until we have a
new set, which is the field of understanding."

1. **A method of every operator.** `extend` stands beside `compose` and
   `generate` on the operator. The grammar file gains no section and no
   rule for it, and nothing is parsed a second time.
2. **It starts from what is active.** "extend should operate on the
   existing conceptual activations, or a top-k select thereof." The
   starting set is the extension of the noun phrase: the concepts its
   words activated, with their trust. The extension of a known concept is
   stored, so nothing is found by matching a vector against the code of
   every concept.
3. **It follows the tree compose left.** "if we have the STM tree, we can
   just iterate over its existing compose(), calling extend() on each
   operator to process the set of conceptual activations." One call for
   each operator of the tree, in the order compose applied them.
4. **A set in, a set out.** An operator of two operands takes the sets of
   both.
5. **The last call leaves the field of understanding.**
6. **An operation on the field, not a lowering.** "it really is an
   operation on the field of active concepts and not necessarily lowering
   the order of a concept." Only a determiner goes from a kind to its
   members.

An idea that is met without its sentence, one recalled from long-term
memory or one a thought concluded, has no tree in short-term memory. "We
already have a mechanism for mapping words to a sentence concept
bidirectionally. So if we walk the idea to the sentence (a WFF), the wff
can select the extension precisely." Its generate walk gives the tree.

## What each operator's extend does

Three rules cover most operators. Narrowing one operand by another is
`intersection`. Joining kinds is `union`. Negation is `not`, which swaps
the two lanes, or `non`, which withholds both.

| operator | when composing | extend |
|---|---|---|
| `intersection` | a field operator | the intersection of its operands' sets |
| `union` | a field operator | their union |
| `not` | swaps the two lanes | swaps the two lanes of the set |
| `non` | withholds both lanes | withholds both lanes of the set |
| `compound` | a modifier selects among its noun's cases | narrowing: the noun's set intersected with the modifier's |
| `adverb` | a modifier applied to a verb | narrowing |
| `verb` | a verb subtyped by its object | narrowing |
| `disjunction` | a probabilistic sum of activations | joining |

### Determiners

"Determiners do lowering/selection on their arguments, to either one or
multiple vectors." A determiner takes a kind to its extension and selects
there.

| word | operator | selects |
|---|---|---|
| "the" | `definite` | "a known identity from MTM" |
| "a" | `lower` | "a single subconcept": "'A cat' should select a generic 'cat', presumably; perhaps it can be a centroid?" One generic member, the centroid of the kind's subconcepts |
| "some" | (no operator in the grammar file) | "some subconcepts" |
| "every" | `universal` | "the entire extension" |

## What the field of understanding is for

- **Memory.** A row of long-term memory holds both sides of one
  sentence-concept: its idea, and references to the strongest members of
  its field of understanding with their signed trust. The row has one
  signed trust of its own, from $-1$ to $1$, which is how far the
  statement is held true or false; the idea is not scaled by it. "A
  semantic vector
  comes from the associated episodic concepts, which may or may not be
  enough to reconstitute the sentence verbatim."
- **Recall.** Extend is the mechanism of recall. A question is extended
  over its own tree like any sentence. One term's result over the rows of
  memory is one product; each operator of the question's tree then
  combines its operands' results as its extend says: narrowing
  intersects, joining unions, `not` and `non` act on the lanes, and a
  determiner selects one row, some or all. The operand a question leaves
  open selects nothing. A recalled row's idea is walked back to its
  sentence and extended, and the part of its field at the question's open
  operand is the answer. A row keeps only the strongest members of its
  field, so the products find the candidates and the walk gives their
  whole fields.
- **Reference.** What "the" selected is in the field. Resolving a
  reference is part of extend, not a search made beside it.
- **Saying.** A member that has a word is said by that word. A member
  without one cannot be said, which is the sense of "non-conceptual". Not
  every concept has a symbol: "we do not symbolize all concepts to begin
  with", and `interpret` is what creates one
  ([Language](Language.md#interpret-word-concept-to-object-concept-2026-09-25)).

## Regions (Alec, 2026-11-13)

"Like "cats have fur" should be a relation in a higher order space about
cat-things and things-with-fur. So every intention (e.g. the property
fur) will have an extension when applied to a set (things with fur) which
will look like a region in conceptual space (the furry things region)."

- A property is an intension. Applied to a set it has an extension, the
  things it holds of, and that extension is a region of conceptual space.
- A subject's set is a region as well: one thing, or the members of a
  kind.
- A sentence that says a property of a kind is a relation between the two
  kinds, in the space of the order above, where each kind is one object.
  It says that the first region lies wholly within the second.
- A claim is **pervaded** when its subject's region lies wholly within
  its property's region by such a relation, and it has an **instance**
  when a trusted object lies in both. Only the first carries a deduction
  ([Logic](Logic.md#existence-and-pervasion-alec-2026-11-13)).

## In the code

- The field is a `ConceptualField` (`bin/Understanding.py`): concept
  references, evidence on two lanes, and each reading's place and time.
- The two lanes are combined by `FieldLogic.reduce` (`bin/FieldLogic.py`),
  which defines `intersection`, `union`, `not` and `non` once.
- The operators are the `GrammarLayer` classes of `bin/Language.py`;
  `compose` and `generate` are their methods today.
- The centroid of a field is `FieldProjection.key`
  (`bin/FieldProjection.py`).
- `CompoundLayer` already narrows inside its compose, through
  `CaseSelection.select_cases`.

## Open

No operator has an `extend` yet. These points are not settled:

- `conjunction` meets two pieces of evidence in one use, which is an
  intersection, and joins two kinds in another ("cats and dogs"), which is
  a union.
- `preposition`: what it does to its object's set.
- `part`, `whole`, `equal` and `implies`: what a sentence that states a
  relation leaves as its field.
- `gloss`, `divide` and `descend` act on the places attention examines and
  are not sentence operators.
- The marker and carrier operators (`surface`, `sign`, `interpret`, `bind`,
  `lift`, `transpose`, `ask`, `generic`): proposed, to pass the content's
  set through, with `interpret` giving the object's set in place of the
  word's.
- How an operator of two operands tells its operands' activations apart:
  proposed, by their places in the sentence.
- Whether what "a" selects is entered in MTM, so that a later "the" finds
  it: proposed, yes.
- Which subconcepts "some" takes: proposed, the strongest of the narrowed
  set.
- The size of the top-k selection.
