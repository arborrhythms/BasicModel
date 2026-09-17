# Future work

Cross-cutting items that are decided in direction but not yet specified or
built, with the decision that opened them and the document that owns the
detail once it exists. Section-local future-work lists elsewhere
([Language.md](Language.md#future-work-nouns-from-partspace-adjectives-from-wholespace),
[SymbolFirewall.md](SymbolFirewall.md#future-work-out-of-scope-for-this-pass))
stay where they are; this file indexes the items that span documents.
Items marked **urgent** block ordinary operation before long.

## 1. Operation profile: optimal versus human

**Decision (Alec, 2026-09-16).** Parameterise optimal versus more-human
operation so that both can be explored in one model. One `<operation>`
element in `model.xml` (`optimal` | `human`, default `optimal`) selects the
human-like behaviour for every item that
[Philosophy.md](Philosophy.md#discrepancies) marks **profile**:

| Item | `optimal` | `human` |
|---|---|---|
| Default belief | Cartesian: comprehension registers at trust `0`; provenance asserts | Spinozan: comprehension asserts at provenance trust; unbelieving is a later revision |
| Verbatim retention | derivations kept losslessly | derivations decay; fused points persist |
| Forgetting | capacity wall, recency only | forgetting model (§2) |

The profile is a switch over documented behaviours, never a hidden
heuristic; each behaviour has its own test under both settings. Params.md
gains the element when the first profile-dependent behaviour lands.

## 2. Forgetting and consolidation (urgent)

Specified in [the forgetting spec](specs/2026-09-16-forgetting.md)
(2026-09-16). Summary: `<ltmCapacity>` is the limit; a pass runs when the
store is nearly full, never mid-sentence and not more often than a minimum
interval, and deletes the lowest-valued unprotected rows down to a target
occupancy. Value combines trust (high trust preferred), utility (a row that
is easily deduced from other rows is not worth keeping: low expectation
surprise for ideas, one-step derivability for relations) and contribution
to luminosity (dissonant rows are forgotten; full coverage gain is optional
because it is expensive). Relations over deleted rows cascade; references
are remapped on compaction. The human profile adds an age term. The earlier
direction here, that isolated rows should be forgotten, is withdrawn: an
isolated row can score high on all three terms.

## 3. Lossy reconstruction trace; inversion as learning

**Decision (Alec, 2026-09-16).** Parts of the reconstruction trace will be
dropped over time, so inversion is a learned approximation from a lossy
trace, not an exact replay. With a complete trace reconstruction teaches
nothing; the loss is what gives the reconstruction objective its role in
learning.

To specify: which parts of a derivation are dropped first (leaf identities
before structure, or the reverse), on what schedule (age, use, profile),
what the reverse chain must recover from the fused point alone when the
trace is gone, and how the reconstruction loss is weighted between a full
trace and none. This is the verbatim/gist asymmetry (Sachs 1967; Reyna &
Brainerd 1995) made dynamic, and it shares its schedule with §2.

## 4. Context across documents

The hard reset at a document boundary clears transient context. People
carry context across texts and conversations. To specify: what survives a
document boundary under the `human` profile (the expectation observation
view, priming heat, the What interaction slots), how a new document's
first sentences are scored against a carried context without penalising a
genuine topic change, and whether the boundary becomes a learned soft
signal rather than a cursor fact. The current per-row document key and
clause-scoped reset in the integrated plan §8.3 and §11 are the base.

## 5. N-ary META and interpretation-time discrimination

**Decision (Alec, 2026-09-16).** META concepts generalise over more than
two concepts: several words and several objects (synonymy and polysemy),
and the discrimination among them happens at interpretation time, from
context, not at binding time. The two-truths spec §3.4 records the
decision; the one-row-per-word law of the binding table is replaced by the
n-ary META. To specify: the META row's member set and its allocation,
what `deref(word)` returns when a word belongs to a META with several
objects (the member set, ranked by context), where in the seal the choice
is made (the chooser that already selects readings), and how a wrong
choice is credited by expectation and reconstruction.
