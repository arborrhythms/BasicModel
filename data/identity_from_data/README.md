# Item 6 part 2: identity lessons, revision 2

Generate with `PYTHONPATH=test .venv/bin/python test/identity_corpus.py`.
The text files contain only an opaque document ID and sentence strings. All
entity labels, mention offsets, candidate lists, roles, counts and grading
answers are in separate `*.labels.jsonl` files. They never enter the learner.
The generated file hashes are in `manifest.json`.

The main 494-document training stream starts with four distinct witnesses of
each singleton, then maximum object support 1, 2 and 3. Every stage contains
all 64 kind/property/verb combinations for each participating noun position;
singleton rehearsal keeps every class present. These are **object counts**, not
a claim that a sentence contains at most three independent atoms. Compound
phrases also contain properties and verbs. The grammar's existing source budget
is unchanged. The 64-document confounded stream is a separate comparison;
each kind always has one property. Do not mix that stream into factorial training.

Stage 5 adds kind-specific follow-up predicates, pronouns with one/two/three
candidates, determiners that are correct on 90% of first/repeat mentions, and
same-kind pairs with distinguishing properties supplied later. The two-candidate
lesson fully crosses target recency, introduction role and initial surface
position (eight cells, four examples each per split). Ordinary and
object-topicalized clauses (`a cat saw a dog .` / `a dog , a cat saw .`)
separate grammatical role from surface order; equally frequent refresh mentions separate
recency from the introduction's role. The three-candidate lesson balances each
kind and target recency/position; its two subject introductions and one object
introduction give a 2:1 role marginal, which the report keeps visible.

Training and evaluation use different kind pairings and different full documents.
Shared words and kind/predicate associations are intentional. The evaluation has
150 documents, including the stage-5 one-row/two-row cases, four held-out
`a black ... the white ...` conflicts, and stage-7 documents in which the same
probe has different referents. Colour and size are exclusive attributes in this
small teaching world. Matching content alone does **not** prove that two identical
individuals are one; noisy determiner examples therefore have irreducible
uncertainty. The scorer reports them and does not impose a perfect-accuracy test.

The neither-candidate control uses the same visible document twice, once for
each hidden answer, with an uninformative `it glittered .` follow-up. Any
deterministic text-only policy has chance accuracy on the paired items. A
separate 16-document `biased_train` stream always rewards recency; its
16-document `reversed_eval` counterpart always opposes recency. The report's
`recent` control is an explicitly labelled **grading sanity control**, not a
trained model. A cold native biased run is a preflight, not a matched learned
comparison. For that comparison, both branches need the same trained warm-up
state and the same number of training presentations.

Run the native baseline, keeping all identity rules:

```
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python test/identity_measurement.py --epochs 0 --out output/identity-cold-baseline
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python test/identity_measurement.py --epochs 1 --train-split biased_train --split reversed_eval --out output/identity-biased
```

`--epochs 0` gives the cold baseline; `--epochs 1` trains once through the
selected raw-text documents, not a convergence claim. The measurement seed fixes one run;
there is no seed search, retry after a native failure, forced grammar, or binding
label supervision. Native failures and unreached cases remain in the denominator.
The observer reads selected order-one references from the chosen journal and
actual minted addresses from the existing writer; it does not guess identities
from cosine, word rows, or position. The metric interface accepts semantic
addresses, so later dictionary-based observers can use the same frozen grading.

Reports include unconditional accuracy, resolved coverage, conditional accuracy,
candidate availability/collisions, candidate count × recency × role cells,
first/second-mention row counts, document-context cases and both controls.
Unresolved bindings are not evidence of chance-level *choice*. A rule can be
retired only after its lesson passes without it; this corpus and baseline retire
none. The noun dictionary's column count is diagnostic, not atom-recovery accuracy.
