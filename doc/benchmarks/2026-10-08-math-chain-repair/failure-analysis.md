# Observed blockers in the declared repair measurement

The measured source remains frozen. No failed training has been retried or
replaced. These notes distinguish observed stack traces from possible causes.
All thirty attempts failed before completing epoch one: 27 compose legality
assertions, two out-of-range action indices and one sentence-reader occupancy
exception. No attempt reached evaluation.

The first paired start failed in all three conditions at the compose
exploration assertion in `bin/Language.py`: `compose has no legal alternative
at the selected exploration round`. The call path is `sentence_pair` →
`compose(cache, exploit)` → `choose_operation` → the operation layer. It is
not a held-out answer failure: no epoch or evaluation completed in these
attempts. The assertion checks the legality of the selected action after a
replay override; the message alone does not distinguish an empty legal menu
from an illegal replayed action. `select_logits` already falls back to the
unmasked logits when masking leaves no departure.

A possible cause to examine after review is the identity of replayed
candidates. `Interpret.bind` prunes columns using availability across the
whole batch (`available.any(dim=(0, 1))`), and conditionally appends mint,
open and unchanged alternatives. The compose replay carries flat action
indices. A change in another row's available columns can therefore change
the numerical layout of the menu. This is a static audit finding, **not a
reproduced diagnosis of the failed trace**. No failing learner was replayed
to investigate it.

Expectation-only start 5 supplies a second observed compose failure:
`index 54 is out of bounds for dimension 1 with size 47`, at
`select_logits`'s `selected.gather(1, action[:, None])` (`Language.py:7194`).
This follows the replay override on the same exploration call path. The
current menu index is demonstrably invalid in that trace; this strengthens
the case for investigating replay/menu identity. It does not, by itself,
establish which conditional pruning changed the menu.
Zero-budget start 9 failed at the same gather with index 36 and menu size 35.

The answer-and-expectation attempt for paired start 2 failed at the
one-or-three-slot assertion. Its recorded path is `ThoughtClosing.close` →
`thought_pair` → `ThoughtAnswer.score` → `_sentence_reader_error` →
`SentenceEndState(value)`. A thought's scored meaning reached the completed
sentence reader with an unsupported occupancy. The saved exception does not
include the exact role mask; a two-slot value is not established by the log.
This differs from the
development failure at the ordinary clause commit; the default sweep did
not cover this particular unforced scored thought.

The observers retain partial question and credit observations. A nonzero
credit gradient does not establish a correct binding or learned chain.
`failed-batches.json` reconstructs the next scheduled batch after each saved
successful-sentence count, using the retained presentation and the unchanged
batching function. It does not replay a model or identify a particular
offending stream. For answer-and-expectation start 2, that batch contains
eight ordinary `what is y ?` questions.

Final chooser displacement and held-out results are unavailable for a run
that exits before the frozen driver saves them.
