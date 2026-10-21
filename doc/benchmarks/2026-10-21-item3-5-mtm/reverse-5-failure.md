# Frozen-source reverse failure: R-selected arrival and conflicting equality

Source: `eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. Fresh reverse draw 5
fails the unchanged D2 assertion at `test_math_chain_section14.py:403`,
row 1 in document/stream 0. Its working read still names x. Both the
first-draw failure and exact replay are retained; no production or test
change preceded the replay.

[Replay validation](reverse-5-replay-validation.json) verifies identical
initial parameters, buffers, initial/model/private-generator RNG states,
source manifest, both trials' readings, costs, keep decisions, retained
rows, Part store, MTM and observed reads. Both outcomes are failures.
Full evidence: [first-draw keep capture](fresh-final-reverse-5-keep.jsonl),
[all 16 document tables](reverse-5-keep-tables.md), and the corresponding
`fresh-final-reverse-5-state/` and `reverse-5-replay-state/` directories.

## Both deciding comparisons in document 0

| Batch / sentence | Trial | R | E (excluded) | A | Keep |
|---|---|---:|---:|---:|---|
| 1 / `x is three.` | exploit / greedy | 0.036067377775907516 | 0 | 0 | discarded |
| 1 / `x is three.` | explore | 0.030056146904826164 | 0 | 0 | explore, `reconstruction` |
| 2 / `y is x plus two.` | exploit / greedy | 0.036067377775907516 | 0.0301087386906147 | 0 | discarded |
| 2 / `y is x plus two.` | explore | 0.032460641115903854 | 0.0301087386906147 | 0 | explore, `reconstruction` |

R+A equals R in each row. No answer targets are supplied. Compose
departures are 13 and 21 respectively.

In batch 1, both trials make the root an equality with pending x.
Greedy's retained complement has `subject_word_id=15` (three). Explore's
complement has `subject_word_id=13` (space): its composition contains a
product of surface groups, then a verb operation. The three word leaf is
still present in that composition. Neither trial has a reference-closing
request in this comparison. Explore wins strictly on R, and its retained
complement is row 0, address `-5467802707417032475`; the pending equality
is row 1, address `-5148403871654289876`. MTM remains empty after batch 1.

In batch 2, greedy uses the plus predicate for the pending x child.
Explore instead makes another equality with pending x, whose complement
is a space/two equality. This reading has one reference-closing mint for
the space child and opens the one MTM entry seen in the final capture;
there is no re-close or eviction. It writes the second x equality at row
20, address `-2656379121610037173`, with complement `('sym', 28)`.
The first equality's complement is an occurrence address. The reader
therefore finds two distinct positively supported complements for x and
leaves it open under the existing ambiguity rule, failing D2.

## Stop and decision

This is a real reverse semantic-certificate failure. It is **not** a
reproduction of the forward `matches(..., 'three')` assertion: all six
fresh forward draws passed, and this reverse draw fails the name-scoped
assertion after its second sentence. Its first comparison nevertheless
exposes the R-only choice of a differently headed `x is three` reading
that motivated the §17 stop instruction. The second comparison introduces
the conflicting equality, as in [reverse draw 2](reverse-2-failure.md).

Codex stops here and returns these readings to Alec for a contract or
mechanism decision. No assertion, scope, budget, objective, keep rule or
ambiguity rule is changed. No attempt is relabeled as a port or a pass.
The remaining reverse draw, eight-file set, decomposition arms and both
eight-seat MM gates have not run on this source. Claude's review and
publication remain pending; earlier-source results are not promoted.
