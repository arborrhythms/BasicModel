# Repair measurement: remaining review issues

The explicit thinking gate completed all 57 selected nodes: 56 passed and one
failed. The seventeen retained 11c nodes, both MM checks, the original 25
certificates and all twelve constituent-ownership cases passed.

The remaining node is
`test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer`.
Its call to `test_output_walk._model` reaches
`test_meronomy_ladder._build_ladder_variant`, which calls `torch.manual_seed(0)`
before constructing the model. The repair measurement's unseeded guard rejects
that call. No seed was applied, and this node did not reach its credit assertion
in the unseeded measurement. This is a fixture/measurement failure, not evidence
that the credit assertion failed. The normal default sweep passed its seeded
fixture; that does not substitute for the requested unseeded check.

The first receipt's observer suppressed and recorded legacy seed calls. This
repair removed the known 11c helper seeds and changed that observer to reject
any remaining explicit seed call. It missed the per-row integration's nested
helper. The frozen observer and its failure are retained, and the node was not
retried. A subsequent authorized repair should provide an unseeded construction
path for that fixture before measuring it.

The original post-report formatting error is fixed: the launcher loaded and
printed the complete 57-node result successfully. Its exit code 1 is the test
failure above, not a result-formatting exception.

The unforced configured run completed all 300 epochs. It opened one thought
episode, at epoch 154. The greedy path performed six `not` acts; the departure
replaced one with `gain`. Both spent all sixteen work units and remained open.
Their equal costs, `0.2989143913984299`, produced no policy gradient. No
two-query chain was observed in that run.

The shared operation scorer changed by L2 `1.4528905087504884`. Analysis of the
[saved tensors](mm-query-configured/movement-analysis.json) places all of that
change in its reduce and apply anchors (L2 approximately `1.40619` and
`0.36541`). The stop anchor, bracket anchors (including the thought anchor),
space prior and operator parameters did not change. This is measured chooser
movement, but it is not evidence of learning a thought chain: the configured
run had one exact credit tie and zero nonzero thought-credit gradients. Its
factory-reported final correctness was `[0.0]`; completion is a runtime result,
not an accuracy or learning success. The focused answer/expectation
certificates passed separately.

The standing thirty are a separate measurement. Their final results belong to
the main receipt. No outcome here authorizes a commit or claims a learning gate.
