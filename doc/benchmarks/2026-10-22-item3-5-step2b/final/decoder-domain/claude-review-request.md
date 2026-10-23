Review frozen BasicModel main source 6a36b7ceeb5a5da9ae6019f62c59c16d27f7f1ce after your N1 HOLD. You are the explicitly requested Claude reviewer. Read-only: Read, Grep and Glob only; do not edit or execute commands or tests. Return LAND/HOLD with concrete remaining findings and file/line evidence. No learning claim.

Governing plan: doc/plans/2026-10-21-step2-one-ltm.md §§36,39–42, dated fixes §§43–66. The user explicitly says §39 supersedes earlier stops: document each fix and every assertion port, stop only to reverse an explicit Alec decision. We took your recommended N1 option (a) as preservation of the earlier bound-word terminal contract, without reversing an explicit decision; option (b)'s broad behavior is withdrawn.

Current receipt: doc/benchmarks/2026-10-22-item3-5-step2b/final/decoder-domain/README.md. Read gate-summary.json and frozen-source.json beside it for complete results from this hash only. source-audit.json, module-coverage.json, expanded-scope.json, assertion-ports.md/assertion-ledger.json, fixture-ledger.json and source-diff.patch record every production/test change since Claude's unchanged §38 baseline bb0045d7a. n1-repair.patch is the exact change from the source you just held. No code or test has changed since the freeze. All three groups reran, including four additional direct reader consumer files. Nothing from 867452a79 or any older source is promoted.

Your N1 review is ../word-form/claude-review.md; the earlier B1–B3 review is ../landing/claude-review.md. You accepted B1–B3 and the earlier follow-ups on 867452a79. Their implementation is unchanged. The narrow change in Spaces.word_surface_for_row now returns None whenever definitions.objects(identity) is nonempty. Other objects and objectless words continue through the prior Interpret.lexical_form path. Thus a word cannot spell through a second definition naming it, ordinary bound words do not gain terminals, and its native code still derives from its own perceptual parts. The captured failing RNG completes all three joint-objective presentations; see ../held-content/rung0-narrow-replay/run.json and working-source.patch. That replay is diagnostic only.

Exact same-round regression port: the loop's surface == b'12' for both word and object is replaced by word surface is None and object surface == b'12'; an additional assertion checks the word had no surface before the second definition. Both exact native-code decode checks and both definition-link assertions remain verbatim. §65 and the receipt declare that this corrects §64's rejected broad contract: it is not presented as a setup fix with unchanged assertions. No other assertion, fixture or retirement changed for N1. The cumulative ledger still names the original three P3 adjacency-only retirements; no new skip.

The four direct reader consumers you named are included: radix_layer_reverse, active_decode_candidates, echoic_decoder, tied_operator_reconstruction. The earlier six added form/definition reader files remain included, as do every direct write_clause/commit_reading file and the full eight-file native set. Every changed test file is selected. Source audit covers 65 source/config/test files.

The receipt explicitly declares your nonblocking notes: interrogative completed Parts now keep live mode and no longer become answer seeds; parsed.mode wins when supplied; invalid thought-shaped stored descriptions also receive no extra holes under the signature guard while live thought records still raise; predicate ports check actual links/content rather than global uniqueness; the narrow-idea diagnostic is a second sighting of the same row; §64's added ordinary word/object observer-key duplicates disappear with the narrow rule. Your five indirect-consumer files remain for step 2's one end full sweep, not claimed as scoped evidence here. Derivation, PartRead, PartClosing, close-time minting and MTM remain unchanged. The three §41 dispositions remain as you accepted them: conservative forgetting stage barrier with forget_pending, question refusal by record kind, complete stored-mode consumer list.

Please check specifically that the N1 repair actually preserves the prior ordinary-word terminal behavior and fixes the dual-role native form fault without broadening other consumers; check the exact regression port and fresh gate evidence. If LAND, publication is main's push plus the WikiOracle gitlink bump, then Codex's requested §§34–35 identity review. Identity implementation awaits Claude's spec. No second full sweep.


§66, subsequent added-consumer failure: the narrow source 85da7c14d completed
1,142 passes/nine existing skips/two echoic-decoder failures, and was
withdrawn without a review request or publication. The fixture supplied
[[2,1],[1,2]], outside the existing decoder_eligibility cube domain; the
inverse correctly rejects the binary split. This unit fixture does not
call any word-surface or LTM reader. Seven relevant production function AST
hashes are identical to the pre-2b §38 baseline; see
../narrow-form/echoic-domain-original/production-comparison.json.
Both captured initial RNGs reproduce the failure and retain the actual
operands, root, output and traces. Changing that literal alone in the
diagnostic to [[1,.5],[.5,1]] passes both eager and compiled cases with
every assertion verbatim: ../narrow-form/echoic-domain-cube/run.json.
The applied test port is echoic-fixture-port.patch beside this receipt:
that literal and a domain comment only. It preserves directions, composed
root, shape/capacity, budget eight, policy, operators, eligibility, the
existing second-walk monkeypatch and all value/trace/no-gradient assertions.
It adds no forcing or menu mask. Production code is byte-identical to
85da7c14d. The existing-fixture port is explicitly under §39, not claimed
as a same-round setup exception. Please verify that diagnosis and port.
All three groups have run again on 6a36b7cee; no evidence from any previous
hash is promoted. All 92 selected files, including all four consumers you
requested, remain in scope. No new assertion changes or retirements for §66.
