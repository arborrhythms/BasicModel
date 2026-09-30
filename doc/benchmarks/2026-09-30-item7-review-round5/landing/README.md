# Item 7 accepted landing after review round 7

Alec authorized the three fixture ports and publication after the checks in
[spec §25](../../../specs/2026-09-16-two-truths-ideas-and-relations.md#25-hand-off-to-codex-claude-2026-09-30-what-to-change-after-review-round-7).
All three ports are applied. Their complete old and new bodies are in the
[ledger](test-ports.json), with the [exact patch](test-ports.patch) and the
[six saved failing proofs](saved-probe.json). The removed todo history is
[preserved verbatim](todo-item7-before-landing.txt).

The shared index fixture primes only inventory operands. The recency fixture
posts no inventory term for a row-free predicate. The vocabulary-renaming test
requires both operands to change and the grammar predicate to stay shared,
and copies only the operands' inventory content. Every other evidence,
retrieval, ownership and oracle-isolation assertion is unchanged.

| Requested check | Result |
|---|---:|
| The six failing cases | [6 passed](cases/run/result.json) |
| Their six complete files | [59 passed](files/run/result.json) |
| Every item-7 case | [202 passed](item7/run/result.json) |
| Documentation links, including todo.md | [Final prose check](docs/run/result.json) |

No case was skipped or stopped in the three code selections. Each selection
completed every collected node once. The bounded runner preserves its 8 GiB
worker guard and uses an 8 GiB pool for these targeted checks, within the
standing 24 GiB aggregate budget. No seed, threshold, model configuration or
protected assertion was changed. No new full sweep was run, as §25 directs.

The [712-file landing manifest](source-manifest.json) differs from the
[swept manifest](../final-source.json) only in the three ported test files.
All production code, configuration, other tests and supporting fixtures match.
The [original full sweep](../full-sweep/summary.json) remains **4,793 passed,
6 failed, 322 skipped and one expected failure**, all 5,122 cases completed
once in 89.6 minutes. Its six fixture failures are resolved by the checks
above; that historical receipt is not rewritten as a new green sweep.
All three round-4 and thirteen round-3 failures already passed in that sweep.

The [XOR table](../final-xor-comparison.md) remains visible: both XOR_exact CLI
gates pass; both XOR_grammar gates fail; MM_20M_xor exact round trips pass
**12/15**, with the three failures at .75, .75 and .25. The
[ten MM_grammar runs](../final-mm-grammar-table.md) all complete, with median
ending MSE .1066178977, against HEAD's .0696157217. Graph release passes at
7.631 GiB. These runtime measurements remain applicable because the landing
changes only test fixtures. The older eight-seed reconstruction campaign is
retained as round-4 evidence, without a re-baseline. The prior
[depth-three campaign stays red](../../2026-09-27-item7-5-landing/README.md).

Item 6.9 is next, in a new session, from its
[plan §8](../../../plans/2026-09-29-item-6-9-xor-grammar.md#8-hand-off-to-codex-after-item-7-is-accepted).
The reviewed implementation is accepted under §25; the remaining XOR_grammar
and intermittent exact-roundtrip work stays in 6.9. NonLayer and
ConjunctionLayer were not changed in this landing. Nanochat is untouched.
