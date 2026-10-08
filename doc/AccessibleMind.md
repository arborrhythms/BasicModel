# Accessible-mind effects

The three grammars share numerical operators and a conceptual dictionary.
`<thought>` effects use the existing controller, chronological thought history,
conceptual activation carrier and ternary LTM store. There is no added language
model, interpreter, policy or semantic store. This is item 1c's implementation;
expectation's negative image is derived at the closing;
[ExpectationRetention](ExpectationRetention.md) gives its gradient and credit contracts.

Item 6.5's mechanism was accepted by Alec on October 7 after spec §10's
review; its learning gates remain pending. For §2.6.4, the latest sentence
source is freshly encoded through learned ConceptualSpace columns; the
predictor no longer unconditionally detaches that encoding. Its observed
target and older context remain detached. For §2.7.3, the existing bounded
situation and already cued frames enumerate individual columns. Their
continuity bands accompany the binding operand while column signatures and
the independence objective contain only content. Binding and minting are
variants of the existing global grammar choice. They do not invoke another
policy or an LTM search. See [the mechanism and limits](specs/2026-09-26-independent-components.md#9-implementation-proposal-october-7).

## Permissions and effects

`Subsystem` enumerates perceptual knowing, order-zero knowing, higher-order
knowing, serial thinking, priming, expectation, LTM, budget, meronymic access
and taxonomic access. A descriptor's read/write sets must fit its grammar's
permissions, and its write target must be in its write set. Structural contexts
carry a geometry-only conceptual capability. An injected LTM, taxonomy,
expectation or controller capability is rejected. Compose's current stream and
priming snapshot remain live owned copies; generate sees its own emitted prefix
and no priming. Thought operands, reads and writes detach.
[Permissions](../bin/AccessibleMind.py), [checked dispatch](../bin/Queries.py).

| Operator | Effect |
|---|---|
| `part`, `equal`, `implies`, `exist` | Conceptual content and an independent evidence pair over codes. |
| `isPart`, `isEqual`, `isImplied`, `isTrue` | Symbolic evidence and witnessing rows read by reference from LTM or the taxonomy. |
| `query` | The best matching cued row, retained as a serial result with its pair and occurrence. |
| `ask` | Attempt to fill an open reference; a nested attempt shares the same history and meter. |
| `not` | Exchanges a pair's poles, retaining its meaning; an uncancelled image concludes absence. |
| `gain` | Sets expectation gain for the next sentence; the predictor remains global. |

`quantize`, `arma` and `expect` are removed from thought. The old `what`,
`lookup`, `chunk` and `true` names raise with their replacements for one release.
Thought conclusions write only `inference` rows, with occurrence addresses and
witness references. Unresolved `question` rows belong to closing. There is no
generic writable-store capability on an executor.

`ThoughtResult` remains the immutable internal effect record and checkpoint
representation. Its serialization kinds describe payload shape; the descriptor's
subsystem write target determines ownership. The answer adapter reads those
owned records without another query. Serial effects live on `ThoughtRecord`;
code/set effects seed the existing row-aligned conceptual activation field.
Higher-order membership descends through the native decomposition used by the
pyramid. It can raise disconnected members without activating intervening rows.
The effect traversal and chooser's read of that field are metered.

## What can enter the chooser

The chooser attends live STM slots, its ordinary thought history, the row's
bounded discourse chain and frames retained by `query`. It no longer reads a
slice of the most recently written LTM rows. Merely writing a fact does not
expose it to the chooser. Each retrieved frame owns detached values, so later
store mutation or compaction cannot alter the held content. Its occurrence
address remains provenance. Frame evidence survives thought-result checkpoints and holds its referenced
LTM occurrences alive while the owning history is retained.

Priming widens retrieval cues with rows whose multiplier is above the neutral
value one. A cleared all-ones mask supplies no extra codes. The code addresses
remain index metadata; their magnitudes never become chooser features.
Open references gate conclusion. One `attentionBudget` covers every nested
attempt; the grammar scorer receives the expectation image as context. Its
paired owner credit uses the answer, next-sentence prediction error and work.

## The existing LTM writer owns the index

LTM retains its `[capacity, 3, width]` slots. Retrieval terms are derived by
unfolding each occupied numerical slot with the current generate MLP and tied
operators, within a fixed bound. Candidates come from the symbolic activation
that already drives semantic priming: values above its neutral value of one.
No per-sentence word list, leaf list or activation snapshot accompanies a row.
Only a terminal emission within numerical tolerance of a candidate counts as
a recovered code; unsuccessful unfolding remains explicitly incomplete.
Structural references follow the actual referenced field.

The durable index is inverted: `(code, role)` maps to store row addresses.
Checkpoint columns `posting_codes`, `posting_roles` and `posting_rows` serialize
those global postings; `leaf_complete` and `index_stream` retain completion and
stream metadata. Appends extend posting lists without copying all prior rows.
Row-to-concept identity remains the allocator's index, independent of semantic
vectors. [Writer](../bin/Layers.py), [index](../bin/MemoryIndex.py).

Cues include the bound roles' leaf codes, occurrence references, priming and
neighbors of previously retrieved frames in the same stream. The reader
examines at most K candidates and charges before reading each record. Scope
and bindings filter candidates; masked role similarity ranks them; contiguity
breaks equal-match ties. Other streams' observations and estimate/question rows
are excluded. `query` retains at most the best row. `isTrue` retains the ended pair and
trust separately; `exist` is the conceptual presence face. Fan increases examined candidates and
work until the cap is reached.

Checkpoint restore rebuilds postings from the columns. Legacy checkpoints
without these columns are reindexed after their real codebook owner is bound;
facts remain shared, while observations with unknown stream ownership are not
exposed as another stream's memories. Store compaction applies
the same row permutation to the index and preserves occurrence IDs. A live
codebook row-removal notification remaps its leaf addresses; removed codes mark
roles incomplete. Rows written before the codebook owner is attached remain
unindexed until binding fills the missing terms with actual codebook rows;
allocator IDs are never treated as row addresses. Old forward leaf lists are
dropped with a warning and their rows require unfolding under the bound owner. Nested writes inherit their parent's stream. Reset clears the
columns and their derived postings.

## Measured limits

The deterministic generativity probe uses distinct concept codes in each chain.
It writes one compound per condition, keeps
its stored idea fixed, then unfolds it with the current generate MLP and shared
operator after 0, 1 and 8 small training updates. Original trees and leaf codes
are used only to score the result; they are not unfold inputs. It compares both
codes and operations.

| Chain length | Composition depth | Exact codes and derivation at updates 0 / 1 / 8 |
|---|---|---|
| 1 | 0 | 1/1, 1/1, 1/1 |
| 2 | 1 | 0/1, 0/1, 0/1 |
| 3 | 2 | 0/1, 0/1, 0/1 |
| 5 | 4 | 0/1, 0/1, 0/1 |

These null results establish no learned compound generativity or chained-episode
recovery. Those historical measurements used the earlier recorded-leaf index.
Under the September 28 decision the index itself depends on generation from
the actual field, making weak generativity a retrieval limitation too. The
historical values are retained; they are not a passing learning claim for the
new index. The small probe
is also not a corpus estimate, multi-seed study or causal-utility comparison.
[Probe](../test/test_mind_generativity.py), [receipts](Testing.md).
