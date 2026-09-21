# Accessible-mind effects

The three grammars share numerical operators and a conceptual dictionary.
`<thought>` effects use the existing controller, chronological thought history,
conceptual activation carrier and ternary LTM store. There is no added language
model, interpreter, policy or semantic store. This is item 1c's implementation;
expectation's negative image is derived at the seal;
[ExpectationRetention](ExpectationRetention.md) gives its gradient and credit contracts.

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
| `part` on order-zero ideas | `Ops.part` vector residual in serial conceptual thinking; scalar support is derived from the same operands. No store read. |
| Higher-order `part`, or explicit `isPart` | Bounded taxonomy traversal produces graded symbolic inclusion. A missing path has value zero and retains any traversal incompleteness. |
| Open-role `part` | A native bound reference enumerates bounded taxonomy members with captured numerical payloads. An unnamed vector cannot offer this form. |
| `equal` | Support on the completed idea; no LTM capability. |
| `quantize` | An existing code, seeded into parallel knowing; no LTM capability. |
| `lookup` | Members from frames already held by `what`, seeded into knowing. |
| `exist` | Summed signed familiarity from indexed matching facts, clipped for support; bounded source provenance is retained without stored meaning frames. Observations do not certify truth. |
| `what` | One best matching, bounded, cued frame is retained in ordinary thought history. If none is retrieved, a declared nested question can use the same controller and meter. |
| `arma` | A detached prior estimate with role-presence logits; never a fact. Its negative-image treatment is subsequent work. |

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
bounded discourse chain and frames retained by `what`. It no longer reads a
slice of the most recently written LTM rows. Merely writing a fact does not
expose it to the chooser. Each retrieved frame owns detached values, so later
store mutation or compaction cannot alter the held content. Its occurrence
address remains provenance. Frame evidence survives thought-result checkpoints and holds its referenced
LTM occurrences alive while the owning history is retained.

Priming widens retrieval cues with rows whose multiplier is above the neutral
value one. A cleared all-ones mask supplies no extra codes. The code addresses
remain index metadata; their magnitudes never become chooser features.
Budget and closure pressure retain their explicit resource features and force
the existing bounded conclusion path at cutoff.

## The existing LTM writer owns the index

LTM retains its `[capacity, 3, width]` idea slots. New registered tensor columns
are `leaf_codes`, per-row/per-role `leaf_offsets`, `leaf_complete`, and
`index_stream`. Posting lists from `(code, role)` to store rows are derived from
these columns. They contain addresses, not another copy of semantic vectors.
The leaf column doubles its allocated capacity when needed; appends write only
new terms. Checkpoints serialize its used prefix, and reset/compaction rebuild
its extent and postings. Row-to-concept identity is cached by the allocator's
row owner at allocation and reconstructed on checkpoint load, so thought
effects do not scan the dictionary to recreate a reverse map.
[Writer and checkpoint](../bin/Layers.py), [index](../bin/MemoryIndex.py).

A canonical meaning supplies native leaf references and nested constituents.
An otherwise unprojected forward program supplies each sealed role's exact
leaf sequence from its completed forest. Neither route guesses the nearest
root code. When no derivation is available, the current generate MLP and tied
operators unfold the detached idea within a bound. Only a terminal emission
within numerical tolerance of a codebook row counts as a recovered code.
Unsuccessful unfolding leaves the role explicitly incomplete.

Cues include the bound roles' leaf codes, occurrence references, priming and
neighbors of previously retrieved frames in the same stream. The reader
examines at most K candidates and charges before reading each record. Scope
and bindings filter candidates; masked role similarity ranks them; contiguity
breaks equal-match ties. Other streams' observations and estimate/question rows
are excluded. `what` retains at most the best row. `exist` aggregates matching
fact support without retaining a frame. Fan increases examined candidates and
work until the cap is reached.

Checkpoint restore rebuilds postings from the columns. Legacy checkpoints
without these columns are reindexed after their real codebook owner is bound;
facts remain shared, while observations with unknown stream ownership are not
exposed as another stream's memories. Store compaction applies
the same row permutation to the index and preserves occurrence IDs. A live
codebook row-removal notification remaps its leaf addresses; removed codes mark
roles incomplete. Rows written before the codebook owner is attached remain
unindexed until binding fills the missing terms with actual codebook rows;
allocator IDs are never treated as row addresses. Existing recorded leaves are
preserved. Nested writes inherit their parent's stream. Reset clears the
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
recovery. The index can retrieve a row through its recorded leaf codes while
its fused idea still fails to regenerate them. This is a design limitation to
measure during further training, not a passing learning claim. The small probe
is also not a corpus estimate, multi-seed study or causal-utility comparison.
[Probe](../test/test_mind_generativity.py), [receipts](Testing.md).
