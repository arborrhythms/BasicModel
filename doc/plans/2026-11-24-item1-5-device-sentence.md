# Item 1.5 proposal for a device-resident sentence

**Decision requested.** Approve a comparison-only reader for recorded grammar actions. It would run beside the current interpreter. Switching a live grammar consumer or MTM writer would be a later decision, informed by the comparison and reviewed with the 1.6 changes.

## Sentence boundary

Staging may admit input bytes, encode static grammar rule names and allocate capacity before execution. From wide-open attention through the single whole-field reconstruction, narrowed looks, candidate scoring, selected backward, optimizer step and retained close, decisions and values stay on the device. The read benchmark should cover the entire execution entry, including its called backward and close. Post-sentence reporting and checkpoint serialization may copy results out. Any pre-existing postprocessing that participates in a decision or optimizer update must be inside the measured boundary.

A strict capture probe on the ordinary four-page reading update stops in `InputSpace._lex_batch` when it turns a tensor into a NumPy array and calls `tobytes()`. Where a host source exists, input admission can carry its actual encoded, clipped bytes forward with the staged slab. Inputs that originate on the accelerator still need a device-native byte path. Preserve ASCII substitution, byte limits, embedded NUL handling, packed-row ownership and provenance. Count admission time in end-to-end measurements. Treat any content, grammar or fixation decision made after admission as sentence work.

## Grammar programs

Keep the current `(kind, local, word)` action encoding. Give the existing trace tensors a stream and path axis. Alongside actions, keep valid bits, stack top, stack seats, two child-action indices, parent action and leaf occurrence. A shift pushes its observed leaf, a transpose exchanges the two existing stack seats, and unary/binary operations take the operands specified by the current rule, including bind and unary offsets. Pre-stage static rule metadata from the configured grammar. Read trainable weights from their live parameters on every update so each chosen operation retains its gradient. Evaluate operators on their valid rows using the same arithmetic and tie order. No padded lane may consume a random draw, advance a clock, receive a gradient or execute an inadmissible operator. Make the same trace available to composition, comprehension and inverse generation. The first oracle is the existing `ReadingIndex` on recorded actions. In comparison mode a per-lane error flag and earliest failing action are checked outside the measured execution. Live use also requires preserving the current errors before any persistent write.

Let `S` count independent sentence streams, `P` count candidates including exploit, `T` bound the staged action trace, and `W` count available word slots. The prototype would use the following scratch arrays. Use 64-bit indices for gathers and preserve the stored action dtype. A validity bit distinguishes an absent seat wherever zero is a valid address.

| Array | Shape | Meaning |
| --- | --- | --- |
| Actions and valid steps | `[S,P,T,3]`, `[S,P,T]` | The existing trace plus its occupied prefix for each candidate. |
| Stack and depth | `[S,P,T]`, `[S,P]` | Action indices in the current forest, in the Python stack’s order. |
| Children and parent | `[S,P,T,2]`, `[S,P,T]` | Ordered operand seats and the consuming action. A transpose records the two exchanged seats and leaves their parents alone. |
| Word to shift | `[S,P,W]` | Action that introduced each observed leaf. Validation catches a repeated or missing shift. |

For example, shifting word 0 at action 0 and word 1 at action 1 leaves stack `[0,1]`. A transpose at action 2 gives `[1,0]`. An ordinary binary reduction at action 3 then records children `[1,0]`, assigns parent 3 to both shifts and leaves root `[3]`. Padded steps after action 3 do nothing. The arrays index the existing trace and value slab. They add no learned parameter or second writable tree.

Time remains sequential. A forecast at look k uses the opening and observations through k−1. After choosing one joint what/where request, read only the selected extent of observed bytes. The sentence has one whole-field reconstruction and keeps the existing clock cadence. Independent streams or candidates may run together at the same optimizer version. Preserve the current draw order and the comparison policy’s explicit dependence on candidate costs.

## Memory, ownership and checkpointing

Use stable tagged addresses with separate domain, identity and validity columns. Physical seats may be reused. A token row holds its current native/transient/symbol IDs, count, order, activity and pending state. Ordered relation rows hold its mentions, timelines and events. Each event stores its role, verb, time and typed references to its actual operands, including before, basis, modifiers, case value and verb meaning. Snapshot an operand at the same point as the current writer so later learning or slot reuse cannot change a historical event. Preserve `None`, zero, and distinct address domains.

| Table | Columns and lifetime |
| --- | --- |
| Token seats | Logical token ID, occupied/kept bits, native/transient/symbol addresses, kind, order, count, time, mass and optional where. The candidate’s support uses the existing feature entries. Preserve reading history, stream clocks and refinement values where a reader distinguishes them. |
| Ordered edges | Source handle, edge kind, target handle, stream, timeline and ordinal. The ordinal preserves the order of mentions, events, modifiers and readings across save/restore. |
| Events | Occurrence handle, verb, role, original time, teaching flags and operand handles. The basis tag distinguishes an integer basis from a tensor basis. |
| Operand storage | Group compatible snapshots by dtype and geometry. Each event owns a reference to a snapshot until the last live event or teaching operation releases it. Reusing an STM seat cannot retarget that reference. |

Follow the wiring plan’s single table for trial and promoted rows. A provisional handle includes its stream, path and local identity until publication. Promotion changes the winner’s retention mark in place. If two winners have the same path-local number, resolve their distinct handles in the existing stream-close order and update every retained reference consistently. Store ordered edges for mentions, event links and modifiers. Snapshot tensors retain their existing dtype, shape, storage ownership and gradient boundary. Presence bits preserve optional metadata through a checkpoint round trip.

Share the read-only opening. Record each candidate's writes, references and proposed allocations in private, ordered tensors keyed by stream and path. A later look sees its own earlier provisional records. After winner selection, replay only that stream’s records in the current close order. Consume the selected backward before publishing new trainable entries. Preserve intentionally paired policy terms, optimizer ownership, external-reference expiry and clock advancement. A rejected candidate can never mutate canonical tables.

For a two-stream test with exploit and three explore paths, force stream 0 to keep path 3 and stream 1 to keep exploit. Give the two losers distinct early writes, including an event referencing a provisional operand. Later looks must see their own earlier writes. After close, compare the winners with isolated runs from the same opening. Check that the losers’ provisional handles are unreachable, their gradients through retained state are zero, and the existing policy comparison still receives its intended gradients.

Keep the current checkpoint adapter at `MediumTermMemory.state/restore`. Convert old records at load and emit the existing records on save. Introduce no smaller limits on legal sentences, events or metadata. Size scratch from the staged batch, existing trace geometry and reachable records. Where a bound cannot be derived, the first prototype must establish lossless overflow handling before switching the live writer. A retry would require the same opening and RNG and must precede any persistent write, including an optimizer update.

## Acceptance and rollout

1. Run the device grammar reader in comparison mode on actual programs. Compare every action, operand, parent, word order, typed reference, output and gradient. Include mixed path winners, absent operands, padding, transpose, unequal lengths and errors.
2. Compare the MTM read view against the checkpoint and Python records. Exercise promotion, lapse, seat reuse, pending events, history snapshots and save/restore across versions. Switch writers only after mixed-winner replay agrees on STM, LTM, clock, optimizer, RNG and gradients.
3. Exercise the complete ordinary training entry with strict fullgraph capture and instrumented execution. Count explicit reads, native scalar/equality operations, device-to-host copies and data-dependent output sizes. Require zero in-sentence synchronization on the actual accelerator. A captured subroutine alone cannot establish this. On Metal, audit the host-binary64 trust update and the clock-band rotation in `RowTime.how_far_back`. Any device replacement must preserve their numerical contracts.
4. Repeat fresh reading, both verbal XOR boundaries, numeric XOR and raised-class gates. Repeat the matched sentences/s and time-to-accuracy measurements. Do not change acceptance thresholds or force fixations to obtain capture.

The integration points for Claude are `ReadingIndex`, `Extension`, `Comprehension`, `WholePaths`, `CandidateGrowth`, `MediumTermMemory` and the reference/grounding writers. The first prototype can share checkpoint fixtures with 1.6 without changing their format.
