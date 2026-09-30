# Round 4 review map

The runtime is frozen at [the final source manifest](ai-native-id-source.json).
No commit is made. The [receipt](README.md) carries completed measurements and
the [three remaining failures](full-sweep/failures.md); this map follows spec
§20's implementation order. The candidate is stopped for Claude's review.

| Step | Change | Saved failing probe | Repair / verification |
|---|---|---|---|
| Z + first AG | LTM occurrences own closed clauses and referenced phrases. Grammar predicates share slot identities without inventory rows. Capacity refusal is atomic. | [Closing identities](z-before/result.json), [full MM_grammar inventory](z-mm-before/result.json), [symbolization refusal](z-symbolization-before/result.json) | [Initial patch](z-initial-repair.patch), [standalone writer patch](z-writer-reference-repair.patch), [final item 7 cases](final3-item7/result.json) |
| AB | Restore all allocators before knowing fields, and publish the actual terminal owner. | [Three checkpoint cases](ab-before/result.json), [owner-label failure after the first repair](ab-after/result.json) | [Allocator patch](ab-repair.patch), [owner-label patch](ab-owner-label-repair.patch), [affected checks](ab-owner-after/result.json) |
| AC | Resolve objects through definitions and exclude word identities without assuming an order. | [Owned-answer probe](ac-ae-af-probes-before/group-00/result.json) | [Patch](ac-repair.patch), [development check](ac-after/result.json), [final production-width check](final3-ac-production/result.json) |
| AD | Count stored sentences by kind, excluding DEF rows. | [Reduction count](ad-reduction-before/result.json), [provisioning count](ae-discovery-before/group-02/result.json) | [Ports](ad-repair.patch), [both affected files](ad-after/result.json) |
| AE | Preserve field discovery when it has witnesses; otherwise admit the read word normally. Read the object's identity through definitions. | [Attended identity](ac-ae-af-probes-before/group-01/result.json), [empty-field discovery](ae-discovery-before/group-00/result.json) | [Patch](ae-initial-repair.patch), [four affected files](ae-after/result.json), [XOR comparison](xor-ae-comparison.md) |
| AF | Close an unindexed lexical predicate as its own unasserted LTM occurrence before referring to it. | [Nested binary relation](ac-ae-af-probes-before/group-02/result.json), [three-slot forest](af-before/group-00/result.json) | [Patch](af-repair.patch), [affected checks](af-after/result.json), [XOR comparison](xor-af-comparison.md) |
| AH | Read an ended idea's point from LTM and a concept's point from the inventory; a relative row has no single point. | [Native observation](ah-before/result.json), [forced owner probe](ah-point-owner-before/result.json) | [Patch](ah-repair.patch), [point-owner check](ah-after/result.json), [completed native observation](ah-native-after/result.json) |
| AI | Preserve the current differentiable root, publish complete batch metadata, refresh host word boundaries, establish clause state before compilation, and reset provisioning's what episode. | [All five original cases](ai-before/result.json), [compiler diagnostic](ai-diagnostics-before/group-00/result.json), [stale-boundary diagnostic](ai-c-trace-v2-before/result.json) | [Patch](ai-initial-repair.patch), [all five checks](ai-after/result.json) |
| AG memory | Recompute property support during backward, release the diagnostic graph, gather selected property rows before expansion, and compact absent predicate columns while retaining global identities. | [Saved-product probe](ag-property-before/result.json), [cache probe](ag-cache-before/result.json), [selected-row probe](ag-selected-property-before/result.json), [predicate-column probes](ag-predicate-before/result.json) | [Ownership measurements](memory-owners.md), [final graph-release gate](final3-core-gates/group-00/result.json), [final item 7 checks](final3-item7/result.json) |

Final validation exposed two further AI defects, both repaired after preserving
the failing assertion:

| Defect | Failing probe | Repair and verification |
|---|---|---|
| A host-only boundary refresh entered the compiled forward. | [Complete-forward failure](explicit-final/group-00/result.json) | [Patch](ai-compiled-boundary-repair.patch), [final complete-forward gate](final3-core-gates/group-01/result.json) |
| An absent native identity carrier became an all-`-1` metadata slab instead of `None`. | [Original unindexed-relation assertion](explicit-final2/group-35/result.json) | [Patch](ai-native-id-repair.patch), [entire unindexed file and adjacent checks](ai-native-id-after/result.json) |

The [six-port ledger](test-ports.json) retains both bodies, and its
[audit](final-port-audit.json) matches them to current source. Earlier item 7
ports remain in the prior receipts. No protected numerical assertion is
weakened. The [scope audit](protected-contract-audit.json) records the unchanged
out-of-scope operators, absence of item 7 seed calls and absence of configuration
changes in this round.
