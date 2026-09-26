# Item 9b: starting contracts, September 25

Work begins from published BasicModel `00a5ded4`, after the
[accepted item 9 parity correction](../2026-09-25-item9-bank-sync/README.md).
The full [latest plan](../../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md)
was read, including section 4e. This is a failing development baseline, not an
implementation landing or a learning-gate result. No production source changed.

## Baseline

The [ten probes](baseline/probe.py) finish in 7.8 seconds, with **6 passing and
4 failing**, one CPU worker, an 8 GiB cap and a 0.51 GiB peak. The
[result](baseline/result.json), [log](baseline/worker-000.log),
[source manifest](baseline/source-manifest.json) and
[verification summary](validation-summary.json) preserve the evidence.

| Contract | Result on the published runtime |
|---|---|
| Checkpoint preserves concept id, definition, dictionary and native poles across modes, aligned binding | Passes in both directions |
| Same checkpoint contract, mixing binding | Serial to parallel passes; parallel to serial raises a store-shape mismatch |
| One active percept reaches all twelve definitions referring to it | Fails: the ninth concept is missing after the eight-row selection |
| A property observed true and false inside one bracket reads `(1, 1)` | Fails: negative evidence is lost |
| Narrowing to the observed-false occurrence reads `(0, 1)` | Fails: negative evidence is lost |
| Narrowing to true, an empty bracket, and padding-only input | All three controls pass |

The checkpoint fixture registers the real ConceptualSpace as a module, so both
its tensor state and structural sidecar go through `save_weights`/`load_weights`.
These are storage and native-read probes; they do not prove that learning in one
mode transfers to the other. The mixing failure comes from the mode-dependent
allocator sizing: this fixture's parallel taper totals 60 rows and its serial
inventory has 64. The checkpoint expansion path permits the aligned binding;
the production configuration uses aligned binding and both directions pass.
The mixing failure remains visible, rather than being generalized to production
or removed from the test.

The [initial eight-case run](initial/result.json) had four passes and four
failures. Its [exact probe](initial/probe.py) and manifest are preserved. The
second run adds the production aligned-binding cases and supplies the matching
model binding/capacity metadata; it retains every initial assertion. Neither
run pins a random seed or marks the failures expected. All 650 existing source
files match the published review manifest; only the new probe differs.

Reproduce the current baseline with:

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py test/test_item9b_field_contract.py --workers 1 --memory-gib 8 --batch-size 10 --max-files 1 --run-dir output/item9b-field-contracts-rerun
```

## Implementation seams and remaining work

The field work starts at `_bind_attended_concepts` and `cs_read_memberships`:
remove definition truncation, pool presence and observed complement before
definition folds, and replace retained conceptual position pairs with the
field's single bracket and interval. Percept events retain their positions for
attribution and refine-before-raise run counts. Preserve the native XOR,
exact-zero and serial reconstruction gates. The current probes cover the
readout contract only; pooling before a multi-literal fold, event attribution,
checkpoint coordinate removal and the native learning gates still need probes.

`interpret` is the mandatory per-word serial translation from an admitted word
row to its object row, including provisional testimony and reverse lexicalization.
It replaces `create_word_object_meta` and its callers. Section 5's explicit
"not optional" decision governs the stale later phrase "routed by the chooser."

Section 4e proposes fixed physical capacities, logical admission within that
reserve, stable where-space slices, and deletion of geometric growth,
optimizer migration and `maxVectors`. The audit located `RadixLayer._grow_to`,
`Codebook.grow_to`, their ownership/migration callers, and the retired
`where_offset` stubs. WholeSpace and symbol dictionaries already have some
capacity-freezing machinery. The production ConceptualSpace dense allocation
and the selected physical capacities still need measurement and reconciliation.
The plan still marks 4e, the mode schedule and the erosion gate as proposals
awaiting Alec's answers; the decided field and `interpret` work can proceed.
