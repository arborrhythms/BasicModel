# Grammar-only reference sites — 2026-10-21

Status: contract source prepared for its first freeze; no gates claimed yet.
Base: `bb1678d170c8b7e98a8c6e2b1abe3250887d0087`, whose production,
test and config files match published `cf61b5678`.
Authority: reconstruction spec §§10, 12–13. This is the first commit of
item 3.5 line 1; the reconstruction term is not implemented here.

The journal uses the reading's effective stamp (reference order where set,
otherwise leaf order). An order-one participant leaf supplies a provisional
bind request with its own content. Explicit formations and rule-declared
modes retain ownership. Both recovered-site paths require that participant
leaf and retain its content and order. Interpretation's identity fallback
is unchanged. Language and ReferenceContext no longer accept the lexical
site flag. Marker decisions use argument occupations or the closing plan's
surface position. PartRead uses only the symbol's own occupations.

Authorized fixture ports:

- `test_g2_generated_leaf_without_object_uses_same_resolver`: its word's
  reading is order one; the four lexical checks are replaced exactly as
  §12 directs. All antecedent, evidence, forward-name and emission checks
  remain.
- `test_selected_open_formation_keeps_one_site_and_its_objectless_cue`:
  the source word reads at order one, and the lexical empty-content check
  becomes equality to its own leaf. Open-role, single-site, leaves and cue
  assertions remain.
- `test_unary_bind_keeps_the_objectless_word_site`: the word reads at
  order one; the lexical empty-content check becomes equality to the leaf.
  The forward-name and bind-mode assertions remain.
- `test_reference_arrivals.equality`: its existing named open subject is
  supplied explicitly as an order-one reading at the equality's participant
  position. This supplies the source leaf that recovery now requires. No
  assertion changes; its own supplied content remains zero. All users of
  this shared semantic fixture get that same reading.
- `test_forced_ordinary_pending_premise_in_both_orders`: provision x if
  needed and set each of its readings to order one in `reference_orders`
  before the first premise. No assertion changes. `ForcedGrammar` and the
  live item-0 pending-premise certificate are unchanged.

Added gates exercise the same word through the actual journal and close at
orders zero and one in participant, marker and predicate positions, and
check that retained argument classification neither reads definitions nor
uses the object-type veto. No occupation copying has been added preemptively;
any need for it must be identified from a gate as §12 specifies.

The first frozen source will run the bind-thought, participant, arrival and
reference-close files with captures and stop on the first failure. Both
pending-premise orders use fresh initializations, full arrival captures and
the keep probe from the outset. The probe records x's actual composition
stamp in each trial. Every outcome remains beside its source hash. A failure
under the forced order stops the round for Claude and Alec. No publication,
full sweep or learning claim is made by this receipt.
