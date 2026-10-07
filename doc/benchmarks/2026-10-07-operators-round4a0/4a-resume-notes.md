# Resume 4a after Claude's address review

The paused 4a implementation remains in the original `WikiOracle/basicmodel`
working tree, unchanged. Its content-key function uses the same
`sentence-identity-v1` byte format as this candidate. No 4a gate campaign has
been started.

When combining the changes after review:

- Keep the address writer's `sentence_content_keys` buffer and compute the
  content key before the upsert. Remove 4a's provisional post-write
  `bind_sentence_content` step; changing a key after addressing would break
  identity.
- Preserve the 4a `meaning_layout` and bipolar operator changes while keeping
  4a-0's address, compaction and checkpoint handling. Do not register the
  sentence-content buffer a second time.
- The context mean should now see one retained row per source occurrence,
  with that row's refreshed timestamp. DEF rows stay excluded. Its identity
  code must use the content key, never the address or the shared format tag.
- Keep the address capacities when applying 4a's wider complements. The
  expectation target change remains the paused 4a change; it was not part of
  this isolated measurement.

The raw MM control's existing absence of sentence closing is recorded in
this receipt. It must not be confused with the grammar gates' four sentence
rows when comparing the two campaigns.
