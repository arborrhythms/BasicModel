# Wrong binding: forward first draw 5

**Stopped for Alec under §20.** Frozen source:
`f56fdf8ff337021ef744070cb3238c9c4fc7e8fa`. Forward draws 1–4 passed,
each with zero bound and sixteen named-open checked sites. Draw 5 failed
at the first checked site after the second batch, `x is three.`

The earlier x occurrence is `('ltm', 'address-v1', -779055165569261151)`
(row index 2). Its raw row still names x (symbol 15) with an open referent.
Its working read instead has reference `('sym', 13)`, no open slot and no
forward name. The captured word map identifies 13 as the space character;
`three` is 31. The token's kind is 13, with three retained occurrences and
count three. This is a **wrong antecedent and a mechanism failure**. The
certificate's assertion remains unchanged.

The old site's arrival rerun selects `('arrival', 2)` in both trials.
Its resolver features are:

| Feature | Value |
|---|---:|
| Candidate coverage | 0 |
| Request coverage | 0 |
| Native prediction | 0.25 |
| Prime | 0 |
| Form | 0 |
| Negative normalized recency | -0.44444445 |
| Candidate role / 2 | 0 |
| Request role / 2 | 0 |

The resolver policy's captured hash is unchanged from initialization. The
winner's logit is **0.25**, probability **0.12150069**; the dummy has logit
**0**, probability **0.09462483**. The explore trial departs at the new
mention's arrival choice, selecting `('arrival', 4)`; it retains the same
wrong choice for the earlier site.

| Row 0 trial | R | A | R + A |
|---|---:|---:|---:|
| Exploit | 0.0360673778 | 0 | 0.0360673778 |
| Explore | 0.0360673778 | 0 | 0.0360673778 |

The diagnostic expectation component is 0.0231600832 in both trials and
does not enter the keep. All eight rows keep greedy on a tie in this batch.
Both trial readings, full menus, features, probabilities, raw/working
reads, initial RNG, private generator state and failure store were captured
from the outset. No replay or replacement draw was needed for this failure.
The exception prevents a complete bound/open certificate for draw 5;
the receipt does not invent counts for unchecked sites.

- [Structured diagnosis and both row-0 menus](wrong-binding-diagnosis.json)
- [Complete keep probe](fresh-final-forward-5-keep.jsonl)
- [Source audit](source-audit.json): all five manifests match all 828 frozen files
- [Initialization fingerprints](fresh-initializations.json): five distinct first draws
- [Gate ledger](gate-summary.json)

Forward draw 6, all six reverse draws, the eight-file set, both decomposition
arms, both MM draws, C's pair, the carrier gate and the new-source regression
set are not run under the explicit stop rule. Claude's final review and
publication are pending. The preceding development receipts retain the
seven unresolved assertion-contract checks and all earlier failures.
Production and test source remain byte-for-byte at the frozen hash.
