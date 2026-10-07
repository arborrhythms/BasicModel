# Identity by pairs: the fixed point before round 3a (Claude, 2026-10-06)

`sim.py` (numpy; reads `/usr/share/dict/words`): 20,000 sampled words plus
the four XOR gate words; parts = boundary-marked adjacent letter pairs
(inventory 606); each pair a fixed sparse binary code (`s` ones of `D`); a
word's form = OR of its pairs' codes, keyed with its length.

| D, s | collisions among 20,004 words | anagram groups separated |
| --- | ---: | ---: |
| 32, 2 | 6 | 236 / 237 |
| 64, 3 | 1 | 236 / 237 |
| 128, 3 | 1 | 236 / 237 |
| 256, 4 | 1 | 236 / 237 |

The one collision and the one unseparated group are the same genuine case,
`calaba` / `cabala` (same pair set, same length): the case collision-minting
is for. Products of sparse codes: with 3 pairs per word (~9 ones of 128) two
random words share no coordinate with probability ≈ .52; with 8 pairs ≈ .01.
So the kernels cannot bind sparse forms directly. On the gate words at
D = 128, all four variants — product on sparse codes (by chance overlap),
product on a fixed dense projection, permutation-OR, permutation-sum — give
unit roots that are affinely XOR-separable (fit MSE ~1e-31, rank 4) and
unbind exactly by bank search (4/4); only the projected product is both
robust for short words and commutative, which conjunction requires.

## Repairs after Codex's construction check (2026-10-07; `sim2_repairs.py`)

Codex's `construction-review.md` (round-3a directory) found: random sparse
bits for the cumulative length atoms are not an exact thermometer
(`aaaaaa`/`aaaaaaa` collide; a binary join can grow strictly at most D
times); the first differing triple can be masked by the existing join
(`cal`/`cab` on `calaba`/`cabala`); and `an`/`and`/`ant` are not part-set
inclusions under boundary pairs (`n#`). With a reserved thermometer block
(32 bits, exact up to that length), a mint that gives each word its own
triple at the first differing position and draws bits from that atom's
seeded stream (then the next position) until the forms separate, and the
witnesses `bana`/`banana`, `cat`/`concat`, `aba`/`ababa`: one collision group
in 20,000 words before minting, zero after (one mint, 3 bits), the length
pair distinct, all three witnesses ordered, the two invalid ones correctly
not.

## The ceiling at order 0 (`sim3b_ceiling.py`, 2026-10-07)

Has-a wholes as the extents of the word's pairs, their symbols the join of
their members; a word's ceiling the meet of its wholes weighted by edge
evidence × narrowing (§12.1); the centroid between L and U; the containment
projection (cap the contained by its containers, decreasing part count).
Static, over the 3a forms of 19,986 dictionary words.

| rule | mean pairwise cosine, forms → centroids | mean \|c − L\| | containment violations before → after projection |
| --- | ---: | ---: | ---: |
| global weight, no narrowing | .520 → .908 | .226 | 26 → 0 |
| global weight × narrowing | .520 → .902 | .221 | 566 → 0 |
| per-coordinate weight × narrowing | .512 → .512 | .001 | 569 → 0 |

Where no whole narrows a coordinate, `U` is 1 there, and any global weight on
it is a common offset — §14's geometry again, narrowing or not. Weighting
the ceiling per coordinate by the narrowing it actually does removes the
offset, and then the ceiling adds almost nothing: a word's own pairs' extents
are determined by those pairs, so their joins are near-everything on every
coordinate the word lacks. The projection never pushes a code below its
parts' join.
