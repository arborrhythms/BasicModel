# Saved decoder and ownership audit

These summaries read saved arrays only. They construct no model and perform no optimizer step. Geometry is diagnostic, not an acceptance bar.

## Decoder stage

Decoder exploration wins 1790/3200 comparisons (55.9375%). Strict-selection violations: 0. All four greedy decoder paths are stable STOP paths; this is stable early stopping, not correct inference.

| Sentence | Kept decoder modal share | Distinct kept decoder paths | Compose modal share | Distinct compose paths |
|---|---:|---:|---:|---:|
| hello world | 0.63250 | 3 | 0.88500 | 5 |
| hello there | 0.49375 | 3 | 0.60750 | 4 |
| loving world | 0.55125 | 2 | 0.74750 | 7 |
| loving there | 0.52625 | 3 | 0.44250 | 7 |

Ownership conflicts: 0; active parameters: 20; inactive: 55; backwards: 1200. The full parameter-by-writer report is in [decoder-xor/ownership/ownership.json](decoder-xor/ownership/ownership.json).

Concept dictionary mean squared off-diagonal cosine: 0.12791129 → 0.19806787. Ending norms: 0.31191942–1.85215211. Full matrices, code rows and step displacements remain in [decoder-xor/ownership/](decoder-xor/ownership/).

## Operators checkpoint

Decoder exploration wins 1094/3200 comparisons (34.1875%). Strict-selection violations: 0. All four greedy decoder paths are stable STOP paths; this is stable early stopping, not correct inference.

| Sentence | Kept decoder modal share | Distinct kept decoder paths | Compose modal share | Distinct compose paths |
|---|---:|---:|---:|---:|
| hello world | 0.55875 | 4 | 0.48000 | 4 |
| hello there | 0.62625 | 4 | 0.32750 | 7 |
| loving world | 0.75500 | 3 | 0.96250 | 3 |
| loving there | 0.69250 | 2 | 0.81500 | 4 |

Ownership conflicts: 0; active parameters: 20; inactive: 55; backwards: 1200. The full parameter-by-writer report is in [checkpoint-xor/ownership/ownership.json](checkpoint-xor/ownership/ownership.json).

Concept dictionary mean squared off-diagonal cosine: 0.11128937 → 0.14797887. Ending norms: 0.34743810–1.90298462. Full matrices, code rows and step displacements remain in [checkpoint-xor/ownership/](checkpoint-xor/ownership/).
