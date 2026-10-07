# Root geometry at the stored precision

The frozen observer computes SVD/pseudoinverse in float64 from float32 data. Its raw rank can count float32 rounding residuals as independent directions. This saved-data report uses the conventional cutoff `max(matrix.shape) × eps(float32) × largest singular value`. It changes no measurement, gate prediction, source, or training.

| Kind / run | Presented MSE | Final root rank | Optimal affine MSE at float32 precision | Retained condition number |
|---|---:|---:|---:|---:|
| sum-01 | 0.25 | 3 | 0.25 | 3.60311 |
| sum-02 | 0.25000006 | 3 | 0.25 | 3.60311 |
| sum-03 | 0.250000119 | 3 | 0.25 | 3.60311 |
| sum-04 | 0.24999994 | 3 | 0.25 | 3.60311 |
| sum-05 | 0.250000119 | 3 | 0.25 | 3.60311 |
| sum-06 | 0.249999955 | 3 | 0.25 | 3.60311 |
| sum-07 | 0.25000003 | 3 | 0.25 | 3.60311 |
| sum-08 | 0.250000119 | 3 | 0.25 | 3.60311 |
| sum-09 | 0.25000006 | 3 | 0.25 | 3.60311 |
| sum-10 | 0.249999955 | 3 | 0.25 | 3.60311 |
| xor-01 | 4.39248359e-06 | 4 | 2.39778e-31 | 5.22121 |
| xor-02 | 0.0207201172 | 4 | 2.39778e-31 | 5.22121 |
| xor-03 | 0.0160247575 | 4 | 2.39778e-31 | 5.22121 |
| xor-04 | 7.50732809e-13 | 4 | 2.39778e-31 | 5.22121 |
| xor-05 | 8.2600593e-14 | 4 | 2.39778e-31 | 5.22121 |
| xor-06 | 3.64993565e-06 | 4 | 2.39778e-31 | 5.22121 |
| xor-07 | 4.33253433e-12 | 4 | 2.39778e-31 | 5.22121 |
| xor-08 | 2.0250468e-13 | 4 | 2.39778e-31 | 5.22121 |
| xor-09 | 0.00370835332 | 4 | 2.39778e-31 | 5.22121 |
| xor-10 | 8.75934553e-07 | 4 | 2.39778e-31 | 5.22121 |

A full-rank final root matrix establishes that an affine reader has a stable separating direction at the precision of the stored roots. It does not establish that the trained reader reached that fit in 400 epochs. The sum-control result and the actual class bar remain the recorded presented outputs. Raw double-precision diagnostics remain in every run audit.
