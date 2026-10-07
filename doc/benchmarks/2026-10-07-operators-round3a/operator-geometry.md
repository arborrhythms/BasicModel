# Centered operator-root geometry

This post-hoc diagnostic answers plan §29 using saved roots only. The four all-disjunction starts (XOR runs 3, 5, 7 and 8) have identical roots. All ten final conjunction root sets are identical. Rows are matched by sentence, centered in float64, and evaluated at the precision of the stored float32 data. There are no new trainings or substituted gate predictions.

| Operator | Centered singular values | Nonzero condition number | Minimum affine weight norm, including bias | Optimal affine MSE |
|---|---|---:|---:|---:|
| conjunction | 0.877855432, 0.800606913, 0.482921475, 1.98894548e-16 | 1.8178 | 2.09393 | 2.39778e-31 |
| disjunction | 0.691669864, 0.523711256, 0.0588574868, 7.29893616e-17 | 11.7516 | 20.4507 | 6.90253e-31 |

The weakest disjunction direction is 8.205 times smaller than the weakest conjunction direction. Disjunction is less well conditioned, but its third singular value is about 0.059, far above float32 rounding. Both root sets are stably affinely separable. The spectra support a difference in convergence difficulty; they do not prove the cause of the comparison reader’s finite-step preference or establish a semantic operator distinction.

The fourth centered singular value is zero up to float64 roundoff by construction: four centered rows have rank at most three. The original float32-centered spectra (including their roughly 1e-7 fourth residuals) remain unchanged in the raw audits.

The final presented-reader errors in runs 2, 3 and 9 are almost entirely a common offset across the four sentences. Their centered error MSEs are 3.75e-8, 3.91e-8 and 5.79e-9 respectively; the saved training curves show late oscillation after much lower earlier errors. These are passing runs, and this description does not replace the final gate MSEs or claim that the precise optimizer cause has been established.

[Raw spectra, root values, combined-root fit and final residuals](operator-geometry.json).
