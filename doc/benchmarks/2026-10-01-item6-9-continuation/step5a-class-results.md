# Step 5a: ten unseeded class-gate runs

8/10 meet all-four-correct and MSE < .05, at the unchanged 400 epochs.

The numeric values and complete observations are in `step5a-class-ten/summary.json`; source hashes are in its source manifest.

Answer columns follow: `hello world`, `hello there`, `loving world`, `loving there`.

| Run | Four answers | MSE | Correct | Bar |
|---|---|---:|---:|---|
| 1 | -0.0031576753, 0.98801452, 0.99177891, 0.0062020421 | 6.491857679e-05 | 4/4 | pass |
| 2 | 0.059029579, 1.060196, 0.95038605, 0.38517153 | 0.03948167714 | 4/4 | pass |
| 3 | -0.042268991, 1.0382855, 0.96386051, 0.52432358 | 0.06986843215 | 3/4 | fail |
| 4 | 0.0086272061, 0.98114932, 0.99992234, 0.016056746 | 0.0001719005275 | 4/4 | pass |
| 5 | -0.04769659, 1.0201687, 0.88558757, 0.10634711 | 0.006770412933 | 4/4 | pass |
| 6 | -0.0011971593, 1.0017329, 0.99850726, 0.0067124665 | 1.293044135e-05 | 4/4 | pass |
| 7 | -0.010849237, 1.0023196, 1.0179772, 0.012526184 | 0.0001507931868 | 4/4 | pass |
| 8 | -0.011835575, 0.98200583, 1.2195053, 0.66292942 | 0.1220304655 | 3/4 | fail |
| 9 | -0.0021267533, 0.97631502, 0.98885083, 0.003967911 | 0.0001763873877 | 4/4 | pass |
| 10 | -0.024803638, 0.87209177, 0.98311597, 0.10263479 | 0.006948676447 | 4/4 | pass |

All ten runs completed normally. The two nonzero pytest exits are the unchanged gate assertion, not resource or execution failures. No initialization was selected; no unsuccessful run was retried. Continue through steps 6–9 under Alec’s October 1 decision.
