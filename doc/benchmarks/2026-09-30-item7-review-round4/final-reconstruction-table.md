# Reconstruction: eight predeclared seeds on each tree

Seeds 0–7 were declared before measurement. The model configuration, driver, 8 GiB guard and 1,200-second deadline are unchanged. Every completed value is reported.

The completed HEAD baseline is reused after exact source and driver verification. Original candidate deadline stops are retained below; only incomplete attempts receive one serial retry with the same seed and limits.

| Seed | HEAD before | Candidate before | HEAD after training | Candidate after training | HEAD GiB | Candidate GiB | Candidate attempts |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.101097988 | 0.1098995563 | 0.1085145287 | 0.1092094779 | 6.611 | 5.649 | deadline stop; serial completion |
| 1 | 0.1222549416 | 0.1105236281 | 0.1188460868 | 0.1077861711 | 6.676 | 5.715 | deadline stop; serial completion |
| 2 | 0.1208517365 | 0.1360428035 | 0.1144352518 | 0.1450556107 | 6.440 | 5.730 | completed once |
| 3 | 0.09851540998 | 0.09771095216 | 0.1013413034 | 0.09769639373 | 5.997 | 5.349 | completed once |
| 4 | 0.09506519139 | 0.1003888082 | 0.1019906513 | 0.1015955992 | 6.493 | 4.895 | completed once |
| 5 | 0.1195291076 | 0.1228054985 | 0.1238037553 | 0.142538365 | 6.157 | 5.683 | completed once |
| 6 | 0.1132729966 | 0.1145265847 | 0.1089693308 | 0.1245986186 | 6.093 | 5.247 | completed once |
| 7 | 0.08667692356 | 0.09421905875 | 0.1120041683 | 0.1181098018 | 6.960 | 5.314 | completed once |

After-training summary (all eight):

| Tree | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| HEAD | 0.1112381346 | 0.1013413034 | 0.1238037553 |
| candidate | 0.1183237548 | 0.09769639373 | 0.1450556107 |

No re-baseline or learning threshold change is made.

[HEAD reuse verification](reconstruction-head-reuse.json); [all original candidate attempts](final3-reconstruction-candidate/processes.json); [completion reconciliation](final3-reconstruction-reconciled.json).
