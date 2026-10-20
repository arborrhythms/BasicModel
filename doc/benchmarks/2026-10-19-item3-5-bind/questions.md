# Bind candidate: reported questions

These are mechanism certificates. All generated cases and companion questions are reported.

## G1: Does the definite site select its generated antecedent with a distractor?

| Case | Generated text | Expected | Selected | Evidence pair | Reverse / open role |
|---|---|---:|---:|---|---|
| 0-1 | a cat ; a dog ; the cat | 12 | 12 | (0.817574, 0.182426) | — |
| 0-2 | a cat ; a bird ; the cat | 12 | 12 | (0.817574, 0.182426) | — |
| 0-3 | a cat ; a horse ; the cat | 12 | 12 | (0.817574, 0.182426) | — |
| 1-0 | a dog ; a cat ; the dog | 12 | 12 | (0.817574, 0.182426) | — |
| 1-2 | a dog ; a bird ; the dog | 12 | 12 | (0.817574, 0.182426) | — |
| 1-3 | a dog ; a horse ; the dog | 12 | 12 | (0.817574, 0.182426) | — |
| 2-0 | a bird ; a cat ; the bird | 12 | 12 | (0.817574, 0.182426) | — |
| 2-1 | a bird ; a dog ; the bird | 12 | 12 | (0.817574, 0.182426) | — |
| 2-3 | a bird ; a horse ; the bird | 12 | 12 | (0.817574, 0.182426) | — |
| 3-0 | a horse ; a cat ; the horse | 12 | 12 | (0.817574, 0.182426) | — |
| 3-1 | a horse ; a dog ; the horse | 12 | 12 | (0.817574, 0.182426) | — |
| 3-2 | a horse ; a bird ; the horse | 12 | 12 | (0.817574, 0.182426) | — |

Exact records: `generated-candidate/G1/*.json`.

## G2: Does the objectless marker select its antecedent and re-emit its leaf?

| Case | Generated text | Expected | Selected | Evidence pair | Reverse / open role |
|---|---|---:|---:|---|---|
| 0-1 | a cat ; a dog ; m01 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 0-2 | a cat ; a bird ; m02 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 0-3 | a cat ; a horse ; m03 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 1-0 | a dog ; a cat ; m10 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 1-2 | a dog ; a bird ; m12 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 1-3 | a dog ; a horse ; m13 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 2-0 | a bird ; a cat ; m20 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 2-1 | a bird ; a dog ; m21 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 2-3 | a bird ; a horse ; m23 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 3-0 | a horse ; a cat ; m30 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 3-1 | a horse ; a dog ; m31 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |
| 3-2 | a horse ; a bird ; m32 chases | 12 | 12 | (0.622459, 0.377541) | 1 leaf |

Exact records: `generated-candidate/G2/*.json`.

## G3: Does the elided site fill from accessible evidence or remain open?

| Case | Generated text | Expected | Selected | Evidence pair | Reverse / open role |
|---|---|---:|---:|---|---|
| 0-0 | dog chases | None | None | (0.000000, 0.000000) | role 2 open |
| 0-1 | dog chases | 12 | 12 | (1.000000, 0.000000) | role 2 filled |
| 1-0 | bird chases | None | None | (0.000000, 0.000000) | role 2 open |
| 1-1 | bird chases | 12 | 12 | (1.000000, 0.000000) | role 2 filled |
| 2-0 | horse chases | None | None | (0.000000, 0.000000) | role 2 open |
| 2-1 | horse chases | 12 | 12 | (1.000000, 0.000000) | role 2 filled |
| 3-0 | cat chases | None | None | (0.000000, 0.000000) | role 2 open |
| 3-1 | cat chases | 12 | 12 | (1.000000, 0.000000) | role 2 filled |

Exact records: `generated-candidate/G3/*.json`.

## Does the unchanged 1-successor certificate close its proved binding?

| Configuration | Document | Answer | Correct binding | Inferences | Work | Chain correct |
|---|---|---|---|---:|---:|---|
| zero | 0 | four | True | 1 | 175 | True |
| zero | 1 | four | True | 1 | 178 | True |
| enabled | 0 | four | True | 1 | 175 | True |
| enabled | 1 | four | True | 1 | 178 | True |

Both configurations keep prime on and φ=.5. `zero` sets P/S/R/D to 0; `enabled` sets all four to .5.

## Does the unchanged 2-successor certificate close its proved binding?

| Configuration | Document | Answer | Correct binding | Inferences | Work | Chain correct |
|---|---|---|---|---:|---:|---|
| zero | 0 | five | True | 2 | 455 | True |
| zero | 1 | five | True | 2 | 465 | True |
| enabled | 0 | five | True | 2 | 455 | True |
| enabled | 1 | five | True | 2 | 465 | True |

Both configurations keep prime on and φ=.5. `zero` sets P/S/R/D to 0; `enabled` sets all four to .5.
