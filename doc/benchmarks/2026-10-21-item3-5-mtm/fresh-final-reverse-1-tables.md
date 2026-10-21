# Arrival keep: one table per document

Source `eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. The reverse
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-reverse-1, batch 1, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 1, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 1, document 2

Kept **explore**; deciding `reconstruction`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0300561469 | 0.0000000000 | 0.0000000000 | 0.0300561469 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]` |

## fresh-final-reverse-1, batch 1, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 14], [2, 15]]; subject_word=14; children=[]` |

## fresh-final-reverse-1, batch 1, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=14; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-1, batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[['clause', 0], 26, ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[13, "ClausePredicate(name='part')", 8220155874494905508]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-5828476475727764754, 13, -1]; forward=[]; subject_word=26; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `29`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154176401 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0157841817 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[['clause', 0], 26, -1]; forward=(); subject_word=12; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0157900378 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `18`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `49`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[-2288343301839160340, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=25; children=[1]`<br>`n1 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |

## fresh-final-reverse-1, batch 2, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `24`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=27; children=[]` |
| explore | 0.0360673778 | 0.0154234963 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=25; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]`<br>`n2 refs=[-1, 26, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[13, "ClausePredicate(name='equal')", -9005012566409249500]; forward=(); subject_word=13; children=[]` |
