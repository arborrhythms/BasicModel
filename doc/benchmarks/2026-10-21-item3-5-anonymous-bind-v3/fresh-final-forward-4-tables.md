# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-4, batch 1, document 0

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 1.1129899025 | 1.1490572691 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 13], [2, 14]]; subject_word=13; children=[]`<br>`n4 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 1

Kept **greedy**; deciding `answer`; compose departure `28`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6397895813 | 0.6758569479 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, -1]; forward=[[0, 15], [2, 17]]; subject_word=15; children=[]` |

## fresh-final-forward-4, batch 1, document 2

Kept **explore**; deciding `answer`; compose departure `29`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=15; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `50`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `18`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 5

Kept **explore**; deciding `reconstruction`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0288539026 | 0.0000000000 | 0.2000000030 | 0.2288539112 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 6]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=14; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n6 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[7]`<br>`n7 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `27`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-4, batch 2, document 0

Kept **explore**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0222744606 | 0.7387439013 | 0.7748112679 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-4, batch 2, document 1

Kept **explore**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0222744606 | 0.7387439013 | 0.7748112679 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-4, batch 2, document 2

Kept **greedy**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0226418711 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0226418711 | 0.4302585125 | 0.4663258791 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |

## fresh-final-forward-4, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## fresh-final-forward-4, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |

## fresh-final-forward-4, batch 2, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |

## fresh-final-forward-4, batch 2, document 6

Kept **greedy**; deciding `answer`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0222683363 | 1.2284821272 | 1.2645494938 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |

## fresh-final-forward-4, batch 2, document 7

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0227986928 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |
| explore | 0.0360673778 | 0.0222683363 | 0.7598754764 | 0.7959428430 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 2]`<br>`n1 refs=[12, 13, -1]; forward=[]; subject_word=15; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]` |

