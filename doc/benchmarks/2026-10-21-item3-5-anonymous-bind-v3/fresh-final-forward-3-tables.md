# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-3, batch 1, document 0

Kept **explore**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `25`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## fresh-final-forward-3, batch 1, document 2

Kept **greedy**; deciding `answer`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 1.0969814062 | 1.1330487728 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-1, "ClausePredicate(name='whole')", -1]; forward=[[0, 13], [2, 16]]; subject_word=13; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 3

Kept **explore**; deciding `reconstruction+answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0324606411 | 0.0000000000 | 0.0000000000 | 0.0324606411 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 4

Kept **explore**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 6]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='part')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n6 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[7]`<br>`n7 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `27`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 1, document 7

Kept **explore**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, ['clause', 1]]; forward=(); subject_word=13; children=[3, 6]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=14; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n6 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-3, batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235234704 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0230509285 | 0.4197224677 | 0.4557898343 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 3]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='whole')", -1]; forward=[[2, 15]]; subject_word=13; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235943403 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0230975207 | 1.2284821272 | 1.2645494938 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[14, "ClausePredicate(name='whole')", -1]; forward=[[2, 13]]; subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 2

Kept **explore**; deciding `answer`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235943403 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0231195856 | 0.7158883214 | 0.7519556880 | `n0 refs=[-1, "ClausePredicate(name='equal')", 14]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-3, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235234704 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0230509285 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[26, "ClausePredicate(name='part')", ['clause', 0]]; forward=(); subject_word=13; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235138964 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0235138964 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 5

Kept **greedy**; deciding `answer`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235943403 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0230975207 | 0.7605170608 | 0.7965844274 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 6

Kept **greedy**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235943403 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0235943403 | 0.7593267560 | 0.7953941226 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

## fresh-final-forward-3, batch 2, document 7

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0235138964 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |
| explore | 0.0360673778 | 0.0230413508 | 0.5079441667 | 0.5440115333 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=38; children=[]` |

