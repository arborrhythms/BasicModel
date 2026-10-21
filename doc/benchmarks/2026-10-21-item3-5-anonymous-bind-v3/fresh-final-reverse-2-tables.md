# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-reverse-2, batch 1, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 1

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.8158883452 | 0.8519557118 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, 14, -1]; forward=[[0, 13]]; subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 2

Kept **explore**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 3]`<br>`n1 refs=[['clause', 0], 14, -1]; forward=(); subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 3]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", -1]; forward=[[2, 14]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 5

Kept **greedy**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.7891821265 | 0.8252494931 | `n0 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 12], [2, 15]]; subject_word=12; children=[]` |

## fresh-final-reverse-2, batch 1, document 6

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 5]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='part')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[14, "ClausePredicate(name='equal')", 13]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

## fresh-final-reverse-2, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `49`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.8129898906 | 0.8490572572 | `n0 refs=[-3909780874639251382, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=24; children=[1]`<br>`n1 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

## fresh-final-reverse-2, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `28`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297309756 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297309756 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, 22]; forward=[[0, 12]]; subject_word=12; children=[]` |

## fresh-final-reverse-2, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", 14]; forward=(); subject_word=13; children=[3]`<br>`n3 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

## fresh-final-reverse-2, batch 2, document 4

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 1.2454719543 | 1.2815393209 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

## fresh-final-reverse-2, batch 2, document 5

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.8129898906 | 0.8490572572 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

## fresh-final-reverse-2, batch 2, document 6

Kept **explore**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.7795791626 | 0.8156465292 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='part')", 25]; forward=[[0, 12]]; subject_word=12; children=[]` |

## fresh-final-reverse-2, batch 2, document 7

Kept **greedy**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0297703501 | 0.7969813347 | 0.8330487013 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |
| explore | 0.0360673778 | 0.0297703501 | 0.8823046088 | 0.9183719754 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=24; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n2 refs=[-1, 25, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=26; children=[]` |

