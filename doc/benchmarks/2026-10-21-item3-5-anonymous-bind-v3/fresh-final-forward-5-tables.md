# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-5, batch 1, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 7]`<br>`n4 refs=[['clause', 0], "ClausePredicate(name='part')", ['clause', 1]]; forward=(); subject_word=16; children=[5, 6]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=16; children=[]`<br>`n6 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n7 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 1

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 1.1129899025 | 1.1490572691 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, "ClausePredicate(name='whole')", -1]; forward=[[0, 13], [2, 12]]; subject_word=13; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 2

Kept **greedy**; deciding `answer`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4484906793 | 0.4845580459 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, 15, -1]; forward=[[0, 13]]; subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 3

Kept **explore**; deciding `answer`; compose departure `29`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=15; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 6

Kept **explore**; deciding `answer`; compose departure `29`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=15; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 1, document 7

Kept **explore**; deciding `answer`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], 16, -1]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-5, batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `4`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0279843062 | 1.1923749447 | 1.2284423113 | `n0 refs=[-1, -1, -1]; forward=(); subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0275713988 | 0.0000000000 | 0.0360673778 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0279575828 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='whole')", 14]; forward=(); subject_word=37; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0276106149 | 0.7782499194 | 0.8143172860 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0275713988 | 0.0000000000 | 0.0360673778 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0283232573 | 0.0000000000 | 0.0360673778 | `n0 refs=[37, "ClausePredicate(name='verb')", 27]; forward=(); subject_word=37; children=[]` |

## fresh-final-forward-5, batch 2, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `4`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0276419073 | 0.0000000000 | 0.0360673778 | `n0 refs=[37, 18, -1]; forward=[]; subject_word=37; children=[]` |
| explore | 0.0360673778 | 0.0283936169 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=37; children=[]` |

