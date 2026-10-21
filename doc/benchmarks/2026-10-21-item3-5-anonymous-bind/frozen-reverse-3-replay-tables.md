# Arrival keep: one table per document

Source `51cce2005be3860f4da5ef15eebfac96812f437b`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## frozen-reverse-3-replay, batch 1, document 0

Kept **explore**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[15, -1, -1]; forward=(); subject_word=15; children=[]` |

## frozen-reverse-3-replay, batch 1, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## frozen-reverse-3-replay, batch 1, document 2

Kept **greedy**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.7891821265 | 0.8252494931 | `n0 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 12], [2, 15]]; subject_word=12; children=[]` |

## frozen-reverse-3-replay, batch 1, document 3

Kept **explore**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## frozen-reverse-3-replay, batch 1, document 4

Kept **greedy**; deciding `answer`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 1.1837731600 | 1.2198405266 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 13], [2, 15]]; subject_word=13; children=[]` |

## frozen-reverse-3-replay, batch 1, document 5

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=14; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## frozen-reverse-3-replay, batch 1, document 6

Kept **greedy**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.7891821265 | 0.8252494931 | `n0 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 12], [2, 15]]; subject_word=12; children=[]` |

## frozen-reverse-3-replay, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## frozen-reverse-3-replay, batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `19`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327186584 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0321325436 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[21, "ClausePredicate(name='whole')", 25]; forward=(); subject_word=29; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0327389650 | 0.3708050251 | 0.4068723917 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `29`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327321626 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0321460478 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[['clause', 0], 29, ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[13, 25, -1]; forward=[]; subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 4

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0324606411 | 0.0327389650 | 0.3708050251 | 0.4032656550 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 5

Kept **greedy**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0327389650 | 0.3708050251 | 0.4068723917 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `50`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |

## frozen-reverse-3-replay, batch 2, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `25`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0327389650 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, 29, -1]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
| explore | 0.0360673778 | 0.0321528502 | 0.3639057875 | 0.3999731541 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=28; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=28; children=[]`<br>`n2 refs=[25, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-3565953245679703555, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=29; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=30; children=[]` |
