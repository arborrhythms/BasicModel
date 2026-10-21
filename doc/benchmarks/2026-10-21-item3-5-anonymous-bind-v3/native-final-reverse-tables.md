# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 0

Kept **explore**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 4]`<br>`n1 refs=[['clause', 1], "ClausePredicate(name='whole')", ['clause', 0]]; forward=(); subject_word=14; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 2

Kept **explore**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 4

Kept **explore**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `12`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203917362 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[20, 33, ['clause', 0]]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203917362 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[20, 33, ['clause', 0]]; forward=(); subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `25`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203758273 | 0.8039329648 | 0.8400003314 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203758273 | 0.8192731738 | 0.8553405404 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", 6200734191045178110]; forward=(); subject_word=33; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=33; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 2

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0204007085 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0204007085 | 0.5707477331 | 0.6068150997 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, 13, -1]; forward=[[0, 32]]; subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 3

Kept **greedy**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203758273 | 0.8039329648 | 0.8400003314 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203758273 | 0.8192731738 | 0.8553405404 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 4]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='part')", ['clause', 1]]; forward=(); subject_word=32; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `18`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0204007085 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0204007085 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 6]`<br>`n4 refs=[13, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=13; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=33; children=[]`<br>`n6 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 5

Kept **greedy**; deciding `answer`; compose departure `49`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203758273 | 0.8039329648 | 0.8400003314 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203758273 | 0.8192731738 | 0.8553405404 | `n0 refs=[5667005226782487560, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=32; children=[1]`<br>`n1 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 6

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203758273 | 0.8039329648 | 0.8400003314 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203758273 | 1.1670105457 | 1.2030779123 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, "ClausePredicate(name='equal')", 13]; forward=[[0, 32]]; subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

## test_forced_ordinary_pending_premise_in_both_orders[True], batch 2, document 7

Kept **greedy**; deciding `answer`; compose departure `27`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0203758273 | 0.8039329648 | 0.8400003314 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |
| explore | 0.0360673778 | 0.0203758273 | 0.8192731738 | 0.8553405404 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=32; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=32; children=[]`<br>`n2 refs=[-1, 33, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=34; children=[]` |

