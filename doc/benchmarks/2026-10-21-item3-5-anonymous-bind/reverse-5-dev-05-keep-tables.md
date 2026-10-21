# Arrival keep: one table per document

Source `working tree on 4159dde45; section 10.4 development`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## reverse-5-dev-05, batch 1, document 0

Kept **explore**; deciding `reconstruction`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0300561469 | 0.0000000000 | 0.3945910633 | 0.4246472120 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## reverse-5-dev-05, batch 1, document 1

Kept **greedy**; deciding `answer`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4945910573 | 0.5306584239 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 4]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", -1]; forward=[[2, 13]]; subject_word=14; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 4

Kept **explore**; deciding `answer`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3000000119 | 0.3360673785 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.3945910633 | 0.4306584299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## reverse-5-dev-05, batch 2, document 0

Kept **explore**; deciding `reconstruction`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0299619418 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0324606411 | 0.0299619418 | 0.4397895336 | 0.4722501636 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 1

Kept **explore**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0307352543 | 0.3639057577 | 0.3999731243 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[['clause', 0], 24, -1]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `25`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `49`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[-2072910214031139291, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=23; children=[1]`<br>`n1 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `48`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0300018881 | 0.3000000119 | 0.3360673785 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0300018881 | 0.3000000119 | 0.3360673785 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='whole')", 14]; forward=(); subject_word=13; children=[3]`<br>`n3 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 5

Kept **explore**; deciding `answer`; compose departure `19`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0301066227 | 0.3564949632 | 0.3925623298 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 5]`<br>`n3 refs=[-2299595933854994912, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=12; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=24; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 6

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0301066227 | 0.5302585363 | 0.5663259029 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

## reverse-5-dev-05, batch 2, document 7

Kept **explore**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0301066227 | 0.4397895336 | 0.4758569002 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[-1, 24, ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |
| explore | 0.0360673778 | 0.0301066227 | 0.3484906852 | 0.3845580518 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=23; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=23; children=[]`<br>`n2 refs=[['clause', 0], 24, ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[14, "ClausePredicate(name='equal')", -5340317820426174101]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=25; children=[]` |

