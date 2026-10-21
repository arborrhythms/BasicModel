# Arrival keep: one table per document

Source `823f157cda4ef6b08744591ade7897476f8551b6`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-reverse-2, batch 1, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 3

Kept **greedy**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.7891821265 | 0.8252494931 | `n0 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 12], [2, 15]]; subject_word=12; children=[]` |

## fresh-final-reverse-2, batch 1, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 5

Kept **explore**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 4]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 6

Kept **explore**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-2, batch 1, document 7

Kept **greedy**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.7891821265 | 0.8252494931 | `n0 refs=[-1, "ClausePredicate(name='equal')", -1]; forward=[[0, 12], [2, 15]]; subject_word=12; children=[]` |

