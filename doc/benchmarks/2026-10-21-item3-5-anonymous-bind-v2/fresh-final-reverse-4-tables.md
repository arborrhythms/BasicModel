# Arrival keep: one table per document

Source `823f157cda4ef6b08744591ade7897476f8551b6`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-reverse-4, batch 1, document 0

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 1

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.8158883452 | 0.8519557118 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 4]`<br>`n2 refs=[-1, "ClausePredicate(name='part')", ['clause', 0]]; forward=[[0, 13]]; subject_word=13; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 5]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='part')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 3

Kept **greedy**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4197224677 | 0.4557898343 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 3]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='whole')", -1]; forward=[[2, 12]]; subject_word=14; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 4

Kept **greedy**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.8394449353 | 0.8755123019 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, "ClausePredicate(name='whole')", -1]; forward=[[0, 14], [2, 12]]; subject_word=14; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

## fresh-final-reverse-4, batch 1, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.4079441726 | 0.4440115392 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]` |

