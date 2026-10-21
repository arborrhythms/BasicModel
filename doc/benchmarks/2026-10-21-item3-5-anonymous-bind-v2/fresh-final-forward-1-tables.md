# Arrival keep: one table per document

Source `823f157cda4ef6b08744591ade7897476f8551b6`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-1, batch 2, document 0

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0313469730 | 0.7605170608 | 0.7965844274 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

## fresh-final-forward-1, batch 2, document 1

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0313221328 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-1, batch 2, document 2

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## fresh-final-forward-1, batch 2, document 3

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0313469730 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='part')", ['clause', 0]]; forward=(); subject_word=14; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

## fresh-final-forward-1, batch 2, document 4

Kept **explore**; deciding `reconstruction`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0300561469 | 0.0313469730 | 0.7394449115 | 0.7695010304 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='equal')", ['clause', 0]]; forward=(); subject_word=14; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

## fresh-final-forward-1, batch 2, document 5

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0319425724 | 0.7605170608 | 0.7965844274 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

## fresh-final-forward-1, batch 2, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0313469730 | 0.7394449115 | 0.7755122781 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 2]`<br>`n1 refs=[-1, "ClausePredicate(name='equal')", 14]; forward=[[0, 15]]; subject_word=15; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

## fresh-final-forward-1, batch 2, document 7

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |
| explore | 0.0360673778 | 0.0319425724 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=22; children=[]` |

