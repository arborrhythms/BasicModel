# Arrival keep: one table per document

Source `823f157cda4ef6b08744591ade7897476f8551b6`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-2, batch 2, document 0

Kept **greedy**; deciding `answer`; compose departure `0`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0188934430 | 0.7784811258 | 0.8145484924 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |

## fresh-final-forward-2, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0188934430 | 0.7593267560 | 0.7953941226 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |

## fresh-final-forward-2, batch 2, document 2

Kept **explore**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0184213668 | 0.7381460071 | 0.7742133737 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-2, batch 2, document 3

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0187958218 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0183312893 | 0.4302585125 | 0.4663258791 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |

## fresh-final-forward-2, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0184065923 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[14, 13, -1]; forward=[]; subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |

## fresh-final-forward-2, batch 2, document 5

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]` |

## fresh-final-forward-2, batch 2, document 6

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0184065923 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[14, "ClausePredicate(name='whole')", 13]; forward=(); subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |

## fresh-final-forward-2, batch 2, document 7

Kept **greedy**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0188934430 | 0.7394449115 | 0.7755122781 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=31; children=[]` |
| explore | 0.0360673778 | 0.0184065923 | 1.1931470633 | 1.2292144299 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='part')", -1]; forward=[[2, 31]]; subject_word=14; children=[]` |

