# Arrival keep: one table per document

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The forward
assertion, budget, forced scope, objective and keep rule are unchanged.
`exploit` is greedy; E is excluded. Readings are before admission.
Complete refs/open fields, compositions, registries, credit costs,
keep costs and departures are retained in the adjacent JSONL files.

## fresh-final-forward-1, batch 1, document 0

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6564949751 | 0.6925623417 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 3]`<br>`n1 refs=[-1, "ClausePredicate(name='part')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 1

Kept **explore**; deciding `answer`; compose departure `13`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], 16, -1]; forward=(); subject_word=13; children=[3, 5]`<br>`n3 refs=[['clause', 0], 15, -1]; forward=(); subject_word=14; children=[4]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 2

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6564949751 | 0.6925623417 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, 13, -1]; forward=[[0, 12]]; subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 3

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6564949751 | 0.6925623417 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 4]`<br>`n3 refs=[-1, 14, -1]; forward=[[0, 13]]; subject_word=13; children=[]`<br>`n4 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 4

Kept **greedy**; deciding `answer`; compose departure `3`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 1.1129899025 | 1.1490572691 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, "ClausePredicate(name='whole')", -1]; forward=[[0, 13], [2, 12]]; subject_word=13; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 5

Kept **greedy**; deciding `answer`; compose departure `21`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6484906673 | 0.6845580339 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[4, 6]`<br>`n4 refs=[-1, "ClausePredicate(name='part')", ['clause', 0]]; forward=[[0, 16]]; subject_word=16; children=[5]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n6 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 6

Kept **explore**; deciding `answer`; compose departure `19`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[3, 6]`<br>`n3 refs=[['clause', 1], "ClausePredicate(name='whole')", ['clause', 0]]; forward=(); subject_word=16; children=[4, 5]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n5 refs=[-1, -1, -1]; forward=(); subject_word=16; children=[]`<br>`n6 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 1, document 7

Kept **greedy**; deciding `answer`; compose departure `49`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0000000000 | 0.2000000030 | 0.2360673845 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=12; children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=12; children=[]`<br>`n2 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[3]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |
| explore | 0.0360673778 | 0.0000000000 | 0.6564949751 | 0.6925623417 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 12]]; subject_word=12; children=[1]`<br>`n1 refs=[-1, 16, ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=17; children=[]` |

## fresh-final-forward-1, batch 2, document 0

Kept **greedy**; deciding `answer`; compose departure `7`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0230197012 | 0.7968126535 | 0.8328800201 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=15; children=[1, 4]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='implies')", ['clause', 1]]; forward=(); subject_word=15; children=[2, 3]`<br>`n2 refs=[23, -1, -1]; forward=(); subject_word=15; children=[]`<br>`n3 refs=[22, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n4 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 1

Kept **greedy**; deciding `answer`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236717444 | 0.4396285117 | 0.4756958783 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0230335072 | 0.6255014539 | 0.6615688205 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, "ClausePredicate(name='part')", 13]; forward=[[0, 14]]; subject_word=14; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 2

Kept **greedy**; deciding `answer`; compose departure `15`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0236629732 | 0.7787045836 | 0.8147719502 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 3

Kept **greedy**; deciding `reconstruction+answer`; compose departure `9`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0751403645 | 0.0230197012 | 0.7793968320 | 0.8545371890 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); subject_word=13; children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); subject_word=13; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 4

Kept **tie:greedy**; deciding `tie:greedy`; compose departure `16`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0230197012 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='implies')", ['clause', 1]]; forward=(); subject_word=14; children=[2, 3]`<br>`n2 refs=[22, -1, -1]; forward=(); subject_word=14; children=[]`<br>`n3 refs=[36, 18, -1]; forward=[]; subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 5

Kept **explore**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0230329931 | 0.7595788836 | 0.7956462502 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

## fresh-final-forward-1, batch 2, document 6

Kept **greedy**; deciding `answer`; compose departure `6`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236818213 | 0.4196139872 | 0.4556813538 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0236818213 | 0.4197224677 | 0.4557898343 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |

## fresh-final-forward-1, batch 2, document 7

Kept **explore**; deciding `answer`; compose departure `17`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0236629732 | 0.7603189349 | 0.7963863015 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; subject_word=15; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); subject_word=36; children=[]` |
| explore | 0.0360673778 | 0.0230329931 | 0.7595788836 | 0.7956462502 | `n0 refs=[-1, "ClausePredicate(name='equal')", 12]; forward=[[0, 15]]; subject_word=15; children=[]` |

