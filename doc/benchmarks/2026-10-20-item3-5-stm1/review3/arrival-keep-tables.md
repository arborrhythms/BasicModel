# D1 arrival keep: one table per document

Source: `266189edab25985ac1d000b80553ee398f873b69`; production and
certificate unchanged. Each arrival batch contains eight document streams.
These are premise-arrival checks, not supplied-answer or learning results.
The original test fails in both exact replays; no failure is regraded.

`exploit` is the greedy trial. E is reported but excluded from keep and
credit. `clause` references index the local node’s children. A equals
supplied-answer cost plus weighted reference-closing credit. All supplied
targets are unavailable (`None`) here. Full readings, open slots, programs,
component costs and departures are in [arrival-keep.jsonl](arrival-keep.jsonl).

## fresh-both-forward-4, document stream 0

Departure: compose round `3`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0338424258 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[1, 3]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='part')", 20]; forward=(); children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); children=[]` |

## fresh-both-forward-4, document stream 1

Departure: compose round `15`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |

## fresh-both-forward-4, document stream 2

Departure: compose round `16`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |

## fresh-both-forward-4, document stream 3

Departure: compose round `15`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |

## fresh-both-forward-4, document stream 4

Departure: compose round `13`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0338424258 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='part')", -1]; forward=[[2, 72]]; children=[]` |

## fresh-both-forward-4, document stream 5

Departure: compose round `3`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0338424258 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[1, 2]`<br>`n1 refs=[45, "ClausePredicate(name='part')", -1]; forward=[[2, 13]]; children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]` |

## fresh-both-forward-4, document stream 6

Departure: compose round `13`. Kept: **explore**. Deciding: `reconstruction`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341606997 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0300561469 | 0.0338329114 | 0.0000000000 | 0.0300561469 | `n0 refs=[-1, "ClausePredicate(name='equal')", 14]; forward=[[0, 15]]; children=[]` |

## fresh-both-forward-4, document stream 7

Departure: compose round `9`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0341997556 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0338755026 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[2, 3]`<br>`n2 refs=[14, "ClausePredicate(name='equal')", -1]; forward=[[2, 13]]; children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 0

Departure: compose round `48`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 1

Departure: compose round `3`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 2

Departure: compose round `12`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 3

Departure: compose round `6`. Kept: **greedy**. Deciding: `answer`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0238175523 | 0.1000000015 | 0.1360673755 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 4

Departure: compose round `0`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[1, 2]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 5

Departure: compose round `6`. Kept: **greedy**. Deciding: `answer`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0238175523 | 0.1000000015 | 0.1360673755 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[['clause', 0], "ClausePredicate(name='equal')", ['clause', 1]]; forward=(); children=[2, 3]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]`<br>`n3 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 6

Departure: compose round `16`. Kept: **tie:greedy**. Deciding: `tie:greedy`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0360673778 | 0.0238175523 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='part')", ['clause', 0]]; forward=(); children=[2]`<br>`n2 refs=[-1, -1, -1]; forward=(); children=[]` |

## arrival-21, document stream 7

Departure: compose round `13`. Kept: **explore**. Deciding: `reconstruction`.

| Trial | R | E (excluded) | A | R + A | Reading before admission |
|---|---:|---:|---:|---:|---|
| exploit | 0.0360673778 | 0.0241408832 | 0.0000000000 | 0.0360673778 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[-1, -1, -1]; forward=(); children=[]` |
| explore | 0.0300561469 | 0.0238175523 | 0.0000000000 | 0.0300561469 | `n0 refs=[-1, "ClausePredicate(name='equal')", ['clause', 0]]; forward=[[0, 15]]; children=[1]`<br>`n1 refs=[14, "ClausePredicate(name='part')", -1]; forward=[[2, 59]]; children=[]` |
