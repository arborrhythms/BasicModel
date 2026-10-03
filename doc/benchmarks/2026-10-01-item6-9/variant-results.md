# Ten unseeded runs of each isolated variant

Each run uses the unchanged XOR_grammar fixture at 400 epochs. The bar is
all four classes correct and MSE below .05. These patches are measurements;
none is applied to the candidate files. All results are retained.

Answers and derivations below follow: hello world, hello there, loving world,
loving there. `L0`, `L1`, `L2` are first word, separator, second word.
Weights are Claude’s pseudoinverse margins, excluding bias. A nonexact
affine solution is explicitly marked, even when its coefficient norm is small.

The final margin is for the four selected roots together: one answer map
must satisfy all four. The best-available search applies each enumerated
derivation to all four sentences, using the captured leaves at that epoch.

| Variant | Passes | Four correct | Median MSE | MSE range |
|---|---:|---:|---:|---|
| a | 0/10 | 1/10 | 0.238214331 | 0.209177909–0.37539441 |
| b | 0/10 | 0/10 | 0.293988354 | 0.252075358–0.395425206 |
| c | 9/10 | 9/10 | 0.00109924769 | 2.41581392e-06–0.287380491 |

## Variant a

### Run 1

[Full record](measurements/a-01/measurement.json): answers **[0.43160212, 0.412848622, 0.428477436, 0.409723967]**; MSE **0.2563847254**; **2/4** correct; bar **FAIL**.

```text
hello world: min(min(not(L0),not(L1)),L2)
hello there: min(min(not(L0),not(L1)),L2)
loving world: min(min(not(L0),not(L1)),L2)
loving there: min(min(not(L0),not(L1)),L2)
```

Final derivation margin: **no exact affine solution**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),not(L2))`; max weight 3.91158715; L2 7.03692708; 24/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,min(L1,not(L2)))`; max weight 14.8777432; L2 22.1974237; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,min(L1,not(L2)))`; max weight 14.650032; L2 21.6405405; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-01/trials.jsonl) · [Equal-parameter comparisons](measurements/a-01/pairs.jsonl) · [Process and guard](measurements/a-01/process.json)

### Run 2

[Full record](measurements/a-02/measurement.json): answers **[0.431177735, 0.496978819, 0.506187379, 0.516350627]**; MSE **0.2373533555**; **2/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,L1),L2)
hello there: min(min(L0,L1),L2)
loving world: min(min(L0,L1),L2)
loving there: min(min(L0,L1),L2)
```

Final derivation margin: **max weight 29.7420873; L2 41.5958196**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),not(L2))`; max weight 4.42087647; L2 6.31988049; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),L2)`; max weight 18.381745; L2 25.5865473; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),L2)`; max weight 18.6026603; L2 25.7868517; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-02/trials.jsonl) · [Equal-parameter comparisons](measurements/a-02/pairs.jsonl) · [Process and guard](measurements/a-02/process.json)

### Run 3

[Full record](measurements/a-03/measurement.json): answers **[0.473920047, 0.45496425, 0.524990082, 0.460940182]**; MSE **0.2399411134**; **3/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,L1),not(L2))
hello there: min(min(L0,L1),L2)
loving world: min(min(not(L0),L1),not(L2))
loving there: min(min(not(L0),L1),L2)
```

Final derivation margin: **max weight 43.5376746; L2 51.0209624**.

- First epoch before updates (epoch 0): `min(not(L0),min(L1,L2))`; max weight 5.92149687; L2 9.08786611; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),L2)`; max weight 3.30060427; L2 5.95984153; 36/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),L2)`; max weight 3.31675561; L2 5.96178586; 36/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-03/trials.jsonl) · [Equal-parameter comparisons](measurements/a-03/pairs.jsonl) · [Process and guard](measurements/a-03/process.json)

### Run 4

[Full record](measurements/a-04/measurement.json): answers **[0.531401753, 0.441343427, 0.510734439, 0.430467308]**; MSE **0.2547919706**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(not(L0),L1),L2)
loving there: max(max(not(L0),L1),L2)
```

Final derivation margin: **max weight 60.1014344; L2 70.9801909**.

- First epoch before updates (epoch 0): `min(not(L0),min(L1,not(L2)))`; max weight 3.5820723; L2 7.19027422; 8/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(L0,L1)),L2)`; max weight 3.18176096; L2 5.92767608; 32/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),L2)`; max weight 3.1893584; L2 5.94010143; 32/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-04/trials.jsonl) · [Equal-parameter comparisons](measurements/a-04/pairs.jsonl) · [Process and guard](measurements/a-04/process.json)

### Run 5

[Full record](measurements/a-05/measurement.json): answers **[0.466894716, 0.595594764, 0.406399459, 0.320648968]**; MSE **0.2091779086**; **3/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,L1),not(L2))
hello there: min(min(L0,L1),L2)
loving world: min(min(L0,L1),not(L2))
loving there: min(min(L0,L1),L2)
```

Final derivation margin: **max weight 12.4007565; L2 16.8675324**.

- First epoch before updates (epoch 0): `min(L0,min(L1,not(L2)))`; max weight 4.84654128; L2 7.69518847; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),min(L1,L2))`; max weight 11.9927829; L2 13.8336845; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),not(L2))`; max weight 11.9902839; L2 13.5475148; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-05/trials.jsonl) · [Equal-parameter comparisons](measurements/a-05/pairs.jsonl) · [Process and guard](measurements/a-05/process.json)

### Run 6

[Full record](measurements/a-06/measurement.json): answers **[0.449632287, 0.568392754, 0.35850203, 0.395382822]**; MSE **0.2390753073**; **3/4** correct; bar **FAIL**.

```text
hello world: max(max(not(L0),L1),L2)
hello there: max(max(not(L0),L1),L2)
loving world: max(max(not(L0),L1),L2)
loving there: max(max(not(L0),L1),L2)
```

Final derivation margin: **max weight 29.2668577; L2 49.1579023**.

- First epoch before updates (epoch 0): `min(not(L0),min(L1,not(L2)))`; max weight 4.57609459; L2 8.91549366; 20/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 17.0301102; L2 22.4536112; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 16.9720981; L2 22.3921105; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-06/trials.jsonl) · [Equal-parameter comparisons](measurements/a-06/pairs.jsonl) · [Process and guard](measurements/a-06/process.json)

### Run 7

[Full record](measurements/a-07/measurement.json): answers **[0.188075513, 0.124848127, 0.803272724, 0.200335503]**; MSE **0.2200247833**; **3/4** correct; bar **FAIL**.

```text
hello world: min(min(not(L0),L1),L2)
hello there: min(min(not(L0),L1),L2)
loving world: not(min(not(min(L0,L1)),L2))
loving there: min(not(min(L0,L1)),L2)
```

Final derivation margin: **max weight 2.70775209; L2 3.20951416**.

- First epoch before updates (epoch 0): `min(L0,min(L1,L2))`; max weight 4.11447629; L2 8.4679064; 8/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 4.35166004; L2 9.2459501; 28/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 4.3412394; L2 9.23767394; 28/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-07/trials.jsonl) · [Equal-parameter comparisons](measurements/a-07/pairs.jsonl) · [Process and guard](measurements/a-07/process.json)

### Run 8

[Full record](measurements/a-08/measurement.json): answers **[0.468795449, 0.549361169, 0.506108463, 0.431515962]**; MSE **0.213244851**; **4/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,L1),not(L2))
hello there: min(min(L0,L1),not(L2))
loving world: min(min(L0,L1),not(L2))
loving there: min(min(L0,L1),not(L2))
```

Final derivation margin: **max weight 18.2308262; L2 24.1128886**.

- First epoch before updates (epoch 0): `min(not(L0),min(L1,not(L2)))`; max weight 4.69966982; L2 7.43975515; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(L0,L1)),L2)`; max weight 13.7726354; L2 20.8735037; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),L2)`; max weight 13.8533384; L2 20.8870316; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-08/trials.jsonl) · [Equal-parameter comparisons](measurements/a-08/pairs.jsonl) · [Process and guard](measurements/a-08/process.json)

### Run 9

[Full record](measurements/a-09/measurement.json): answers **[0.467176378, 0.454094529, 0.557709157, 0.412445545]**; MSE **0.220499767**; **3/4** correct; bar **FAIL**.

```text
hello world: max(max(not(L0),L1),L2)
hello there: max(max(not(L0),L1),L2)
loving world: max(max(L0,L1),L2)
loving there: max(max(L0,L1),L2)
```

Final derivation margin: **max weight 25.8466616; L2 36.2061831**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),L2)`; max weight 4.13550213; L2 7.57622389; 20/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 3.40401697; L2 6.1085834; 40/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 3.40736135; L2 6.1096655; 40/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-09/trials.jsonl) · [Equal-parameter comparisons](measurements/a-09/pairs.jsonl) · [Process and guard](measurements/a-09/process.json)

### Run 10

[Full record](measurements/a-10/measurement.json): answers **[0.101042062, 0.112436652, 0.184351146, 0.195745736]**; MSE **0.3753944102**; **2/4** correct; bar **FAIL**.

```text
hello world: not(min(min(not(L0),not(L1)),L2))
hello there: not(min(min(not(L0),not(L1)),L2))
loving world: not(min(min(not(L0),not(L1)),L2))
loving there: not(min(min(not(L0),not(L1)),L2))
```

Final derivation margin: **max weight 84289880.2; L2 121737575**.

- First epoch before updates (epoch 0): `min(L0,min(L1,L2))`; max weight 4.58327914; L2 9.72619633; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),min(L1,L2))`; max weight 7.26015616; L2 12.0270068; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),min(L1,L2))`; max weight 7.28244127; L2 12.0149184; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/a-10/trials.jsonl) · [Equal-parameter comparisons](measurements/a-10/pairs.jsonl) · [Process and guard](measurements/a-10/process.json)


## Variant b

### Run 1

[Full record](measurements/b-01/measurement.json): answers **[0.68102777, 0.705595732, 0.67844069, 0.712356389]**; MSE **0.2903311777**; **2/4** correct; bar **FAIL**.

```text
hello world: min(min(not(L0),not(L1)),L2)
hello there: min(min(not(L0),not(L1)),L2)
loving world: min(min(not(L0),not(L1)),L2)
loving there: min(min(not(L0),not(L1)),L2)
```

Final derivation margin: **max weight 89.2210535; L2 95.6247277**.

- First epoch before updates (epoch 0): `min(not(L0),not(min(L1,L2)))`; max weight 5.93913093; L2 9.73774152; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),not(min(L1,not(L2))))`; max weight 5.18306846; L2 9.44189293; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),not(min(L1,not(L2))))`; max weight 5.22435331; L2 9.38813754; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-01/trials.jsonl) · [Equal-parameter comparisons](measurements/b-01/pairs.jsonl) · [Process and guard](measurements/b-01/process.json)

### Run 2

[Full record](measurements/b-02/measurement.json): answers **[0.966001511, 1.32522941, 0.602472961, 0.620274127]**; MSE **0.395425206**; **2/4** correct; bar **FAIL**.

```text
hello world: min(min(not(L0),L1),L2)
hello there: min(min(not(L0),L1),L2)
loving world: min(min(not(L0),L1),L2)
loving there: min(min(not(L0),L1),L2)
```

Final derivation margin: **max weight 4.55251084; L2 6.45810153**.

- First epoch before updates (epoch 0): `min(not(min(L0,L1)),not(L2))`; max weight 5.02877063; L2 7.95022538; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),not(L2))`; max weight 4.12729579; L2 6.28125276; 56/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),not(L2))`; max weight 4.08963902; L2 6.27820183; 56/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-02/trials.jsonl) · [Equal-parameter comparisons](measurements/b-02/pairs.jsonl) · [Process and guard](measurements/b-02/process.json)

### Run 3

[Full record](measurements/b-03/measurement.json): answers **[0.492521644, 0.560155511, 0.509905815, 0.59279263]**; MSE **0.2569090391**; **3/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(L0,L1),L2)
loving there: max(max(L0,L1),L2)
```

Final derivation margin: **max weight 84.9080793; L2 116.848462**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 4.54653296; L2 9.61807856; 28/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),min(L1,L2))`; max weight 11.906803; L2 17.2887904; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),min(L1,L2))`; max weight 11.9310378; L2 16.8606403; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-03/trials.jsonl) · [Equal-parameter comparisons](measurements/b-03/pairs.jsonl) · [Process and guard](measurements/b-03/process.json)

### Run 4

[Full record](measurements/b-04/measurement.json): answers **[0.317565918, 0.318789244, 0.181159139, 0.183927536]**; MSE **0.3173064754**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(not(L0),L1),L2)
loving there: max(max(not(L0),L1),L2)
```

Final derivation margin: **max weight 160.35581; L2 239.473621**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),not(L2))`; max weight 3.57848802; L2 6.71633901; 36/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),L2)`; max weight 3.04774807; L2 6.0148038; 16/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),L2)`; max weight 3.06824972; L2 5.97313329; 16/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-04/trials.jsonl) · [Equal-parameter comparisons](measurements/b-04/pairs.jsonl) · [Process and guard](measurements/b-04/process.json)

### Run 5

[Full record](measurements/b-05/measurement.json): answers **[0.468715429, 0.913524985, 0.540324986, 0.949666619]**; MSE **0.335084972**; **3/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,L1),L2)
hello there: min(min(L0,L1),not(L2))
loving world: min(min(L0,L1),L2)
loving there: min(min(L0,L1),not(L2))
```

Final derivation margin: **max weight 24.1662936; L2 26.8788428**.

- First epoch before updates (epoch 0): `min(L0,min(L1,L2))`; max weight 4.09928427; L2 6.26706618; 24/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,not(L2))))`; max weight 16.8990683; L2 21.626009; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,not(L2))))`; max weight 17.9947767; L2 22.6040048; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-05/trials.jsonl) · [Equal-parameter comparisons](measurements/b-05/pairs.jsonl) · [Process and guard](measurements/b-05/process.json)

### Run 6

[Full record](measurements/b-06/measurement.json): answers **[0.44429338, 0.446147025, 0.465188503, 0.467042148]**; MSE **0.2520753577**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(not(L0),L1),L2)
hello there: max(max(not(L0),L1),L2)
loving world: max(max(L0,L1),L2)
loving there: max(max(L0,L1),L2)
```

Final derivation margin: **no exact affine solution**.

- First epoch before updates (epoch 0): `min(not(min(L0,L1)),not(L2))`; max weight 4.58012451; L2 8.95367715; 4/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 4.85445957; L2 7.9298434; 20/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),L2)`; max weight 4.77826905; L2 6.5882177; 20/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-06/trials.jsonl) · [Equal-parameter comparisons](measurements/b-06/pairs.jsonl) · [Process and guard](measurements/b-06/process.json)

### Run 7

[Full record](measurements/b-07/measurement.json): answers **[0.263706565, 0.262699246, 0.302506626, 0.3015486]**; MSE **0.2976455298**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(not(L0),L1),L2)
loving there: max(max(not(L0),L1),L2)
```

Final derivation margin: **max weight 2511.52313; L2 3558.01912**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 3.12073514; L2 6.05992233; 36/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),not(L2))`; max weight 4.49641835; L2 8.57350277; 12/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),not(L2))`; max weight 4.46883088; L2 8.6983326; 12/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-07/trials.jsonl) · [Equal-parameter comparisons](measurements/b-07/pairs.jsonl) · [Process and guard](measurements/b-07/process.json)

### Run 8

[Full record](measurements/b-08/measurement.json): answers **[0.624091506, 0.744674206, 0.625379324, 0.717403889]**; MSE **0.2774226149**; **2/4** correct; bar **FAIL**.

```text
hello world: min(min(L0,not(L1)),L2)
hello there: min(min(L0,not(L1)),L2)
loving world: min(min(not(L0),not(L1)),L2)
loving there: min(min(not(L0),not(L1)),L2)
```

Final derivation margin: **max weight 108.943396; L2 126.947159**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 4.62423375; L2 6.90390065; 4/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),not(min(L1,L2)))`; max weight 4.72816321; L2 8.90442723; 4/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),not(min(L1,L2)))`; max weight 4.63373848; L2 8.83321374; 4/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-08/trials.jsonl) · [Equal-parameter comparisons](measurements/b-08/pairs.jsonl) · [Process and guard](measurements/b-08/process.json)

### Run 9

[Full record](measurements/b-09/measurement.json): answers **[0.141317815, 0.165380746, 0.0723940432, 0.0574738681]**; MSE **0.3950790201**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(L0,L1),L2)
loving there: max(max(L0,L1),L2)
```

Final derivation margin: **max weight 23.3840166; L2 28.5989779**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,not(L2))))`; max weight 4.25453135; L2 7.36132863; 8/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 11.5755773; L2 14.5920306; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 10.8334695; L2 13.9715269; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-09/trials.jsonl) · [Equal-parameter comparisons](measurements/b-09/pairs.jsonl) · [Process and guard](measurements/b-09/process.json)

### Run 10

[Full record](measurements/b-10/measurement.json): answers **[0.515677214, 0.395870179, 0.511353731, 0.394905031]**; MSE **0.2564052472**; **2/4** correct; bar **FAIL**.

```text
hello world: max(max(L0,L1),L2)
hello there: max(max(L0,L1),L2)
loving world: max(max(not(L0),L1),L2)
loving there: max(max(not(L0),L1),L2)
```

Final derivation margin: **max weight 406.167034; L2 481.595159**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 4.87179175; L2 9.33902665; 4/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),not(L2))`; max weight 3.09164255; L2 5.5305371; 40/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),not(L2))`; max weight 3.06480741; L2 5.51108126; 40/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/b-10/trials.jsonl) · [Equal-parameter comparisons](measurements/b-10/pairs.jsonl) · [Process and guard](measurements/b-10/process.json)


## Variant c

### Run 1

[Full record](measurements/c-01/measurement.json): answers **[1.03663135, 0.98138392, 0.72964561, 0.0384624898]**; MSE **0.2873804912**; **3/4** correct; bar **FAIL**.

```text
hello world: not(min(min(not(L0),L1),L2))
hello there: not(min(min(not(L0),L1),L2))
loving world: min(min(not(L0),L1),L2)
loving there: min(min(not(L0),L1),L2)
```

Final derivation margin: **max weight 4.02665179; L2 6.56962159**.

- First epoch before updates (epoch 0): `min(not(L0),not(min(L1,not(L2))))`; max weight 2.56824061; L2 5.38182337; 64/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),not(min(L1,not(L2))))`; max weight 3.06137458; L2 5.6639647; 20/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),not(min(L1,not(L2))))`; max weight 3.08351372; L2 5.80636623; 20/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-01/trials.jsonl) · [Equal-parameter comparisons](measurements/c-01/pairs.jsonl) · [Process and guard](measurements/c-01/process.json)

### Run 2

[Full record](measurements/c-02/measurement.json): answers **[0.105662525, 1.06404352, 0.827253997, 0.0679905415]**; MSE **0.01243250925**; **4/4** correct; bar **PASS**.

```text
hello world: not(min(min(L0,L1),L2))
hello there: min(min(L0,L1),L2)
loving world: not(min(min(L0,L1),L2))
loving there: min(min(L0,L1),L2)
```

Final derivation margin: **max weight 2.36760186; L2 3.20416657**.

- First epoch before updates (epoch 0): `min(not(min(L0,L1)),not(L2))`; max weight 3.53974038; L2 4.45487528; 28/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 4.7515877; L2 8.96876863; 4/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 4.92649663; L2 9.15454252; 4/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-02/trials.jsonl) · [Equal-parameter comparisons](measurements/c-02/pairs.jsonl) · [Process and guard](measurements/c-02/process.json)

### Run 3

[Full record](measurements/c-03/measurement.json): answers **[0.0152574778, 0.969215512, 0.986330867, -0.0070425272]**; MSE **0.0003542294259**; **4/4** correct; bar **PASS**.

```text
hello world: min(min(not(L0),L1),L2)
hello there: min(min(not(L0),L1),L2)
loving world: min(min(L0,L1),L2)
loving there: min(min(L0,L1),L2)
```

Final derivation margin: **max weight 2.64734014; L2 3.82265665**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 5.74496505; L2 9.60732232; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 7.74671541; L2 13.5631315; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,not(L2))))`; max weight 7.69702125; L2 12.1934079; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-03/trials.jsonl) · [Equal-parameter comparisons](measurements/c-03/pairs.jsonl) · [Process and guard](measurements/c-03/process.json)

### Run 4

[Full record](measurements/c-04/measurement.json): answers **[0.00657570362, 0.989049792, 1.00165987, -0.0011806488]**; MSE **4.182400688e-05**; **4/4** correct; bar **PASS**.

```text
hello world: min(min(L0,L1),L2)
hello there: not(min(min(L0,L1),L2))
loving world: min(min(L0,L1),L2)
loving there: not(min(min(L0,L1),L2))
```

Final derivation margin: **max weight 1.53162557; L2 1.94497646**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,L2)))`; max weight 13.0635961; L2 14.9339631; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,not(L2))))`; max weight 8.09800804; L2 12.1678025; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,not(L2))))`; max weight 8.17565122; L2 12.2337144; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-04/trials.jsonl) · [Equal-parameter comparisons](measurements/c-04/pairs.jsonl) · [Process and guard](measurements/c-04/process.json)

### Run 5

[Full record](measurements/c-05/measurement.json): answers **[0.00541770458, 1.0098573, 0.985422313, 0.00281190872]**; MSE **8.673340217e-05**; **4/4** correct; bar **PASS**.

```text
hello world: min(min(L0,L1),not(L2))
hello there: min(min(L0,L1),L2)
loving world: min(min(not(L0),L1),not(L2))
loving there: min(min(not(L0),L1),L2)
```

Final derivation margin: **max weight 1.87388577; L2 3.04866787**.

- First epoch before updates (epoch 0): `min(not(L0),min(L1,not(L2)))`; max weight 6.20743377; L2 11.0419048; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(L0,not(min(L1,L2)))`; max weight 2.65721169; L2 4.53750863; 64/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(L0,not(min(L1,L2)))`; max weight 2.68010021; L2 4.55149739; 64/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-05/trials.jsonl) · [Equal-parameter comparisons](measurements/c-05/pairs.jsonl) · [Process and guard](measurements/c-05/process.json)

### Run 6

[Full record](measurements/c-06/measurement.json): answers **[0.0896017253, 0.815571189, 0.891831279, 0.0627318323]**; MSE **0.01441955264**; **4/4** correct; bar **PASS**.

```text
hello world: min(min(L0,L1),L2)
hello there: min(min(L0,L1),not(L2))
loving world: min(min(not(L0),L1),L2)
loving there: min(min(not(L0),L1),not(L2))
```

Final derivation margin: **max weight 2.28259558; L2 3.76381692**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),not(L2))`; max weight 5.08472979; L2 8.67876613; 0/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(L0,L1)),L2)`; max weight 2.83252587; L2 5.07258179; 64/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),L2)`; max weight 2.8466809; L2 5.04249643; 64/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-06/trials.jsonl) · [Equal-parameter comparisons](measurements/c-06/pairs.jsonl) · [Process and guard](measurements/c-06/process.json)

### Run 7

[Full record](measurements/c-07/measurement.json): answers **[0.0221112967, 1.00292981, 1.01057947, -0.00457173586]**; MSE **0.0001575797735**; **4/4** correct; bar **PASS**.

```text
hello world: min(min(L0,L1),L2)
hello there: min(min(L0,L1),not(L2))
loving world: min(min(not(L0),L1),L2)
loving there: min(min(not(L0),L1),not(L2))
```

Final derivation margin: **max weight 2.68118427; L2 3.68281084**.

- First epoch before updates (epoch 0): `min(not(min(not(L0),L1)),not(L2))`; max weight 4.23981083; L2 7.48038236; 4/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(L0,L1)),not(L2))`; max weight 4.19557279; L2 5.95848598; 32/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(L0,L1)),not(L2))`; max weight 4.14604922; L2 5.9255346; 32/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-07/trials.jsonl) · [Equal-parameter comparisons](measurements/c-07/pairs.jsonl) · [Process and guard](measurements/c-07/process.json)

### Run 8

[Full record](measurements/c-08/measurement.json): answers **[0.0636415482, 1.04838574, 0.970704377, -0.0112873316]**; MSE **0.001844265955**; **4/4** correct; bar **PASS**.

```text
hello world: not(min(min(L0,L1),L2))
hello there: not(min(min(L0,L1),L2))
loving world: not(min(min(L0,L1),L2))
loving there: not(min(min(L0,L1),L2))
```

Final derivation margin: **max weight 2.22326719; L2 3.38779302**.

- First epoch before updates (epoch 0): `min(not(min(L0,L1)),not(L2))`; max weight 3.75193229; L2 7.17440239; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),L2)`; max weight 2.00607566; L2 3.64891994; 64/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),min(L1,not(L2)))`; max weight 2.04772288; L2 3.29392253; 64/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-08/trials.jsonl) · [Equal-parameter comparisons](measurements/c-08/pairs.jsonl) · [Process and guard](measurements/c-08/process.json)

### Run 9

[Full record](measurements/c-09/measurement.json): answers **[0.0768100917, 0.776862502, 0.962239504, -0.00192064047]**; MSE **0.01427991927**; **4/4** correct; bar **PASS**.

```text
hello world: not(min(min(L0,L1),L2))
hello there: not(min(min(L0,L1),not(L2)))
loving world: min(min(not(L0),L1),L2)
loving there: not(min(min(not(L0),L1),L2))
```

Final derivation margin: **max weight 1.57624513; L2 1.86176174**.

- First epoch before updates (epoch 0): `min(not(min(L0,L1)),L2)`; max weight 4.43694495; L2 8.04514263; 12/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(L0),min(L1,L2))`; max weight 3.60443243; L2 5.57256211; 28/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(L0),min(L1,L2))`; max weight 3.62114563; L2 5.58268825; 24/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-09/trials.jsonl) · [Equal-parameter comparisons](measurements/c-09/pairs.jsonl) · [Process and guard](measurements/c-09/process.json)

### Run 10

[Full record](measurements/c-10/measurement.json): answers **[-0.00246906281, 1.00102019, 1.00079668, -0.00137531757]**; MSE **2.415813917e-06**; **4/4** correct; bar **PASS**.

```text
hello world: not(min(min(L0,L1),not(L2)))
hello there: min(min(L0,L1),L2)
loving world: min(min(not(L0),L1),not(L2))
loving there: min(min(not(L0),L1),L2)
```

Final derivation margin: **max weight 1.46368544; L2 2.09974418**.

- First epoch before updates (epoch 0): `min(L0,not(min(L1,not(L2))))`; max weight 3.3982837; L2 6.57705025; 60/128 enumerated derivations readable at weight ≤5.
- Last epoch before updates (epoch 399): `min(not(min(not(L0),L1)),not(L2))`; max weight 5.02505042; L2 11.5471206; 0/128 enumerated derivations readable at weight ≤5.
- Final evaluation after training: `min(not(min(not(L0),L1)),not(L2))`; max weight 5.07906667; L2 11.5898617; 0/128 enumerated derivations readable at weight ≤5.

[All trial choices](measurements/c-10/trials.jsonl) · [Equal-parameter comparisons](measurements/c-10/pairs.jsonl) · [Process and guard](measurements/c-10/process.json)

