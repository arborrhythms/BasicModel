# Native sentence-pair timing and reconstruction

These are fresh unseeded runs of the same current `data/MM_ladder.xml` protocol: four validation batches, two training warm-ups, five measured training batches, and four final validation batches, with two sentences per batch. They are not paired initializations. Compilation/setup affects total run time; the timing breakdown below uses the five warm training batches. Any later graph-capture work stays in those batches; no timing outlier is dropped.

| Source stage | Result | Before MSE | Training MSE | After MSE | Warm sentences/s | Peak worker GiB | Total seconds |
|---|---|---|---|---|---|---|---|
| [native-before](native-before/source-manifest.json) | [completed](native-before/driver.process.json) | 0.123003203 | 0.109315856 | 0.105721978 | 1.242274174 | 5.744 | 1224.15 |
| [native-prepair](native-prepair/source-manifest.json) | [completed](native-prepair/driver.process.json) | 0.126346681 | 0.110405068 | 0.108204592 | 0.707889601 | 5.290 | 920.73 |
| [native-after](native-after/source-manifest.json) | [completed](native-after/driver.process.json) | 0.114950193 | 0.113841595 | 0.118921064 | 0.491403460 | 5.349 | 1457.98 |

| Mean seconds per warm batch | Before item | Before step 5 | After step 5 |
|---|---|---|---|
| exploit_compose | 0.346739241 | 0.359940992 | 0.489593517 |
| explore_compose | 0.354745367 | 1.507843792 | 2.415393542 |
| sentence_scoring | 0.099152692 | 0.107438450 | 0.120008058 |
| exploit_backward | 0.325659650 | 0.344848800 | 0.431520875 |
| explore_backward | 0.325847750 | 0.343172500 | 0.422123950 |
| batch_backward | 0.000322817 | 0.000331667 | 0.000445825 |
| snapshot | 0.000000659 | 0.000000708 | 0.000001050 |
| restore | 0.000378733 | 0.000402267 | 0.001194042 |
| other | 0.148791776 | 0.152461976 | 0.178097008 |
| total | 1.601638684 | 2.816441150 | 4.058377866 |

Measured after/before-step-5 ratios: warm batch time **1.4410**, whole-run peak worker memory **1.0111**. Peak memory includes model construction, graph capture, evaluation and training; it is not an isolated allocation count for the two trial graphs.

The implementation retains both trial graphs through the existing shared saved-value hooks; it does not take a full model snapshot. Both costs precede both optimizer steps, and each backward restores its own perception pullback. Prefix-sharing remains future work.

The historical packed/single fixture remains blocked by its retired `WholeSpace.propertyBasis` configuration element. The failed attempts are recorded separately; no configuration was changed to manufacture a new parity result.
