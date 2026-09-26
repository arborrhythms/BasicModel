# Full erosion measurements

**Historical measurement.** Interleave rows used serial-first processing and
a separate label read-back. They are void for the parallel-first schedule.

Each row preserves a single run; there is no averaging across seeds.

| Probes | Seed | Schedule | CP offline | CP read-back | Within offline | Within read-back | Alternatives/concept | Lexical coverage |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| smoke | 0 | serial | 0.000000 | 0.000000 | 0.000000 | 1.414214 | 0.184524 | 4/4 |
| smoke | 0 | parallel | 0.000000 | 0.000000 | 0.500000 | 1.866025 | 0.000000 | 4/4 |
| smoke | 0 | interleave:2 | 0.000000 | -0.012472 | 0.000000 | 1.732051 | 0.184524 | 4/4 |
| smoke | 1 | serial | 0.000000 | 0.000000 | 0.000000 | 1.414214 | 0.184524 | 4/4 |
| smoke | 1 | parallel | 0.000000 | 0.000000 | 0.500000 | 1.866025 | 0.000000 | 4/4 |
| smoke | 1 | interleave:2 | 0.000000 | -0.012472 | 1.868647 | 1.732051 | 0.184524 | 4/4 |
| smoke | 2 | serial | 0.000000 | 0.000000 | 0.000000 | 1.414214 | 0.184524 | 4/4 |
| smoke | 2 | parallel | 0.000000 | 0.000000 | 0.500000 | 1.866025 | 0.000000 | 4/4 |
| smoke | 2 | interleave:2 | 0.000000 | -0.012472 | 0.000000 | 1.732051 | 0.184524 | 4/4 |
| fineweb | 0 | serial | 3.993629 | -0.099544 | 1.743104 | 1.253756 | 0.184524 | 36/68 |
| fineweb | 0 | parallel | 1.207074 | -0.067686 | 2.470575 | 0.545909 | 0.000000 | 17/68 |
| fineweb | 0 | interleave:2 | 5.289717 | 0.022804 | 5.226126 | 0.714237 | 0.184524 | 36/68 |
| fineweb | 1 | serial | 5.613404 | -0.084822 | 1.796675 | 1.153144 | 0.184524 | 36/68 |
| fineweb | 1 | parallel | 1.426923 | -0.062798 | 1.976987 | 0.508029 | 0.000000 | 17/68 |
| fineweb | 1 | interleave:2 | 3.905974 | -0.001512 | 5.327150 | 0.790025 | 0.184524 | 36/68 |
| fineweb | 2 | serial | 4.422914 | -0.093754 | 1.848517 | 1.214464 | 0.184524 | 36/68 |
| fineweb | 2 | parallel | 1.496482 | -0.062797 | 2.038432 | 0.508019 | 0.000000 | 17/68 |
| fineweb | 2 | interleave:2 | 4.496124 | 0.010632 | 5.922278 | 0.755871 | 0.184524 | 36/68 |
