# Final XOR_grammar runs

All runs are unseeded, 400 epochs, candidate only. Four answers and four grammar reconstructions are from the existing final evaluation. Input order is hello world, hello there, loving world, loving there. Every unavailable flag is retained in the JSON.

| Gate | Run | Answers | MSE | Correct | Reconstructions | Word multisets | Gate outcome |
|---|---:|---|---:|---:|---|---:|---|
| class | 1 | 0.147302091, 0.754580081, 0.778507829, 0.119591802 | 0.0363224560162 | 4/4 | 'world world hello'; 'there there hello'; 'world world there'; 'there there world' | 0/4 | PASS |
| class | 2 | 0.106560946, 0.796530545, 0.800655961, 0.209710062 | 0.0341178525276 | 4/4 | 'world world there'; 'there there world'; 'there there world'; 'there there there' | 0/4 | PASS |
| class | 3 | 0.0487349331, 0.811623812, 0.807855666, 0.150653511 | 0.0243691519227 | 4/4 | 'hello hello hello'; 'there there hello'; 'there there there'; 'there there there' | 0/4 | PASS |
| class | 4 | 0.192735255, 0.863617361, 0.912393272, 0.271387994 | 0.0342683712525 | 4/4 | 'there there there'; 'there there there'; 'hello hello there'; 'there there there' | 0/4 | PASS |
| class | 5 | 0.0763066709, 0.891216159, 0.942453265, 0.0626423955 | 0.00622308212924 | 4/4 | 'world world world'; 'there there world'; 'loving loving world'; 'loving loving there' | 0/4 | PASS |
| class | 6 | 0.0704433024, 0.886878729, 0.873637319, 0.0996176898 | 0.0109124730533 | 4/4 | 'world world loving'; 'loving loving world'; 'hello hello world'; 'hello hello world' | 0/4 | PASS |
| class | 7 | 0.0390765965, 0.709302902, 1.05353236, 0.000239819288 | 0.0222243885859 | 4/4 | 'world world hello'; 'there there hello'; 'world world world'; 'world world there' | 0/4 | PASS |
| class | 8 | 0.0487456322, 0.928587437, 0.92021054, 0.198506147 | 0.0133117347505 | 4/4 | 'world world loving'; 'world world there'; 'world world there'; 'world world there' | 0/4 | PASS |
| class | 9 | 0.102257609, 0.925530076, 0.963170171, 0.0556710958 | 0.00511452387077 | 4/4 | 'world world hello'; 'hello hello hello'; 'world world world'; 'world world world' | 0/4 | PASS |
| class | 10 | 0.124248207, 0.979367852, 0.96972847, 0.0987567902 | 0.00663314287754 | 4/4 | 'world world loving'; 'world world loving'; 'world world loving'; 'there there loving' | 0/4 | PASS |
| reconstruction | 1 | 0.149095148, 0.860105157, 0.751583159, 0.105810434 | 0.0286766762524 | 4/4 | 'there there loving'; 'there there loving'; 'there there loving'; 'there there there' | 0/4 | FAIL |
| reconstruction | 2 | 0.0734218955, 0.966059983, 0.908166587, 0.121473014 | 0.00743294210084 | 4/4 | 'hello hello world'; 'hello hello there'; 'world world there'; 'hello hello there' | 0/4 | FAIL |
| reconstruction | 3 | 0.220543444, 0.758733749, 0.781666636, 0.0303335786 | 0.0388595995164 | 4/4 | 'world world hello'; 'there there world'; 'loving loving world'; 'loving loving there' | 0/4 | FAIL |
| reconstruction | 4 | 0.0635762513, 0.902943909, 0.882522166, 0.0849323869 | 0.00861909409183 | 4/4 | 'world world there'; 'there there there'; 'loving loving world'; 'loving loving there' | 0/4 | FAIL |
| reconstruction | 5 | 0.0280781984, 0.746297956, 0.931318521, 0.0678795874 | 0.0186194741201 | 4/4 | 'world world there'; 'there there there'; 'there there hello'; 'there there world' | 0/4 | FAIL |
| reconstruction | 6 | 0.0325102806, 0.948505223, 0.975395143, 0.161741078 | 0.00761855142277 | 4/4 | 'world world hello'; 'hello hello there'; 'world world there'; 'there there there' | 0/4 | FAIL |
| reconstruction | 7 | 0.0904423594, 0.989656448, 0.853503704, 0.239799619 | 0.0218129578256 | 4/4 | 'world world world'; 'there there world'; 'world world there'; 'there there world' | 0/4 | FAIL |
| reconstruction | 8 | 0.0813472867, 0.824270189, 0.782340586, 0.161706001 | 0.0277556996591 | 4/4 | 'loving loving there'; 'there there loving'; 'there there loving'; 'there there world' | 0/4 | FAIL |
| reconstruction | 9 | 0.280592561, 1.04428005, 0.968555927, 0.0137731731 | 0.0204678345466 | 4/4 | 'hello hello world'; 'there there there'; 'loving loving loving'; 'there there loving' | 0/4 | FAIL |
| reconstruction | 10 | 0.185915977, 0.923764229, 0.835146546, 0.172803491 | 0.0243535877557 | 4/4 | 'world world hello'; 'there there hello'; 'loving loving there'; 'there there loving' | 0/4 | FAIL |
