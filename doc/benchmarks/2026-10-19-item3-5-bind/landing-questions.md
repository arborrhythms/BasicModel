# Landing: native question receipts

One table per companion question. No question is dropped.

## zero: chain 1, document 0

| Measurement | Result |
|---|---|
| pair | `[3, 1]` |
| answer | `"four"` |
| bound_correct | `true` |
| expected_chain_length | `1` |
| inference_count | `1` |
| chain_correct | `true` |
| work | `175` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", -284838226991552333]` |

Full diagnostics: [decomposition-1/questions.jsonl](bind-decomposition-zero-landing-reports/test_forced_native_decomposition_through_training_driver[1]/decomposition-1/questions.jsonl).

## zero: chain 1, document 1

| Measurement | Result |
|---|---|
| pair | `[3, 1]` |
| answer | `"four"` |
| bound_correct | `true` |
| expected_chain_length | `1` |
| inference_count | `1` |
| chain_correct | `true` |
| work | `178` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", 573548959258110783]` |

Full diagnostics: [decomposition-1/questions.jsonl](bind-decomposition-zero-landing-reports/test_forced_native_decomposition_through_training_driver[1]/decomposition-1/questions.jsonl).

## zero: chain 2, document 0

| Measurement | Result |
|---|---|
| pair | `[3, 2]` |
| answer | `"five"` |
| bound_correct | `true` |
| expected_chain_length | `2` |
| inference_count | `2` |
| chain_correct | `true` |
| work | `455` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", -3336032979067523229]` |

Full diagnostics: [decomposition-2/questions.jsonl](bind-decomposition-zero-landing-reports/test_forced_native_decomposition_through_training_driver[2]/decomposition-2/questions.jsonl).

## zero: chain 2, document 1

| Measurement | Result |
|---|---|
| pair | `[3, 2]` |
| answer | `"five"` |
| bound_correct | `true` |
| expected_chain_length | `2` |
| inference_count | `2` |
| chain_correct | `true` |
| work | `465` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", 1450819904323826519]` |

Full diagnostics: [decomposition-2/questions.jsonl](bind-decomposition-zero-landing-reports/test_forced_native_decomposition_through_training_driver[2]/decomposition-2/questions.jsonl).

## enabled: chain 1, document 0

| Measurement | Result |
|---|---|
| pair | `[3, 1]` |
| answer | `"four"` |
| bound_correct | `true` |
| expected_chain_length | `1` |
| inference_count | `1` |
| chain_correct | `true` |
| work | `175` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", -293069186471588769]` |

Full diagnostics: [decomposition-1/questions.jsonl](bind-decomposition-enabled-landing-reports/test_forced_native_decomposition_through_training_driver[1]/decomposition-1/questions.jsonl).

## enabled: chain 1, document 1

| Measurement | Result |
|---|---|
| pair | `[3, 1]` |
| answer | `"four"` |
| bound_correct | `true` |
| expected_chain_length | `1` |
| inference_count | `1` |
| chain_correct | `true` |
| work | `178` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", -3238957211043607807]` |

Full diagnostics: [decomposition-1/questions.jsonl](bind-decomposition-enabled-landing-reports/test_forced_native_decomposition_through_training_driver[1]/decomposition-1/questions.jsonl).

## enabled: chain 2, document 0

| Measurement | Result |
|---|---|
| pair | `[3, 2]` |
| answer | `"five"` |
| bound_correct | `true` |
| expected_chain_length | `2` |
| inference_count | `2` |
| chain_correct | `true` |
| work | `455` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", 78251609764091395]` |

Full diagnostics: [decomposition-2/questions.jsonl](bind-decomposition-enabled-landing-reports/test_forced_native_decomposition_through_training_driver[2]/decomposition-2/questions.jsonl).

## enabled: chain 2, document 1

| Measurement | Result |
|---|---|
| pair | `[3, 2]` |
| answer | `"five"` |
| bound_correct | `true` |
| expected_chain_length | `2` |
| inference_count | `2` |
| chain_correct | `true` |
| work | `465` |
| open | `[]` |
| committed_binding | `["ltm", "address-v1", 9061003334637649752]` |

Full diagnostics: [decomposition-2/questions.jsonl](bind-decomposition-enabled-landing-reports/test_forced_native_decomposition_through_training_driver[2]/decomposition-2/questions.jsonl).

## G1: every generated site at the landing

| Case | Recorded result |
|---|---|
| `0-1` | `{"case": "0-1", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a cat ; a dog ; the cat"}` |
| `0-2` | `{"case": "0-2", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a cat ; a bird ; the cat"}` |
| `0-3` | `{"case": "0-3", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a cat ; a horse ; the cat"}` |
| `1-0` | `{"case": "1-0", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a dog ; a cat ; the dog"}` |
| `1-2` | `{"case": "1-2", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a dog ; a bird ; the dog"}` |
| `1-3` | `{"case": "1-3", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a dog ; a horse ; the dog"}` |
| `2-0` | `{"case": "2-0", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a bird ; a cat ; the bird"}` |
| `2-1` | `{"case": "2-1", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a bird ; a dog ; the bird"}` |
| `2-3` | `{"case": "2-3", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a bird ; a horse ; the bird"}` |
| `3-0` | `{"case": "3-0", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a horse ; a cat ; the horse"}` |
| `3-1` | `{"case": "3-1", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a horse ; a dog ; the horse"}` |
| `3-2` | `{"case": "3-2", "chosen": 12, "expected": 12, "pair": [0.8175744414329529, 0.18242555856704712], "text": "a horse ; a bird ; the horse"}` |

## G2: every generated site at the landing

| Case | Recorded result |
|---|---|
| `0-1` | `{"case": "0-1", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a cat ; a dog ; m01 chases"}` |
| `0-2` | `{"case": "0-2", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a cat ; a bird ; m02 chases"}` |
| `0-3` | `{"case": "0-3", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a cat ; a horse ; m03 chases"}` |
| `1-0` | `{"case": "1-0", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a dog ; a cat ; m10 chases"}` |
| `1-2` | `{"case": "1-2", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a dog ; a bird ; m12 chases"}` |
| `1-3` | `{"case": "1-3", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a dog ; a horse ; m13 chases"}` |
| `2-0` | `{"case": "2-0", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a bird ; a cat ; m20 chases"}` |
| `2-1` | `{"case": "2-1", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a bird ; a dog ; m21 chases"}` |
| `2-3` | `{"case": "2-3", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a bird ; a horse ; m23 chases"}` |
| `3-0` | `{"case": "3-0", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a horse ; a cat ; m30 chases"}` |
| `3-1` | `{"case": "3-1", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a horse ; a dog ; m31 chases"}` |
| `3-2` | `{"case": "3-2", "chosen": 12, "emitted": 1, "expected": 12, "pair": [0.622459352016449, 0.377540647983551], "text": "a horse ; a bird ; m32 chases"}` |

## G3: every generated site at the landing

| Case | Recorded result |
|---|---|
| `0-0` | `{"case": "0-0", "chosen": null, "expected": null, "open_roles": [["referent", 2], ["referent", 0], ["relation", 1]], "pair": [0.0, 0.0], "text": "dog chases"}` |
| `0-1` | `{"case": "0-1", "chosen": 12, "expected": 12, "open_roles": [["referent", 0], ["relation", 1]], "pair": [1.0, 0.0], "text": "dog chases"}` |
| `1-0` | `{"case": "1-0", "chosen": null, "expected": null, "open_roles": [["referent", 2], ["referent", 0], ["relation", 1]], "pair": [0.0, 0.0], "text": "bird chases"}` |
| `1-1` | `{"case": "1-1", "chosen": 12, "expected": 12, "open_roles": [["referent", 0], ["relation", 1]], "pair": [1.0, 0.0], "text": "bird chases"}` |
| `2-0` | `{"case": "2-0", "chosen": null, "expected": null, "open_roles": [["referent", 2], ["referent", 0], ["relation", 1]], "pair": [0.0, 0.0], "text": "horse chases"}` |
| `2-1` | `{"case": "2-1", "chosen": 12, "expected": 12, "open_roles": [["referent", 0], ["relation", 1]], "pair": [1.0, 0.0], "text": "horse chases"}` |
| `3-0` | `{"case": "3-0", "chosen": null, "expected": null, "open_roles": [["referent", 2], ["referent", 0], ["relation", 1]], "pair": [0.0, 0.0], "text": "cat chases"}` |
| `3-1` | `{"case": "3-1", "chosen": 12, "expected": 12, "open_roles": [["referent", 0], ["relation", 1]], "pair": [1.0, 0.0], "text": "cat chases"}` |
