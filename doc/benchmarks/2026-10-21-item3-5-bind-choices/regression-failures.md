# Frozen-source regression failures

Source `575cab06cb7d2ecb94a0c02cf64cb71bc6de33b0`: 53 passed, six failed. Existing assertions are unchanged. These failures are not waived. Three require a zero-score candidate to bind; three expect count two where the current noun-opening/refinement path records three occurrences. Claude must assess mechanism correctness and the exact decision needed from Alec.

## test/test_reference_arrivals.py::test_arrival_bind_refines_and_reads_three_through_the_token

```text
test/test_reference_arrivals.py:92: in test_arrival_bind_refines_and_reads_three_through_the_token
    assert entry.kind == words['three'] and entry.count == 2
E   AssertionError: assert (2 == 2 and 3 == 2)
E    +  where 2 = Token(occurrences=(('ltm', 'address-v1', 6722543677319378513), ('ltm', 'address-v1', -4641033302401945218), ('ltm', 'address-v1', 7737524838193717520)), kind=2, count=3, when=9).kind
E    +  and   3 = Token(occurrences=(('ltm', 'address-v1', 6722543677319378513), ('ltm', 'address-v1', -4641033302401945218), ('ltm', 'address-v1', 7737524838193717520)), kind=2, count=3, when=9).count
```

## test/test_reference_arrivals.py::test_bound_use_does_not_assign_a_kind_before_arrival

```text
test/test_reference_arrivals.py:173: in test_bound_use_does_not_assign_a_kind_before_arrival
    assert plan['items'][0]['reference'] == token
E   AssertionError: assert ('ltm', 'addr...3115727082221) == ('ltm', 'addr...3677319378513)
E     
E     At index 2 diff: 929043115727082221 != 6722543677319378513
E     Use -v to get more diff
```

## test/test_reference_arrivals.py::test_named_trial_plans_only_refine_the_kept_close[False]

```text
test/test_reference_arrivals.py:227: in test_named_trial_plans_only_refine_the_kept_close
    assert pool.entries[token].count == 2 and pool.entries[token].kind == words['three']
E   AssertionError: assert (3 == 2)
E    +  where 3 = Token(occurrences=(('ltm', 'address-v1', 6722543677319378513), ('ltm', 'address-v1', -333381716561690737), ('ltm', 'address-v1', -5284066160171494554)), kind=2, count=3, when=9).count
```

## test/test_reference_arrivals.py::test_named_trial_plans_only_refine_the_kept_close[True]

```text
test/test_reference_arrivals.py:227: in test_named_trial_plans_only_refine_the_kept_close
    assert pool.entries[token].count == 2 and pool.entries[token].kind == words['three']
E   AssertionError: assert (3 == 2)
E    +  where 3 = Token(occurrences=(('ltm', 'address-v1', 6722543677319378513), ('ltm', 'address-v1', -333381716561690737), ('ltm', 'address-v1', -5284066160171494554)), kind=2, count=3, when=9).count
```

## test/test_reference_arrivals.py::test_named_bind_to_a_native_individual_creates_no_pool_entry

```text
test/test_reference_arrivals.py:247: in test_named_bind_to_a_native_individual_creates_no_pool_entry
    assert plan['items'][0]['reference'] == native
E   AssertionError: assert ('ltm', 'address-v1', 6722543677319378513) == 4
```

## test/test_reference_arrivals.py::test_named_bind_to_a_retained_unpooled_occurrence_keeps_its_kind

```text
test/test_reference_arrivals.py:262: in test_named_bind_to_a_retained_unpooled_occurrence_keeps_its_kind
    assert plan['items'][0]['reference'] == store.occurrence_of(antecedent)
E   AssertionError: assert ('ltm', 'addr...7406946228817) == ('ltm', 'addr...1716561690737)
E     
E     At index 2 diff: 5014767406946228817 != -333381716561690737
E     Use -v to get more diff
```
