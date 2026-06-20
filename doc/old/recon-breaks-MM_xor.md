# Recon — REAL fullgraph=True compile (MM_xor.xml)

status: BROKEN
unique_graphs: 0   calls_captured: 0

- [non-grammar] first blocking break:
    UserError: Consider annotating your code using torch._check*(). Could not guard on data-dependent expression Eq(u0, 1) (unhinted: Eq(u0, 1)).  (Size-like symbols: none)

consider using data-dependent friendly APIs such as guard_or_false, guard_or_true and statically_known_true.
Caused by: if not bool(any_pos.any()):  # bin/Spaces.py:6510 in next_word (_dynamo/variables/tensor.py:1951 in evaluate_expr)
For more information, run with TORCH_LOGS="dynamic"
For extended logs when we create symbols, also add TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u0"
If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
For more debugging help, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing

User Stack (most recent call last):
  (snipped, see stack below for prefix)
  File "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/Models.py", line 2170, in forward
    return self._forward_per_stage(inputData)
  File "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/Models.py", line 5585, in _forward_per_stage
    body_sub = s
