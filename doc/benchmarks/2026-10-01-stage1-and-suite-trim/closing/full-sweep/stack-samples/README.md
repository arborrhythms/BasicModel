# Two samples of the running sweep

Each process was sampled once for one second at 10 ms intervals with macOS
`sample`, without restarting the test or changing source. The calls' elapsed
times include this external observation. This is a brief snapshot, not a
whole-test attribution of runtime. Full native stacks and command results are
saved here.

- Worker 467 was running
  `test_output_percept_readout.py::test_model_checkpoint_preserves_readout_and_adam[False]`.
  All 78 main-thread samples were below `dynamo_call_callback`. Within that
  path, 31 samples were loading a dynamic Python extension and 42 were in
  `read`. This places the observed work in the Dynamo compilation callback,
  including extension loading and waiting for input, rather than establishing
  any expensive numerical training kernel as the bottleneck.
- Worker 475 was running
  `test_output_walk.py::test_held_answer_idea_ignores_later_staging_and_memory_context`.
  The dominant main-thread branch (78 of 82 samples) was also below
  `dynamo_call_callback`, with deeply nested Python evaluation and PyTorch
  Python-dispatch frames. This is consistent with graph capture; the sample
  does not identify which model-level operation caused that capture.

These two cases were outside the 25 cases selected for the earlier cProfile
pass. Their final durations are included in the full-sweep runtime table.
