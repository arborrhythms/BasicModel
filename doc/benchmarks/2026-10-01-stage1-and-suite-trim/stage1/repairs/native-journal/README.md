# Part 1 — item 7 operation-record memory repair

The four numerical records lang[22:26] now have capacity for one open sentence:
three rounds per word of the longest sentence in this batch, plus the bounded
closing. A separate small integer map translates unchanged global trace addresses
into those local records. Packed sentences reuse the bank only after their two
trials have trained and the closing has discarded its records. Inactive columns
consume no record and do not split a sentence. Word steps gather their local
windows and publish each numerical journal once.

All four records retain their existing autograd behavior. In particular, operation
values remain live through `_program_entries` into concluded meanings. No detach
was added, so no objective's reach is intentionally cut. The regression checks
that the compact program boundary carries the numerical frame gradient; existing
compiled-journal and live-reference tests pass unchanged.

Before repair: both batch-28 measurement arms exceeded 24 GiB before their first
trial cost or update (sampled peaks 24.39 / 24.33 GiB). See `../../before-journal-repair-24gib/`.
The original allocation/graph diagnostics and this receipt's saved red compact-
layout and padding probes precede their respective repairs. No assertion, seed,
production configuration or ordinary sweep guard changed.

The same graph diagnostic on a read-only `git archive` snapshot of HEAD d679df2b
confirms the defect already exists at the item 7 landing: each operation-values
scatter has shape [28,4864,3096], 1,686,601,728 bytes. The native batch's longest
sentence has three words. The repair's numerical banks use 25 slots; the value
bank is 8,668,800 bytes (8.27 MiB). Capture-only measured peaks were 10.00 GiB on
HEAD and 3.78 GiB on the candidate; these are not full training peaks. These graph
captures intentionally exit 73 before execution or any training. The padding
follow-up changes only host-side ordinal staging, preserving the graph's native
shapes. Full benchmark peak is reported by part 2.

Validation: 29 focused cases pass, including eager/compiled packed endings,
compiled journal gradients, end-state storage and live references. That group
ran in 795.68 seconds; the compiled packed test took 498.98 seconds (previously
595.94), and the complete group peaked at 3.56 GiB. After the saved padding probe
and correction, all three compact-layout/program-gradient cases pass. The full
old/new Models.py files and changed method bodies are retained, including the
intermediate implementation and the padding repair. No test port was needed.
