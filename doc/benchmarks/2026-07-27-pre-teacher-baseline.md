# Pre-Teacher one-hour baseline

The comparison artifact is preserved at:

```text
output/runs/basicmodel_1h_b24_20260726_075933/
```

The final completed run in `launchd.out.log` reported:

| Metric | Value |
|---|---:|
| Device / compile | MPS, Inductor `default`, fullgraph |
| Batch / word capacity | B24 / W256 |
| Training cap | 3,600 s between completed bricks |
| Measured epoch interval | 3,637.495 s |
| Optimizer bricks | 71 |
| Complete sentences | 24,746 |
| Throughput | **6.803 sentences/s** |
| Output loss | 0.0000 |
| Reconstruction loss | 3.0047 |
| Total process elapsed | 1:03:22.799 |

Artifact identities:

```text
c165b1e6ae9b290c71b74c7e8a7ea68cda4288645bcba59937f682a37f0ed5b3  BasicModel.ckpt
541f550e84269958d9675f56951a48b9561060a82f80ac1baa4833c63c07f734  BasicModel.xml
```

The older canonical pre-Teacher checkpoint was moved out of `data/` before
the new basin was started:

```text
output/checkpoints/BasicModel.pre_teacher_20260722.ckpt
d297040394422811feb106503a15e3ff4e7a532773b10bf809f17e999f8af92c
```

The accepted performance comparison is the final 6.803 sentences/s entry, not
the earlier appended attempts in the same launch log. A Teacher comparison
should use B24 for exact throughput parity; B28 may be reported separately as
the preferred operating point.
