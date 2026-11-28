# NVIDIA GeForce RTX 4070 Ti SUPER — hardware report

**Commit:** 3174eb3ac63b2faf6c364b096450eb5e88cfcac2  
**OS:** Windows 10 build 10.0.26200.9711  
**Adapter:** NVIDIA GeForce RTX 4070 Ti SUPER (vendor 10de, device 2705)  
**Driver:** 32.0.16.2101  
**Debug layer:** enabled, no errors (0), no warnings (0)

## Lane range

Native wave width 32. Forced wave64 skipped — 64-lane fixture requires SM 6.6, WaveOps, and 64-lane waves. Measured lane range 32–32.

## Per-test results

| Test | Result | Details |
|---|---|---|
| bvh_waves_hardware | Passed | native_wave_width=32, logical_width=64, 38 groups, 32 fields/thread, 0 mismatches |
| bvh_wave64_hardware | Skipped | Requires SM 6.6, WaveOps, and 64-lane waves (lane range 32–32) |
| bvh_differential_hardware | Passed | 3 epochs × 136 comparisons (408 total); faults: cross_half_address_swap, stale_scene, high_word_alias |

## Summary

| Adapter | Default native lanes | Forced wave64 | BVH differential | Evidence |
|---|---:|---|---|---|
| NVIDIA GeForce RTX 4070 Ti SUPER (driver 32.0.16.1088) | 32 | Skipped, lane range 32–32 | 408 queries passed | Hardware coverage above this row |
