# FastHistograms

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Octogonapus.github.io/FastHistograms.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Octogonapus.github.io/FastHistograms.jl/dev)
[![Build Status](https://github.com/Octogonapus/FastHistograms.jl/workflows/CI/badge.svg)](https://github.com/Octogonapus/FastHistograms.jl/actions)
[![Coverage](https://codecov.io/gh/Octogonapus/FastHistograms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Octogonapus/FastHistograms.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

FastHistograms declares and implements a minimal histogram interface with a focus on speed.
This package does not aim to implement general histogram algorithms that work for a wide array of data types; for that
purpose, consider [StatsBase](https://github.com/JuliaStats/StatsBase.jl).

## Example

```julia
julia> using FastHistograms, Random

# Create a 2D histogram for 8-bit integer data.
# Use fixed-width bins with an optimized bin search algorithm (Arithmetic) for fixed-width bins.
# Don't use any parallelization because our data are small.
julia> h = create_fast_histogram(
    FastHistograms.FixedWidth(),
    FastHistograms.Arithmetic(),
    FastHistograms.NoParallelization(),
    Val{2}(), # 2D histogram
    0x00,     # Lowest bucket edge
    0xff,     # Highest bucket edge
    4,        # Number of buckets
);

# Create two random images to compute the joint histogram for
julia> img1 = rand(0x00:0xff, 32, 32);

julia> img2 = rand(0x00:0xff, 32, 32);

# Compute the histogram bin counts
julia> increment_bins!(h, img1, img2)

# Get the bin counts
julia> counts(h)
4×4 Matrix{Int64}:
 61  64  67  64
 65  59  72  65
 61  66  71  61
 53  67  63  65
```

## Benchmarks

These benchmarks were run on a t3.medium EC2 instance. Processor shielding was not used. ASLR was enabled.

```text
Benchmarking type=UInt8, size=(40, 80)
┌───────────────┬───────────────────┬───────────────────┬─────────────────┬────────────┬───────────────┐
│               │ FastHistograms:   | FastHistograms:   │ FastHistograms: │ StatsBase: │ StatsBase:    │
|               | FixedWidth,       | VariableWidth,    | FixedWidth,     | FixedWidth | VariableWidth |
|               | Arithmetic,       | BinarySearch,     | Arithmetic,     |            |               |
|               | NoParallelization │ NoParallelization | SIMD            |            |               |
├───────────────┼───────────────────┼───────────────────┼─────────────────┼────────────┼───────────────┤
│ Min Time (ns) │           19197.0 │           49013.0 │         19061.0 │    47274.0 │       53584.0 │
│  GC Time (ns) │               0.0 │               0.0 │             0.0 │        0.0 │           0.0 │
│    Allocs (B) │               0.0 │               0.0 │             0.0 │        2.0 │           2.0 │
│    Memory (B) │               0.0 │               0.0 │             0.0 │     2224.0 │         224.0 │
└───────────────┴───────────────────┴───────────────────┴─────────────────┴────────────┴───────────────┘
Benchmarking type=UInt8, size=(256, 256)
┌───────────────┬───────────────────┬───────────────────┬─────────────────┬────────────┬───────────────┐
│               │ FastHistograms:   | FastHistograms:   │ FastHistograms: │ StatsBase: │ StatsBase:    │
|               | FixedWidth,       | VariableWidth,    | FixedWidth,     | FixedWidth | VariableWidth |
|               | Arithmetic,       | BinarySearch,     | Arithmetic,     |            |               |
|               | NoParallelization │ NoParallelization | SIMD            |            |               |
├───────────────┼───────────────────┼───────────────────┼─────────────────┼────────────┼───────────────┤
│ Min Time (ns) │          375058.0 │          987333.0 │        372359.0 │   962661.0 │     1.09619e6 │
│  GC Time (ns) │               0.0 │               0.0 │             0.0 │        0.0 │           0.0 │
│    Allocs (B) │               0.0 │               0.0 │             0.0 │        2.0 │           2.0 │
│    Memory (B) │               0.0 │               0.0 │             0.0 │     2224.0 │         224.0 │
└───────────────┴───────────────────┴───────────────────┴─────────────────┴────────────┴───────────────┘
Benchmarking type=Float32, size=(40, 80)
┌───────────────┬───────────────────┬───────────────────┬─────────────────┬────────────┬───────────────┐
│               │ FastHistograms:   | FastHistograms:   │ FastHistograms: │ StatsBase: │ StatsBase:    │
|               | FixedWidth,       | VariableWidth,    | FixedWidth,     | FixedWidth | VariableWidth |
|               | Arithmetic,       | BinarySearch,     | Arithmetic,     |            |               |
|               | NoParallelization │ NoParallelization | SIMD            |            |               |
├───────────────┼───────────────────┼───────────────────┼─────────────────┼────────────┼───────────────┤
│ Min Time (ns) │           15592.0 │          100289.0 │         17634.0 │    74005.0 │      107417.0 │
│  GC Time (ns) │               0.0 │               0.0 │             0.0 │        0.0 │           0.0 │
│    Allocs (B) │               0.0 │               0.0 │             0.0 │        2.0 │           2.0 │
│    Memory (B) │               0.0 │               0.0 │             0.0 │     2224.0 │         224.0 │
└───────────────┴───────────────────┴───────────────────┴─────────────────┴────────────┴───────────────┘
Benchmarking type=Float32, size=(256, 256)
┌───────────────┬───────────────────┬───────────────────┬─────────────────┬────────────┬───────────────┐
│               │ FastHistograms:   | FastHistograms:   │ FastHistograms: │ StatsBase: │ StatsBase:    │
|               | FixedWidth,       | VariableWidth,    | FixedWidth,     | FixedWidth | VariableWidth |
|               | Arithmetic,       | BinarySearch,     | Arithmetic,     |            |               |
|               | NoParallelization │ NoParallelization | SIMD            |            |               |
├───────────────┼───────────────────┼───────────────────┼─────────────────┼────────────┼───────────────┤
│ Min Time (ns) │          309092.0 │         2.13267e6 │        332810.0 │  1.50458e6 │     2.30337e6 │
│  GC Time (ns) │               0.0 │               0.0 │             0.0 │        0.0 │           0.0 │
│    Allocs (B) │               0.0 │               0.0 │             0.0 │        2.0 │           2.0 │
│    Memory (B) │               0.0 │               0.0 │             0.0 │     2224.0 │         224.0 │
└───────────────┴───────────────────┴───────────────────┴─────────────────┴────────────┴───────────────┘
┌───────────────┬─────────────────┬────────────┐
│               │ FastHistograms: | StatsBase: │
|               | FixedWidth,     | FixedWidth |
|               | Arithmetic,     |            |
|               | PrivateThreads  |            |
├───────────────┼─────────────────┼────────────┤
│ Min Time (ns) │       2.57596e8 │  3.01149e9 │
│  GC Time (ns) │             0.0 │        0.0 │
│    Allocs (B) │            11.0 │        2.0 │
│    Memory (B) │          1056.0 │     4288.0 │
└───────────────┴─────────────────┴────────────┘

```
