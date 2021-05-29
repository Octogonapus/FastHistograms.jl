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

These benchmarks were run on a not-exactly-benchmark-ready system, so read them with a grain of salt.
The CPU is an i9-9900K. Processor shielding was not used. Swap was enabled. ASLR was enabled.
CPU frequency scaling and boosting were enabled. Hyperthreading was enabled. The effect of IRQs on benchmark results
have not been investigated. The CPU used does not have AVX512, so SIMD results are not relevant.

I hope to improve the quality of these benchmarks in the future.

```text
Benchmarking type=UInt8, size=(40, 80)
┌────────────────────┬───────────┬─────────────────┬─────────────────────┬───────────┐
│               Rows │ StatsBase │ FH (NoParallel) │ FH (NoParallel, BS) │ FH (SIMD) │
│             String │   Float64 │         Float64 │             Float64 │   Float64 │
├────────────────────┼───────────┼─────────────────┼─────────────────────┼───────────┤
│      Min Time (ns) │   32627.0 │         13031.0 │             13315.0 │   14679.0 │
│       GC Time (ns) │       0.0 │             0.0 │                 0.0 │       0.0 │
│         Allocs (B) │       2.0 │             0.0 │                 0.0 │       0.0 │
│         Memory (B) │    2224.0 │             0.0 │                 0.0 │       0.0 │
└────────────────────┴───────────┴─────────────────┴─────────────────────┴───────────┘
Benchmarking type=UInt8, size=(256, 256)
┌────────────────────┬───────────┬─────────────────┬─────────────────────┬───────────┐
│               Rows │ StatsBase │ FH (NoParallel) │ FH (NoParallel, BS) │ FH (SIMD) │
│             String │   Float64 │         Float64 │             Float64 │   Float64 │
├────────────────────┼───────────┼─────────────────┼─────────────────────┼───────────┤
│      Min Time (ns) │  671187.0 │        256721.0 │            256691.0 │  289822.0 │
│       GC Time (ns) │       0.0 │             0.0 │                 0.0 │       0.0 │
│         Allocs (B) │       2.0 │             0.0 │                 0.0 │       0.0 │
│         Memory (B) │    2224.0 │             0.0 │                 0.0 │       0.0 │
└────────────────────┴───────────┴─────────────────┴─────────────────────┴───────────┘
Benchmarking type=Float32, size=(40, 80)
┌────────────────────┬───────────┬─────────────────┬─────────────────────┬───────────┐
│               Rows │ StatsBase │ FH (NoParallel) │ FH (NoParallel, BS) │ FH (SIMD) │
│             String │   Float64 │         Float64 │             Float64 │   Float64 │
├────────────────────┼───────────┼─────────────────┼─────────────────────┼───────────┤
│      Min Time (ns) │   51020.0 │         10815.0 │             10818.0 │   14021.0 │
│       GC Time (ns) │       0.0 │             0.0 │                 0.0 │       0.0 │
│         Allocs (B) │       2.0 │             0.0 │                 0.0 │       0.0 │
│         Memory (B) │    2224.0 │             0.0 │                 0.0 │       0.0 │
└────────────────────┴───────────┴─────────────────┴─────────────────────┴───────────┘
Benchmarking type=Float32, size=(256, 256)
┌────────────────────┬───────────┬─────────────────┬─────────────────────┬───────────┐
│               Rows │ StatsBase │ FH (NoParallel) │ FH (NoParallel, BS) │ FH (SIMD) │
│             String │   Float64 │         Float64 │             Float64 │   Float64 │
├────────────────────┼───────────┼─────────────────┼─────────────────────┼───────────┤
│      Min Time (ns) │ 1.05138e6 │        214891.0 │            214821.0 │  271832.0 │
│       GC Time (ns) │       0.0 │             0.0 │                 0.0 │       0.0 │
│         Allocs (B) │       2.0 │             0.0 │                 0.0 │       0.0 │
│         Memory (B) │    2224.0 │             0.0 │                 0.0 │       0.0 │
└────────────────────┴───────────┴─────────────────┴─────────────────────┴───────────┘
```
