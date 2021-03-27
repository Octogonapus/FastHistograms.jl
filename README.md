# FastHistograms

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Octogonapus.github.io/FastHistograms.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Octogonapus.github.io/FastHistograms.jl/dev)
[![Build Status](https://github.com/Octogonapus/FastHistograms.jl/workflows/CI/badge.svg)](https://github.com/Octogonapus/FastHistograms.jl/actions)
[![Coverage](https://codecov.io/gh/Octogonapus/FastHistograms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Octogonapus/FastHistograms.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

FastHistograms declares and implements a minimal histogram interface with a focus on speed.
This package does not aim to implement general histogram algorithms that work for a wide array of data types; for that
purpose, consider [StatsBase](https://github.com/JuliaStats/StatsBase.jl).

Currently, the only implemented algorithm is for fixed-width bins on (small) 2D data.

## Example

```julia
using FastHistograms, Random

julia> h = SingleThreadFixedWidth2DHistogram()

julia> img1 = zeros(10, 10)
julia> img1[:, 6:end] .= 0xff

julia> img2 = zeros(10, 10)
julia> img2[6:end, :] .= 0xff

# Compute the histogram bin counts
julia> calc_hist!(h, img1, img2)

julia> @show counts(h) # Get the bin counts

16×16 Matrix{Int64}:
 25  0  0  0  0  0  0  0  0  0  0  0  0  0  0  25
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0
 25  0  0  0  0  0  0  0  0  0  0  0  0  0  0  25
```

## Benchmarks

With two 40x80 8-bit images and 16 bins per dimension, FastHistograms runs in 12 μs and StatsBase runs in 194 μs.
