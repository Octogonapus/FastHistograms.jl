module FastHistograms

import Base: eltype, @propagate_inbounds
using ComputedFieldTypes
using LoopVectorization

include("traits.jl")
include("bin_search.jl")
include("bin_update.jl")
include("fixed_width_histogram.jl")

export create_fast_histogram, bin_search, increment_bins!, counts, zero!

"""
    create_fast_histogram(
        ::BinType,
        ::BinSearchAlgorithm,
        ::HistogramParallelization,
        args...
    )

Creates a histogram with the given `BinType`, `BinSearchAlgorithm`, and `HistogramParallelization` traits.
Methods of this function will also require additional arguments (here `args...`) that depend on the combination of
traits selected.
"""
create_fast_histogram(
    ::BinType,
    ::BinSearchAlgorithm,
    ::HistogramParallelization,
    args...,
) = error("Not implemented.")

"""
    counts(hist)

Returns the count in each bin of the histogram. This function takes one argument: the histogram structure.

This function is not required to be thread-safe.
"""
function counts end

"""
    zero!(hist)

Sets the count in each bin to zero. This function takes one argument: the histogram structure.

This function is not required to be thread-safe.
"""
function zero! end

"""
FastHistograms declares and implements a minimal histogram interface with a focus on speed.

```julia-repl
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
"""
FastHistograms

end
