module FastHistograms

import Base: eltype, @propagate_inbounds
using ComputedFieldTypes
using LoopVectorization

include("traits.jl")
include("bin_search.jl")
include("bin_update.jl")

include("single_thread_fixed_width_2d_hist.jl")

export create_fast_histogram, bin_search, bin_update!, counts, zero!

function create_fast_histogram end
function bin_search end
function bin_update! end

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

```julia
using FastHistograms, Random

h = FixedWidthHistogram()

img1 = zeros(10, 10)
img1[:, 6:end] .= 0xff

img2 = zeros(10, 10)
img2[6:end, :] .= 0xff

# Compute the histogram bin counts
calc_hist!(h, img1, img2)

@show counts(h) # Get the bin counts

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
"""
FastHistograms

end
