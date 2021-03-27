module FastHistograms

abstract type FastHistogram end

include("single_thread_fixed_width_2d_hist.jl")

export FastHistogram, SingleThreadFixedWidth2DHistogram
export calc_hist!, counts, zero!, bin_type

"""
    calc_hist!(hist, img)
    calc_hist!(hist, img1, img2)

Computes the histogram of the given data. This function takes two or three arguments. The first argument must be the
histogram structure (a subtype of `AbstractHistogram`). The second and optional third argument(s) must be the data
to operate on. The type of the histogram determines whether one or two additional arguments are needed. For example,
a 2D histogram requires two additional arguments. A typical 1D histogram requires only one.

Implementations of this function are free to implement any optimizations they wish as long as the histogram is correct
once this function exits. This function is not required to be thread-safe. Implementations are encouraged to create
descriptive names for their subtypes of `AbstractHistogram` that describe which optimizations are implemented. For
example, `SingleThreadFixedWidth2DHistogram` does not implement thread-level parallelization by design.
"""
function calc_hist! end

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
    bin_type(hist)

Returns the type of the bin used in the histogram. This function takes one argument: the histogram structure.

This function is not required to be thread-safe.
"""
function bin_type end

"""
FastHistograms declares and implements a minimal histogram interface with a focus on speed.

```julia
using FastHistograms, Random

h = SingleThreadFixedWidth2DHistogram()

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
