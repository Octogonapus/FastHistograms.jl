"""
A trait for the ways the bin search step can be implemented.

Histograms that operate on real-valued data must implement the following functions, in addition to any trait-specific functions:
- `get_weights(hist)::AbstractArray{Int,N}` Returns the weights (i.e. counts) array for an N-dimensional histogram.

Histograms that operate on text data must implement the following functions, in addition to any trait-specific functions:
- `get_table(hist)::AbstractDict{String,Int}` Returns the table for the histogram.
"""
abstract type BinSearchAlgorithm end

"""
Basic arithmetic to determine the bin to update, compatible only with the FixedWidth bin type.

Requires these functions to be defined:
- `binmin(hist, axis)::Int` Returns the value of the lowest bin edge for the axis. The implementation should use `@propagate_inbounds` for good performance.
- `norm(hist, axis)::Float32` Returns the inverse of the size of the bin range for the axis (`1 / (last_bin - first_bin)`). The implementation should use `@propagate_inbounds` for good performance.
- `nbins(hist, axis)::Int` Returns the number of bins for the axis. The implementation should use `@propagate_inbounds` for good performance.
"""
struct Arithmetic <: BinSearchAlgorithm end

"""
Uses binary search to find the bin to update. Meant to be used with the VariableWidth bin type.

Requires these functions to be defined:
- `bin_edges(hist, axis)::Vector{Int}` Returns a sorted vector of the bin edges for the axis. The implementation should use `@propagate_inbounds` for good performance.
"""
struct BinarySearch <: BinSearchAlgorithm end

"""
Uses a hash function to find the bin to update. Compatible only with the UnboundedWidth bin type.
"""
struct HashFunction <: BinSearchAlgorithm end

BinSearchAlgorithm(t) = error("BinSearchAlgorithm not defined for $(typeof(t))")

"""
    bin_search(h, axis, data)

Returns the index of the bin to increment.
"""
bin_search(h, axis, data) = bin_search(BinSearchAlgorithm(h), h, axis, data)

function bin_search(::Arithmetic, h, axis, data)
    # Using `min(nbins, max(1, ceil(<computed index>)))` here is consistent with StatsBase, but it's 2 μs slower than
    # truncating. Therefore, we add 1 and then truncate to get the same result.
    return @inbounds clamp(trunc(Int, (data - binmin(h, axis)) * norm(h, axis) * nbins(h, axis) + 1), 1, nbins(h, axis))
end

function bin_search(::BinarySearch, h, axis, data)
    @inbounds idxs = searchsorted(bin_edges(h, axis), data)
    return @inbounds clamp(last(idxs), 1, nbins(h, axis))
end
