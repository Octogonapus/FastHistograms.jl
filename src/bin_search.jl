"A trait for the ways the bin search step can be implemented."
abstract type BinSearchAlgorithm end

"""
Basic arithmetic to determine the bin to update, compatible only with the FixedWidth bin type.

Requires these functions to be defined:
- `binmin(hist)::Int` Returns the value of the lowest bin edge.
- `norm(hist)::Float32` Returns the inverse of the size of the bin range (`1 / (last_bin - first_bin)`).
- `nbins(hist)::Int` Returns the number of bins.
"""
struct Arithmetic <: BinSearchAlgorithm end

"""
Uses binary search to find the bin to update. Meant to be used with the VariableWidth bin type.

Requires these functions to be defined:
- `bin_ranges(hist)::Vector{Int}`
"""
struct BinarySearch <: BinSearchAlgorithm end

BinSearchAlgorithm(::Type) = error("")

"""
    bin_search(h, data)

Returns the index of the bin to increment.
"""
bin_search(h, data) = bin_search(BinSearchAlgorithm(h), h, data)

function bin_search(::Arithmetic, h, data)
    # Using `min(nbins, max(1, ceil(<computed index>)))` here is consistent with StatsBase, but it's 2 μs slower than
    # truncating. Therefore, we add 1 and then truncate to get the same result.
    return clamp(trunc(Int, (data - binmin(h)) * norm(h) * nbins(h) + 1), 1, nbins(h))
end

function bin_search(::BinarySearch, h, data)
    idxs = searchsorted(bin_ranges(h), data)
    return clamp(last(idxs), 1, nbins(h))
end
