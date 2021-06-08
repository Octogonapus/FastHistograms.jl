"A trait for the type of bins a histogram may have."
abstract type BinType end
"Each bin has the same predetermined width."
struct FixedWidth <: BinType end
"Bins have possibly different predetermined widths."
struct VariableWidth <: BinType end
"Bin widths are not known before computing the histogram (i.e. text data). Only 1D histograms are supported."
struct UnboundedWidth <: BinType end

"A trait for the ways the bin search and bin update steps can be parallelized."
abstract type HistogramParallelization end

"No threading nor vectorization."
struct NoParallelization <: HistogramParallelization end

"""
Threads that have private bin data structures that are reduced after their private updates.

Requires these functions to be defined for real-valued histograms:
- `get_subweights(hist)::AbstractArray{Int,N+1}` Returns the weights (i.e. counts) array for an N-dimensional histogram.

Requires these functions to be defined for text histograms:
- `get_subtable(hist)::AbstractVector{AbstractDict{String,Int}}` Returns a vector of independent tables.
"""
struct PrivateThreads <: HistogramParallelization end

"""
SIMD vectorization.

Requires these functions to be defined for real-valued histograms:
- `get_subweights(hist)::AbstractArray{Int,N+1}` Returns the weights (i.e. counts) array for an N-dimensional histogram.

Requires these functions to be defined for text histograms:
- `get_subtable(hist)::AbstractVector{AbstractDict{String,Int}}` Returns a vector of independent tables.
"""
struct SIMD <: HistogramParallelization end
