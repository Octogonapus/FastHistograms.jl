var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastHistograms","category":"page"},{"location":"#FastHistograms","page":"Home","title":"FastHistograms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FastHistograms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastHistograms]","category":"page"},{"location":"#FastHistograms.FastHistograms","page":"Home","title":"FastHistograms.FastHistograms","text":"FastHistograms declares and implements a minimal histogram interface with a focus on speed.\n\njulia> using FastHistograms, Random\n\n# Create a 2D histogram for 8-bit integer data.\n# Use fixed-width bins with an optimized bin search algorithm (Arithmetic) for fixed-width bins.\n# Don't use any parallelization because our data are small.\njulia> h = create_fast_histogram(\n    FastHistograms.FixedWidth(),\n    FastHistograms.Arithmetic(),\n    FastHistograms.NoParallelization(),\n    Val{2}(), # 2D histogram\n    0x00,     # Lowest bucket edge\n    0xff,     # Highest bucket edge\n    4,        # Number of buckets\n);\n\n# Create two random images to compute the joint histogram for\njulia> img1 = rand(0x00:0xff, 32, 32);\n\njulia> img2 = rand(0x00:0xff, 32, 32);\n\n# Compute the histogram bin counts\njulia> increment_bins!(h, img1, img2)\n\n# Get the bin counts\njulia> counts(h)\n4×4 Matrix{Int64}:\n 61  64  67  64\n 65  59  72  65\n 61  66  71  61\n 53  67  63  65\n\n\n\n\n\n","category":"module"},{"location":"#FastHistograms.Arithmetic","page":"Home","title":"FastHistograms.Arithmetic","text":"Basic arithmetic to determine the bin to update, compatible only with the FixedWidth bin type.\n\nRequires these functions to be defined:\n\nbinmin(hist, axis)::Int Returns the value of the lowest bin edge for the axis. The implementation should use @propagate_inbounds for good performance.\nnorm(hist, axis)::Float32 Returns the inverse of the size of the bin range for the axis (1 / (last_bin - first_bin)). The implementation should use @propagate_inbounds for good performance.\nnbins(hist, axis)::Int Returns the number of bins for the axis. The implementation should use @propagate_inbounds for good performance.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.BinSearchAlgorithm","page":"Home","title":"FastHistograms.BinSearchAlgorithm","text":"A trait for the ways the bin search step can be implemented.\n\nHistograms that operate on real-valued data must implement the following functions, in addition to any trait-specific functions:\n\nget_weights(hist)::AbstractArray{Int,N} Returns the weights (i.e. counts) array for an N-dimensional histogram.\n\nHistograms that operate on text data must implement the following functions, in addition to any trait-specific functions:\n\nget_table(hist)::AbstractDict{String,Int} Returns the table for the histogram.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.BinType","page":"Home","title":"FastHistograms.BinType","text":"A trait for the type of bins a histogram may have.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.BinarySearch","page":"Home","title":"FastHistograms.BinarySearch","text":"Uses binary search to find the bin to update. Meant to be used with the VariableWidth bin type.\n\nRequires these functions to be defined:\n\nbin_edges(hist, axis)::Vector{Int} Returns a sorted vector of the bin edges for the axis. The implementation should use @propagate_inbounds for good performance.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.FixedWidth","page":"Home","title":"FastHistograms.FixedWidth","text":"Each bin has the same predetermined width.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.HashFunction","page":"Home","title":"FastHistograms.HashFunction","text":"Uses a hash function to find the bin to update. Compatible only with the UnboundedWidth bin type.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.HistogramParallelization","page":"Home","title":"FastHistograms.HistogramParallelization","text":"A trait for the ways the bin search and bin update steps can be parallelized.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.NoParallelization","page":"Home","title":"FastHistograms.NoParallelization","text":"No threading nor vectorization.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.PrivateThreads","page":"Home","title":"FastHistograms.PrivateThreads","text":"Threads that have private bin data structures that are reduced after their private updates.\n\nRequires these functions to be defined for real-valued histograms:\n\nget_subweights(hist)::AbstractArray{Int,N+1} Returns the weights (i.e. counts) array for an N-dimensional histogram.\n\nRequires these functions to be defined for text histograms:\n\nget_subtable(hist)::AbstractVector{AbstractDict{String,Int}} Returns a vector of independent tables.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.SIMD","page":"Home","title":"FastHistograms.SIMD","text":"SIMD vectorization.\n\nRequires these functions to be defined for real-valued histograms:\n\nget_subweights(hist)::AbstractArray{Int,N+1} Returns the weights (i.e. counts) array for an N-dimensional histogram.\n\nRequires these functions to be defined for text histograms:\n\nget_subtable(hist)::AbstractVector{AbstractDict{String,Int}} Returns a vector of independent tables.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.UnboundedWidth","page":"Home","title":"FastHistograms.UnboundedWidth","text":"Bin widths are not known before computing the histogram (i.e. text data). Only 1D histograms are supported.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.VariableWidth","page":"Home","title":"FastHistograms.VariableWidth","text":"Bins have possibly different predetermined widths.\n\n\n\n\n\n","category":"type"},{"location":"#FastHistograms.bin_search-Tuple{Any, Any, Any}","page":"Home","title":"FastHistograms.bin_search","text":"bin_search(h, axis, data)\n\nReturns the index of the bin to increment.\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.counts-Tuple{Any}","page":"Home","title":"FastHistograms.counts","text":"counts(h)\n\nReturns the bin counts of the histogram h. All histograms must implement this.\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.create_fast_histogram","page":"Home","title":"FastHistograms.create_fast_histogram","text":"create_fast_histogram(\n    ::BinType,\n    ::BinSearchAlgorithm,\n    ::HistogramParallelization,\n    args...\n)\n\nCreates a histogram with the given BinType, BinSearchAlgorithm, and HistogramParallelization traits. Methods of this function will also require additional arguments (here args...) that depend on the combination of traits selected.\n\n\n\n\n\n","category":"function"},{"location":"#FastHistograms.create_fast_histogram-Union{Tuple{P}, Tuple{FastHistograms.UnboundedWidth, FastHistograms.HashFunction, P}} where P<:FastHistograms.HistogramParallelization","page":"Home","title":"FastHistograms.create_fast_histogram","text":"create_fast_histogram(::UnboundedWidth, ::HashFunction, ::P) where {P<:HistogramParallelization}\n\nCreates a histogram for 1D text data. P can be any parallelization scheme.\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.create_fast_histogram-Union{Tuple{P}, Tuple{FastHistograms.VariableWidth, FastHistograms.BinarySearch, P, AbstractVector{var\"#s97\"} where var\"#s97\"<:(AbstractVector{T} where T)}} where P<:FastHistograms.HistogramParallelization","page":"Home","title":"FastHistograms.create_fast_histogram","text":"create_fast_histogram(\n    ::VariableWidth,\n    ::BinarySearch,\n    ::P,\n    edges::AbstractVector{<:AbstractVector}, # Vector of edges, one edge vector per dimension\n) where {P<:HistogramParallelization}\n\nCreates a histogram with variable-width bins (i.e. bins of possibly different widths). P can be any parallelization scheme. The edges define the bin edges for each axis of the histogram. Provide one element for each dimension. Each element has the form (first edge, second edge, ..., nth edge).\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.create_fast_histogram-Union{Tuple{P}, Tuple{S}, Tuple{E}, Tuple{FastHistograms.FixedWidth, S, P, AbstractArray{Tuple{E, E, Int64}, 1}}} where {E<:Real, S<:FastHistograms.BinSearchAlgorithm, P<:FastHistograms.HistogramParallelization}","page":"Home","title":"FastHistograms.create_fast_histogram","text":"create_fast_histogram(\n    ::FixedWidth,\n    ::S,\n    ::P,\n    axes_data::AbstractVector{Tuple{E,E,Int}}, # first, last, nbins\n) where {E<:Real,S<:BinSearchAlgorithm,P<:HistogramParallelization}\n\nCreates a histogram with fixed-width bins. S and P can be any bin search algorithm or parallelization scheme, respectively. The axes_data define the range of each axis of the histogram. Provide one element for each dimension. Each element has the form (first_bin, last_bin, nbins).\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.get_subweights","page":"Home","title":"FastHistograms.get_subweights","text":"get_subweights(h)\n\nReturns the subweights array. All histograms implementing SIMD and PrivateThreads parallelization must implement this.\n\n\n\n\n\n","category":"function"},{"location":"#FastHistograms.get_weights","page":"Home","title":"FastHistograms.get_weights","text":"get_weights(h)\n\nReturns the weights array. All histograms must implement this.\n\n\n\n\n\n","category":"function"},{"location":"#FastHistograms.increment_bins!-Tuple{Any, Any, Any}","page":"Home","title":"FastHistograms.increment_bins!","text":"increment_bins!(h, data1, data2)\n\nIncrements the bin counts for a 2D histogram h using the data data1 and data2. Elements of data that are outside the range of the histogram's bins will NOT be filtered out, they will be considered as members of the closest bin.\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.increment_bins!-Tuple{Any, Any}","page":"Home","title":"FastHistograms.increment_bins!","text":"increment_bins!(h, data)\n\nIncrements the bin counts for a 1D histogram h using the data. Elements of data that are outside the range of the histogram's bins will NOT be filtered out, they will be considered as members of the closest bin.\n\n\n\n\n\n","category":"method"},{"location":"#FastHistograms.zero!-Tuple{Any}","page":"Home","title":"FastHistograms.zero!","text":"zero!(h)\n\nSets all bin counts of the histogram h to zero. All histograms must implement this.\n\n\n\n\n\n","category":"method"}]
}
