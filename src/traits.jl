"A trait for the type of bins a histogram may have."
abstract type BinType end
"Each bin has the same predetermined width."
struct FixedWidth <: BinType end
"Bins have possibly different predetermined widths."
struct VariableWidth <: BinType end
"Bin widths are not known before computing the histogram."
struct UnboundedWidth <: BinType end

"A trait for the ways the bin search step can be implemented."
abstract type BinSearchAlgorithm end
"Basic arithmetic to determine the bin to update, compatible only with the FixedWidth bin type."
struct Arithmetic <: BinSearchAlgorithm end
"Uses binary search to find the bin to update. Meant to be used with the VariableWidth bin type."
struct BinarySearch <: BinSearchAlgorithm end

"A trait for the ways the bin search and bin update steps can be parallelized."
abstract type HistogramParallelization end
"No threading nor vectorization."
struct NoParallelization <: HistogramParallelization end
"Threads that share a common bin data structure and update it atomically."
struct SharedThreads <: HistogramParallelization end
"Threads that have private bin data structures that are reduced after their private updates."
struct PrivateThreads <: HistogramParallelization end
"SIMD."
struct SIMD <: HistogramParallelization end

abstract type FastHistogram{T<:BinType,S<:BinSearchAlgorithm,P<:HistogramParallelization} end
