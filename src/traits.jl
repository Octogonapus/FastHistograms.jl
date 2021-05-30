"A trait for the type of bins a histogram may have."
abstract type BinType end
"Each bin has the same predetermined width."
struct FixedWidth <: BinType end
"Bins have possibly different predetermined widths."
struct VariableWidth <: BinType end
"Bin widths are not known before computing the histogram."
struct UnboundedWidth <: BinType end

"A trait for the ways the bin search and bin update steps can be parallelized."
abstract type HistogramParallelization end
"No threading nor vectorization."
struct NoParallelization <: HistogramParallelization end
"Threads that have private bin data structures that are reduced after their private updates."
struct PrivateThreads <: HistogramParallelization end
"SIMD."
struct SIMD <: HistogramParallelization end
