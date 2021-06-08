struct TextHistogram{P<:HistogramParallelization}
    table::AbstractDict
    subtable::AbstractVector{<:AbstractDict}
end

BinType(::TextHistogram{P}) where {P} = UnboundedWidth()
BinSearchAlgorithm(::TextHistogram{P}) where {P} = HashFunction()
HistogramParallelization(::TextHistogram{P}) where {P} = P()
eltype(::TextHistogram{P}) where {P} = String

"""
    create_fast_histogram(::UnboundedWidth, ::HashFunction, ::P) where {P<:HistogramParallelization}

Creates a histogram for 1D text data.
`P` can be any parallelization scheme.
"""
function create_fast_histogram(::UnboundedWidth, ::HashFunction, ::P) where {P<:HistogramParallelization}
    table = Dict{String,Int}()
    subtable = map(x -> Dict{String,Int}(), 1:Threads.nthreads())
    TextHistogram{P}(table, subtable)
end

get_table(h::TextHistogram) = h.table

# For PrivateThreads and SIMD
get_subtable(h::TextHistogram) = h.subtable

function zero!(h::TextHistogram)
    empty!(get_table(h))
    empty!.(get_subtable(h))
    nothing
end

counts(h::TextHistogram) = h.table
