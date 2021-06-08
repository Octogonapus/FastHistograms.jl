struct Axis{E<:Real}
    norm::Float32
    nbins::Int
    binmin::E
    bin_edges::Vector{E}
end

@computed struct RealHistogram{N,B<:BinType,E<:Real,S<:BinSearchAlgorithm,P<:HistogramParallelization}
    weights::Array{Int,N}
    subweights::Array{Int,N + 1}
    axes::SVector{N,Axis{E}}
end

BinType(::RealHistogram{N,B,E,S,P}) where {N,B,E,S,P} = B()
BinSearchAlgorithm(::RealHistogram{N,B,E,S,P}) where {N,B,E,S,P} = S()
HistogramParallelization(::RealHistogram{N,B,E,S,P}) where {N,B,E,S,P} = P()
eltype(::RealHistogram{N,B,E,S,P}) where {N,B,E,S,P} = E

"""
    create_fast_histogram(
        ::FixedWidth,
        ::B,
        ::P,
        ::Dims,
        first_bin::E,
        last_bin::E,
        nbins::Int,
    ) where {N,E<:Real,B<:BinSearchAlgorithm,P<:HistogramParallelization,Dims}

Creates a histogram for fixed-width bins.
`B` and `P` can be any bin search algorithm or parallelization scheme, respectively.
`Dims` is a `Val` representing the number of dimensions of the histogram; it can be `Val{1}()` or `Val{2}()`.
The `first_bin` and `last_bin` are the values of the lowest and highest bins, respectively.
`nbins` is the number of bins (not the number of edges).
"""
function create_fast_histogram(
    ::FixedWidth,
    ::S,
    ::P,
    axes_data::AbstractVector{Tuple{E,E,Int}}, # first, last, nbins
) where {E<:Real,S<:BinSearchAlgorithm,P<:HistogramParallelization}
    axes = map(axes_data) do axis
        first_bin, last_bin, nbins = axis

        bin_edges = if S <: Arithmetic
            # Arithmetic does not need `bin_edges(h)` so don't bother computing them
            []
        else
            tmp_range = range(first_bin; stop = last_bin, length = nbins + 1)
            if E <: Integer
                ceil.(E, tmp_range)
            else
                collect(E, tmp_range)
            end
        end

        norm = 1 / (last_bin - first_bin)

        Axis{E}(norm, nbins, first_bin, bin_edges)
    end

    weights = zeros(Int, map(x -> x.nbins, axes)...)

    subweights = if P <: SIMD
        zeros(Int, map(x -> x.nbins, axes)..., 4)
    elseif P <: PrivateThreads
        zeros(Int, map(x -> x.nbins, axes)..., Threads.nthreads())
    else
        zeros(Int, map(x -> x.nbins, axes)..., 1)
    end

    RealHistogram{length(axes_data),FixedWidth,E,S,P}(weights, subweights, SVector{length(axes)}(axes))
end

function create_fast_histogram(
    ::VariableWidth,
    ::BinarySearch,
    ::P,
    edges::AbstractVector{<:AbstractVector}, # Vector of edges, one edge vector per dimension
) where {P<:HistogramParallelization}
    E = eltype(first(edges))
    axes = map(edges) do edge
        bin_edges = collect(edge)
        first_bin = minimum(edge)
        last_bin = maximum(edge)
        nbins = length(edge) - 1
        norm = 1 / (last_bin - first_bin)
        Axis{E}(norm, nbins, first_bin, bin_edges)
    end

    weights = zeros(Int, map(x -> x.nbins, axes)...)

    subweights = if P <: SIMD
        zeros(Int, map(x -> x.nbins, axes)..., 4)
    elseif P <: PrivateThreads
        zeros(Int, map(x -> x.nbins, axes)..., Threads.nthreads())
    else
        zeros(Int, map(x -> x.nbins, axes)..., 1)
    end

    RealHistogram{length(axes),VariableWidth,E,BinarySearch,P}(weights, subweights, SVector{length(axes)}(axes))
end

# For Arithmetic
@propagate_inbounds nbins(h::RealHistogram, axis) = h.axes[axis].nbins
@propagate_inbounds binmin(h::RealHistogram, axis) = h.axes[axis].binmin
@propagate_inbounds norm(h::RealHistogram, axis) = h.axes[axis].norm

# For BinarySearch
@propagate_inbounds bin_edges(h::RealHistogram, axis) = h.axes[axis].bin_edges

get_weights(h::RealHistogram) = h.weights

# For SIMD and PrivateThreads
get_subweights(h::RealHistogram) = h.subweights
