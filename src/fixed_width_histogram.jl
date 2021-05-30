struct Axis{E<:Real}
    norm::Float32
    nbins::Int
    binmin::E
    bin_edges::Vector{E}
end

@computed struct FastHistogram{N,B<:BinType,E<:Real,S<:BinSearchAlgorithm,P<:HistogramParallelization}
    weights::Array{Int,N}
    subweights::Array{Int,N + 1}
    axes::SVector{N,Axis{E}}
end

BinType(::FastHistogram{N,B,E,S,P}) where {N,B,E,S,P} = B()
BinSearchAlgorithm(::FastHistogram{N,B,E,S,P}) where {N,B,E,S,P} = S()
HistogramParallelization(::FastHistogram{N,B,E,S,P}) where {N,B,E,S,P} = P()
eltype(::FastHistogram{N,B,E,S,P}) where {N,B,E,S,P} = E

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

    FastHistogram{length(axes_data),FixedWidth,E,S,P}(weights, subweights, SVector{length(axes)}(axes))
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

    FastHistogram{length(axes),VariableWidth,E,BinarySearch,P}(weights, subweights, SVector{length(axes)}(axes))
end

# For Arithmetic
@propagate_inbounds nbins(h::FastHistogram, axis) = h.axes[axis].nbins
@propagate_inbounds binmin(h::FastHistogram, axis) = h.axes[axis].binmin
@propagate_inbounds norm(h::FastHistogram, axis) = h.axes[axis].norm

# For BinarySearch
@propagate_inbounds bin_edges(h::FastHistogram, axis) = h.axes[axis].bin_edges

@propagate_inbounds increment_weight!(h::FastHistogram, is...) =
    h.weights[is...] += 1

# For SIMD
@propagate_inbounds increment_subweight!(h::FastHistogram, is...) = h.subweights[is...] += 1
sum_subweights!(h::FastHistogram) = sum!(h.weights, h.subweights)

# TODO: Move to bin_update.jl once LoopVectorization issues are sorted out
function increment_bins!(::Arithmetic, ::SIMD, h::FastHistogram, data::Union{AbstractVector,AbstractMatrix})
    rows = size(data, 1)
    align_rows = floor(Int, rows / 3)

    for c = 1:size(data, 2)
        r = 1

        while r < align_rows
            @turbo for i = 0:2
                @inbounds tx = data[r+i, c]
                @inbounds h.subweights[bin_search(h, 1, tx), i+1] += 1
                # TODO: Figure out how to use @inbounds increment_subweight!(h, bin_search(h, tx), i+1)
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = data[r2, c]
            @inbounds h.subweights[bin_search(h, 1, tx), 1] += 1
            # TODO: Figure out how to use @inbounds increment_subweight!(h, bin_search(h, tx), 1)
        end
    end

    sum_subweights!(h)

    nothing
end

# TODO: Move to bin_update.jl once LoopVectorization issues are sorted out
function increment_bins!(
    ::Arithmetic,
    ::SIMD,
    h::FastHistogram,
    img1::Union{AbstractVector,AbstractMatrix},
    img2::Union{AbstractVector,AbstractMatrix},
)
    rows = size(img1, 1)
    align_rows = floor(Int, rows / 4)

    for c = 1:size(img1, 2)
        r = 1

        while r < align_rows
            @turbo for i = 0:3
                @inbounds tx = img1[r+i, c]
                @inbounds ty = img2[r+i, c]
                ix = bin_search(h, 1, tx)
                iy = bin_search(h, 2, ty)
                @inbounds h.subweights[ix, iy, i+1] += 1
                # TODO: Figure out how to use @inbounds increment_subweight!(h, ix, iy, i+1)
            end
            r += 4
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = bin_search(h, 1, tx)
            iy = bin_search(h, 2, ty)
            @inbounds h.subweights[ix, iy, 1] += 1
            # TODO: Figure out how to use @inbounds increment_subweight!(h, ix, iy, 1)
        end
    end

    sum_subweights!(h)

    nothing
end

function increment_bins!(
    ::BinSearchAlgorithm,
    ::PrivateThreads,
    h::FastHistogram,
    data::Union{AbstractVector,AbstractMatrix},
)
    nthreads = size(h.subweights, 2)
    Threads.@threads for thread_idx in 1:nthreads
        if length(data) <= nthreads
            if thread_idx == 1
                start_idx = 1
                end_idx = length(data)
            else
                # Make an empty range
                start_idx = 1
                end_idx = 0
            end
        else
            step = length(data) ÷ nthreads
            start_idx = (thread_idx-1) * step + 1
            end_idx = step*thread_idx
            if end_idx + step > length(data) - (length(data) % nthreads)
                end_idx = length(data)
            end
        end

        for i in start_idx:end_idx
            @inbounds i = bin_search(h, 1, data[i])
            @inbounds h.subweights[i, thread_idx] += 1
        end
    end

    sum!(h.weights, h.subweights)

    nothing
end

function increment_bins!(
    ::BinSearchAlgorithm,
    ::PrivateThreads,
    h::FastHistogram,
    data1::Union{AbstractVector,AbstractMatrix},
    data2::Union{AbstractVector,AbstractMatrix},
)
    nthreads = size(h.subweights, 3)
    Threads.@threads for thread_idx in 1:nthreads
        if length(data1) <= nthreads
            if thread_idx == 1
                start_idx = 1
                end_idx = length(data1)
            else
                # Make an empty range
                start_idx = 1
                end_idx = 0
            end
        else
            step = length(data1) ÷ nthreads
            start_idx = (thread_idx-1) * step + 1
            end_idx = step*thread_idx
            if end_idx + step > length(data1) - (length(data1) % nthreads)
                end_idx = length(data1)
            end
        end

        for i in start_idx:end_idx
            @inbounds ix = bin_search(h, 1, data1[i])
            @inbounds iy = bin_search(h, 2, data2[i])
            @inbounds h.subweights[ix, iy, thread_idx] += 1
        end
    end

    sum!(h.weights, h.subweights)

    nothing
end

counts(h::FastHistogram) = h.weights

function zero!(h::FastHistogram)
    h.weights .= 0
    h.subweights .= 0
end
