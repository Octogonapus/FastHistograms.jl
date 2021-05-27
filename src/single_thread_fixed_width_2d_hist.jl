@computed struct SingleThreadFixedWidth2DHistogram{
    N,
    BinEltype<:Real,
    P<:HistogramParallelization,
} <: FastHistogram{FixedWidth,Arithmetic,P}
    weights::Array{Int,N}
    subweights::Array{Int,N + 1}
    norm::Float32
    nbins::Int
    binmin::BinEltype
end

BinType(::SingleThreadFixedWidth2DHistogram) = FixedWidth()
BinSearchAlgorithm(::SingleThreadFixedWidth2DHistogram) = Arithmetic()
HistogramParallelization(
    ::SingleThreadFixedWidth2DHistogram{N,B,P},
) where {N,B,P<:HistogramParallelization} = P()
eltype(::SingleThreadFixedWidth2DHistogram{N,B,P}) where {N,B,P} = B

"""
Creates a fixed-width histogram for small 1D data. The `first_bin` and `last_bin` are the values of the lowest
and highest bins, respectively. `nbins` is the number of bins (not the number of edges).
"""
function create_fast_histogram(
    ::FixedWidth,
    ::Arithmetic,
    ::P,
    ::Val{1},
    first_bin::BinEltype,
    last_bin::BinEltype,
    nbins::Int,
) where {N,BinEltype<:Real,P<:HistogramParallelization}
    norm = 1 / (last_bin - first_bin)

    weights = zeros(Int, nbins)
    subweights = zeros(Int, nbins, 3)

    SingleThreadFixedWidth2DHistogram{1,BinEltype,P}(
        weights,
        subweights,
        norm,
        nbins,
        first_bin,
    )
end

"""
Creates a fixed-width histogram for small 2D data. The `first_bin` and `last_bin` are the values of the lowest
and highest bins, respectively. `nbins` is the number of bins per axis (not the number of edges).
"""
function create_fast_histogram(
    ::FixedWidth,
    ::Arithmetic,
    ::P,
    ::Val{2},
    first_bin::BinEltype,
    last_bin::BinEltype,
    nbins::Int,
) where {N,BinEltype<:Real,P<:HistogramParallelization}
    norm = 1 / (last_bin - first_bin)

    weights = zeros(Int, nbins, nbins)
    subweights = zeros(Int, nbins, nbins, 3)

    SingleThreadFixedWidth2DHistogram{2,BinEltype,P}(
        weights,
        subweights,
        norm,
        nbins,
        first_bin,
    )
end

# TODO: Type trait definition is slow?
# nbins(h::SingleThreadFixedWidth2DHistogram) = h.nbins
# binmin(h::SingleThreadFixedWidth2DHistogram) = h.binmin
# norm(h::SingleThreadFixedWidth2DHistogram) = h.norm
# function bin_search(::Arithmetic, h, data) where {T,P}
#     return min(nbins(h), max(1, trunc(Int, (data - binmin(h)) * norm(h) * nbins(h) + 1)))
# end

function bin_search(h::SingleThreadFixedWidth2DHistogram{N,B,P}, data) where {N,B,P}
    # Using `min(nbins, max(1, ceil(<computed index>)))` here is consistent with StatsBase, but it's 2 μs slower than
    # truncating. Therefore, we add 1 and then truncate to get the same result.
    return min(h.nbins, max(1, trunc(Int, (data - h.binmin) * h.norm * h.nbins + 1)))
end

function bin_update!(
    h::SingleThreadFixedWidth2DHistogram{1,B,NoParallelization},
    data::Union{AbstractVector,AbstractMatrix},
) where {N,B}
    rows = size(data, 1)
    cols = size(data, 2)
    align_rows = floor(Int, rows / 3)

    subweights = h.subweights

    for c = 1:cols
        r = 1

        while r < align_rows
            for i = 0:2
                @inbounds tx = data[r+i, c]
                @inbounds subweights[bin_search(h, tx), i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = data[r2, c]
            @inbounds subweights[bin_search(h, tx), 1] += 1
        end
    end

    sum!(h.weights, subweights)

    nothing
end

function bin_update!(
    h::SingleThreadFixedWidth2DHistogram{2,B,NoParallelization},
    img1::Union{AbstractVector,AbstractMatrix},
    img2::Union{AbstractVector,AbstractMatrix},
) where {N,B}
    rows = size(img1, 1)
    cols = size(img1, 2)
    align_rows = floor(Int, rows / 3)

    subweights = h.subweights

    for c = 1:cols
        r = 1

        while r < align_rows
            for i = 0:2
                @inbounds tx = img1[r+i, c]
                @inbounds ty = img2[r+i, c]
                ix = bin_search(h, tx)
                iy = bin_search(h, ty)
                @inbounds subweights[ix, iy, i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = bin_search(h, tx)
            iy = bin_search(h, ty)
            @inbounds subweights[ix, iy, 1] += 1
        end
    end

    sum!(h.weights, subweights)

    nothing
end

function bin_update!(
    h::SingleThreadFixedWidth2DHistogram{1,B,SIMD},
    data::Union{AbstractVector,AbstractMatrix},
) where {N,B}
    rows = size(data, 1)
    cols = size(data, 2)
    align_rows = floor(Int, rows / 3)

    subweights = h.subweights

    for c = 1:cols
        r = 1

        while r < align_rows
            # TODO: LoopVectorization regression on this code
            for i = 0:2 # @avx
                @inbounds tx = data[r+i, c]
                @inbounds subweights[bin_search(h, tx), i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = data[r2, c]
            @inbounds subweights[bin_search(h, tx), 1] += 1
        end
    end

    sum!(h.weights, subweights)

    nothing
end

function bin_update!(
    h::SingleThreadFixedWidth2DHistogram{2,B,SIMD},
    img1::Union{AbstractVector,AbstractMatrix},
    img2::Union{AbstractVector,AbstractMatrix},
) where {N,B}
    rows = size(img1, 1)
    cols = size(img1, 2)
    align_rows = floor(Int, rows / 3)

    subweights = h.subweights

    for c = 1:cols
        r = 1

        while r < align_rows
            # TODO: LoopVectorization regression on this code
            for i = 0:2 # @avx
                @inbounds tx = img1[r+i, c]
                @inbounds ty = img2[r+i, c]
                ix = bin_search(h, tx)
                iy = bin_search(h, ty)
                @inbounds subweights[ix, iy, i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = bin_search(h, tx)
            iy = bin_search(h, ty)
            @inbounds subweights[ix, iy, 1] += 1
        end
    end

    sum!(h.weights, subweights)

    nothing
end

counts(h::SingleThreadFixedWidth2DHistogram) = h.weights

function zero!(h::SingleThreadFixedWidth2DHistogram)
    h.weights .= 0
    h.subweights .= 0
end
