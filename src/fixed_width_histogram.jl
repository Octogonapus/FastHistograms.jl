@computed struct FixedWidthHistogram{
    N,
    BinEltype<:Real,
    B<:BinSearchAlgorithm,
    P<:HistogramParallelization,
}
    weights::Array{Int,N}
    subweights::Array{Int,N + 1}
    norm::Float32
    nbins::Int
    binmin::BinEltype
    bin_ranges::Vector{BinEltype}
end

BinType(::FixedWidthHistogram) = FixedWidth()
BinSearchAlgorithm(::FixedWidthHistogram{N,E,B,P}) where {N,E,B,P} = B()
HistogramParallelization(::FixedWidthHistogram{N,E,B,P}) where {N,E,B,P} = P()

eltype(::FixedWidthHistogram{N,E,B,P}) where {N,E,B,P} = E

"""
Creates a histogram for fixed-width bins. The `first_bin` and `last_bin` are the values of the lowest
and highest bins, respectively. `nbins` is the number of bins (not the number of edges).
"""
function create_fast_histogram(
    ::FixedWidth,
    ::B,
    ::P,
    ::Dims,
    first_bin::BinEltype,
    last_bin::BinEltype,
    nbins::Int,
) where {N,BinEltype<:Real,B<:BinSearchAlgorithm,P<:HistogramParallelization,Dims}
    norm = 1 / (last_bin - first_bin)

    weights = create_weights(Dims, nbins)
    subweights = create_subweights(Dims, nbins)

    if B <: Arithmetic
        # Arithmetic does not need `bin_ranges(h)` so don't bother computing them
        bin_ranges = []
    else
        tmp_range = range(first_bin; stop = last_bin, length = nbins + 1)
        bin_ranges = if BinEltype <: Integer
            ceil.(BinEltype, tmp_range)
        else
            collect(BinEltype, tmp_range)
        end
    end

    FixedWidthHistogram{dims_number(Dims),BinEltype,B,P}(
        weights,
        subweights,
        norm,
        nbins,
        first_bin,
        bin_ranges,
    )
end

create_weights(::Type{Val{1}}, nbins) = zeros(Int, nbins)
create_weights(::Type{Val{2}}, nbins) = zeros(Int, nbins, nbins)
create_subweights(::Type{Val{1}}, nbins) = zeros(Int, nbins, 4)
create_subweights(::Type{Val{2}}, nbins) = zeros(Int, nbins, nbins, 4)
dims_number(::Type{Val{1}}) = 1
dims_number(::Type{Val{2}}) = 2

# For Arithmetic
nbins(h::FixedWidthHistogram) = h.nbins
binmin(h::FixedWidthHistogram) = h.binmin
norm(h::FixedWidthHistogram) = h.norm

# For BinarySearch
bin_ranges(h::FixedWidthHistogram) = h.bin_ranges

@propagate_inbounds increment_weight!(h::FixedWidthHistogram, is...) = h.weights[is...] += 1

# For SIMD
@propagate_inbounds increment_subweight!(h::FixedWidthHistogram, is...) =
    h.subweights[is...] += 1
sum_subweights!(h::FixedWidthHistogram) = sum!(h.weights, h.subweights)

# TODO: Move to bin_update.jl once LoopVectorization issues are sorted out
function increment_bins!(
    ::Arithmetic,
    ::SIMD,
    h::FixedWidthHistogram,
    data::Union{AbstractVector,AbstractMatrix},
)
    rows = size(data, 1)
    align_rows = floor(Int, rows / 3)

    for c = 1:size(data, 2)
        r = 1

        while r < align_rows
            @turbo for i = 0:2
                @inbounds tx = data[r+i, c]
                @inbounds h.subweights[bin_search(h, tx), i+1] += 1
                # TODO: Figure out how to use @inbounds increment_subweight!(h, bin_search(h, tx), i+1)
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = data[r2, c]
            @inbounds h.subweights[bin_search(h, tx), 1] += 1
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
    h::FixedWidthHistogram,
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
                ix = bin_search(h, tx)
                iy = bin_search(h, ty)
                @inbounds h.subweights[ix, iy, i+1] += 1
                # TODO: Figure out how to use @inbounds increment_subweight!(h, ix, iy, i+1)
            end
            r += 4
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = bin_search(h, tx)
            iy = bin_search(h, ty)
            @inbounds h.subweights[ix, iy, 1] += 1
            # TODO: Figure out how to use @inbounds increment_subweight!(h, ix, iy, 1)
        end
    end

    sum_subweights!(h)

    nothing
end

counts(h::FixedWidthHistogram) = h.weights

function zero!(h::FixedWidthHistogram)
    h.weights .= 0
    h.subweights .= 0
end
