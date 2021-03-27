"""
A 2D histogram with fixed-width bins. The histogram computation is single-threaded by design and is therefore best
suited to small data.
"""
struct SingleThreadFixedWidth2DHistogram{BinType<:Integer} <: FastHistogram
    weights::Array{Int,2}
    subweights::Array{Int,3}
    norm::Float32
    nbins::Int
    binmin::BinType
end

"""
    SingleThreadFixedWidth2DHistogram(first_bin::BinType, last_bin::BinType, nbins::Int) where {BinType<:Int}

A 2D histogram with fixed-width bins. The histogram computation is single-threaded by design and is therefore best
suited to small data.

The `first_bin` and `last_bin` are the values of the lowest and highest bins, respectively. `nbins` is the total number
of bins.
"""
function SingleThreadFixedWidth2DHistogram(
    first_bin::BinType,
    last_bin::BinType,
    nbins::Int,
) where {BinType<:Integer}
    norm = 1 / (last_bin - first_bin)

    weights = zeros(Int, nbins, nbins)
    subweights = zeros(Int, nbins, nbins, 3)

    SingleThreadFixedWidth2DHistogram{BinType}(weights, subweights, norm, nbins, first_bin)
end

"""
    SingleThreadFixedWidth2DHistogram(bins = 0x00:UInt8(16):0xff)

A 2D histogram with fixed-width bins. The histogram computation is single-threaded by design and is therefore best
suited to small data.

The `bins` must be given in strictly increasing order (e.g., `[0, 64, 128, 192]`).
"""
SingleThreadFixedWidth2DHistogram(bins = 0x00:UInt8(16):0xff) =
    SingleThreadFixedWidth2DHistogram(first(bins), last(bins), length(bins))

function calc_hist!(h::SingleThreadFixedWidth2DHistogram, img1, img2)
    rows = size(img1, 1)
    cols = size(img1, 2)
    align_rows = floor(Int, rows / 3)

    binmin = h.binmin # @avx doesn't like operating on sub-fields
    nbins = h.nbins
    norm = h.norm
    subweights = h.subweights

    for c = 1:cols
        r = 1

        while r < align_rows
            for i = 0:2 # @avx
                @inbounds tx = img1[r+i, c]
                @inbounds ty = img2[r+i, c]
                ix = min(nbins, max(1, trunc(Int, (tx - binmin) * norm * nbins)))
                iy = min(nbins, max(1, trunc(Int, (ty - binmin) * norm * nbins)))
                @inbounds subweights[ix, iy, i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = min(nbins, max(1, trunc(Int, (tx - binmin) * norm * nbins)))
            iy = min(nbins, max(1, trunc(Int, (ty - binmin) * norm * nbins)))
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

bin_type(h::SingleThreadFixedWidth2DHistogram) = typeof(h.binmin)
