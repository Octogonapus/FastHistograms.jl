
"""
    increment_bins!(h, data)

Increments the bin counts for a 1D histogram `h` using the `data`. Elements of `data` that are outside the range of
the histogram's bins will NOT be filtered out, they will be considered as members of the closest bin.
"""
increment_bins!(h, data) = increment_bins!(BinSearchAlgorithm(h), HistogramParallelization(h), h, data)

"""
    increment_bins!(h, data1, data2)

Increments the bin counts for a 2D histogram `h` using the data `data1` and `data2`. Elements of `data` that are
outside the range of the histogram's bins will NOT be filtered out, they will be considered as members of the closest
bin.
"""
increment_bins!(h, data1, data2) = increment_bins!(BinSearchAlgorithm(h), HistogramParallelization(h), h, data1, data2)

"""
    increment_weight!(h, is...)

Increments the bin count at the index `is`. All histograms must implement this.
The implementation should use `@propagate_inbounds` for good performance.
"""
increment_weight!

"""
    increment_subweight!(h, is...)

Increments the bin count at the index `is`. All histograms implementing SIMD parallelization must implement this.
The implementation should use `@propagate_inbounds` for good performance.
"""
increment_subweight!

"""
    sum_subweights!(h)

Sums the subweights into the weights matrix to produce the final weights.
All histograms implementing SIMD parallelization must implement this.
"""
sum_subweights!

"""
    counts(h)

Returns the bin counts of the histogram `h`. All histograms must implement this.
"""
counts

"""
    zero!(h)

Sets all bin counts of the histogram `h` to zero. All histograms must implement this.
"""
zero!

function increment_bins!(::BinSearchAlgorithm, ::NoParallelization, h, data::Union{AbstractVector,AbstractMatrix})
    for c = 1:size(data, 2)
        for r = 1:size(data, 1)
            @inbounds x = data[r, c]
            @inbounds i = bin_search(h, 1, x)
            @inbounds increment_weight!(h, i)
        end
    end

    nothing
end

function increment_bins!(
    ::BinSearchAlgorithm,
    ::NoParallelization,
    h,
    data1::Union{AbstractVector,AbstractMatrix},
    data2::Union{AbstractVector,AbstractMatrix},
)
    for c = 1:size(data1, 2)
        for r = 1:size(data1, 1)
            @inbounds tx = data1[r, c]
            @inbounds ty = data2[r, c]
            @inbounds ix = bin_search(h, 1, tx)
            @inbounds iy = bin_search(h, 2, ty)
            @inbounds increment_weight!(h, ix, iy)
        end
    end

    nothing
end
