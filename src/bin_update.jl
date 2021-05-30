
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
    get_weights(h)

Returns the weights array. All histograms must implement this.
"""
get_weights

"""
    get_subweights(h)

Returns the subweights array. All histograms implementing SIMD and PrivateThreads parallelization must implement this.
"""
get_subweights

"""
    counts(h)

Returns the bin counts of the histogram `h`. All histograms must implement this.
"""
counts(h) = get_weights(h)

"""
    zero!(h)

Sets all bin counts of the histogram `h` to zero. All histograms must implement this.
"""
function zero!(h)
    get_weights(h) .= 0
    get_subweights(h) .= 0
end

function increment_bins!(::BinSearchAlgorithm, ::NoParallelization, h, data)
    for d in data
        @inbounds i = bin_search(h, 1, d)
        @inbounds get_weights(h)[i] += 1
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
            @inbounds get_weights(h)[ix, iy] += 1
        end
    end

    nothing
end

function increment_bins!(::Arithmetic, ::SIMD, h, data::Union{AbstractVector,AbstractMatrix})
    rows = size(data, 1)
    align_rows = floor(Int, rows / 3)

    w = get_subweights(h)
    for c = 1:size(data, 2)
        r = 1

        while r < align_rows
            @turbo for i = 0:2
                @inbounds tx = data[r+i, c]
                @inbounds w[bin_search(h, 1, tx), i+1] += 1
            end
            r += 3
        end

        for r2 = r:rows
            @inbounds tx = data[r2, c]
            @inbounds w[bin_search(h, 1, tx), 1] += 1
        end
    end

    sum_subweights!(h)

    nothing
end

function increment_bins!(
    ::Arithmetic,
    ::SIMD,
    h,
    img1::Union{AbstractVector,AbstractMatrix},
    img2::Union{AbstractVector,AbstractMatrix},
)
    rows = size(img1, 1)
    align_rows = floor(Int, rows / 4)

    w = get_subweights(h)
    for c = 1:size(img1, 2)
        r = 1

        while r < align_rows
            @turbo for i = 0:3
                @inbounds tx = img1[r+i, c]
                @inbounds ty = img2[r+i, c]
                ix = bin_search(h, 1, tx)
                iy = bin_search(h, 2, ty)
                @inbounds w[ix, iy, i+1] += 1
            end
            r += 4
        end

        for r2 = r:rows
            @inbounds tx = img1[r2, c]
            @inbounds ty = img2[r2, c]
            ix = bin_search(h, 1, tx)
            iy = bin_search(h, 2, ty)
            @inbounds w[ix, iy, 1] += 1
        end
    end

    sum_subweights!(h)

    nothing
end

function increment_bins!(
    ::BinSearchAlgorithm,
    ::PrivateThreads,
    h,
    data::Union{AbstractVector,AbstractMatrix},
)
    nthreads = size(get_subweights(h), 2)
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
            @inbounds get_subweights(h)[i, thread_idx] += 1
        end
    end

    sum!(h.weights, get_subweights(h))

    nothing
end

function increment_bins!(
    ::BinSearchAlgorithm,
    ::PrivateThreads,
    h,
    data1::Union{AbstractVector,AbstractMatrix},
    data2::Union{AbstractVector,AbstractMatrix},
)
    nthreads = size(get_subweights(h), 3)
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
            @inbounds get_subweights(h)[ix, iy, thread_idx] += 1
        end
    end

    sum!(h.weights, get_subweights(h))

    nothing
end

sum_subweights!(h) = sum!(get_weights(h), get_subweights(h))
