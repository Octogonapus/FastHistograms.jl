using FastHistograms, StatsBase, BenchmarkTools

function bench(img1, img2)
    bins = 0x00:UInt8(16):0xff
    h = SingleThreadFixedWidth2DHistogram(bins)
    hist_bench = @benchmarkable calc_hist!($h, $img1, $img2)

    img1_vec = vec(img1)
    img2_vec = vec(img2)
    nbins = length(bins)
    stats_base_bench =
        @benchmarkable fit(StatsBase.Histogram, ($img1_vec, $img2_vec), nbins = $nbins)

    return hist_bench, stats_base_bench
end

function run_all_benchmarks()
    img1 = rand(UInt8, 40, 80)
    img2 = rand(UInt8, 40, 80)
    run.(bench(img1, img2))
end
