using FastHistograms, StatsBase, BenchmarkTools, DataFrames

function bench_noparallel(img1, img2)
    h_noparallel = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.NoParallelization(),
        Val{2}(),
        0x00,
        0xff,
        16,
    )
    hist_bench_noparallel = @benchmarkable bin_update!($h_noparallel, $img1, $img2)

    h_simd = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.SIMD(),
        Val{2}(),
        0x00,
        0xff,
        16,
    )
    hist_bench_simd = @benchmarkable bin_update!($h_simd, $img1, $img2)

    img1_vec = vec(img1)
    img2_vec = vec(img2)
    stats_base_bench = @benchmarkable StatsBase.fit(
        StatsBase.Histogram,
        ($img1_vec, $img2_vec),
        (0x00:UInt8(16):0x10f, 0x00:UInt8(16):0x10f),
    )

    return stats_base_bench, hist_bench_noparallel, hist_bench_simd
end

function run_all_benchmarks()
    img1 = rand(UInt8, 40, 80)
    img2 = rand(UInt8, 40, 80)
    run.(bench_noparallel(img1, img2)) # SB=32μs, FH_np=13μs, FH_simd=12μs
end
