using FastHistograms, StatsBase, BenchmarkTools, DataFrames, PrettyTables

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

bench_df_row(trial) = [time(trial), gctime(trial), allocs(trial), memory(trial)]

function run_all_benchmarks()
    img1 = rand(UInt8, 40, 80)
    img2 = rand(UInt8, 40, 80)
    tune!.(bench_noparallel(img1, img2))
    results = run.(bench_noparallel(img1, img2))
    df = DataFrame(
        "Rows" => ["Expected Time (ns)", "Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "StatsBase" => [32000, bench_df_row(results[1])...],
        "FH (NoParallel)" => [12000, bench_df_row(results[2])...],
        "FH (SIMD)" => [15000, bench_df_row(results[3])...],
    )
    pretty_table(df)
end

run_all_benchmarks()
