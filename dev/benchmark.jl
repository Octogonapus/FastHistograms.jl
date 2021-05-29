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
    hist_bench_noparallel = @benchmarkable increment_bins!($h_noparallel, $img1, $img2)

    h_noparallel_bs = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.BinarySearch(),
        FastHistograms.NoParallelization(),
        Val{2}(),
        0x00,
        0xff,
        16,
    )
    hist_bench_noparallel_bs = @benchmarkable increment_bins!($h_noparallel, $img1, $img2)

    h_simd = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.SIMD(),
        Val{2}(),
        0x00,
        0xff,
        16,
    )
    hist_bench_simd = @benchmarkable increment_bins!($h_simd, $img1, $img2)

    img1_vec = vec(img1)
    img2_vec = vec(img2)
    stats_base_bench = @benchmarkable StatsBase.fit(
        StatsBase.Histogram,
        ($img1_vec, $img2_vec),
        (0x00:UInt8(16):0x10f, 0x00:UInt8(16):0x10f),
    )

    return stats_base_bench, hist_bench_noparallel, hist_bench_noparallel_bs, hist_bench_simd
end

bench_df_row(trial) = [time(trial), gctime(trial), allocs(trial), memory(trial)]

function run_all_benchmarks(img_type, img_size)
    img1 = rand(img_type, img_size...)
    img2 = rand(img_type, img_size...)
    tune!.(bench_noparallel(img1, img2))
    results = run.(bench_noparallel(img1, img2))
    df = DataFrame(
        "Rows" => ["Expected Time (ns)", "Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "StatsBase" => [32000, bench_df_row(results[1])...],
        "FH (NoParallel)" => [12000, bench_df_row(results[2])...],
        "FH (NoParallel, BS)" => [12000, bench_df_row(results[3])...],
        "FH (SIMD)" => [15000, bench_df_row(results[4])...],
    )
    pretty_table(df)
end

function run_all_benchmarks()
    types = [UInt8, Float32]
    sizes = [(40, 80), (256, 256)]
    for img_type in types
        for img_size in sizes
            println("Benchmarking type=$img_type, size=$img_size")
            run_all_benchmarks(img_type, img_size)
        end
    end
end

run_all_benchmarks()
