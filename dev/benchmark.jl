using FastHistograms, StatsBase, BenchmarkTools, DataFrames, PrettyTables

function bench_noparallel(img1, img2)
    h_noparallel = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.NoParallelization(),
        [(0x00, 0xff, 16), (0x00, 0xff, 16)],
    )
    hist_bench_noparallel = @benchmarkable increment_bins!($h_noparallel, $img1, $img2)

    h_noparallel_bs = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.BinarySearch(),
        FastHistograms.NoParallelization(),
        [(0x00, 0xff, 16), (0x00, 0xff, 16)],
    )
    hist_bench_noparallel_bs = @benchmarkable increment_bins!($h_noparallel, $img1, $img2)

    h_simd = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.SIMD(),
        [(0x00, 0xff, 16), (0x00, 0xff, 16)],
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

function bench_parallel(data)
    img_vec = vec(data)
    stats_base_bench = @benchmarkable StatsBase.fit(
        StatsBase.Histogram,
        $img_vec,
        range(Float32(0.0); stop = Float32(512.0), length = 513),
    )

    h_private_threads = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.PrivateThreads(),
        [(Float32(0.0), Float32(512.0), 512)],
    )
    hist_bench_private_threads = @benchmarkable increment_bins!($h_private_threads, $data)

    return stats_base_bench, hist_bench_private_threads
end

bench_df_row(trial) = [time(trial), gctime(trial), allocs(trial), memory(trial)]

function run_all_benchmarks(img_type, img_size)
    img1 = rand(img_type, img_size...)
    img2 = rand(img_type, img_size...)
    tune!.(bench_noparallel(img1, img2))
    results_noparallel = run.(bench_noparallel(img1, img2))

    df = DataFrame(
        "Rows" => ["Expected Time (ns)", "Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "StatsBase" => [32000, bench_df_row(results_noparallel[1])...],
        "FH (NoParallel)" => [12000, bench_df_row(results_noparallel[2])...],
        "FH (NoParallel, BS)" => [12000, bench_df_row(results_noparallel[3])...],
        "FH (SIMD)" => [15000, bench_df_row(results_noparallel[4])...],
    )
    pretty_table(df)
end

function run_all_benchmarks()
    # types = [UInt8, Float32]
    # sizes = [(40, 80), (256, 256)]
    # for img_type in types
    #     for img_size in sizes
    #         println("Benchmarking type=$img_type, size=$img_size")
    #         run_all_benchmarks(img_type, img_size)
    #     end
    # end

    ahtl_data = rand(Float32, 100000000) .* (512 - Float32(0.01))
    tune!.(bench_parallel(ahtl_data))
    results_parallel = run.(bench_parallel(ahtl_data))
    df = DataFrame(
        "Rows" => ["Expected Time (ns)", "Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "StatsBase" => [0, bench_df_row(results_parallel[1])...],
        "FH (PrivateThreads, AHTL)" => [108249000, bench_df_row(results_parallel[2])...],
    )
    pretty_table(df)
end

run_all_benchmarks()
