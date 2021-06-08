using FastHistograms, StatsBase, BenchmarkTools, DataFrames, PrettyTables

function bench_noparallel(img1, img2)
    h_fw_a_np = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.NoParallelization(),
        [(0x00, 0xff, 16), (0x00, 0xff, 16)],
    )
    bench_fh_fw_a_np = @benchmarkable increment_bins!($h_fw_a_np, $img1, $img2)

    edge1 = [0.0, 0.1, 0.5, 0.9, 1.0]
    edge2 = [0.0, 0.5, 0.6, 1.0]
    h_vw_bs_np = create_fast_histogram(
        FastHistograms.VariableWidth(),
        FastHistograms.BinarySearch(),
        FastHistograms.NoParallelization(),
        [edge1, edge2],
    )
    bench_fh_vw_bs_np = @benchmarkable increment_bins!($h_vw_bs_np, $img1, $img2)

    h_fw_a_simd = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.SIMD(),
        [(0x00, 0xff, 16), (0x00, 0xff, 16)],
    )
    bench_fh_fw_a_simd = @benchmarkable increment_bins!($h_fw_a_simd, $img1, $img2)

    img1_vec = vec(img1)
    img2_vec = vec(img2)
    bench_sb_fw = @benchmarkable StatsBase.fit(
        StatsBase.Histogram,
        ($img1_vec, $img2_vec),
        (0x00:UInt8(16):0x10f, 0x00:UInt8(16):0x10f),
    )

    bench_sb_vw = @benchmarkable StatsBase.fit(StatsBase.Histogram, ($img1_vec, $img2_vec), ($edge1, $edge2))

    return bench_fh_fw_a_np, bench_fh_vw_bs_np, bench_fh_fw_a_simd, bench_sb_fw, bench_sb_vw
end

function bench_parallel(data)
    img_vec = vec(data)
    bench_sb_fw = @benchmarkable StatsBase.fit(
        StatsBase.Histogram,
        $img_vec,
        range(Float32(0.0); stop = Float32(512.0), length = 513),
    )

    h_fw_a_pt = create_fast_histogram(
        FastHistograms.FixedWidth(),
        FastHistograms.Arithmetic(),
        FastHistograms.PrivateThreads(),
        [(Float32(0.0), Float32(512.0), 512)],
    )
    bench_fh_fw_a_pt = @benchmarkable increment_bins!($h_fw_a_pt, $data)

    return bench_fh_fw_a_pt, bench_sb_fw
end

bench_df_row(trial) = [time(trial), gctime(trial), allocs(trial), memory(trial)]

function run_all_benchmarks(img_type, img_size)
    img1 = rand(img_type, img_size...)
    img2 = rand(img_type, img_size...)
    tune!.(bench_noparallel(img1, img2))
    results_noparallel = run.(bench_noparallel(img1, img2))

    df = DataFrame(
        "Rows" => ["Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "FastHistograms:\nFixedWidth,\nArithmetic,\nNoParallelization" => bench_df_row(results_noparallel[1]),
        "FastHistograms:\nVariableWidth,\nBinarySearch,\nNoParallelization" => bench_df_row(results_noparallel[2]),
        "FastHistograms:\nFixedWidth,\nArithmetic,\nSIMD" => bench_df_row(results_noparallel[3]),
        "StatsBase:\nFixedWidth" => bench_df_row(results_noparallel[4]),
        "StatsBase:\nVariableWidth" => bench_df_row(results_noparallel[5]),
    )
    pretty_table(df, autowrap = true, linebreaks = true)
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

    ahtl_data = rand(Float32, 100000000) .* (512 - Float32(0.01))
    tune!.(bench_parallel(ahtl_data))
    results_parallel = run.(bench_parallel(ahtl_data))
    df = DataFrame(
        "Rows" => ["Min Time (ns)", "GC Time (ns)", "Allocs (B)", "Memory (B)"],
        "FastHistograms:\nFixedWidth,\nArithmetic,\nPrivateThreads" => bench_df_row(results_parallel[1]), # AHTL is 108249000 ns
        "StatsBase:\nFixedWidth" => bench_df_row(results_parallel[2]),
    )
    pretty_table(df, autowrap = true, linebreaks = true)
end

run_all_benchmarks()
