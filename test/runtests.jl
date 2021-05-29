using FastHistograms
using FastHistograms: BinType, FixedWidth
using FastHistograms: BinSearchAlgorithm, Arithmetic, BinarySearch
using FastHistograms: HistogramParallelization, NoParallelization, SIMD
using Test
using Random
import StatsBase
using BenchmarkTools

function test_parameterized_hist(bin_type, search_algorithm, parallelization)
    @testset "2D 16x16 corners" begin
        h = create_fast_histogram(bin_type, search_algorithm, parallelization, Val{2}(), 0x00, 0xff, 16)

        # Sanity check traits
        @test BinType(h) == bin_type
        @test BinSearchAlgorithm(h) == search_algorithm
        @test HistogramParallelization(h) == parallelization

        @test counts(h) == zeros(16, 16)

        img1 = zeros(10, 10)
        img1[:, 6:end] .= 0xff

        img2 = zeros(10, 10)
        img2[6:end, :] .= 0xff

        increment_bins!(h, img1, img2)
        for (i, v) in pairs(counts(h))
            if i == CartesianIndex((1, 1)) ||
               i == CartesianIndex((16, 1)) ||
               i == CartesianIndex((1, 16)) ||
               i == CartesianIndex((16, 16))
                @test v == 25
            else
                @test v == 0
            end
        end

        # Test zero! works
        zero!(h)
        @test counts(h) == zeros(16, 16)

        # FixedWidth should be able to run without allocating in all cases
        if bin_type isa FixedWidth
            bench = @benchmarkable increment_bins!($h, $img1, $img2)
            @test allocs(run(bench)) == 0
        end
    end

    @testset "2D 4x4" begin
        h = create_fast_histogram(bin_type, search_algorithm, parallelization, Val{2}(), 0x00, 0xff, 4)
        @test counts(h) == zeros(4, 4)

        img1 = [
            0x00 0x40
            0x80 0xc0
        ]

        img2 = [
            0x00 0x00
            0x40 0x00
        ]

        # Bin search steps
        # 0x00 0x00 = 1 1
        # 0x80 0x40 = 3 2
        # 0x40 0x00 = 2 1
        # 0xc0 0x00 = 4 1
        @test bin_search(h, 0x00) == 1
        @test bin_search(h, 0x40) == 2
        @test bin_search(h, 0x80) == 3
        @test bin_search(h, 0xc0) == 4

        increment_bins!(h, img1, img2)
        @test counts(h) == [
            1 0 0 0
            1 0 0 0
            0 1 0 0
            1 0 0 0
        ]
    end

    @testset "regressons" begin
        @testset "regression 1 2x2" begin
            img1 = [
                0x00 0x40
                0x80 0xc0
            ]

            img2 = [
                0x00 0x00
                0x40 0x00
            ]

            h = create_fast_histogram(bin_type, search_algorithm, parallelization, Val{2}(), 0x00, 0xff, 4)

            increment_bins!(h, img1, img2)

            fh_counts = counts(h)
            sb_counts =
                StatsBase.fit(
                    StatsBase.Histogram,
                    (vec(img1), vec(img2)),
                    (0x00:UInt8(64):0x10f, 0x00:UInt8(64):0x10f),
                ).weights

            @test fh_counts == sb_counts
        end

        @testset "regression 1 4x1" begin
            img1 = [0x00, 0x40, 0x80, 0xc0]
            img2 = [0x00, 0x00, 0x40, 0x00]

            h = create_fast_histogram(bin_type, search_algorithm, parallelization, Val{1}(), 0x00, 0xff, 4)

            increment_bins!(h, img1)

            fh_counts = counts(h)
            sb_counts = StatsBase.fit(StatsBase.Histogram, vec(img1), 0x00:UInt8(64):0x10f).weights

            @test fh_counts == sb_counts
        end

        @testset "random regressions (2D, UInt8)" begin
            for i = 1:100
                img_size = rand(2:64, 2)
                img1 = rand(0x00:0xff, img_size...)
                img2 = rand(0x00:0xff, img_size...)

                h = create_fast_histogram(bin_type, search_algorithm, parallelization, Val{2}(), 0x00, 0xff, 16)

                increment_bins!(h, img1, img2)

                fh_counts = counts(h)

                # StatsBase needs one more edge than the number of weights, but that doesn't fit evenly into
                # the 8-bit range [0x00, 0xff], so we need to extend the upper bound a bit to get it to do
                # the "logical equivalent" to what FastHistograms does.
                sb_counts =
                    StatsBase.fit(
                        StatsBase.Histogram,
                        (vec(img1), vec(img2)),
                        (0x00:UInt8(16):0x10f, 0x00:UInt8(16):0x10f),
                    ).weights

                @test fh_counts == sb_counts

                if fh_counts != sb_counts
                    break
                end
            end
        end

        @testset "random regressions (2D, Float32)" begin
            for i = 1:100
                img_size = rand(2:64, 2)
                img1 = rand(img_size...)
                img2 = rand(img_size...)

                h = create_fast_histogram(
                    bin_type,
                    search_algorithm,
                    parallelization,
                    Val{2}(),
                    Float32(0.0),
                    Float32(1.0),
                    16,
                )

                increment_bins!(h, img1, img2)

                fh_counts = counts(h)
                sb_counts =
                    StatsBase.fit(
                        StatsBase.Histogram,
                        (vec(img1), vec(img2)),
                        # StatsBase takes the edges (number of bins plus one), so add 1 to the length
                        (
                            range(Float32(0.0); stop = Float32(1.0), length = 17),
                            range(Float32(0.0); stop = Float32(1.0), length = 17),
                        ),
                    ).weights

                @test fh_counts == sb_counts

                if fh_counts != sb_counts
                    break
                end
            end
        end

        @testset "random regressions (1D, Float32)" begin
            for i = 1:100
                img_size = rand(2:64, 2)
                img1 = rand(img_size...)

                h = create_fast_histogram(
                    bin_type,
                    search_algorithm,
                    parallelization,
                    Val{1}(),
                    Float32(0.0),
                    Float32(1.0),
                    16,
                )

                increment_bins!(h, img1)

                fh_counts = counts(h)
                sb_counts =
                    StatsBase.fit(
                        StatsBase.Histogram,
                        vec(img1),
                        # StatsBase takes the edges (number of bins plus one), so add 1 to the length
                        range(Float32(0.0); stop = Float32(1.0), length = 17),
                    ).weights

                @test fh_counts == sb_counts

                if fh_counts != sb_counts
                    break
                end
            end
        end
    end
end

invalid_combination(::BinType, ::BinSearchAlgorithm, ::HistogramParallelization) = false
invalid_combination(::BinType, ::BinarySearch, ::SIMD) = true

@testset "FastHistograms.jl" begin
    bin_types = [FixedWidth()]
    search_algorithms = [Arithmetic(), BinarySearch()]
    parallelizations = [NoParallelization(), SIMD()]

    @testset "histogram computations BinType=$(bin_type)" for bin_type in bin_types
        @testset "BinSearchAlgorithm=$(search_algorithm)" for search_algorithm in search_algorithms
            @testset "HistogramParallelization=$(parallelization)" for parallelization in parallelizations
                if invalid_combination(bin_type, search_algorithm, parallelization)
                    break
                end

                test_parameterized_hist(bin_type, search_algorithm, parallelization)
            end
        end
    end

    @testset "zero bins" begin
        h = create_fast_histogram(FixedWidth(), Arithmetic(), NoParallelization(), Val{2}(), 0x00, 0xff, 16)

        increment_bins!(h, rand(UInt8, 10, 10), rand(UInt8, 10, 10))

        @test any(counts(h) .!= 0)

        zero!(h)
        @test all(counts(h) .== 0)
    end

    @testset "default hist bin type" begin
        h = create_fast_histogram(FixedWidth(), Arithmetic(), NoParallelization(), Val{2}(), 0x00, 0xff, 16)

        @test eltype(h) == UInt8
    end

    @testset "custom hist bin type" begin
        h = create_fast_histogram(FixedWidth(), Arithmetic(), NoParallelization(), Val{2}(), Int32(0), Int32(16), 16)

        @test eltype(h) == Int32
    end
end
