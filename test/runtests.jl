using FastHistograms
using Test
using Random
import StatsBase

@testset "FastHistograms.jl" begin
    @testset "histogram computations parameterized by parallelization" for parallelization in
                                                                           [
        FastHistograms.NoParallelization(),
        FastHistograms.SIMD(),
    ]
        @testset "single threaded fixed width 2D" begin
            h = create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                parallelization,
                Val{2}(),
                0x00,
                0xff,
                16,
            )
            @test counts(h) == zeros(16, 16)

            img1 = zeros(10, 10)
            img1[:, 6:end] .= 0xff

            img2 = zeros(10, 10)
            img2[6:end, :] .= 0xff

            bin_update!(h, img1, img2)
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
        end

        @testset "4x4" begin
            h = create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                parallelization,
                Val{2}(),
                0x00,
                0xff,
                4,
            )
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

            bin_update!(h, img1, img2)
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

                h = create_fast_histogram(
                    FastHistograms.FixedWidth(),
                    FastHistograms.Arithmetic(),
                    parallelization,
                    Val{2}(),
                    0x00,
                    0xff,
                    4,
                )

                bin_update!(h, img1, img2)

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

                h = create_fast_histogram(
                    FastHistograms.FixedWidth(),
                    FastHistograms.Arithmetic(),
                    parallelization,
                    Val{1}(),
                    0x00,
                    0xff,
                    4,
                )

                bin_update!(h, img1)

                fh_counts = counts(h)
                sb_counts =
                    StatsBase.fit(
                        StatsBase.Histogram,
                        vec(img1),
                        0x00:UInt8(64):0x10f,
                    ).weights

                @test fh_counts == sb_counts
            end

            @testset "random regressions (2D, UInt8)" begin
                for i = 1:100
                    img_size = rand(2:64, 2)
                    img1 = rand(0x00:0xff, img_size...)
                    img2 = rand(0x00:0xff, img_size...)

                    h = create_fast_histogram(
                        FastHistograms.FixedWidth(),
                        FastHistograms.Arithmetic(),
                        parallelization,
                        Val{2}(),
                        0x00,
                        0xff,
                        16,
                    )

                    bin_update!(h, img1, img2)

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
                        FastHistograms.FixedWidth(),
                        FastHistograms.Arithmetic(),
                        parallelization,
                        Val{2}(),
                        Float32(0.0),
                        Float32(1.0),
                        16,
                    )

                    bin_update!(h, img1, img2)

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
                        FastHistograms.FixedWidth(),
                        FastHistograms.Arithmetic(),
                        parallelization,
                        Val{1}(),
                        Float32(0.0),
                        Float32(1.0),
                        16,
                    )

                    bin_update!(h, img1)

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

    @testset "zero bins" begin
        h = create_fast_histogram(
            FastHistograms.FixedWidth(),
            FastHistograms.Arithmetic(),
            FastHistograms.NoParallelization(),
            Val{2}(),
            0x00,
            0xff,
            16,
        )

        bin_update!(h, rand(UInt8, 10, 10), rand(UInt8, 10, 10))

        @test any(counts(h) .!= 0)

        zero!(h)
        @test all(counts(h) .== 0)
    end

    @testset "default hist bin type" begin
        h = create_fast_histogram(
            FastHistograms.FixedWidth(),
            FastHistograms.Arithmetic(),
            FastHistograms.NoParallelization(),
            Val{2}(),
            0x00,
            0xff,
            16,
        )

        @test eltype(h) == UInt8
    end

    @testset "custom hist bin type" begin
        h = create_fast_histogram(
            FastHistograms.FixedWidth(),
            FastHistograms.Arithmetic(),
            FastHistograms.NoParallelization(),
            Val{2}(),
            Int32(0),
            Int32(16),
            16,
        )

        @test eltype(h) == Int32
    end
end
