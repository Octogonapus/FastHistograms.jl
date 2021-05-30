function test_parameterized_hist(bin_type::FixedWidth, search_algorithm, parallelization)
    @testset "2D 16x16 corners" begin
        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(0x00, 0xff, 16), (0x00, 0xff, 16)])

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
        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(0x00, 0xff, 4), (0x00, 0xff, 4)])
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
        @test bin_search(h, 1, 0x00) == 1
        @test bin_search(h, 1, 0x40) == 2
        @test bin_search(h, 1, 0x80) == 3
        @test bin_search(h, 1, 0xc0) == 4

        increment_bins!(h, img1, img2)
        @test counts(h) == [
            1 0 0 0
            1 0 0 0
            0 1 0 0
            1 0 0 0
        ]
    end

    @testset "regression 1 2x2" begin
        img1 = [
            0x00 0x40
            0x80 0xc0
        ]

        img2 = [
            0x00 0x00
            0x40 0x00
        ]

        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(0x00, 0xff, 4), (0x00, 0xff, 4)])

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

    @testset "regression 1 1x4 and 4x1" begin
        img1 = [0x00, 0x40, 0x80, 0xc0]

        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(0x00, 0xff, 4)])

        increment_bins!(h, img1)
        fh_counts = counts(h)
        sb_counts = StatsBase.fit(StatsBase.Histogram, vec(img1), 0x00:UInt8(64):0x10f).weights
        @test fh_counts == sb_counts

        # Also test the transpose of img1
        zero!(h)
        increment_bins!(h, transpose(img1))
        fh_counts = counts(h)
        sb_counts = StatsBase.fit(StatsBase.Histogram, vec(transpose(img1)), 0x00:UInt8(64):0x10f).weights
        @test fh_counts == sb_counts
    end

    @testset "random regressions (2D, UInt8, square)" begin
        for i = 1:100
            img_size = rand(2:64, 2)
            img1 = rand(0x00:0xff, img_size...)
            img2 = rand(0x00:0xff, img_size...)

            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(0x00, 0xff, 16), (0x00, 0xff, 16)])

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

    @testset "random regressions (2D, Float32, square)" begin
        for i = 1:100
            img_size = rand(2:64, 2)
            img1 = rand(img_size...)
            img2 = rand(img_size...)

            h = create_fast_histogram(
                bin_type,
                search_algorithm,
                parallelization,
                [(Float32(0.0), Float32(1.0), 16), (Float32(0.0), Float32(1.0), 16)],
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

    @testset "random regressions (2D, Float32, non-square)" begin
        for i = 1:100
            img_size = rand(2:64, 2)
            img1 = rand(img_size...)
            img2 = rand(img_size...)

            h = create_fast_histogram(
                bin_type,
                search_algorithm,
                parallelization,
                [(Float32(0.0), Float32(1.0), 8), (Float32(0.0), Float32(1.0), 16)],
            )

            increment_bins!(h, img1, img2)

            fh_counts = counts(h)
            sb_counts =
                StatsBase.fit(
                    StatsBase.Histogram,
                    (vec(img1), vec(img2)),
                    # StatsBase takes the edges (number of bins plus one), so add 1 to the length
                    (
                        range(Float32(0.0); stop = Float32(1.0), length = 9),
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

            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [(Float32(0.0), Float32(1.0), 16)])

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
