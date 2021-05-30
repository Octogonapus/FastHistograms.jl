function test_parameterized_hist(bin_type::VariableWidth, search_algorithm, parallelization)
    @testset "regression 1 2x2" begin
        img1 = [
            0x00 0x40
            0x80 0xc0
        ]

        img2 = [
            0x00 0x00
            0x40 0x00
        ]

        edge = range(0x00; stop = 0xff, length = 5)
        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [edge, edge])

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

        h = create_fast_histogram(bin_type, search_algorithm, parallelization, [range(0x00; stop = 0xff, length = 5)])

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

            edge = range(0x00; stop = 0xff, length = 17)
            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [edge, edge])

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

            edge = range(Float32(0.0); stop = Float32(1.0), length = 17)
            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [edge, edge])

            increment_bins!(h, img1, img2)

            fh_counts = counts(h)
            sb_counts =
                StatsBase.fit(
                    StatsBase.Histogram,
                    (vec(img1), vec(img2)),
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
                [
                    range(Float32(0.0); stop = Float32(1.0), length = 9),
                    range(Float32(0.0); stop = Float32(1.0), length = 17),
                ],
            )

            increment_bins!(h, img1, img2)

            fh_counts = counts(h)
            sb_counts =
                StatsBase.fit(
                    StatsBase.Histogram,
                    (vec(img1), vec(img2)),
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

            h = create_fast_histogram(
                bin_type,
                search_algorithm,
                parallelization,
                [range(Float32(0.0); stop = Float32(1.0), length = 17)],
            )

            increment_bins!(h, img1)

            fh_counts = counts(h)
            sb_counts =
                StatsBase.fit(
                    StatsBase.Histogram,
                    vec(img1),
                    range(Float32(0.0); stop = Float32(1.0), length = 17),
                ).weights

            @test fh_counts == sb_counts

            if fh_counts != sb_counts
                break
            end
        end
    end

    @testset "random regressions (2D, Float32, non-square, variable width)" begin
        for i = 1:100
            img_size = rand(2:64, 2)
            img1 = rand(img_size...)
            img2 = rand(img_size...)

            edge1 = [0.0, 0.1, 0.5, 0.9, 1.0]
            edge2 = [0.0, 0.5, 0.6, 1.0]
            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [edge1, edge2])

            increment_bins!(h, img1, img2)

            fh_counts = counts(h)
            sb_counts = StatsBase.fit(StatsBase.Histogram, (vec(img1), vec(img2)), (edge1, edge2)).weights

            @test fh_counts == sb_counts

            if fh_counts != sb_counts
                break
            end
        end
    end

    @testset "random regressions (1D, Float32, variable width)" begin
        for i = 1:100
            img_size = rand(2:64, 2)
            img1 = rand(img_size...)

            edge = [0.0, 0.1, 0.5, 0.9, 1.0]
            h = create_fast_histogram(bin_type, search_algorithm, parallelization, [edge])

            increment_bins!(h, img1)

            fh_counts = counts(h)
            sb_counts = StatsBase.fit(StatsBase.Histogram, vec(img1), edge).weights

            @test fh_counts == sb_counts

            if fh_counts != sb_counts
                break
            end
        end
    end
end
