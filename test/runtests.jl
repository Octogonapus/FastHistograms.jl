using FastHistograms
using Test
using Random

@testset "single threaded fixed width 2D" begin
    h = SingleThreadFixedWidth2DHistogram()

    img1 = zeros(10, 10)
    img1[:, 6:end] .= 0xff

    img2 = zeros(10, 10)
    img2[6:end, :] .= 0xff

    calc_hist!(h, img1, img2)
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

@testset "zero bins" begin
    h = SingleThreadFixedWidth2DHistogram()
    calc_hist!(h, rand(UInt8, 10, 10), rand(UInt8, 10, 10))

    @test any(counts(h) .!= 0)

    zero!(h)
    @test all(counts(h) .== 0)
end

@testset "default hist bin type" begin
    h = SingleThreadFixedWidth2DHistogram()
    @test bin_type(h) == UInt8
end

@testset "custom hist bin type" begin
    h = SingleThreadFixedWidth2DHistogram(Int32(0), Int32(16), 16)
    @test bin_type(h) == Int32
end
