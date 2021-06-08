using FastHistograms
using FastHistograms: BinType, FixedWidth, VariableWidth, UnboundedWidth
using FastHistograms: BinSearchAlgorithm, Arithmetic, BinarySearch, HashFunction
using FastHistograms: HistogramParallelization, NoParallelization, SIMD, PrivateThreads
using Test
using Random
import StatsBase
using BenchmarkTools
using Downloads

const run_1D_tests = true
const run_2D_tests = true

function get_test_resources()
    path1 = joinpath(@__DIR__, "pride_and_prejudice.txt")
    if !isfile(path1)
        Downloads.download("https://www.gutenberg.org/files/1342/1342-0.txt", path1)
    end
end

get_test_resources()

include("test_fixed_width.jl")
include("test_variable_width.jl")
include("test_text_hist.jl")

invalid_combination(::BinType, ::BinSearchAlgorithm, ::HistogramParallelization) = false
# SIMD binary search needs a special implementation that I don't have yet
invalid_combination(::BinType, ::BinarySearch, ::SIMD) = true
invalid_combination(::Union{FixedWidth, VariableWidth}, ::BinarySearch, ::SIMD) = true
# Arithmetic only works for fixed width
invalid_combination(::Union{VariableWidth,UnboundedWidth}, ::Arithmetic, ::HistogramParallelization) = true
# Text histograms can only use UnboundedWidth, HashFunction, and NoParallelization or PrivateThreads. SIMD not implemented yet.
invalid_combination(::Union{FixedWidth,VariableWidth}, ::Union{Arithmetic,BinarySearch}, ::HistogramParallelization) =
    true
invalid_combination(::UnboundedWidth, ::HashFunction, ::SIMD) = true

@testset "FastHistograms.jl" begin
    bin_types = [FixedWidth(), VariableWidth()]
    search_algorithms = [Arithmetic(), BinarySearch()]
    parallelizations = [NoParallelization(), SIMD(), PrivateThreads()]

    @testset "histogram computations BinType=$(bin_type)" for bin_type in bin_types
        @testset "BinSearchAlgorithm=$(search_algorithm)" for search_algorithm in search_algorithms
            @testset "HistogramParallelization=$(parallelization)" for parallelization in parallelizations
                if invalid_combination(bin_type, search_algorithm, parallelization)
                    continue
                end

                test_parameterized_hist(bin_type, search_algorithm, parallelization)
            end
        end
    end

    @testset "zero bins" begin
        h = create_fast_histogram(FixedWidth(), Arithmetic(), NoParallelization(), [(0x00, 0xff, 16), (0x00, 0xff, 16)])

        increment_bins!(h, rand(UInt8, 10, 10), rand(UInt8, 10, 10))

        @test any(counts(h) .!= 0)

        zero!(h)
        @test all(counts(h) .== 0)
    end

    @testset "default hist bin type" begin
        h = create_fast_histogram(FixedWidth(), Arithmetic(), NoParallelization(), [(0x00, 0xff, 16), (0x00, 0xff, 16)])

        @test eltype(h) == UInt8
    end

    @testset "custom hist bin type" begin
        h = create_fast_histogram(
            FixedWidth(),
            Arithmetic(),
            NoParallelization(),
            [(Int32(0), Int32(16), 16), (Int32(0), Int32(16), 16)],
        )

        @test eltype(h) == Int32
    end

    @testset "AHTL fixed data" begin
        # Generate the data in the same way the AHTL test does
        data = rand(Float32, 100000000) .* (512 - Float32(0.01))

        img_vec = vec(data)
        sb_counts = StatsBase.fit(
            StatsBase.Histogram,
            img_vec,
            range(Float32(0.0); stop = Float32(512.0), length = 513),
        ).weights

        h = create_fast_histogram(
            FastHistograms.FixedWidth(),
            FastHistograms.Arithmetic(),
            FastHistograms.PrivateThreads(),
            [(Float32(0.0), Float32(512.0), 512)],
        )
        increment_bins!(h, data)
        fh_counts = counts(h)
        @test fh_counts == sb_counts
    end

    let bin_type = UnboundedWidth(), search_algorithm = HashFunction()
        @testset "HistogramParallelization=$(parallelization)" for parallelization in parallelizations
            if invalid_combination(bin_type, search_algorithm, parallelization)
                continue
            end

            @show parallelization
            test_text_hist(parallelization)
        end
    end
end
