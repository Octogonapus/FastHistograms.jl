using FastHistograms
using FastHistograms: BinType, FixedWidth, VariableWidth
using FastHistograms: BinSearchAlgorithm, Arithmetic, BinarySearch
using FastHistograms: HistogramParallelization, NoParallelization, SIMD
using Test
using Random
import StatsBase
using BenchmarkTools

include("test_fixed_width.jl")
include("test_variable_width.jl")

invalid_combination(::BinType, ::BinSearchAlgorithm, ::HistogramParallelization) = false
invalid_combination(::BinType, ::BinarySearch, ::SIMD) = true
invalid_combination(::VariableWidth, ::Arithmetic, ::HistogramParallelization) = true

@testset "FastHistograms.jl" begin
    bin_types = [FixedWidth(), VariableWidth()]
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
end
