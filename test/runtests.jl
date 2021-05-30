using FastHistograms
using FastHistograms: BinType, FixedWidth, VariableWidth, UnboundedWidth
using FastHistograms: BinSearchAlgorithm, Arithmetic, BinarySearch
using FastHistograms: HistogramParallelization, NoParallelization, SIMD, PrivateThreads
using Test
using Random
import StatsBase
using BenchmarkTools

const run_1D_tests = true
const run_2D_tests = true

include("test_fixed_width.jl")
include("test_variable_width.jl")

invalid_combination(::BinType, ::BinSearchAlgorithm, ::HistogramParallelization) = false
# SIMD binary search needs a special implementation that I don't have yet
invalid_combination(::BinType, ::BinarySearch, ::SIMD) = true
# Arithmetic only works for fixed width
invalid_combination(::Union{VariableWidth,UnboundedWidth}, ::Arithmetic, ::HistogramParallelization) = true

@testset "FastHistograms.jl" begin
    bin_types = [FixedWidth(), VariableWidth()]
    search_algorithms = [Arithmetic(), BinarySearch()]
    parallelizations = [NoParallelization(), SIMD(), PrivateThreads()]

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

    @testset "partition_data_to_threads" begin
        @test FastHistograms.partition_data_to_threads(1:1, 3) == [1:1]
        @test FastHistograms.partition_data_to_threads(1:5, 3) == [1:1, 2:5]
        @test FastHistograms.partition_data_to_threads(1:30, 3) == [1:10, 11:20, 21:30]
        @test FastHistograms.partition_data_to_threads(1:31, 3) == [1:10, 11:20, 21:31]
        @test FastHistograms.partition_data_to_threads(1:32, 3) == [1:10, 11:20, 21:32]
        @test FastHistograms.partition_data_to_threads(1:33, 3) == [1:11, 12:22, 23:33]
    end
end
