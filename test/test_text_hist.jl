function test_text_hist(parallelization::HistogramParallelization)
    @testset "1D text" begin
        strings = split(lowercase(read(joinpath(@__DIR__, "pride_and_prejudice.txt"), String)))
        expected = Dict([s => count(isequal(s), strings) for s in unique(strings)])

        h = create_fast_histogram(UnboundedWidth(), HashFunction(), parallelization)

        increment_bins!(h, strings)
        @test counts(h) == expected

        zero!(h)
        increment_bins!(h, strings)
        @test counts(h) == expected

        zero!(h)
        @test isempty(counts(h))
    end
end
