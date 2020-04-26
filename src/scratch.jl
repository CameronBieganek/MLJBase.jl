

y = categorical(
    ['b', 'c', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'a', 'a', 'a']
)


scv = StratifiedCV(nfolds=3)

MLJBase.train_test_pairs(scv, 1:12, nothing, y)
MLJBase.my_train_test_pairs1(scv, 1:12, nothing, y)
MLJBase.my_train_test_pairs2(scv, 1:12, nothing, y)


# Benchmark.

using BenchmarkTools

cv = CV(nfolds=5)
scv = StratifiedCV(nfolds=5)

N = 300_000
rows = 1:(10N)

y = (
    vcat(fill(:a, N), fill(:b, 2N), fill(:c, 3N), fill(:d, 4N))
    |> shuffle
    |> categorical
);


@btime MLJBase.train_test_pairs($scv, $rows, nothing, $y);
@btime MLJBase.my_train_test_pairs1($scv, $rows, nothing, $y);
@btime MLJBase.my_train_test_pairs2($scv, $rows, nothing, $y);


# Attempting to replicate the performance of StatsBase.countmap
function my_countmap(x::AbstractVector{T}) where {T}
    d = Dict{T, Int}()
    for v in x
        d[v] = get(d, v, 0) + 1
    end
    d
end

function my_countmap2(x::AbstractVector{T}) where {T}
    d = Dict{T, Int}()
    for v in x
        if haskey(d, v)
            d[v] += 1
        else
            d[v] = 1
        end
    end
    d
end
