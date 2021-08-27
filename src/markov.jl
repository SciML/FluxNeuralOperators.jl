export MarkovOperator

struct MarkovOperator{F, T}
    m::F
    Δt::T
end

Flux.@functor MarkovOperator

function (mo::MarkovOperator)(𝐱)
    state = 𝐱
    for _ in 1:mo.Δt
        state = mo.m(state)
    end

    return state
end
