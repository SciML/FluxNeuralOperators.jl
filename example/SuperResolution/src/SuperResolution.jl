module SuperResolution

using NeuralOperators
using Flux
using Flux.Losses: mse
using Flux.Data: DataLoader
using GeometricFlux
using Graphs
using CUDA
using JLD2
using ProgressMeter: Progress, next!

include("data.jl")
include("models.jl")

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @info "model updated!"
end

function get_model()
    f = jldopen(joinpath(@__DIR__, "../model/model.jld2"))
    model = f["model"]
    close(f)

    return model
end

loss(m, 𝐱, 𝐲) = mse(m(𝐱), 𝐲)
loss(m, loader::DataLoader, device) = sum(loss(m, 𝐱 |> device, 𝐲 |> device) for (𝐱, 𝐲) in loader)/length(loader)

end
