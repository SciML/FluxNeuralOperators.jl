export l₂loss

function l₂loss(𝐲̂, 𝐲; agg=mean, grid_normalize=true)
    feature_dims = 2:(ndims(𝐲)-1)
    loss = agg(.√(sum(abs2, 𝐲̂-𝐲, dims=feature_dims)))

    return grid_normalize ? loss/prod(feature_dims) : loss
end
