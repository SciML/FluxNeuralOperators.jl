export l₂loss

function l₂loss(𝐲̂, 𝐲; agg=mean, is_grid_normalized=true)
    feature_dims = 2:(ndims(𝐲)-1)
    loss = agg(.√(sum(abs2, 𝐲̂-𝐲, dims=feature_dims)))

    return is_grid_normalized ? loss/prod(feature_dims) : loss
end
