export l₂loss

function l₂loss(𝐲̂, 𝐲)
    feature_dims = 2:(ndims(𝐲)-1)

    loss = sum(.√(sum(abs2, 𝐲̂-𝐲, dims=feature_dims)))
    y_norm = sum(.√(sum(abs2, 𝐲, dims=feature_dims)))

    return loss / y_norm
end
