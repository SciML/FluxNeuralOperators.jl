# Temporarily capture certain calls like AMDGPU for ComplexFloats
@inline __batched_mul(x, y) = __batched_mul(x, y, get_device((x, y)))
@inline function __batched_mul(
        x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3}, _)
    return x ⊠ y
end
@inline function __batched_mul(
        x::AbstractArray{<:Complex, 3}, y::AbstractArray{<:Complex, 3}, ::LuxAMDGPUDevice)
    # FIXME: This is not good for performance but that is okay for now
    return stack(*, eachslice(x; dims=3), eachslice(y; dims=3))
end
