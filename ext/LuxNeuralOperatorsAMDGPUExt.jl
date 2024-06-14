module LuxNeuralOperatorsAMDGPUExt

using AMDGPU: AnyROCArray
using LuxNeuralOperators: LuxNeuralOperators

# This should be upstreamed to NNlib before we release this package
@inline function LuxNeuralOperators.__batched_mul(
        x::AnyROCArray{<:Union{ComplexF16, ComplexF32, ComplexF64}, 3},
        y::AnyROCArray{<:Union{ComplexF16, ComplexF32, ComplexF64}, 3})
    # FIXME: This is not good for performance but that is okay for now
    return stack(*, eachslice(x; dims=3), eachslice(y; dims=3))
end

end