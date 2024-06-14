# Temporarily capture certain calls like AMDGPU for ComplexFloats
@inline __batched_mul(x, y) = x ⊠ y
