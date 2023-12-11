module LuxNeuralOperators

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArrayInterface, FFTW, Lux, Random, TransmuteDims

    import ChainRulesCore as CRC
    import Lux.Experimental: @compact
    import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer,
        initialparameters, initialstates
    import Random: AbstractRNG
end

__default_rng() = Xoshiro(0)

include("transform.jl")
include("layers.jl")
include("fno.jl")

export FourierTransform
export SpectralConv, OperatorConv
export SpectralKernel, OperatorKernel
export FourierNeuralOperator

end
