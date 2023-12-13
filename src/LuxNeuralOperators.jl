module LuxNeuralOperators

import PrecompileTools: @recompile_invalidations
import Reexport: @reexport

@recompile_invalidations begin
    using ArrayInterface, FFTW, Lux, Random

    import ChainRulesCore as CRC
    import Lux.Experimental: @compact
    import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer,
        initialparameters, initialstates
    import Random: AbstractRNG
end

@reexport using Lux, Random

__default_rng() = Xoshiro(0)

const True = Val(true)
const False = Val(false)

include("transform.jl")
include("layers.jl")
include("fno.jl")

export FourierTransform
export SpectralConv, OperatorConv
export SpectralKernel, OperatorKernel
export FourierNeuralOperator

end
