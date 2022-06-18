struct NOMAD{T1, T2}
    approximator_net::T1
    decoder_net::T2
end

function NOMAD(architecture_approximator::Tuple, architecture_decoder::Tuple,
                        act_approximator = identity, act_decoder=true;
                        init_approximator = Flux.glorot_uniform,
                        init_decoder = Flux.glorot_uniform,
                        bias_approximator=true, bias_decoder=true)

    approximator_net = construct_subnet(architecture_approximator, act_approximator;
                                    init=init_approximator, bias=bias_approximator)

    decoder_net = construct_subnet(architecture_decoder, act_decoder;
                                    init=init_decoder, bias=bias_decoder)

    return NOMAD{typeof(approximator_net), typeof(decoder_net)}(approximator_net, decoder_net)
end

Flux.@functor NOMAD

function (a::NOMAD)(x::AbstractArray, y::AbstractVecOrMat)
    # Assign the parameters
    approximator, decoder = a.approximator_net, a.decoder_net

    return decoder(cat(approximator(x), y', dims=1))'
end

# Print nicely
function Base.show(io::IO, l::NOMAD)
    print(io, "NOMAD with\napproximator net: (",l.approximator_net)
    print(io, ")\n")
    print(io, "Decoder net: (", l.decoder_net)
    print(io, ")\n")
end
