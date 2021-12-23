function get_filter(base::Symbol, k)
    if base == :legendre
        return legendre_filter(k)
    elseif base == :chebyshev
        return chebyshev_filter(k)
    else
        throw(ArgumentError("base must be one of :legendre or :chebyshev."))
    end
end

function legendre_ϕ_ψ(k)
    # TODO: row-major -> column major
    ϕ_coefs = zeros(k, k)
    ϕ_2x_coefs = zeros(k, k)

    p = Polynomial([-1, 2])  # 2x-1
    p2 = Polynomial([-1, 4])  # 4x-1

    for ki in 0:(k-1)
        l = convert(Polynomial, gen_poly(Legendre, ki))  # Legendre of n=ki
        ϕ_coefs[ki+1, 1:(ki+1)] .= sqrt(2*ki+1) .* coeffs(l(p))
        ϕ_2x_coefs[ki+1, 1:(ki+1)] .= sqrt(2*(2*ki+1)) .* coeffs(l(p2))
    end
    
    ψ1_coefs .= ϕ_2x_coefs
    ψ2_coefs = zeros(k, k)
    for ki in 0:(k-1)
        for i in 0:(k-1)
            a = ϕ_2x_coefs[ki+1, 1:(ki+1)]
            b = ϕ_coefs[i+1, 1:(i+1)]
            proj_ = proj_factor(a, b)
            view(ψ1_coefs, ki+1, :) .-= proj_ .* view(ϕ_coefs, i+1, :)
            view(ψ2_coefs, ki+1, :) .-= proj_ .* view(ϕ_coefs, i+1, :)
        end

        for j in 0:(k-1)
            a = ϕ_2x_coefs[ki+1, 1:(ki+1)]
            b = ψ1_coefs[j+1, :]
            proj_ = proj_factor(a, b)
            view(ψ1_coefs, ki+1, :) .-= proj_ .* view(ψ1_coefs, j+1, :)
            view(ψ2_coefs, ki+1, :) .-= proj_ .* view(ψ2_coefs, j+1, :)
        end

        a = ψ1_coefs[ki+1, :]
        norm1 = proj_factor(a, a)

        a = ψ2_coefs[ki+1, :]
        norm2 = proj_factor(a, a, complement=true)
        norm_ = sqrt(norm1 + norm2)
        ψ1_coefs[ki+1, :] ./= norm_
        ψ2_coefs[ki+1, :] ./= norm_
        zero_out!(ψ1_coefs)
        zero_out!(ψ2_coefs)
    end

    ϕ = [Polynomial(ϕ_coefs[i,:]) for i in 1:k]
    ψ1 = [Polynomial(ψ1_coefs[i,:]) for i in 1:k]
    ψ2 = [Polynomial(ψ2_coefs[i,:]) for i in 1:k]

    return ϕ, ψ1, ψ2
end

# function chebyshev_ϕ_ψ(k)
#     ϕ_coefs = zeros(k, k)
#     ϕ_2x_coefs = zeros(k, k)

#     p = Polynomial([-1, 2])  # 2x-1
#     p2 = Polynomial([-1, 4])  # 4x-1

#     for ki in 0:(k-1)
#         if ki == 0
#             ϕ_coefs[ki+1, 1:(ki+1)] .= sqrt(2/π)
#             ϕ_2x_coefs[ki+1, 1:(ki+1)] .= sqrt(4/π)
#         else
#             c = convert(Polynomial, gen_poly(Chebyshev, ki))  # Chebyshev of n=ki
#             ϕ_coefs[ki+1, 1:(ki+1)] .= 2/sqrt(π) .* coeffs(c(p))
#             ϕ_2x_coefs[ki+1, 1:(ki+1)] .= sqrt(2) * 2/sqrt(π) .* coeffs(c(p2))
#         end
#     end

#     ϕ = [ϕ_(ϕ_coefs[i, :]) for i in 1:k]

#     k_use = 2k
            
#     # phi = [partial(phi_, phi_coeff[i,:]) for i in range(k)]
    
#     # x = Symbol('x')
#     # kUse = 2*k
#     # roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
#     # x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
#     # # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
#     # # not needed for our purpose here, we use even k always to avoid
#     # wm = np.pi / kUse / 2
    
#     # psi1_coeff = np.zeros((k, k))
#     # psi2_coeff = np.zeros((k, k))

#     # psi1 = [[] for _ in range(k)]
#     # psi2 = [[] for _ in range(k)]

#     # for ki in range(k):
#     #     psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
#     #     for i in range(k):
#     #         proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
#     #         psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
#     #         psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

#     #     for j in range(ki):
#     #         proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
#     #         psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
#     #         psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

#     #     psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5)
#     #     psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5, ub = 1)

#     #     norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
#     #     norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

#     #     norm_ = np.sqrt(norm1 + norm2)
#     #     psi1_coeff[ki,:] /= norm_
#     #     psi2_coeff[ki,:] /= norm_
#     #     psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
#     #     psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

#     #     psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
#     #     psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
    
#     # return phi, psi1, psi2
# end

function legendre_filter(k)
    # x = Symbol('x')
    # H0 = np.zeros((k,k))
    # H1 = np.zeros((k,k))
    # G0 = np.zeros((k,k))
    # G1 = np.zeros((k,k))
    # PHI0 = np.zeros((k,k))
    # PHI1 = np.zeros((k,k))
    # phi, psi1, psi2 = get_phi_psi(k, base)

    # ----------------------------------------------------------

    # roots = Poly(legendre(k, 2*x-1)).all_roots()
    # x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
    # wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
    
    # for ki in range(k):
    #     for kpi in range(k):
    #         H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
    #         G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
    #         H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
    #         G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
            
    # PHI0 = np.eye(k)
    # PHI1 = np.eye(k)

    # ----------------------------------------------------------

    # H0[np.abs(H0)<1e-8] = 0
    # H1[np.abs(H1)<1e-8] = 0
    # G0[np.abs(G0)<1e-8] = 0
    # G1[np.abs(G1)<1e-8] = 0
        
    # return H0, H1, G0, G1, PHI0, PHI1
end

function chebyshev_filter(k)
    # x = Symbol('x')
    # H0 = np.zeros((k,k))
    # H1 = np.zeros((k,k))
    # G0 = np.zeros((k,k))
    # G1 = np.zeros((k,k))
    # PHI0 = np.zeros((k,k))
    # PHI1 = np.zeros((k,k))
    # phi, psi1, psi2 = get_phi_psi(k, base)

    # ----------------------------------------------------------

    # x = Symbol('x')
    # kUse = 2*k
    # roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
    # x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
    # # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
    # # not needed for our purpose here, we use even k always to avoid
    # wm = np.pi / kUse / 2

    # for ki in range(k):
    #     for kpi in range(k):
    #         H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
    #         G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
    #         H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
    #         G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

    #         PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
    #         PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
            
    # PHI0[np.abs(PHI0)<1e-8] = 0
    # PHI1[np.abs(PHI1)<1e-8] = 0

    # ----------------------------------------------------------

    # H0[np.abs(H0)<1e-8] = 0
    # H1[np.abs(H1)<1e-8] = 0
    # G0[np.abs(G0)<1e-8] = 0
    # G1[np.abs(G1)<1e-8] = 0
        
    # return H0, H1, G0, G1, PHI0, PHI1
end
