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

    for ki in 0:(k - 1)
        l = convert(Polynomial, gen_poly(Legendre, ki))  # Legendre of n=ki
        ϕ_coefs[ki + 1, 1:(ki + 1)] .= sqrt(2 * ki + 1) .* coeffs(l(p))
        ϕ_2x_coefs[ki + 1, 1:(ki + 1)] .= sqrt(2 * (2 * ki + 1)) .* coeffs(l(p2))
    end

    ψ1_coefs = zeros(k, k)
    ψ2_coefs = zeros(k, k)
    for ki in 0:(k - 1)
        ψ1_coefs[ki + 1, :] .= ϕ_2x_coefs[ki + 1, :]
        for i in 0:(k - 1)
            a = ϕ_2x_coefs[ki + 1, 1:(ki + 1)]
            b = ϕ_coefs[i + 1, 1:(i + 1)]
            proj_ = proj_factor(a, b)
            ψ1_coefs[ki + 1, :] .-= proj_ .* view(ϕ_coefs, i + 1, :)
            ψ2_coefs[ki + 1, :] .-= proj_ .* view(ϕ_coefs, i + 1, :)
        end

        for j in 0:(k - 1)
            a = ϕ_2x_coefs[ki + 1, 1:(ki + 1)]
            b = ψ1_coefs[j + 1, :]
            proj_ = proj_factor(a, b)
            ψ1_coefs[ki + 1, :] .-= proj_ .* view(ψ1_coefs, j + 1, :)
            ψ2_coefs[ki + 1, :] .-= proj_ .* view(ψ2_coefs, j + 1, :)
        end

        a = ψ1_coefs[ki + 1, :]
        norm1 = proj_factor(a, a)

        a = ψ2_coefs[ki + 1, :]
        norm2 = proj_factor(a, a, complement = true)
        norm_ = sqrt(norm1 + norm2)
        ψ1_coefs[ki + 1, :] ./= norm_
        ψ2_coefs[ki + 1, :] ./= norm_
        zero_out!(ψ1_coefs)
        zero_out!(ψ2_coefs)
    end

    ϕ = [Polynomial(ϕ_coefs[i, :]) for i in 1:k]
    ψ1 = [Polynomial(ψ1_coefs[i, :]) for i in 1:k]
    ψ2 = [Polynomial(ψ2_coefs[i, :]) for i in 1:k]

    return ϕ, ψ1, ψ2
end

function chebyshev_ϕ_ψ(k)
    ϕ_coefs = zeros(k, k)
    ϕ_2x_coefs = zeros(k, k)

    p = Polynomial([-1, 2])  # 2x-1
    p2 = Polynomial([-1, 4])  # 4x-1

    for ki in 0:(k - 1)
        if ki == 0
            ϕ_coefs[ki + 1, 1:(ki + 1)] .= sqrt(2 / π)
            ϕ_2x_coefs[ki + 1, 1:(ki + 1)] .= sqrt(4 / π)
        else
            c = convert(Polynomial, gen_poly(Chebyshev, ki))  # Chebyshev of n=ki
            ϕ_coefs[ki + 1, 1:(ki + 1)] .= 2 / sqrt(π) .* coeffs(c(p))
            ϕ_2x_coefs[ki + 1, 1:(ki + 1)] .= sqrt(2) * 2 / sqrt(π) .* coeffs(c(p2))
        end
    end

    ϕ = [ϕ_(ϕ_coefs[i, :]) for i in 1:k]

    k_use = 2k
    c = convert(Polynomial, gen_poly(Chebyshev, k_use))
    x_m = roots(c(p))
    # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
    # not needed for our purpose here, we use even k always to avoid
    wm = π / k_use / 2

    ψ1_coefs = zeros(k, k)
    ψ2_coefs = zeros(k, k)

    ψ1 = Array{Any, 1}(undef, k)
    ψ2 = Array{Any, 1}(undef, k)

    for ki in 0:(k - 1)
        ψ1_coefs[ki + 1, :] .= ϕ_2x_coefs[ki + 1, :]
        for i in 0:(k - 1)
            proj_ = sum(wm .* ϕ[i + 1].(x_m) .* sqrt(2) .* ϕ[ki + 1].(2 * x_m))
            ψ1_coefs[ki + 1, :] .-= proj_ .* view(ϕ_coefs, i + 1, :)
            ψ2_coefs[ki + 1, :] .-= proj_ .* view(ϕ_coefs, i + 1, :)
        end

        for j in 0:(ki - 1)
            proj_ = sum(wm .* ψ1[j + 1].(x_m) .* sqrt(2) .* ϕ[ki + 1].(2 * x_m))
            ψ1_coefs[ki + 1, :] .-= proj_ .* view(ψ1_coefs, j + 1, :)
            ψ2_coefs[ki + 1, :] .-= proj_ .* view(ψ2_coefs, j + 1, :)
        end

        ψ1[ki + 1] = ϕ_(ψ1_coefs[ki + 1, :]; lb = 0.0, ub = 0.5)
        ψ2[ki + 1] = ϕ_(ψ2_coefs[ki + 1, :]; lb = 0.5, ub = 1.0)

        norm1 = sum(wm .* ψ1[ki + 1].(x_m) .* ψ1[ki + 1].(x_m))
        norm2 = sum(wm .* ψ2[ki + 1].(x_m) .* ψ2[ki + 1].(x_m))

        norm_ = sqrt(norm1 + norm2)
        ψ1_coefs[ki + 1, :] ./= norm_
        ψ2_coefs[ki + 1, :] ./= norm_
        zero_out!(ψ1_coefs)
        zero_out!(ψ2_coefs)

        ψ1[ki + 1] = ϕ_(ψ1_coefs[ki + 1, :]; lb = 0.0, ub = 0.5 + 1e-16)
        ψ2[ki + 1] = ϕ_(ψ2_coefs[ki + 1, :]; lb = 0.5 + 1e-16, ub = 1.0)
    end

    return ϕ, ψ1, ψ2
end

function legendre_filter(k)
    H0 = zeros(k, k)
    H1 = zeros(k, k)
    G0 = zeros(k, k)
    G1 = zeros(k, k)
    ϕ, ψ1, ψ2 = legendre_ϕ_ψ(k)

    l = convert(Polynomial, gen_poly(Legendre, k))
    x_m = roots(l(Polynomial([-1, 2])))  # 2x-1
    m = 2 .* x_m .- 1
    wm = 1 ./ k ./ legendre_der.(k, m) ./ gen_poly(Legendre, k - 1).(m)

    for ki in 0:(k - 1)
        for kpi in 0:(k - 1)
            H0[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ϕ[ki + 1].(x_m / 2) .* ϕ[kpi + 1].(x_m))
            G0[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ψ(ψ1, ψ2, ki, x_m / 2) .* ϕ[kpi + 1].(x_m))
            H1[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ϕ[ki + 1].((x_m .+ 1) / 2) .* ϕ[kpi + 1].(x_m))
            G1[ki + 1, kpi + 1] = 1 / sqrt(2) * sum(wm .* ψ(ψ1, ψ2, ki, (x_m .+ 1) / 2) .*
                                      ϕ[kpi + 1].(x_m))
        end
    end

    zero_out!(H0)
    zero_out!(H1)
    zero_out!(G0)
    zero_out!(G1)

    return H0, H1, G0, G1, I(k), I(k)
end

function chebyshev_filter(k)
    H0 = zeros(k, k)
    H1 = zeros(k, k)
    G0 = zeros(k, k)
    G1 = zeros(k, k)
    Φ0 = zeros(k, k)
    Φ1 = zeros(k, k)
    ϕ, ψ1, ψ2 = chebyshev_ϕ_ψ(k)

    k_use = 2k
    c = convert(Polynomial, gen_poly(Chebyshev, k_use))
    x_m = roots(c(Polynomial([-1, 2])))  # 2x-1
    # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
    # not needed for our purpose here, we use even k always to avoid
    wm = π / k_use / 2

    for ki in 0:(k - 1)
        for kpi in 0:(k - 1)
            H0[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ϕ[ki + 1].(x_m / 2) .* ϕ[kpi + 1].(x_m))
            H1[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ϕ[ki + 1].((x_m .+ 1) / 2) .* ϕ[kpi + 1].(x_m))
            G0[ki + 1, kpi + 1] = 1 / sqrt(2) *
                                  sum(wm .* ψ(ψ1, ψ2, ki, x_m / 2) .* ϕ[kpi + 1].(x_m))
            G1[ki + 1, kpi + 1] = 1 / sqrt(2) * sum(wm .* ψ(ψ1, ψ2, ki, (x_m .+ 1) / 2) .*
                                      ϕ[kpi + 1].(x_m))
            Φ0[ki + 1, kpi + 1] = 2 * sum(wm .* ϕ[ki + 1].(2x_m) .* ϕ[kpi + 1].(2x_m))
            Φ1[ki + 1, kpi + 1] = 2 * sum(wm .* ϕ[ki + 1].(2 .* x_m .- 1) .*
                                      ϕ[kpi + 1].(2 .* x_m .- 1))
        end
    end

    zero_out!(H0)
    zero_out!(H1)
    zero_out!(G0)
    zero_out!(G1)
    zero_out!(Φ0)
    zero_out!(Φ1)

    return H0, H1, G0, G1, Φ0, Φ1
end
