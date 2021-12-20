# function ϕ_(ϕ_coefs; lb::Real=0., ub::Real=1.)
#     mask = 
#     return Polynomial(ϕ_coefs)
# end

# def phi_(phi_c, x, lb = 0, ub = 1):
# mask = np.logical_or(x<lb, x>ub) * 1.0
# return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

function ψ(ψ1, ψ2, i, inp)
    mask = (inp ≤ 0.5) * 1.0
    return ψ1[i](inp) * mask + ψ2[i](inp) * (1-mask)
end

zero_out!(x; tol=1e-8) = (x[abs.(x) .< tol] .= 0)

function gen_poly(poly, n)
    x = zeros(n+1)
    x[end] = 1
    return poly(x)
end

function convolve(a, b)
    n = length(b)
    y = similar(a, length(a)+n-1)
    for i in 1:length(a)
        y[i:(i+n-1)] .+= a[i] .* b
    end
    return y
end

function proj_factor(a, b; complement::Bool=false)
    prod_ = convolve(a, b)
    zero_out!(prod_)
    r = collect(1:length(prod_))
    s = complement ? (1 .- 0.5 .^ r) : (0.5 .^ r)
    proj_ = sum(prod_ ./ r .* s)
    return proj_
end
