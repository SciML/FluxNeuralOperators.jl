function grf(N, m, γ, τ, σ, type)
    (type == "dirichlet") && (m = 0)
    (type == "periodic") && (my_const = 2π) || (my_const = π)
    my_eigs = sqrt(2)*(abs(σ).*((my_const.*(1:N)').^2 + τ^2).^(-γ/2));

    (type == "dirichlet") && (α = zeros(N, 1)) || (α = my_eigs .* randn(N, 1))

    (type == "neumann") && (β = zeros(N, 1)) || (β = my_eigs .* randn(N, 1))

    a = α / 2
    b = -β / 2;

    #=
    c = [
        [a[end:-1:1]...] - im*[b[end:-1:1]...] ;
        m + 0im ;
        a + b*im
    ]

    (type == "periodic") &&
        (return chebfun(t -> chebfun(c, [0 1], 'trig', 'coeffs')(t - 0.5), [0 1], 'trig')) ||
        (return chebfun(t -> chebfun(c, [-pi pi], 'trig', 'coeffs')(pi*t), [0 1]))
    =#
end

function burgers(init, tspan, s, viscosity)
    #=
    S = spinop([0 1], tspan);
    dt = tspan[2] - tspan[1];
    S.lin = @(u) + visc*diff(u,2);
    S.nonlin = @(u) - 0.5*diff(u.^2);
    S.init = init;
    u = spin(S,s,dt,'plot','off');
    =#
end

function gen_burgers(; n=1, γ=2.5, τ=7,  σ=7^2, viscosity=1e-3, grid_size=1024, η=200)
    input = zeros(N, s);
    if steps == 1
        output = zeros(N, s);
    else
        output = zeros(N, steps, s);
    end

    tspan = LinRange(0, 1, η+1)
    x = LinRange(0, 1, grid_size+1)

    #=
    tspan = linspace(0,1,steps+1);
    x = linspace(0,1,s+1);
    for j=1:N
        u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
        u = burgers1(u0, tspan, s, visc);

        u0eval = u0(x);
        input(j,:) = u0eval(1:end-1);

        if steps == 1
            output(j,:) = u.values;
        else
            for k=2:(steps+1)
                output(j,k,:) = u{k}.values;
            end
        end

        disp(j);
    end
    =#
end
