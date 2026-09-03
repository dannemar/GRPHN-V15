using GLMakie
using Statistics
using Printf
using LinearAlgebra

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function compute_matrix_pencil(x::Vector{Float64}, L::Int, r::Int, dt::Float64)
    N = length(x)
    K = N - L + 1

    # 1. Build Hankel Matrix
    H = zeros(Float64, L, K)
    for j in 1:K
        for i in 1:L
            H[i, j] = x[i + j - 1]
        end
    end

    # 2. SVD and Truncation
    F = svd(H)
    U_r = F.U[:, 1:r]

    # 3. Time-shifted matrices directly from U
    U1 = U_r[1:end-1, :]
    U2 = U_r[2:end, :]

    # 4. Extract discrete poles via Generalized Eigenvalue Problem
    z = eigvals(pinv(U1) * U2)

    # 5. Stability Enforcement
    # Autocorrelation cannot grow exponentially.
    # Force any |z| > 1 to lie on the unit circle (|z| = 1).
    # Prevent absolute zeros to avoid log(0) domain errors.
    for i in eachindex(z)
        mag = abs(z[i])
        if mag > 1.0
            z[i] = z[i] / mag
        elseif mag < 1e-12
            z[i] = 1e-12 + 0im
        end
    end

    # 6. Convert to continuous-time eigenvalues (poles)
    ω = log.(Complex.(z)) ./ dt

    # 7. Global Least-Squares Fit for Amplitudes
    t_vec = (0:N-1) .* dt
    M = zeros(ComplexF64, N, r)
    for j in 1:r
        for i in 1:N
            M[i, j] = exp(ω[j] * t_vec[i])
        end
    end

    # Solve M * b = x for an overdetermined system
    b = M \ Complex.(x)

    # 8. Evaluate the closed-form reconstruction
    x_rec = real.(M * b)

    return x_rec, ω, b
end

function analyze_vacf_prony()
    data_dir = "../../data/"
    lattice_configs = [(256, 256, 100)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 200
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    L = 661  # Pencil parameter / window length
    r = 97   # Cutoff rank for the series (now stable for larger r)

    # Figure Setup: 2x2 layout
    fig = Figure(size=(1600, 1200), fontsize=20)
    lwidth = 2.5

    ax_signal = Axis(fig[1, 1], xlabel="Time", ylabel="C_v(t)", title="Original vs Matrix Pencil Reconstruction")
    ax_error  = Axis(fig[1, 2], xlabel="Time", ylabel="Error", title="Reconstruction Error", yscale =log10)
    ax_eigs   = Axis(fig[2, 1], xlabel="Real(ω) [Decay/Growth Rate]", ylabel="Imag(ω) [Oscillation Frequency]", title="Continuous Poles (ω)")

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        e_str = @sprintf("%.2f", energy)
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        mean_cv, _, _, T_steps = process_observable(autocorr_cv_path, R)

        t_phy = (1:T_steps) .* dt
        mean_cv[end-10:end] = mean_cv[end-10:end] .* 0

        eq_cv = mean_cv[eq_idx:end]
        centered_cv = eq_cv .- mean(eq_cv)
        t_eq = t_phy[eq_idx:end]

        # Compute SVD-Prony (Matrix Pencil)
        reconstructed_cv, omega, amplitudes = compute_matrix_pencil(centered_cv, L, r, dt)

        # Calculate Error
        error_cv = centered_cv .- reconstructed_cv

        # Plot 1: Signal vs Reconstruction
        lines!(ax_signal, t_eq, centered_cv, color=(colors[1], 0.5), linewidth=lwidth + 2, label="Original")
        lines!(ax_signal, t_eq, reconstructed_cv, color=colors[2], linewidth=lwidth, label="Prony Form ($r modes)")
        axislegend(ax_signal, position=:rt)

        # Plot 2: Error
        lines!(ax_error, t_eq, abs.(error_cv), color=colors[3], linewidth=lwidth)

        # Plot 3: Continuous Poles
        scatter!(ax_eigs, real.(omega), imag.(omega), color=colors[4], markersize=12)
        vlines!(ax_eigs, 0.0, color=:black, linestyle=:dash)
    end

    save("../../img/VACF_Matrix_Pencil_Analysis.png", fig)
    display(fig)
end

analyze_vacf_prony()
