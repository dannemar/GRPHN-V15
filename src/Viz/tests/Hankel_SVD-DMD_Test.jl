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

function create_hankel_matrix(x::Vector{Float64}, L::Int)
    N = length(x)
    K = N - L + 1
    H = zeros(Float64, L, K)
    for j in 1:K
        for i in 1:L
            H[i, j] = x[i + j - 1]
        end
    end
    return H
end

function compute_dmd_1d(H::Matrix{Float64}, r::Int, dt::Float64, N_total::Int)
    # 1. Create time-shifted matrices
    X1 = H[:, 1:end-1]
    X2 = H[:, 2:end]

    # 2. SVD of X1 and Truncation
    F = svd(X1)
    U_r = F.U[:, 1:r]
    S_r = Diagonal(F.S[1:r])
    V_r = F.V[:, 1:r]

    # 3. Compute reduced linear operator A_tilde
    A_tilde = U_r' * X2 * V_r * inv(S_r)

    # 4. Eigendecomposition of A_tilde
    eig = eigen(A_tilde)
    Λ = eig.values
    W = eig.vectors

    # 5. Compute exact DMD modes and continuous-time eigenvalues
    Φ = X2 * V_r * inv(S_r) * W
    ω = log.(Complex.(Λ)) ./ dt

    # 6. Compute mode amplitudes b from the initial state
    x1 = H[:, 1]
    b = pinv(Φ) * x1

    # 7. Evaluate the closed-form continuous analytical expression
    t_vec = (0:N_total-1) .* dt
    x_dmd = zeros(ComplexF64, N_total)

    for (idx, t) in enumerate(t_vec)
        val = zero(ComplexF64)
        for i in 1:r
            # First row of Φ corresponds to the 1D signal space
            val += b[i] * Φ[1, i] * exp(ω[i] * t)
        end
        x_dmd[idx] = val
    end

    # Return real part of reconstruction, eigenvalues, and amplitudes
    return real.(x_dmd), ω, b
end

function analyze_vacf_dmd()
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

    L = 200  # Hankel window length
    r = 20   # Cutoff rank for DMD approximation.

    # Figure Setup: 2x2 layout
    fig = Figure(size=(1600, 1200), fontsize=20)
    lwidth = 2.5

    ax_signal = Axis(fig[1, 1], xlabel="Time", ylabel="C_v(t)", title="Original vs Continuous DMD Reconstruction")
    ax_error  = Axis(fig[1, 2], xlabel="Time", ylabel="Error", title="Reconstruction Error")
    ax_eigs   = Axis(fig[2, 1], xlabel="Real(ω) [Decay/Growth Rate]", ylabel="Imag(ω) [Oscillation Frequency]", title="Continuous Eigenvalues (ω)")

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
        N_eq = length(centered_cv)

        # Build Hankel Matrix
        H = create_hankel_matrix(centered_cv, L)

        # Compute DMD and Reconstruct
        reconstructed_cv, omega, amplitudes = compute_dmd_1d(H, r, dt, N_eq)

        # Calculate Error
        error_cv = centered_cv .- reconstructed_cv

        # Plot 1: Signal vs Reconstruction
        lines!(ax_signal, t_eq[1:1000], centered_cv[1:1000], color=(colors[1], 0.5), linewidth=lwidth + 2, label="Original")
        lines!(ax_signal, t_eq[1:1000], reconstructed_cv[1:1000], color=colors[2], linewidth=lwidth, label="DMD Form ($r modes)")
        axislegend(ax_signal, position=:rt)

        # Plot 2: Error
        lines!(ax_error, t_eq, error_cv, color=colors[3], linewidth=lwidth)

        # Plot 3: Continuous Eigenvalues
        # Negative Real(ω) denotes decay; Positive denotes growth. Imag(ω) denotes frequency.
        scatter!(ax_eigs, real.(omega), imag.(omega), color=colors[4], markersize=12)

        # Add a vertical line at 0 to separate stable (decaying) from unstable (growing) modes
        vlines!(ax_eigs, 0.0, color=:black, linestyle=:dash)
    end

    save("../../img/VACF_Hankel_DMD_Analysis.png", fig)
    display(fig)
end

analyze_vacf_dmd()
