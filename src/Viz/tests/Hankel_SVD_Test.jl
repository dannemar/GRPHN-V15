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

function diagonal_averaging(H_tilde::Matrix{Float64})
    L, K = size(H_tilde)
    N = L + K - 1
    x_rec = zeros(Float64, N)
    counts = zeros(Int, N)

    for j in 1:K
        for i in 1:L
            idx = i + j - 1
            x_rec[idx] += H_tilde[i, j]
            counts[idx] += 1
        end
    end

    # Divide by the number of elements on each skew-diagonal to get the mean
    return x_rec ./ counts
end

function analyze_vacf_svd()
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

    L = 100
    r = 16 # Number of modes used for reconstruction

    # Figure Setup: 3x2 layout
    fig = Figure(size=(1600, 1800), fontsize=20)
    lwidth = 2.5

    ax_signal = Axis(fig[1, 1], xlabel="Time", ylabel="Equilibrated C_v(t)", title="Original Signal")
    ax_scree  = Axis(fig[1, 2], xlabel="Index", ylabel="Log(Singular Value)", title="Scree Plot", yscale=log10)
    ax_u_vecs = Axis(fig[2, 1], xlabel="Window Index", ylabel="Amplitude", title="Top 3 Left Singular Vectors (U)")
    ax_v_vecs = Axis(fig[2, 2], xlabel="Sequence Index", ylabel="Amplitude", title="Top 3 Right Singular Vectors (V)")
    ax_recon  = Axis(fig[3, 1], xlabel="Time", ylabel="Approximation", title="Reconstructed Signal (Top $r Modes)")
    ax_error  = Axis(fig[3, 2], xlabel="Time", ylabel="Error", title="Reconstruction Error (Original - Approx)", yscale=log10)

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

        # 1. Build Hankel Matrix
        H = create_hankel_matrix(centered_cv, L)

        # 2. Perform SVD
        F = svd(H)

        # 3. Truncation and Matrix Reconstruction
        H_tilde = zeros(Float64, size(H))
        for i in 1:r
            # Add rank-1 matrix for each mode: σ_i * u_i * v_i^T
            H_tilde .+= F.S[i] .* (F.U[:, i] * F.V[:, i]')
        end

        # 4. Hankelization (Diagonal Averaging)
        reconstructed_cv = diagonal_averaging(H_tilde)

        # 5. Error Calculation
        error_cv = centered_cv .- reconstructed_cv

        # Plot 1: Original Signal
        lines!(ax_signal, t_eq, centered_cv, color=colors[1], linewidth=lwidth)

        # Plot 2: Scree Plot
        scatterlines!(ax_scree, 1:length(F.S), F.S, color=colors[2], markersize=8, linewidth=2)

        # Plot 3: Top 3 Left Singular Vectors (U)
        lines!(ax_u_vecs, F.U[:, 1], color=colors[1], linewidth=lwidth, label="Mode 1")
        lines!(ax_u_vecs, F.U[:, 2], color=colors[2], linewidth=lwidth, label="Mode 2")
        lines!(ax_u_vecs, F.U[:, 3], color=colors[3], linewidth=lwidth, label="Mode 3")
        axislegend(ax_u_vecs, position=:rt)

        # Plot 4: Top 3 Right Singular Vectors (V)
        lines!(ax_v_vecs, F.V[:, 1], color=colors[1], linewidth=lwidth, label="Mode 1")
        lines!(ax_v_vecs, F.V[:, 2], color=colors[2], linewidth=lwidth, label="Mode 2")
        lines!(ax_v_vecs, F.V[:, 3], color=colors[3], linewidth=lwidth, label="Mode 3")
        axislegend(ax_v_vecs, position=:rt)

        # Plot 5: Reconstructed Signal
        lines!(ax_recon, t_eq, reconstructed_cv, color=colors[4], linewidth=lwidth)

        # Plot 6: Error
        lines!(ax_error, t_eq, abs.(error_cv), color=colors[5], linewidth=lwidth)
    end

    save("../../img/VACF_Hankel_SVD_Reconstruction.png", fig)
    display(fig)
end

analyze_vacf_svd()
