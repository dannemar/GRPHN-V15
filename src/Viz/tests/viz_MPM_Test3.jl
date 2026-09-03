using GLMakie
using Statistics
using LinearAlgebra
using Polynomials

# ==========================================
# Data Processing Function
# ==========================================
function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

# ==========================================
# Matrix Pencil Method (MPM)
# ==========================================
function matrix_pencil_method(data::Vector{Float64}, dt::Float64, M::Int; L::Int=div(length(data), 3))
    N = length(data)

    # 1. Form the Hankel Matrix (Data Windowing)
    Y = zeros(Float64, N - L, L + 1)
    for i in 1:(N - L)
        Y[i, :] = data[i : i + L]
    end

    # 2. Singular Value Decomposition
    F = svd(Y)
    S_vals = F.S # Capture singular values

    # 3. Truncate to M physical modes
    V_M = F.V[:, 1:M]

    # 4. Form time-shifted subspace matrices
    V1 = V_M[1:end-1, :]
    V2 = V_M[2:end, :]

    # 5. Calculate roots (poles) via eigenvalues of the sub-matrix
    Z_mat = pinv(V1) * V2
    roots_z = eigvals(Z_mat)

    # 6. Solve for complex amplitudes using a constrained Vandermonde matrix
    Z = zeros(ComplexF64, N, M)
    for n in 1:N
        for k in 1:M
            Z[n, k] = roots_z[k]^(n - 1)
        end
    end

    h = Z \ complex.(data)

    # 7. Extract physical parameters
    results = []
    for k in 1:M
        σ = log(abs(roots_z[k])) / dt
        ω = angle(roots_z[k]) / dt

        if abs(ω) > 1e-5
            A = 2 * abs(h[k])
            ϕ = angle(h[k])
        else
            A = real(h[k])
            ϕ = 0.0
        end

        push!(results, (A=A, σ=σ, ω=ω, ϕ=ϕ, z=roots_z[k], h=h[k]))
    end

    return results, S_vals
end

# ==========================================
# Signal Reconstruction
# ==========================================
function reconstruct_prony(t_array::AbstractVector{Float64}, dt::Float64, prony_results)
    y_recon = zeros(Float64, length(t_array))
    for (i, t) in enumerate(t_array)
        val = 0.0 + 0.0im
        n = round(Int, t / dt)
        for res in prony_results
            val += res.h * (res.z ^ n)
        end
        y_recon[i] = real(val)
    end
    return y_recon
end

# ==========================================
# Main Execution Routine
# ==========================================
function main()
    # 1. Configuration and Data Loading
    data_dir = "../../../data/"
    w, h, energy = 256, 256, 500
    s, tt, eqt_str, dt_str, R, of = 41, 600, "500.000", "0.010", 200, 1

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    strEnd = "_w-$(w)_h-$(h)_H-$(energy).00_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"
    file_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

    mean_crel, _, _, T_steps = process_observable(file_path, R)
    t_phy = (1:T_steps) .* dt

    t_eq = t_phy[eq_idx:end]
    t_fit = t_eq .- t_eq[1]
    eq_crel = mean_crel[eq_idx:end]

    # 2. Fit the Data using the Matrix Pencil Method
    M = 20
    mpm_params, S_vals = matrix_pencil_method(eq_crel, dt, M)

    # 3. Print Extracted Physical Parameters
    println("--- Extracted MPM Parameters ---")
    for (i, prm) in enumerate(mpm_params)
        if prm.ω >= -1e-5
            if abs(prm.ω) < 1e-5
                println("Offset Mode:  Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5))")
            else
                println("Oscillation:  Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5)), Freq=$(round(prm.ω, digits=5)), Phase=$(round(prm.ϕ, digits=5))")
            end
        end
    end
    println("----------------------------------")

    # 4. Reconstruct the curve and calculate error
    fit_crel = reconstruct_prony(t_fit, dt, mpm_params)
    abs_error = abs.(eq_crel .- fit_crel)

    # 5. Figure Setup & Plotting
    fig = Figure(size=(1400, 1400), fontsize=20)

    # Top Axis: Original vs Reconstruction
    ax1 = Axis(fig[1, 1],
              xlabel="Time",
              ylabel="Bond Length C_B(t)",
              title="MPM Fit vs Original Data")
    lines!(ax1, t_fit, eq_crel, color=:blue, linewidth=2.5, label="Original C_rel")
    lines!(ax1, t_fit, fit_crel, color=:red, linewidth=2.5, linestyle=:dash, label="MPM Model (M=$M)")
    axislegend(ax1, position=:rt)

    # Middle Axis: Error Plot
    ax2 = Axis(fig[2, 1],
              xlabel="Time",
              ylabel="Absolute Error",
              yscale=log10)
    lines!(ax2, t_fit, abs_error, color=:black, linewidth=2.0, label="|Original - Fit|")
    axislegend(ax2, position=:rt)

    # Bottom Axis: Singular Value Spectrum
    ax3 = Axis(fig[3, 1],
              xlabel="Singular Value Index",
              ylabel="Magnitude",
              yscale=log10,
              title="SVD Spectrum (Identify Noise Floor)")

    for i in 1:length(S_vals)
        println("$(i):\t $(S_vals[i])")
    end

    scatterlines!(ax3, 1:length(S_vals), S_vals, color=:purple, markersize=8, linewidth=1.5)

    linkxaxes!(ax1, ax2)
    display(fig)
end

main()
