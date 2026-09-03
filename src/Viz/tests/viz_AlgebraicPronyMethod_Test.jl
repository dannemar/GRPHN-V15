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
# Algebraic Prony Method
# ==========================================
function algebraic_prony(data::Vector{Float64}, dt::Float64, p::Int)
    N = length(data)

    # Form the Y matrix and y vector (Linear difference equation)
    Y = zeros(Float64, N - p, p)
    y = zeros(Float64, N - p)

    for i in 1:(N - p)
        Y[i, :] = data[i+p-1 : -1 : i]
        y[i] = data[i+p]
    end

    # Solve linear least squares for polynomial coefficients
    c = Y \ (-y)

    # Construct characteristic polynomial and find roots
    coeffs = vcat(reverse(c), 1.0)
    poly = Polynomial(coeffs)
    roots_z = roots(poly)

    # Set up the Vandermonde matrix
    Z = zeros(ComplexF64, N, p)
    for n in 1:N
        for k in 1:p
            Z[n, k] = roots_z[k]^(n - 1)
        end
    end

    # Solve for complex amplitudes
    h = Z \ data

    # Extract physical parameters
    results = []
    for k in 1:p
        σ = log(abs(roots_z[k])) / dt
        ω = angle(roots_z[k]) / dt

        # Determine amplitude and phase
        if abs(ω) > 1e-5
            A = 2 * abs(h[k])
            ϕ = angle(h[k])
        else
            A = real(h[k])
            ϕ = 0.0
        end

        # Store physical parameters along with raw complex components for exact reconstruction
        push!(results, (A=A, σ=σ, ω=ω, ϕ=ϕ, z=roots_z[k], h=h[k]))
    end

    return results
end

# ==========================================
# Signal Reconstruction
# ==========================================
function reconstruct_prony(t_array::AbstractVector{Float64}, dt::Float64, prony_results)
    y_recon = zeros(Float64, length(t_array))
    for (i, t) in enumerate(t_array)
        val = 0.0 + 0.0im
        # Ensure n is handled strictly as an integer index
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
    data_dir = "../../data/"
    w, h, energy = 256, 256, 500
    s, tt, eqt_str, dt_str, R, of = 41, 600, "500.000", "0.010", 200, 1

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    strEnd = "_w-$(w)_h-$(h)_H-$(energy).00_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"
    file_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

    # Process data
    mean_crel, _, _, T_steps = process_observable(file_path, R)
    t_phy = (1:T_steps) .* dt

    # Isolate equilibrated data
    t_eq = t_phy[eq_idx:end]
    t_fit = t_eq .- t_eq[1]
    eq_crel = mean_crel[eq_idx:end]

    # 2. Fit the Data using Prony's Method
    # Order p=5 maps to 1 constant offset + 2 complex conjugate pairs (oscillations)
    p = 1500
    prony_params = algebraic_prony(eq_crel, dt, p)

    # 3. Print Extracted Physical Parameters
    println("--- Extracted Prony Parameters ---")
    for (i, prm) in enumerate(prony_params)
        # Filter negative frequencies for the printout to isolate the physical terms
        if prm.ω >= -1e-5
            if abs(prm.ω) < 1e-5
                println("Offset Mode:  Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5))")
            else
                println("Oscillation:  Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5)), Freq=$(round(prm.ω, digits=5)), Phase=$(round(prm.ϕ, digits=5))")
            end
        end
    end
    println("----------------------------------")

    # 4. Reconstruct the curve
    fit_crel = reconstruct_prony(t_fit, dt, prony_params)

    # 5. Figure Setup & Plotting
    fig = Figure(size=(1400, 1000), fontsize=20)
    ax = Axis(fig[1, 1],
              xlabel="Time (shifted to t=0)",
              ylabel="Bond Length C_B(t)",
              title="Algebraic Prony Fit vs Original Data")

    # Original Data
    lines!(ax, t_fit, eq_crel, color=:blue, linewidth=2.5, label="Original C_rel")

    # Reconstructed Model
    lines!(ax, t_fit, fit_crel, color=:red, linewidth=2.5, linestyle=:dash, label="Algebraic Prony Model (p=$p)")

    axislegend(ax, position=:rt)
    display(fig)
end

# Run the routine
main()
