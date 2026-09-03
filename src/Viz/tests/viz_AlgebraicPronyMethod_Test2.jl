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
# Dominant Mode Extraction
# ==========================================
function extract_dominant_modes(prony_params; max_modes::Int=10)
    valid_modes = []
    for p in prony_params
        # Filter out unstable or growing modes
        if p.σ < 0.0
            # Calculate energy contribution of the mode
            energy = -(p.A^2) / (2 * p.σ)
            # Store mode with calculated energy
            push!(valid_modes, (A=p.A, σ=p.σ, ω=p.ω, ϕ=p.ϕ, z=p.z, h=p.h, Energy=energy))
        end
    end

    # Sort descending by Energy
    sort!(valid_modes, by = x -> x.Energy, rev=true)

    # Truncate to max_modes
    return valid_modes[1:min(length(valid_modes), max_modes)]
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
    data_dir = "../../../data/"
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
    p = 1000
    prony_params = algebraic_prony(eq_crel, dt, p)

    # 3. Extract the Dominant Modes
    num_dominant = 23
    dominant_modes = extract_dominant_modes(prony_params, max_modes=num_dominant)

    # 4. Print Extracted Physical Parameters
    println("--- Top $num_dominant Dominant Prony Parameters ---")
    for (i, prm) in enumerate(dominant_modes)
        # Filter negative frequencies for the printout
        if prm.ω >= -1e-5
            if abs(prm.ω) < 1e-5
                println("Mode $i (Offset):      Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5)), Energy=$(round(prm.Energy, digits=5))")
            else
                println("Mode $i (Oscillation): Amp=$(round(prm.A, digits=5)), Decay=$(round(prm.σ, digits=5)), Freq=$(round(prm.ω, digits=5)), Phase=$(round(prm.ϕ, digits=5)), Energy=$(round(prm.Energy, digits=5))")
            end
        end
    end
    println("----------------------------------")

    # 5. Reconstruct the curve using ONLY dominant modes
    fit_crel = reconstruct_prony(t_fit, dt, dominant_modes)

    # 6. Figure Setup & Plotting
    fig = Figure(size=(900, 600), fontsize=28)
    ax = Axis(fig[1, 1],
              xlabel=L"\text{Time (shifted to }t=0)",
              ylabel=L" C_B(t)",
              #title="Dominant Prony Modes vs Original Data")
              )

    # Original Data
    lines!(ax, t_fit, eq_crel,  linewidth=2.5, label=L"\text{Original} C_{B}")

    # Reconstructed Model
    lines!(ax, t_fit, fit_crel,  linewidth=2.5, label=L"\text{Fitted Result}")

    axislegend(ax, position=:rt)
    save("../../../img/PronyFit_p-$(p)_modes-$(num_dominant).png", fig)
    display(fig)
end

# Run the routine
main()
