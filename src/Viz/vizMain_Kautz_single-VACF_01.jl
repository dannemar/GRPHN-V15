#=
# Extract the VACF, compute internal FFT for initial peak guesses,
# fit the optimized Kautz function basis, plot the time-domain result,
# and print the primitive parameters for the closed-form equation.
=#

using GLMakie
using Statistics
using Printf
using FFTW
using LinearAlgebra
using LsqFit

# -----------------------------------------------------------------------------
# Kautz Basis Implementation & Fitting
# -----------------------------------------------------------------------------

# Generates the primitive damped sinusoids before orthogonalization
function build_kautz_primitives(N::Int, freqs::AbstractVector{T}, dampings::AbstractVector{T}) where T <: Real
    M = length(freqs)
    V = zeros(T, N, 2 * M)
    k = 0:(N - 1)

    for i in 1:M
        w = freqs[i]
        r = dampings[i]

        r_clamped = clamp(r, T(1e-6), T(0.999999))
        decay = r_clamped .^ k

        V[:, 2i - 1] = decay .* cos.(w .* k)
        V[:, 2i]     = decay .* sin.(w .* k)
    end

    return V
end

function kautz_model(k_dummy, p::AbstractVector{T}, y_target::Vector{Float64}) where T <: Real
    N = length(y_target)
    M = div(length(p), 2)
    freqs = p[1:M]
    dampings = p[M+1:end]

    # Generate primitives and orthogonalize strictly for stable projection
    V = build_kautz_primitives(N, freqs, dampings)
    F = qr(V)
    Q = Matrix(F.Q)

    c = Q' * y_target
    return Q * c
end

function fit_kautz(y::Vector{Float64}, initial_freqs::Vector{Float64}, initial_dampings::Vector{Float64})
    N = length(y)
    k_dummy = collect(1.0:N)
    p0 = vcat(initial_freqs, initial_dampings)
    M = length(initial_freqs)

    lb = vcat(fill(0.0, M), fill(0.01, M))
    ub = vcat(fill(pi, M), fill(0.9999, M))

    model(x, p) = kautz_model(x, p, y)
    fit = curve_fit(model, k_dummy, y, p0, lower=lb, upper=ub)

    best_p = fit.param
    best_freqs = best_p[1:M]
    best_dampings = best_p[M+1:end]

    # Construct final matrices
    V_final = build_kautz_primitives(N, best_freqs, best_dampings)
    F_final = qr(V_final)
    Q_final = Matrix(F_final.Q)

    # Projection onto orthogonal basis
    c_ortho = Q_final' * y
    y_fit = Q_final * c_ortho

    # Compute primitive coefficients: a = R \ c
    prim_coeffs = F_final.R \ c_ortho

    return y_fit, best_freqs, best_dampings, prim_coeffs
end

# -----------------------------------------------------------------------------
# Data Processing & Plotting
# -----------------------------------------------------------------------------

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function extract_all_peaks(magnitudes, frequencies)
    peaks = Tuple{Float64, Float64}[]
    for k in 2:(length(magnitudes)-1)
        if magnitudes[k] > magnitudes[k-1] && magnitudes[k] > magnitudes[k+1]
            push!(peaks, (frequencies[k], magnitudes[k]))
        end
    end
    sort!(peaks, by = x -> x[2], rev=true)
    return peaks
end

function plot_vacf_kautz()
    data_dir = "../../data/"
    lattice_configs = [(256,256,100)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R_val = 200
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    fig = Figure(size=(1200, 800), fontsize=23)
    lwidth = 2.5

    ax_auto_cv = Axis(fig[1, 1], xlabel="Time", ylabel="Velocity C_v(t)")
    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R_val)_of-$(of)"

        # Process only VACF
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        mean_cv, _, _, T_steps = process_observable(autocorr_cv_path, R_val)

        t_phy = (1:T_steps) .* dt
        mean_cv[end-10:end] .= 0.0

        eq_cv = mean_cv[eq_idx:end]
        cv_mean_val = mean(eq_cv)
        centered_cv = eq_cv .- cv_mean_val

        # Internal FFT for initial peak extraction
        cv_fft = fft(centered_cv)
        N_obs_len = length(eq_cv)
        fs = 1 / dt
        axis_samples = fftfreq(N_obs_len, fs) .* (2*pi)

        pos_idx = 1:div(N_obs_len, 2)
        freqs_pos = axis_samples[pos_idx]
        cv_fft_abs_pos = abs.(cv_fft)[pos_idx]

        freq_stop_cnt = 150
        cv_peaks = extract_all_peaks(cv_fft_abs_pos[1:freq_stop_cnt], freqs_pos[1:freq_stop_cnt])

        # Plot original time-domain VACF
        lines!(ax_auto_cv, t_phy[eq_idx:end], eq_cv, color=c, linewidth=lwidth, label="C_v ($lbl)")

        # Kautz Fitting and Parameter Extraction
        num_modes = min(3, length(cv_peaks))
        if num_modes > 0
            top_peaks = cv_peaks[1:num_modes]
            initial_w = [p[1] * dt for p in top_peaks]
            initial_r = fill(0.98, num_modes)

            fit_data, opt_w, opt_r, prim_coeffs = fit_kautz(centered_cv, initial_w, initial_r)
            fit_shifted = fit_data .+ cv_mean_val

            lines!(ax_auto_cv, t_phy[eq_idx:end], fit_shifted, linewidth=2.5,  color=:red, label="Kautz Fit")
            #
            # Print parameters to standard output
            println("\n=======================================================")
            println("VACF Kautz Model Parameters for: $lbl")
            println("-------------------------------------------------------")
            println("Closed-form approximation (where 'k' is the time step index):")
            println("C_v(k) = Σ [ A_i * (r_i)^k * cos(w_i * k) + B_i * (r_i)^k * sin(w_i * k) ] + Mean")
            println("\nMean Constant = $cv_mean_val")
            println("-------------------------------------------------------")

            for i in 1:num_modes
                @printf("Mode %d:\n", i)
                @printf("  Frequency (w_%d) = %.6f rad/sample\n", i, opt_w[i])
                @printf("  Damping   (r_%d) = %.6f\n", i, opt_r[i])
                @printf("  Cosine A  (A_%d) = %.6f\n", i, prim_coeffs[2i - 1])
                @printf("  Sine B    (B_%d) = %.6f\n\n", i, prim_coeffs[2i])
            end
            println("=======================================================\n")
        end
    end

    axislegend(ax_auto_cv, position=:rt)
    save("../../img/VACF_Kautz_Fit.png", fig)
    display(fig)
end

plot_vacf_kautz()
