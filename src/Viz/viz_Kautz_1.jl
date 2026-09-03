#=
# Compute and plot the FFT of each ACF
# Extract the peaks by seeing if a 'point's neighbours switch slopes
# plot identified peaks with frequency and amplitude annotations
# Fit and model the ACF using an optimized Kautz function basis
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
# 1. Update matrix builder to accept generic Real types and initialize V with type T
function build_kautz_matrix(N::Int, freqs::AbstractVector{T}, dampings::AbstractVector{T}) where T <: Real
    M = length(freqs)
    # Ensure the matrix V matches the input type T to hold ForwardDiff.Dual numbers
    V = zeros(T, N, 2 * M)
    k = 0:(N - 1)

    for i in 1:M
        w = freqs[i]
        r = dampings[i]

        # Clamp r using the generic type T to maintain Dual number compatibility
        r_clamped = clamp(r, T(1e-6), T(0.999999))
        decay = r_clamped .^ k

        V[:, 2i - 1] = decay .* cos.(w .* k)
        V[:, 2i]     = decay .* sin.(w .* k)
    end

    # Orthogonalize the primitive vectors via QR decomposition
    F = qr(V)
    return Matrix(F.Q)
end

# 2. Update model parameter vector to accept generic Real types
function kautz_model(k_dummy, p::AbstractVector{T}, y_target::Vector{Float64}) where T <: Real
    N = length(y_target)
    M = div(length(p), 2)
    freqs = p[1:M]
    dampings = p[M+1:end]

    Q = build_kautz_matrix(N, freqs, dampings)
    c = Q' * y_target
    return Q * c
end

function fit_kautz(y::Vector{Float64}, initial_freqs::Vector{Float64}, initial_dampings::Vector{Float64})
    N = length(y)
    k_dummy = collect(1.0:N)
    p0 = vcat(initial_freqs, initial_dampings)
    M = length(initial_freqs)

    # Set boundaries: frequencies between 0 and pi, damping strictly between 0 and 1
    lb = vcat(fill(0.0, M), fill(0.01, M))
    ub = vcat(fill(pi, M), fill(0.9999, M))

    model(x, p) = kautz_model(x, p, y)
    fit = curve_fit(model, k_dummy, y, p0, lower=lb, upper=ub)

    best_p = fit.param
    best_freqs = best_p[1:M]
    best_dampings = best_p[M+1:end]

    Q_final = build_kautz_matrix(N, best_freqs, best_dampings)
    c_final = Q_final' * y
    y_fit = Q_final * c_final

    return y_fit, best_freqs, best_dampings, c_final
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

function plot_all_observables()
    data_dir = "../../data/"
    lattice_configs = [(256,256,100)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 200
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    int_scheme_selected = "ABA864"

    fig = Figure(size=(2400, 2600), fontsize=23)
    lwidth = 2.5

    println("Equilibration Calc: $(eq_idx)")

    ax_auto_ce = Axis(fig[1, 1], xlabel="Time", ylabel="Energy C_E(t)")
    ax_auto_cd = Axis(fig[2, 1], xlabel="Time", ylabel="Absolute C_D(t)")
    ax_auto_cv = Axis(fig[3, 1], xlabel="Time", ylabel="Velocity C_v(t)")
    ax_auto_crel = Axis(fig[4, 1], xlabel="Time", ylabel="Bond Length C_B(t)")

    ax_ce_fft_peaks = Axis(fig[1, 2], xlabel=L"\omega", ylabel=L"|FT(C_E)|", xlabelsize = 42)
    ax_cd_fft_peaks = Axis(fig[2, 2], xlabel=L"\omega", ylabel=L"|FT(C_D)|", xlabelsize = 42)
    ax_cv_fft_peaks = Axis(fig[3, 2], xlabel=L"\omega", ylabel=L"|FT(C_v)|", xlabelsize = 42)
    ax_crel_fft_peaks = Axis(fig[4, 2], xlabel=L"\omega", ylabel=L"|FT(C_B)|", xlabelsize = 42)

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        autocorr_cd_path = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        autocorr_ce_path = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

        mean_cd, _, _, T_steps = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)
        mean_ce, _, _, _ = process_observable(autocorr_ce_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)

        t_phy = (1:T_steps) .* dt

        mean_cd[end-10:end] = mean_cd[end-10:end] .* 0
        mean_cv[end-10:end] = mean_cv[end-10:end] .* 0
        mean_ce[end-10:end] = mean_ce[end-10:end] .* 0
        mean_crel[end-10:end] = mean_crel[end-10:end] .* 0

        eq_cd = mean_cd[eq_idx:end]
        eq_cv = mean_cv[eq_idx:end]
        eq_ce = mean_ce[eq_idx:end]
        eq_crel = mean_crel[eq_idx:end]

        centered_cd = eq_cd .- mean(eq_cd)
        centered_cv = eq_cv .- mean(eq_cv)
        centered_ce = eq_ce .- mean(eq_ce)
        centered_crel = eq_crel .- mean(eq_crel)

        cd_fft = fft(centered_cd)
        cv_fft = fft(centered_cv)
        ce_fft = fft(centered_ce)
        crel_fft = fft(centered_crel)

        N_obs_len = length(eq_cd)
        fs = 1 / dt
        axis_samples = fftfreq(N_obs_len, fs) .* (2*pi)

        pos_idx = 1:div(N_obs_len, 2)
        freqs_pos = axis_samples[pos_idx]

        cd_fft_abs_pos = abs.(cd_fft)[pos_idx]
        cv_fft_abs_pos = abs.(cv_fft)[pos_idx]
        ce_fft_abs_pos = abs.(ce_fft)[pos_idx]
        crel_fft_abs_pos = abs.(crel_fft)[pos_idx]

        freq_stop_cnt = 150

        ce_peaks = extract_all_peaks(ce_fft_abs_pos[1:freq_stop_cnt], freqs_pos[1:freq_stop_cnt])
        cd_peaks = extract_all_peaks(cd_fft_abs_pos[1:freq_stop_cnt], freqs_pos[1:freq_stop_cnt])
        cv_peaks = extract_all_peaks(cv_fft_abs_pos[1:freq_stop_cnt], freqs_pos[1:freq_stop_cnt])
        crel_peaks = extract_all_peaks(crel_fft_abs_pos[1:freq_stop_cnt], freqs_pos[1:freq_stop_cnt])

        # Plot Column 1: Time Domain Data
        lines!(ax_auto_ce, t_phy[eq_idx:end], eq_ce, color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_auto_cd, t_phy[eq_idx:end], eq_cd, color=c, linewidth=lwidth + 2.5, label="C_D ($lbl)")
        lines!(ax_auto_cv, t_phy[eq_idx:end], eq_cv, color=c, linewidth=lwidth, label="C_v ($lbl)")
        lines!(ax_auto_crel, t_phy[eq_idx:end], eq_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)")

        # Function to apply Kautz fitting and plot the overlay
        function apply_kautz_and_plot(ax, peaks, centered_data, original_data, label_name)
            num_modes = min(3, length(peaks))
            if num_modes > 0
                top_peaks = peaks[1:num_modes]
                # Convert continuous angular frequency to discrete angular frequency
                initial_w = [p[1] * dt for p in top_peaks]
                initial_r = fill(0.98, num_modes)

                fit_data, _, _, _ = fit_kautz(centered_data, initial_w, initial_r)

                # Re-add the mean to match the uncentered observable
                fit_shifted = fit_data .+ mean(original_data)

                lines!(ax, t_phy[eq_idx:end], fit_shifted, color=:black, linewidth=2.0, linestyle=:dash, label="Kautz Fit ($label_name)")
            end
        end

        # Execute fitting for all available observables
        apply_kautz_and_plot(ax_auto_ce, ce_peaks, centered_ce, eq_ce, "C_E")
        apply_kautz_and_plot(ax_auto_cd, cd_peaks, centered_cd, eq_cd, "C_D")
        apply_kautz_and_plot(ax_auto_cv, cv_peaks, centered_cv, eq_cv, "C_v")
        apply_kautz_and_plot(ax_auto_crel, crel_peaks, centered_crel, eq_crel, "C_rel")

        # Plot Column 2: Frequency Domain Annotations
        lines!(ax_ce_fft_peaks, freqs_pos[1:freq_stop_cnt], ce_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_E FFT ($lbl)")
        if !isempty(ce_peaks)
            scatter!(ax_ce_fft_peaks, [p[1] for p in ce_peaks], [p[2] for p in ce_peaks], color=:red, markersize=12, label="Peaks")
            text!(ax_ce_fft_peaks, [Point2f(p[1], p[2]) for p in ce_peaks], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in ce_peaks], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end

        lines!(ax_cd_fft_peaks, freqs_pos[1:freq_stop_cnt], cd_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_D FFT ($lbl)")
        if !isempty(cd_peaks)
            scatter!(ax_cd_fft_peaks, [p[1] for p in cd_peaks], [p[2] for p in cd_peaks], color=:red, markersize=12, label="Peaks")
            text!(ax_cd_fft_peaks, [Point2f(p[1], p[2]) for p in cd_peaks], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in cd_peaks], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end

        lines!(ax_cv_fft_peaks, freqs_pos[1:freq_stop_cnt], cv_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_v FFT ($lbl)")
        if !isempty(cv_peaks)
            scatter!(ax_cv_fft_peaks, [p[1] for p in cv_peaks], [p[2] for p in cv_peaks], color=:red, markersize=12, label="Peaks")
            text!(ax_cv_fft_peaks, [Point2f(p[1], p[2]) for p in cv_peaks], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in cv_peaks], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end

        lines!(ax_crel_fft_peaks, freqs_pos[1:freq_stop_cnt], crel_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_rel FFT ($lbl)")
        if !isempty(crel_peaks)
            scatter!(ax_crel_fft_peaks, [p[1] for p in crel_peaks], [p[2] for p in crel_peaks], color=:red, markersize=12, label="Peaks")
            text!(ax_crel_fft_peaks, [Point2f(p[1], p[2]) for p in crel_peaks], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in crel_peaks], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end
    end

    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_cd, position=:rt)
    axislegend(ax_auto_cv, position=:rt)
    axislegend(ax_auto_crel, position=:rt)

    axislegend(ax_ce_fft_peaks, position=:rt)
    axislegend(ax_cd_fft_peaks, position=:rt)
    axislegend(ax_cv_fft_peaks, position=:rt)
    axislegend(ax_crel_fft_peaks, position=:rt)

    save("../../img/AB_viz_obsrv_LINLIN_FFTW_4x2_annotatedPeaks_Kautz.png", fig)
    display(fig)
end

plot_all_observables()
