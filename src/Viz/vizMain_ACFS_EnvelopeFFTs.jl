using GLMakie
using Statistics
using Printf
using Peaks
using Interpolations
using FFTW

# ==========================================================
# 1. Core Processing Functions
# ==========================================================

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function extract_upper_envelope(C_raw::Vector{Float64})
    peak_indices, peak_values = findmaxima(C_raw)

    idx_vec = collect(Int, peak_indices)
    val_vec = collect(Float64, peak_values)

    if isempty(idx_vec) || idx_vec[1] != 1
        pushfirst!(idx_vec, 1)
        pushfirst!(val_vec, C_raw[1])
    end
    if idx_vec[end] != length(C_raw)
        push!(idx_vec, length(C_raw))
        push!(val_vec, C_raw[end])
    end

    itp = linear_interpolation(idx_vec, val_vec)
    return itp(1:length(C_raw))
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

function prep_and_fft(signal::Vector{Float64}, dt::Float64, is_oscillatory::Bool)
    N = length(signal)

    # 1. Baseline Correction
    # Only center signals that oscillate around 0.
    # Positive decaying signals (C_E, Envelopes) have a theoretical baseline of 0.
    if is_oscillatory
        sig_base = signal .- mean(signal)
    else
        sig_base = signal
    end

    # 2. Asymmetric Apodization (Half-Hann Window)
    # Starts at 1.0 at t=0, tapers to 0.0 at t=T_max.
    # Preserves the ACF peak while eliminating zero-padding boundary steps.
    half_hann = 0.5 .* (1 .+ cos.(π .* (0:N-1) ./ (N-1)))
    windowed = sig_base .* half_hann

    # 3. Zero-Pad
    N_pad = nextpow(2, N * 4)
    padded = zeros(N_pad)
    padded[1:N] .= windowed

    # 4. Perform FFT
    fft_result = fft(padded)

    # Calculate frequencies in rad/s (ω)
    freqs = fftfreq(N_pad, 1/dt) .* (2π)

    # Extract strictly positive frequencies
    pos_idx = 1:div(N_pad, 2)
    return freqs[pos_idx], abs.(fft_result)[pos_idx]
end

# ==========================================================
# 2. Main Plotting Routine
# ==========================================================

function plot_all_observables()
    data_dir = "../../data/"

    # Configurations: (W, H, Target Energy)
    lattice_configs = [(256, 256, 100)] # Adjusted for testing based on provided files

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 200
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    # Figure Setup: 4 Rows x 3 Columns
    fig = Figure(size=(3600, 2600), fontsize=23)
    lwidth = 2.5

    println("Equilibration Calc: $(eq_idx)")

    # Define Axes
    ax_t_ce = Axis(fig[1, 1], xlabel="Time", ylabel="Energy C_E(t)")
    ax_f_ce = Axis(fig[1, 2], xlabel=L"\omega", ylabel=L"|FT(C_E)|")

    ax_t_cd = Axis(fig[2, 1], xlabel="Time", ylabel="Absolute C_D(t)")
    ax_f_cd = Axis(fig[2, 2], xlabel=L"\omega", ylabel=L"|FT(C_D)|")
    ax_fe_cd = Axis(fig[2, 3], xlabel=L"\omega", ylabel=L"|FT(Env\ C_D)|")

    ax_t_cv = Axis(fig[3, 1], xlabel="Time", ylabel="Velocity C_v(t)")
    ax_f_cv = Axis(fig[3, 2], xlabel=L"\omega", ylabel=L"|FT(C_v)|")
    ax_fe_cv = Axis(fig[3, 3], xlabel=L"\omega", ylabel=L"|FT(Env\ C_v)|")

    ax_t_crel = Axis(fig[4, 1], xlabel="Time", ylabel="Bond Length C_{rel}(t)")
    ax_f_crel = Axis(fig[4, 2], xlabel=L"\omega", ylabel=L"|FT(C_{rel})|")
    ax_fe_crel = Axis(fig[4, 3], xlabel=L"\omega", ylabel=L"|FT(Env\ C_{rel})|")

    colors = Makie.wong_colors()

    # Boundary for plotting and peak extraction in frequency domain (rad/s)
    omega_max = 9.0

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # File paths
        autocorr_ce_path = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_cd_path = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

        # Process data
        mean_ce, _, _, T_steps = process_observable(autocorr_ce_path, R)
        mean_cd, _, _, _ = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)

        t_phy = (1:T_steps) .* dt

        # Isolate equilibrated data
        eq_ce = mean_ce[eq_idx:end]
        eq_cd = mean_cd[eq_idx:end]
        eq_cv = mean_cv[eq_idx:end]
        eq_crel = mean_crel[eq_idx:end]
        t_eq = t_phy[eq_idx:end]

        # Extract Envelopes
        env_cd = extract_upper_envelope(eq_cd)
        env_cv = extract_upper_envelope(eq_cv)
        env_crel = extract_upper_envelope(eq_crel)

        # --------------------------------------------------
        # TIME DOMAIN PLOTS (Column 1)
        # --------------------------------------------------
        lines!(ax_t_ce, t_eq, eq_ce, color=c, linewidth=lwidth, label="C_E ($lbl)")

        lines!(ax_t_cd, t_eq, eq_cd, color=c, linewidth=lwidth, label="C_D ($lbl)", alpha=0.5)
        lines!(ax_t_cd, t_eq, env_cd, color=c, linestyle=:dash, linewidth=lwidth, label="Env C_D ($lbl)")

        lines!(ax_t_cv, t_eq, eq_cv, color=c, linewidth=lwidth, label="C_v ($lbl)", alpha=0.5)
        lines!(ax_t_cv, t_eq, env_cv, color=c, linestyle=:dash, linewidth=lwidth, label="Env C_v ($lbl)")

        lines!(ax_t_crel, t_eq, eq_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)", alpha=0.5)
        lines!(ax_t_crel, t_eq, env_crel, color=c, linestyle=:dash, linewidth=lwidth, label="Env C_rel ($lbl)")

        # --------------------------------------------------
        # FFT PROCESSING & PLOTS (Columns 2 & 3)
        # --------------------------------------------------

        function process_and_plot_fft!(ax, signal, label_prefix, is_oscillatory)
            freqs, mags = prep_and_fft(signal, dt, is_oscillatory)

            # Restrict viewing and peak extraction
            idx_stop = findlast(x -> x <= omega_max, freqs)
            idx_stop = isnothing(idx_stop) ? length(freqs) : idx_stop

            f_view = freqs[1:idx_stop]
            m_view = mags[1:idx_stop]

            peaks = extract_all_peaks(m_view, f_view)

            lines!(ax, f_view, m_view, color=c, linewidth=lwidth, label="$(label_prefix) ($lbl)")
            if !isempty(peaks)
                scatter!(ax, [p[1] for p in peaks], [p[2] for p in peaks], color=:red, markersize=12)
                text!(ax, [Point2f(p[1], p[2]) for p in peaks], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in peaks], align=(:left, :bottom), offset=(4, 4), fontsize=16)
            end
        end

        # Raw FFTs (Column 2)
        # C_E does not oscillate around 0, so is_oscillatory = false
        process_and_plot_fft!(ax_f_ce, eq_ce, "FFT C_E", false)

        # C_D, C_V, C_rel oscillate around 0, so is_oscillatory = true
        process_and_plot_fft!(ax_f_cd, eq_cd, "FFT C_D", true)
        process_and_plot_fft!(ax_f_cv, eq_cv, "FFT C_v", true)
        process_and_plot_fft!(ax_f_crel, eq_crel, "FFT C_rel", true)

        # Envelope FFTs (Column 3)
        # Envelopes are strictly positive, so is_oscillatory = false
        process_and_plot_fft!(ax_fe_cd, env_cd, "FFT Env C_D", false)
        process_and_plot_fft!(ax_fe_cv, env_cv, "FFT Env C_v", false)
        process_and_plot_fft!(ax_fe_crel, env_crel, "FFT Env C_rel", false)
    end

    # Add Legends
    axislegend(ax_t_ce, position=:rt); axislegend(ax_f_ce, position=:rt)
    axislegend(ax_t_cd, position=:rt); axislegend(ax_f_cd, position=:rt); axislegend(ax_fe_cd, position=:rt)
    axislegend(ax_t_cv, position=:rt); axislegend(ax_f_cv, position=:rt); axislegend(ax_fe_cv, position=:rt)
    axislegend(ax_t_crel, position=:rt); axislegend(ax_f_crel, position=:rt); axislegend(ax_fe_crel, position=:rt)

    # Save and display
    save("../../img/Combined_viz_ACFs_Envelopes_FFT.png", fig)
    #display(fig)
end

plot_all_observables()
