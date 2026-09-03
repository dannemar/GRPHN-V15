using GLMakie
using Statistics
using Printf
using FFTW

# ==============================================================================
# 1. DATA LOADING & NORMALIZATION
# ==============================================================================
function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function normalize_acf(signal::AbstractVector{Float64})
    c0 = signal[1]
    if c0 == 0.0
        @warn "C(0) is zero; cannot normalize time series."
        return signal
    end
    return signal ./ c0
end

# ==============================================================================
# 2. TAPERING & SYMMETRIC MIRRORING
# ==============================================================================
function apply_tail_taper(signal::AbstractVector{Float64}, taper_fraction::Float64 = 0.10)
    N = length(signal)
    N_flat = floor(Int, (1.0 - taper_fraction) * N)

    window = ones(Float64, N)
    for i in (N_flat + 1):N
        theta = pi * (i - N_flat) / (N - N_flat)
        window[i] = 0.5 * (1.0 + cos(theta))
    end

    return signal .* window
end

function mirror_signal(signal::AbstractVector{Float64}, t_lag::AbstractVector{Float64})
    sym_signal = vcat(reverse(signal[2:end]), signal)
    sym_time   = vcat(-reverse(t_lag[2:end]), t_lag)
    return sym_signal, sym_time
end

# ==============================================================================
# 3. SPECTRAL ANALYSIS (DFT / PSD) & PEAK EXTRACTION
# ==============================================================================
"""
Computes the Power Spectral Density (PSD) of a symmetric time series.
Applies ifftshift to wrap tau=0 to index 1, computes FFT, and scales by dt.
Returns positive angular frequencies (omega) and their corresponding amplitudes.
"""
function compute_psd(sym_signal::AbstractVector{Float64}, dt::Float64)
    L = length(sym_signal)

    # 1. Zero-phase wrap: shift center element (tau=0) to index 1
    wrapped_signal = ifftshift(sym_signal)

    # 2. Compute FFT and scale by dt to approximate the continuous integral
    raw_fft = fft(wrapped_signal) .* dt

    # 3. Isolate real power (Wiener-Khinchin theorem)
    psd = abs.(raw_fft)

    # Generate angular frequency axis: \omega = 2\pi f
    fs = 1.0 / dt
    omega = fftfreq(L, fs) .* (2 * pi)

    # Extract only positive frequencies (up to Nyquist)
    pos_idx = 1:div(L, 2)+1
    return omega[pos_idx], psd[pos_idx]
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

# ==============================================================================
# 4. FULL EXECUTION & PLOTTING (4x2 GRID)
# ==============================================================================
function plot_step3_psd()
    data_dir = "../../data/"
    lattice_configs = [(256, 256, 200)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 200

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1
    taper_fraction = 0.250

    fig = Figure(size=(2400, 2600), fontsize=23)
    lwidth = 2.5

    # Column 1: Time Domain Axes
    ax_ce_time   = Axis(fig[1, 1], ylabel=L"C_E(\tau) / C_E(0)")
    ax_cd_time   = Axis(fig[2, 1], ylabel=L"C_D(\tau) / C_D(0)")
    ax_cv_time   = Axis(fig[3, 1], ylabel=L"C_v(\tau) / C_v(0)")
    ax_crel_time = Axis(fig[4, 1], xlabel=L"\text{Symmetric Lag Time } (\tau)", ylabel=L"C_{rel}(\tau) / C_{rel}(0)")

    # Column 2: Frequency Domain Axes
    ax_ce_psd   = Axis(fig[1, 2], ylabel=L"|\mathcal{F}\{C_E\}|", xlabelsize=36)
    ax_cd_psd   = Axis(fig[2, 2], ylabel=L"|\mathcal{F}\{C_D\}|", xlabelsize=36)
    ax_cv_psd   = Axis(fig[3, 2], ylabel=L"|\mathcal{F}\{C_v\}|", xlabelsize=36)
    ax_crel_psd = Axis(fig[4, 2], xlabel=L"\omega", ylabel=L"|\mathcal{F}\{C_{rel}\}|", xlabelsize=36)

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # 1. Load and process
        mean_ce, _, _, _   = process_observable(joinpath(data_dir, "acf_C_E" * strEnd * ".bin"), R)
        mean_cd, _, _, _   = process_observable(joinpath(data_dir, "acf_C_D" * strEnd * ".bin"), R)
        mean_cv, _, _, _   = process_observable(joinpath(data_dir, "acf_C_v" * strEnd * ".bin"), R)
        mean_crel, _, _, _ = process_observable(joinpath(data_dir, "acf_C_rel" * strEnd * ".bin"), R)

        norm_ce   = normalize_acf(mean_ce[eq_idx:end])
        norm_cd   = normalize_acf(mean_cd[eq_idx:end])
        norm_cv   = normalize_acf(mean_cv[eq_idx:end])
        norm_crel = normalize_acf(mean_crel[eq_idx:end])

        t_lag = (0:(length(norm_ce) - 1)) .* dt

        # 2. Taper and Mirror
        taper_ce   = apply_tail_taper(norm_ce, taper_fraction)
        taper_cd   = apply_tail_taper(norm_cd, taper_fraction)
        taper_cv   = apply_tail_taper(norm_cv, taper_fraction)
        taper_crel = apply_tail_taper(norm_crel, taper_fraction)

        sym_ce, t_sym     = mirror_signal(taper_ce, t_lag)
        sym_cd, _         = mirror_signal(taper_cd, t_lag)
        sym_cv, _         = mirror_signal(taper_cv, t_lag)
        sym_crel, _       = mirror_signal(taper_crel, t_lag)



        omega_max_target = 5.0  # Set the max physical frequency you want to observe (in rad/s)
        delta_omega = (2 * pi) / (length(sym_ce) * dt)
        freq_stop_cnt = round(Int, omega_max_target / delta_omega)


        # 3. Compute PSDs
        omega, psd_ce   = compute_psd(sym_ce, dt)
        _,     psd_cd   = compute_psd(sym_cd, dt)
        _,     psd_cv   = compute_psd(sym_cv, dt)
        _,     psd_crel = compute_psd(sym_crel, dt)

        # Extract peaks within the viewing window
        view_idx = 1:min(freq_stop_cnt, length(omega))
        peaks_ce   = extract_all_peaks(psd_ce[view_idx], omega[view_idx])
        peaks_cd   = extract_all_peaks(psd_cd[view_idx], omega[view_idx])
        peaks_cv   = extract_all_peaks(psd_cv[view_idx], omega[view_idx])
        peaks_crel = extract_all_peaks(psd_crel[view_idx], omega[view_idx])

        # --- Plot Column 1: Conditioned Time Series ---
        lines!(ax_ce_time,   t_sym, sym_ce,   color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_cd_time,   t_sym, sym_cd,   color=c, linewidth=lwidth, label="C_D ($lbl)")
        lines!(ax_cv_time,   t_sym, sym_cv,   color=c, linewidth=lwidth, label="C_v ($lbl)")
        lines!(ax_crel_time, t_sym, sym_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)")

        # --- Plot Column 2: Power Spectral Density & Peaks ---
        lines!(ax_ce_psd, omega[view_idx], psd_ce[view_idx], color=c, linewidth=lwidth, label="PSD ($lbl)")
        #=
        if !isempty(peaks_ce)
            scatter!(ax_ce_psd, [p[1] for p in peaks_ce], [p[2] for p in peaks_ce], color=:red, markersize=12)
            text!(ax_ce_psd, [Point2f(p[1], p[2]) for p in peaks_ce], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in peaks_ce], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end
=#
        lines!(ax_cd_psd, omega[view_idx], psd_cd[view_idx], color=c, linewidth=lwidth, label="PSD ($lbl)")

        #=
        if !isempty(peaks_cd)
            scatter!(ax_cd_psd, [p[1] for p in peaks_cd], [p[2] for p in peaks_cd], color=:red, markersize=12)
            text!(ax_cd_psd, [Point2f(p[1], p[2]) for p in peaks_cd], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in peaks_cd], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end
=#
        lines!(ax_cv_psd, omega[view_idx], psd_cv[view_idx], color=c, linewidth=lwidth, label="PSD ($lbl)")
        #=
        if !isempty(peaks_cv)
            scatter!(ax_cv_psd, [p[1] for p in peaks_cv], [p[2] for p in peaks_cv], color=:red, markersize=12)
            text!(ax_cv_psd, [Point2f(p[1], p[2]) for p in peaks_cv], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in peaks_cv], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end
=#
        lines!(ax_crel_psd, omega[view_idx], psd_crel[view_idx], color=c, linewidth=lwidth, label="PSD ($lbl)")
    #=
        if !isempty(peaks_crel)
            scatter!(ax_crel_psd, [p[1] for p in peaks_crel], [p[2] for p in peaks_crel], color=:red, markersize=12)
            text!(ax_crel_psd, [Point2f(p[1], p[2]) for p in peaks_crel], text=[@sprintf("(%.2f, %.2f)", p[1], p[2]) for p in peaks_crel], align=(:left, :bottom), offset=(4, 4), fontsize=16)
        end
    =#
    end

    # Legends
    axislegend(ax_ce_time,   position=:rt)
    axislegend(ax_cd_time,   position=:rt)
    axislegend(ax_cv_time,   position=:rt)
    axislegend(ax_crel_time, position=:rt)

    axislegend(ax_ce_psd,   position=:rt)
    axislegend(ax_cd_psd,   position=:rt)
    axislegend(ax_cv_psd,   position=:rt)
    axislegend(ax_crel_psd, position=:rt)

    # Save and display
    save("../../img/Step3_PSD_4x2_annotatedPeaks.png", fig)
    display(fig)
end

plot_step3_psd()
