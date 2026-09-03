using GLMakie
using Statistics
using Printf

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
"""
Applies a one-sided cosine taper to the final `taper_fraction` of the signal,
forcing the tail smoothly to zero while leaving the earlier lags unmodified.
"""
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

"""
Mirrors a one-sided ACF sequence [C(0), C(1), ..., C(N)] across tau = 0
to produce a two-sided symmetric array of length 2N - 1.
"""
function mirror_signal(signal::AbstractVector{Float64}, t_lag::AbstractVector{Float64})
    # Reverse lags from N down to 2 for the negative axis
    sym_signal = vcat(reverse(signal[2:end]), signal)
    sym_time   = vcat(-reverse(t_lag[2:end]), t_lag)
    return sym_signal, sym_time
end

# ==============================================================================
# 3. STEP 2 PLOTTING: MIRRORED & TAPERED VALIDATION
# ==============================================================================
function plot_step2_mirrored_taper()
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

    # Taper the final 10% of the time series
    taper_fraction = 0.10

    fig = Figure(size=(1200, 1600), fontsize=22)
    lwidth = 2.5

    ax_ce   = Axis(fig[1, 1], ylabel=L"C_E(\tau) / C_E(0)")
    ax_cd   = Axis(fig[2, 1], ylabel=L"C_D(\tau) / C_D(0)")
    ax_cv   = Axis(fig[3, 1], ylabel=L"C_v(\tau) / C_v(0)")
    ax_crel = Axis(fig[4, 1], xlabel=L"\text{Symmetric Lag Time } (\tau)", ylabel=L"C_{rel}(\tau) / C_{rel}(0)")

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # Load and normalize
        mean_ce, _, _, _   = process_observable(joinpath(data_dir, "acf_C_E" * strEnd * ".bin"), R)
        mean_cd, _, _, _   = process_observable(joinpath(data_dir, "acf_C_D" * strEnd * ".bin"), R)
        mean_cv, _, _, _   = process_observable(joinpath(data_dir, "acf_C_v" * strEnd * ".bin"), R)
        mean_crel, _, _, _ = process_observable(joinpath(data_dir, "acf_C_rel" * strEnd * ".bin"), R)

        norm_ce   = normalize_acf(mean_ce[eq_idx:end])
        norm_cd   = normalize_acf(mean_cd[eq_idx:end])
        norm_cv   = normalize_acf(mean_cv[eq_idx:end])
        norm_crel = normalize_acf(mean_crel[eq_idx:end])

        t_lag = (0:(length(norm_ce) - 1)) .* dt

        # Apply 10% tail taper
        taper_ce   = apply_tail_taper(norm_ce, taper_fraction)
        taper_cd   = apply_tail_taper(norm_cd, taper_fraction)
        taper_cv   = apply_tail_taper(norm_cv, taper_fraction)
        taper_crel = apply_tail_taper(norm_crel, taper_fraction)

        # Mirror both raw and tapered signals for comparison
        raw_sym_ce, t_sym     = mirror_signal(norm_ce, t_lag)
        taper_sym_ce, _       = mirror_signal(taper_ce, t_lag)

        raw_sym_cd, _         = mirror_signal(norm_cd, t_lag)
        taper_sym_cd, _       = mirror_signal(taper_cd, t_lag)

        raw_sym_cv, _         = mirror_signal(norm_cv, t_lag)
        taper_sym_cv, _       = mirror_signal(taper_cv, t_lag)

        raw_sym_crel, _       = mirror_signal(norm_crel, t_lag)
        taper_sym_crel, _     = mirror_signal(taper_crel, t_lag)

        # Plot raw mirrored signal in background (gray) to prove preservation of early lags
        lines!(ax_ce,   t_sym, raw_sym_ce,   color=(:gray, 0.5), linewidth=lwidth, label="Raw Un-tapered")
        lines!(ax_cd,   t_sym, raw_sym_cd,   color=(:gray, 0.5), linewidth=lwidth)
        lines!(ax_cv,   t_sym, raw_sym_cv,   color=(:gray, 0.5), linewidth=lwidth)
        lines!(ax_crel, t_sym, raw_sym_crel, color=(:gray, 0.5), linewidth=lwidth)

        # Plot tapered mirrored signal in foreground
        lines!(ax_ce,   t_sym, taper_sym_ce,   color=c, linewidth=lwidth, label="Tapered ($lbl)")
        lines!(ax_cd,   t_sym, taper_sym_cd,   color=c, linewidth=lwidth, label="Tapered ($lbl)")
        lines!(ax_cv,   t_sym, taper_sym_cv,   color=c, linewidth=lwidth, label="Tapered ($lbl)")
        lines!(ax_crel, t_sym, taper_sym_crel, color=c, linewidth=lwidth, label="Tapered ($lbl)")
    end

    axislegend(ax_ce,   position=:rt)
    axislegend(ax_cd,   position=:rt)
    axislegend(ax_cv,   position=:rt)
    axislegend(ax_crel, position=:rt)

    #save("../../img/Step2_Mirrored_Tapered_ACFs.png", fig)
    display(fig)
end

plot_step2_mirrored_taper()
