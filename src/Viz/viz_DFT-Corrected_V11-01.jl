using GLMakie
using Statistics
using Printf

# ==============================================================================
# 1. DATA LOADING & PROCESSING
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

# Normalizes an ACF sequence by its lag-0 value: C(t) / C(0)
function normalize_acf(signal::AbstractVector{Float64})
    c0 = signal[1]
    if c0 == 0.0
        @warn "C(0) is zero; cannot normalize time series."
        return signal
    end
    return signal ./ c0
end

# ==============================================================================
# 2. STEP 1 PLOTTING: RAW NORMALIZED TIME SERIES
# ==============================================================================
function plot_step1_time_domain()
    data_dir = "../../data/"
    lattice_configs = [(256, 256, 200)]

    # Simulation parameters
    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 200

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    println("Equilibration Index: $(eq_idx)")

    # Figure Setup: 4 Rows x 1 Column for initial inspection
    fig = Figure(size=(1200, 1600), fontsize=22)
    lwidth = 2.5

    ax_ce   = Axis(fig[1, 1], ylabel=L"C_E(\tau) / C_E(0)")
    ax_cd   = Axis(fig[2, 1], ylabel=L"C_D(\tau) / C_D(0)")
    ax_cv   = Axis(fig[3, 1], ylabel=L"C_v(\tau) / C_v(0)")
    ax_crel = Axis(fig[4, 1], xlabel=L"\text{Lag Time } (\tau)", ylabel=L"C_{rel}(\tau) / C_{rel}(0)")

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # File paths
        autocorr_ce_path   = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_cd_path   = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_cv_path   = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

        # Load raw data
        mean_ce, _, _, _   = process_observable(autocorr_ce_path, R)
        mean_cd, _, _, _   = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _   = process_observable(autocorr_cv_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)

        # Isolate equilibrated region
        eq_ce   = mean_ce[eq_idx:end]
        eq_cd   = mean_cd[eq_idx:end]
        eq_cv   = mean_cv[eq_idx:end]
        eq_crel = mean_crel[eq_idx:end]

        # Normalize C(tau) / C(0)
        norm_ce   = normalize_acf(eq_ce)
        norm_cd   = normalize_acf(eq_cd)
        norm_cv   = normalize_acf(eq_cv)
        norm_crel = normalize_acf(eq_crel)

        # Define lag time axis starting at tau = 0
        N_lags = length(norm_ce)
        t_lag = (0:(N_lags - 1)) .* dt

        # Plot normalized ACFs
        lines!(ax_ce,   t_lag, norm_ce,   color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_cd,   t_lag, norm_cd,   color=c, linewidth=lwidth, label="C_D ($lbl)")
        lines!(ax_cv,   t_lag, norm_cv,   color=c, linewidth=lwidth, label="C_v ($lbl)")
        lines!(ax_crel, t_lag, norm_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)")
    end

    # Add legends
    axislegend(ax_ce,   position=:rt)
    axislegend(ax_cd,   position=:rt)
    axislegend(ax_cv,   position=:rt)
    axislegend(ax_crel, position=:rt)

    # Save and display
    #save("../../img/Step1_Normalized_ACFs.png", fig)
    display(fig)
end

plot_step1_time_domain()
