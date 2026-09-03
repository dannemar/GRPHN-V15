using GLMakie
using Statistics
using Printf
using DSP

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function extract_envelope(C_raw::Vector{Float64}; cutoff_noise=0.05)
    # 1. Isolate the oscillating component (Baseline Subtraction)
    tail_mean = mean(C_raw[div(end, 2):end])
    C_osc = C_raw .- tail_mean

    # 2. Filter high-frequency noise
    my_filter = digitalfilter(Lowpass(cutoff_noise), Butterworth(4))
    C_osc_filtered = filtfilt(my_filter, C_osc)

    # 3. Pad boundaries to prevent FFT/Hilbert ringing
    pad_len = min(500, div(length(C_osc_filtered), 4))
    pad_start = reverse(C_osc_filtered[2:(pad_len + 1)])
    pad_end = zeros(pad_len)

    C_padded = vcat(pad_start, C_osc_filtered, pad_end)

    # 4. Apply Hilbert Transform
    analytic_sig = hilbert(C_padded)
    env_padded = abs.(analytic_sig)

    # 5. Reconstruct physical envelope
    env_osc = env_padded[(pad_len + 1) : (pad_len + length(C_raw))]
    E_final = env_osc .+ tail_mean

    return E_final
end

function plot_all_observables()
    data_dir = "../../data/"

    # Define the list of configurations: (W, H, Target Energy)
    lattice_configs = [(500,500,500)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 25
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    # Figure Setup: 5 Rows x 1 Column
    fig = Figure(size=(1600, 2500), fontsize=23)

    lwidth = 2.5 * 2

    println("Equilibration Calc: $(eq_idx)")

    # Define axes for linear-linear scale
    ax_auto_ce = Axis(fig[1, 1], xlabel="Time", ylabel="Energy C_E(t)")
    ax_auto_cd = Axis(fig[2, 1], xlabel="Time", ylabel="Absolute C_D(t)")
    ax_auto_cv = Axis(fig[3, 1], xlabel="Time", ylabel="Velocity C_V(t)")
    ax_auto_crel = Axis(fig[4, 1], xlabel="Time", ylabel="Length C_rel(t)")
    ax_comp = Axis(fig[5, 1], xlabel="Time", ylabel="Envelope Comparison")

    colors = Makie.wong_colors()

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

        # Process raw data
        mean_ce, _, _, T_steps = process_observable(autocorr_ce_path, R)
        mean_cd, _, _, _ = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)

        t_phy = (1:T_steps) .* dt

        # Extract post-equilibration time series
        data_ce = mean_ce[eq_idx:end]
        data_cd = mean_cd[eq_idx:end]
        data_cv = mean_cv[eq_idx:end]
        data_crel = mean_crel[eq_idx:end]

        # Calculate envelopes (omitted for C_E)
        env_cd = extract_envelope(data_cd, cutoff_noise=0.05)
        env_cv = extract_envelope(data_cv, cutoff_noise=0.05)
        env_crel = extract_envelope(data_crel, cutoff_noise=0.05)

        # Plot Row 1: C_E (Raw only)
        lines!(ax_auto_ce, t_phy[eq_idx:end], data_ce, color=c, linewidth=lwidth, label="C_E ($lbl)")

        # Plot Row 2: C_D
        lines!(ax_auto_cd, t_phy[eq_idx:end], data_cd, color=c, linewidth=lwidth, label="C_D ($lbl)")
        lines!(ax_auto_cd, t_phy[eq_idx:end], env_cd, color=:black, linestyle=:dash, linewidth=2.0, label="Env C_D")

        # Plot Row 3: C_V
        lines!(ax_auto_cv, t_phy[eq_idx:end], data_cv, color=c, linewidth=lwidth, label="C_V ($lbl)")
        lines!(ax_auto_cv, t_phy[eq_idx:end], env_cv, color=:black, linestyle=:dash, linewidth=2.0, label="Env C_V")

        # Plot Row 4: C_rel
        lines!(ax_auto_crel, t_phy[eq_idx:end], data_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)")
        lines!(ax_auto_crel, t_phy[eq_idx:end], env_crel, color=:black, linestyle=:dash, linewidth=2.0, label="Env C_rel")

        # Plot Row 5: Comparison Plot
        # Using distinct colors from the wong_colors palette to differentiate the lines
        lines!(ax_comp, t_phy[eq_idx:end], data_ce, color=colors[1], linewidth=lwidth, label="C_E (Raw)")
        lines!(ax_comp, t_phy[eq_idx:end], env_cd, color=colors[2], linestyle=:dash, linewidth=2.0, label="Env C_D")
        lines!(ax_comp, t_phy[eq_idx:end], env_cv, color=colors[3], linestyle=:dash, linewidth=2.0, label="Env C_V")
        lines!(ax_comp, t_phy[eq_idx:end], env_crel, color=colors[4], linestyle=:dash, linewidth=2.0, label="Env C_rel")
    end

    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_cd, position=:rt)
    axislegend(ax_auto_cv, position=:rt)
    axislegend(ax_auto_crel, position=:rt)
    axislegend(ax_comp, position=:rt)

    save("../../img/ACFs_LINLIN_Env-HilbertTransform.png", fig)
end

plot_all_observables()
