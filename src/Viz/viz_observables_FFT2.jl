using GLMakie
using Statistics
using Printf
using FFTW

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function plot_all_observables()
    data_dir = "../../data/"

    # Define the list of configurations: (W, H, Target Energy)
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

    # Figure Setup: 4 Rows x 2 Columns
    fig = Figure(size=(2400, 2600), fontsize=23)
    lwidth = 2.5

    println("Equilibration Calc: $(eq_idx)")

    # Define axes for time-domain (Column 1)
    ax_auto_ce = Axis(fig[1, 1], xlabel="Time", ylabel="Energy C_E(t)")
    ax_auto_cd = Axis(fig[2, 1], xlabel="Time", ylabel="Absolute C_D(t)")
    ax_auto_cv = Axis(fig[3, 1], xlabel="Time", ylabel="Velocity C_v(t)")
    ax_auto_crel = Axis(fig[4, 1], xlabel="Time", ylabel="Length C_rel(t)")

    # Define axes for frequency-domain (Column 2)
    ax_ce_fft = Axis(fig[1, 2], xlabel="Frequency (Hz)", ylabel="Magnitude FT(C_E)")
    ax_cd_fft = Axis(fig[2, 2], xlabel="Frequency (Hz)", ylabel="Magnitude FT(C_D)")
    ax_cv_fft = Axis(fig[3, 2], xlabel="Frequency (Hz)", ylabel="Magnitude FT(C_v)")
    ax_crel_fft = Axis(fig[4, 2], xlabel="Frequency (Hz)", ylabel="Magnitude FT(C_rel)")

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # File paths
        autocorr_cd_path = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        autocorr_ce_path = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

        # Process data
        mean_cd, _, _, T_steps = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)
        mean_ce, _, _, _ = process_observable(autocorr_ce_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)

        t_phy = (1:T_steps) .* dt

        # Isolate equilibrated data
        eq_cd = mean_cd[eq_idx:end]
        eq_cv = mean_cv[eq_idx:end]
        eq_ce = mean_ce[eq_idx:end]
        eq_crel = mean_crel[eq_idx:end]

        # Center the equilibrated data (subtract the mean)
        centered_cd = eq_cd .- mean(eq_cd)
        centered_cv = eq_cv .- mean(eq_cv)
        centered_ce = eq_ce .- mean(eq_ce)
        centered_crel = eq_crel .- mean(eq_crel)

        # Perform FFT strictly on centered data
        cd_fft = fft(centered_cd)
        cv_fft = fft(centered_cv)
        ce_fft = fft(centered_ce)
        crel_fft = fft(centered_crel)

        # Set up frequencies
        N_obs_len = length(eq_cd)
        fs = 1 / dt
        axis_samples = fftfreq(N_obs_len, fs)

        # Extract positive indices and corresponding magnitudes
        pos_idx = 1:div(N_obs_len, 2)
        freqs_pos = axis_samples[pos_idx]

        cd_fft_abs_pos = abs.(cd_fft)[pos_idx]
        cv_fft_abs_pos = abs.(cv_fft)[pos_idx]
        ce_fft_abs_pos = abs.(ce_fft)[pos_idx]
        crel_fft_abs_pos = abs.(crel_fft)[pos_idx]

        # Plot Column 1: Time Domain (Raw Equilibrated Data)
        lines!(ax_auto_cd, t_phy[eq_idx:end], eq_cd, color=c, linewidth=lwidth + 2.5, label="C_D ($lbl)")
        lines!(ax_auto_cv, t_phy[eq_idx:end], eq_cv, color=c, linewidth=lwidth, label="C_v ($lbl)")
        lines!(ax_auto_ce, t_phy[eq_idx:end], eq_ce, color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_auto_crel, t_phy[eq_idx:end], eq_crel, color=c, linewidth=lwidth, label="C_rel ($lbl)")

        # Plot Column 2: Frequency Domain (FFT Magnitudes)
        freq_stop_cnt = 150
        lines!(ax_cd_fft, freqs_pos[1:freq_stop_cnt], cd_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_D FFT ($lbl)")
        lines!(ax_cv_fft, freqs_pos[1:freq_stop_cnt], cv_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_v FFT ($lbl)")
        lines!(ax_ce_fft, freqs_pos[1:freq_stop_cnt], ce_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_E FFT ($lbl)")
        lines!(ax_crel_fft, freqs_pos[1:freq_stop_cnt], crel_fft_abs_pos[1:freq_stop_cnt], color=c, linewidth=lwidth, label="C_rel FFT ($lbl)")
    end

    # Add legends
    axislegend(ax_auto_cd, position=:rt)
    axislegend(ax_auto_cv, position=:rt)
    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_crel, position=:rt)
    axislegend(ax_cd_fft, position=:rt)
    axislegend(ax_cv_fft, position=:rt)
    axislegend(ax_ce_fft, position=:rt)
    axislegend(ax_crel_fft, position=:rt)

    # Save and display
    save("../../img/AB_viz_obsrv_LINLIN_FFTW_$(int_scheme_selected)_4x2.png", fig)
end

plot_all_observables()
