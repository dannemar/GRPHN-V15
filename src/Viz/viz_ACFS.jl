using GLMakie
using Statistics
using Printf # Added for formatting the energy string

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
    lattice_configs = [(100,100,500)]#[(1024,1024,100), (1024,1024,200), (1024,1024,600), (1024,1024,700)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 25
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    int_scheme_selected = "ABA864"

    # Figure Setup: 4 Rows x 1 Column
    fig = Figure(size=(1600, 2000), fontsize=23)

    lwidth = 2.5 * 2

    println("Equilibration Calc: $(eq_idx)")

    # Define axes for linear-linear scale in the requested order
    ax_auto_ce = Axis(fig[1, 1],
        xlabel="Time",
        ylabel="Energy C_E(t)"
    )
    ax_auto_cd = Axis(fig[2, 1],
        xlabel="Time",
        ylabel="Absolute C_D(t)"
    )
    ax_auto_cv = Axis(fig[3, 1],
        xlabel="Time",
        ylabel="Velocity C_V(t)"
    )
    ax_auto_crel = Axis(fig[4, 1],
        xlabel="Time",
        ylabel="Length C_rel(t)"
    )

    # Use Makie's color palette to differentiate sizes
    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        # Assign a consistent color for this specific configuration
        c = colors[mod1(idx, length(colors))]

        # Format the energy to 2 decimal places for the filename (e.g., 100.0 -> "100.00")
        e_str = @sprintf("%.2f", energy)

        # Update label to include the energy/temperature
        lbl = "$(w)x$(h), E=$(energy)"

        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # File paths
        autocorr_ce_path = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_cd_path = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")
        # temp_path = joinpath(data_dir, "temperature" * strEnd * ".bin")

        # Process data
        mean_ce, _, _, T_steps = process_observable(autocorr_ce_path, R)
        mean_cd, _, _, _ = process_observable(autocorr_cd_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)
        # mean_temp, _, _, _ = process_observable(temp_path, R)

        t_phy = (1:T_steps) .* dt

        # Plot Row 1: Autocorrelation C_E (Energy) - Raw data
        lines!(ax_auto_ce, t_phy[eq_idx:end],
            mean_ce[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_E ($lbl)")

        # Plot Row 2: Autocorrelation C_D (Absolute) - Raw data
        lines!(ax_auto_cd, t_phy[eq_idx:end],
            mean_cd[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_D ($lbl)")

        # Plot Row 3: Autocorrelation C_V (Velocity) - Raw data
        lines!(ax_auto_cv, t_phy[eq_idx:end],
            mean_cv[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_V ($lbl)")

        # Plot Row 4: Autocorrelation C_rel (Relative Length) - Raw data
        lines!(ax_auto_crel, t_phy[eq_idx:end],
            mean_crel[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_rel ($lbl)")
    end

    # Add legends to all subplots
    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_cd, position=:rt)
    axislegend(ax_auto_cv, position=:rt)
    axislegend(ax_auto_crel, position=:rt)

    # Save and display
    save("../../img/ACFs_LINLIN_combined.png", fig)
end

plot_all_observables()
