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
    lattice_configs = [(256,256,100),(256,256,200),(256,256,300), (256,256,400), (256,256,500), (256,256,600)]#[(100,100,500),(200,200,500),(256,256,500), (300,300,500), (500,500,500), (700,700,500), (1000,1000,500)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.001"
    of = 1
    R = 200
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    # Figure Setup: 4 Rows x 1 Column
    fig = Figure(size=(1600, 2000), fontsize=23)

    lwidth = 2.5 * 2

    println("Equilibration Calc: $(eq_idx)")

    # Define axes for linear-linear scale
    ax_auto_cd = Axis(fig[1, 1],
        xlabel="Time",
        ylabel="Absolute C_D(t)"
    )
    ax_auto_ce = Axis(fig[2, 1],
        xlabel="Time",
        ylabel="Energy C_E(t)"
    )
    ax_auto_crel = Axis(fig[3, 1],
        xlabel="Time",
        ylabel="Length C_rel(t)"
    )
    ax_auto_cv = Axis(fig[4, 1],
        xlabel="Time",
        ylabel="Velocity C_v(t)"
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

        # Appended _kineticIC to match the C++ output
        #strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)_KineticIC"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # File paths
        temp_path = joinpath(data_dir, "temperature" * strEnd * ".bin")
        autocorr_cd_path = joinpath(data_dir, "acf_C_D" * strEnd * ".bin")
        autocorr_crel_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")
        autocorr_ce_path = joinpath(data_dir, "acf_C_E" * strEnd * ".bin")
        autocorr_cv_path = joinpath(data_dir, "acf_C_v" * strEnd * ".bin")

        # Process data
        mean_temp, _, _, T_steps = process_observable(temp_path, R)
        mean_cd, _, _, _ = process_observable(autocorr_cd_path, R)
        mean_crel, _, _, _ = process_observable(autocorr_crel_path, R)
        mean_ce, _, _, _ = process_observable(autocorr_ce_path, R)
        mean_cv, _, _, _ = process_observable(autocorr_cv_path, R)

        t_phy = (1:T_steps) .* dt
        avg_temp_val = mean(mean_temp[eq_idx:end])

        # Plot Row 1: Autocorrelation C_D (Absolute) - Raw data
        lines!(ax_auto_cd, t_phy[eq_idx:end],
            mean_cd[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_D ($lbl)")

        # Plot Row 2: Autocorrelation C_E (Energy) - Raw data
        lines!(ax_auto_ce, t_phy[eq_idx:end],
            mean_ce[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_E ($lbl)")

        # Plot Row 3: Autocorrelation C_rel (Relative Length) - Raw data
        lines!(ax_auto_crel, t_phy[eq_idx:end],
            mean_crel[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_rel ($lbl)")

        # Plot Row 4: Autocorrelation C_v (Velocity) - Raw data
        lines!(ax_auto_cv, t_phy[eq_idx:end],
            mean_cv[eq_idx:end],
            color=c,
            linewidth=lwidth,
            label="C_v ($lbl)")
    end

    # Add legends to all subplots
    axislegend(ax_auto_cd, position=:rt)
    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_crel, position=:rt)
    axislegend(ax_auto_cv, position=:rt)

    # Save and display
    save("../../img/TEMP_viz_obsrv_LINLIN_combined_4x1_Raw.png", fig)
    display(fig)
end

plot_all_observables()
