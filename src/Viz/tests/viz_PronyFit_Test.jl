using GLMakie
using Statistics

# ==========================================
# Data Processing Function
# ==========================================
function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

# ==========================================
# Two-Term Prony Series Model
# ==========================================
function prony_series(t, C0, A1, σ1, ω1, ϕ1, A2, σ2, ω2, ϕ2)
    term1 = A1 .* exp.(σ1 .* t) .* cos.(ω1 .* t .+ ϕ1)
    term2 = A2 .* exp.(σ2 .* t) .* cos.(ω2 .* t .+ ϕ2)
    return C0 .+ term1 .+ term2
end

# ==========================================
# Main Plotting Routine
# ==========================================
function interactive_prony_fit()
    # 1. Configuration and Data Loading
    data_dir = "../../../data/"
    # Adjust this filename/parameters to match the specific file you want to load
    w, h, energy = 256, 256, 500
    s, tt, eqt_str, dt_str, R, of = 41, 600, "500.000", "0.010", 200, 1

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    strEnd = "_w-$(w)_h-$(h)_H-$(energy).00_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"
    # Fallback name matching your previous script conventions:
    # strEnd = "_w-256_h-256_H-500.00_s-41_tt-600_eqt-500.000_dt-0.010_r-200_of-1"

    file_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

    # Process data
    mean_crel, _, _, T_steps = process_observable(file_path, R)
    t_phy = (1:T_steps) .* dt

    # Isolate equilibrated data
    t_eq = t_phy[eq_idx:end]
    # Shift time so the fitting window starts at t=0 for the Prony series
    t_fit = t_eq .- t_eq[1]
    eq_crel = mean_crel[eq_idx:end]
    mean_offset = mean(eq_crel)

    # 2. Figure Setup
    fig = Figure(size=(1400, 1000), fontsize=20)
    ax = Axis(fig[1, 1], xlabel="Time (shifted to t=0)", ylabel="Bond Length C_B(t)", title="Interactive Prony Series Fit")

    # 3. Interactive UI Elements (SliderGrid)
    # Range parameters can be adjusted here if your extracted values fall outside these bounds.
    # Note: Click the value box next to the slider in the UI to manually type exact values.
    sg = SliderGrid(fig[2, 1],
            (label = "Offset (C₀)", range = -0.5:0.001:0.5, startvalue = mean_offset),
            (label = "Amp₁ (A₁)", range = 0.0:0.001:1, startvalue = 0.074),
            (label = "Decay₁ (σ₁)", range = -2.0:0.001:0.0, startvalue = -0.1),
            (label = "Freq₁ (ω₁)", range = 0:0.01:5.0, startvalue = 3.39),
            (label = "Phase₁ (ϕ₁)", range = -pi:0.01:pi, startvalue = 0.0),
            (label = "Amp₂ (A₂)", range = 0.0:0.001:0.2, startvalue = 0.043),
            (label = "Decay₂ (σ₂)", range = -2.0:0.001:0.0, startvalue = -0.1),
            (label = "Freq₂ (ω₂)", range = 0:0.01:5.0, startvalue = 4.02),
            (label = "Phase₂ (ϕ₂)", range = -pi:0.01:pi, startvalue = 0.0)
        )

    # Extract observables from sliders
    C0_obs = sg.sliders[1].value
    A1_obs = sg.sliders[2].value
    σ1_obs = sg.sliders[3].value
    ω1_obs = sg.sliders[4].value
    ϕ1_obs = sg.sliders[5].value
    A2_obs = sg.sliders[6].value
    σ2_obs = sg.sliders[7].value
    ω2_obs = sg.sliders[8].value
    ϕ2_obs = sg.sliders[9].value

    # 4. Bind Observables to the Model
    # This automatically updates `prony_line` whenever a slider changes
    prony_line = lift(C0_obs, A1_obs, σ1_obs, ω1_obs, ϕ1_obs, A2_obs, σ2_obs, ω2_obs, ϕ2_obs) do C0, A1, σ1, ω1, ϕ1, A2, σ2, ω2, ϕ2
        prony_series(t_fit, C0, A1, σ1, ω1, ϕ1, A2, σ2, ω2, ϕ2)
    end

    # 5. Plotting
    # Original Data
    lines!(ax, t_fit, eq_crel, color=:blue, linewidth=2.5, label="Original C_rel")

    # Overlay Interactive Model
    lines!(ax, t_fit, prony_line, color=:red, linewidth=2.5, label="Prony Model")

    axislegend(ax, position=:rt)

    # 6. Display the interactive figure
    display(fig)
end

# Run the routine
interactive_prony_fit()
