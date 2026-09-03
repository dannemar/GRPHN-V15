using GLMakie
using Statistics

# ==========================================
# Data Processing Function
# ==========================================
function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path))) #[cite: 5]
    T_steps = div(length(raw_data), R) #[cite: 5]
    mat = reshape(raw_data, (R, T_steps)) #[cite: 5]

    mean_vals = vec(mean(mat, dims=1)) #[cite: 5]
    std_vals = vec(std(mat, dims=1)) #[cite: 5]
    sem_vals = std_vals ./ sqrt(R) #[cite: 5]

    return mean_vals, std_vals, sem_vals, T_steps #[cite: 5]
end

# ==========================================
# Generalized Multi-Pole Model (Two Modes)
# ==========================================
function multi_pole_model(t, C0, α1, ω1, c1, c2, α2, ω2, c3, c4)
    # First dominant mode
    term1 = exp.(-α1 .* t) .* (c1 .* cos.(ω1 .* t) .+ c2 .* sin.(ω1 .* t))

    # Second dominant mode
    term2 = exp.(-α2 .* t) .* (c3 .* cos.(ω2 .* t) .+ c4 .* sin.(ω2 .* t))

    return C0 .+ term1 .+ term2
end

# ==========================================
# Main Interactive Routine
# ==========================================
function interactive_multipole_fit()
    # 1. Configuration and Data Loading
    data_dir = "../../../data/" #[cite: 5]
    w, h, energy = 256, 256, 500 #[cite: 5]
    s, tt, eqt_str, dt_str, R, of = 41, 600, "500.000", "0.010", 200, 1 #[cite: 5]

    dt = parse(Float64, dt_str) #[cite: 5]
    eqt = parse(Float64, eqt_str) #[cite: 5]
    eq_idx = Int(ceil(eqt / dt)) + 1 #[cite: 5]

    strEnd = "_w-$(w)_h-$(h)_H-$(energy).00_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)" #[cite: 5]
    file_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin") #[cite: 5]

    # Process data
    mean_crel, _, _, T_steps = process_observable(file_path, R) #[cite: 5]
    t_phy = (1:T_steps) .* dt #[cite: 5]

    # Isolate equilibrated data
    t_eq = t_phy[eq_idx:end] #[cite: 5]
    t_fit = t_eq .- t_eq[1] #[cite: 5]
    eq_crel = mean_crel[eq_idx:end] #[cite: 5]
    mean_offset = mean(eq_crel) #[cite: 5]

    # 2. Figure Setup
    fig = Figure(size=(1400, 1100), fontsize=20) #[cite: 5]
    ax = Axis(fig[1, 1], xlabel="Time (shifted to t=0)", ylabel="Observable ACF", title="Interactive Multi-Pole (Generalized Kautz) Fit")

    # 3. Interactive UI Elements (SliderGrid)
    # The starting frequencies (3.39 and 4.02) are based on the MPM extraction from the Prony model.
    sg = SliderGrid(fig[2, 1],
            (label = "Offset (C₀)", range = -0.5:0.001:0.5, startvalue = mean_offset),

            # Mode 1 Parameters
            (label = "Mode 1 Decay (α₁)", range = 0.0:0.001:2.0, startvalue = 0.1),
            (label = "Mode 1 Freq (ω₁)", range = 0.0:0.01:10.0, startvalue = 3.39),
            (label = "Mode 1 Cosine (c₁)", range = -0.5:0.001:1.5, startvalue = 0.074),
            (label = "Mode 1 Sine (c₂)", range = -0.5:0.001:0.5, startvalue = 0.0),

            # Mode 2 Parameters
            (label = "Mode 2 Decay (α₂)", range = 0.0:0.001:2.0, startvalue = 0.1),
            (label = "Mode 2 Freq (ω₂)", range = 0.0:0.01:10.0, startvalue = 4.02),
            (label = "Mode 2 Cosine (c₃)", range = -0.5:0.001:1.5, startvalue = 0.043),
            (label = "Mode 2 Sine (c₄)", range = -0.5:0.001:0.5, startvalue = 0.0)
        )

    # Extract observables from sliders
    C0_obs = sg.sliders[1].value

    α1_obs = sg.sliders[2].value
    ω1_obs = sg.sliders[3].value
    c1_obs = sg.sliders[4].value
    c2_obs = sg.sliders[5].value

    α2_obs = sg.sliders[6].value
    ω2_obs = sg.sliders[7].value
    c3_obs = sg.sliders[8].value
    c4_obs = sg.sliders[9].value

    # 4. Bind Observables to the Model
    fit_line = lift(C0_obs, α1_obs, ω1_obs, c1_obs, c2_obs, α2_obs, ω2_obs, c3_obs, c4_obs) do C0, α1, ω1, c1, c2, α2, ω2, c3, c4
        multi_pole_model(t_fit, C0, α1, ω1, c1, c2, α2, ω2, c3, c4)
    end

    # 5. Plotting
    lines!(ax, t_fit, eq_crel, color=:blue, linewidth=2.5, label="Original ACF Data") #[cite: 5]
    lines!(ax, t_fit, fit_line, color=:red, linewidth=2.5, label="Multi-Pole Fit (2 Modes)")

    axislegend(ax, position=:rt) #[cite: 5]

    # 6. Display the interactive figure
    display(fig) #[cite: 5]
end

# Execute the routine
interactive_multipole_fit()
