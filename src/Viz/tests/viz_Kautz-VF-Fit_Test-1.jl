using GLMakie
using Statistics
using LinearAlgebra

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
# Time-Domain Pole Extraction (ERA / Kung's Method)
# ==========================================
function extract_poles(y::AbstractVector{Float64}, dt::Float64, M::Int)
    N = length(y)
    L = div(N, 2)

    H0 = zeros(Float64, N - L, L)
    H1 = zeros(Float64, N - L, L)

    for i in 1:(N - L)
        for j in 1:L
            H0[i, j] = y[i + j - 1]
            H1[i, j] = y[i + j]
        end
    end

    F = svd(H0)

    M_eff = min(M, length(F.S))
    U_trunc = F.U[:, 1:M_eff]
    S_trunc = Diagonal(F.S[1:M_eff])
    V_trunc = F.V[:, 1:M_eff]

    A = inv(S_trunc) * U_trunc' * H1 * V_trunc
    z = eigvals(A)
    s = log.(Complex.(z)) ./ dt

    return s
end

# ==========================================
# Main Execution Routine
# ==========================================
function main()
    # 1. Configuration and Data Loading
    data_dir = "../../../data/"
    w, h, energy = 256, 256, 500
    s_seed, tt, eqt_str, dt_str, R, of = 41, 600, "500.000", "0.010", 200, 1

    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    strEnd = "_w-$(w)_h-$(h)_H-$(energy).00_s-$(s_seed)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"
    file_path = joinpath(data_dir, "acf_C_rel" * strEnd * ".bin")

    mean_crel, _, _, T_steps = process_observable(file_path, R)
    t_phy = (1:T_steps) .* dt

    t_eq = t_phy[eq_idx:end]
    t_fit = t_eq .- t_eq[1]
    eq_crel = mean_crel[eq_idx:end]

    # 2. Time-Domain Pole Extraction
    N_window = min(1000, length(eq_crel))
    transient_data = eq_crel[1:N_window] .- mean(eq_crel[1:N_window])

    println("Running Time-Domain ERA for Pole Extraction...")
    extracted_poles = extract_poles(transient_data, dt, 12)

    # 3. Extract continuous-time Kautz parameters
    stable_modes = filter(p -> real(p) < 0.0 && imag(p) > 1e-5, extracted_poles)
    sort!(stable_modes, by = p -> abs(real(p)))
    kautz_poles = stable_modes[1:min(3, length(stable_modes))]

    println("\n--- Extracted 3-Term Kautz Poles ---")
    for (i, p) in enumerate(kautz_poles)
        println("Mode $i: α (Decay) = $(round(-real(p), digits=4)), ω (Freq) = $(round(imag(p), digits=4)) rad/s")
    end

    # 4. Construct the Generalized Kautz Basis in Time Domain
    N_c = 2 * length(kautz_poles) + 1
    H_mat = zeros(Float64, length(t_fit), N_c)
    H_mat[:, 1] .= 1.0

    col_idx = 2
    for p in kautz_poles
        α = -real(p)
        ω = imag(p)
        H_mat[:, col_idx]     = exp.(-α .* t_fit) .* cos.(ω .* t_fit)
        H_mat[:, col_idx + 1] = exp.(-α .* t_fit) .* sin.(ω .* t_fit)
        col_idx += 2
    end

    # 5. Constrained Least Squares Formulation (KKT System)
    # Constraints: 1) C(0) = eq_crel[1], 2) C'(0) = 0.0
    A_eq = zeros(Float64, 2, N_c)
    b_eq = [eq_crel[1], 0.0]

    # Constraint 1: Match initial value exactly
    A_eq[1, :] .= H_mat[1, :]

    # Constraint 2: Force initial derivative to zero (time-reversal symmetry)
    A_eq[2, 1] = 0.0 # Derivative of constant offset is 0
    col_idx = 2
    for p in kautz_poles
        α = -real(p)
        ω = imag(p)
        A_eq[2, col_idx]     = -α # d/dt [exp(-αt)cos(ωt)] at t=0
        A_eq[2, col_idx + 1] =  ω # d/dt [exp(-αt)sin(ωt)] at t=0
        col_idx += 2
    end

    # Construct the block KKT matrix: [H^T H, A_eq^T ; A_eq, 0]
    KKT_matrix = [H_mat' * H_mat   A_eq';
                  A_eq             zeros(Float64, 2, 2)]
    KKT_rhs = [H_mat' * eq_crel; b_eq]

    # Solve constrained system
    sol = KKT_matrix \ KKT_rhs
    coefficients = sol[1:N_c] # Extract parameters, discard Lagrange multipliers

    println("\n--- Extracted Linear Coefficients ---")
    println("C0 (Offset): ", round(coefficients[1], digits=5))
    col_idx = 2
    for i in 1:length(kautz_poles)
        println("Mode $i: c$(col_idx-1) (cos) = ", round(coefficients[col_idx], digits=5),
                ", c$(col_idx) (sin) = ", round(coefficients[col_idx+1], digits=5))
        col_idx += 2
    end

    # Generate the model curve
    fit_curve = H_mat * coefficients

    # 6. Plotting Results
    fig = Figure(size=(1200, 800), fontsize=20)
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="C_rel(t)", title="3-Term Constrained Kautz Approximation")

    lines!(ax, t_fit, eq_crel, color=:blue, linewidth=2.5, label="Original ACF")
    lines!(ax, t_fit, fit_curve, color=:red, linewidth=2.5, linestyle=:dash, label="Constrained ERA Model")

    axislegend(ax, position=:rt)
    display(fig)
end

main()
