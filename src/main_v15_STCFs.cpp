#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include "sim_constants.hpp"
#include "auxiliary.hpp"
#include "observables.hpp"

int main(int argc, char* argv[]) {
    SimParams params;
    get_parameters(argc, argv, params);

    int T_steps = int(params.totaltime / params.dt);
    int N = params.W * params.H;
    int N_total = N * params.num_realizations;

    int T_equilibration = static_cast<int>(params.eqltime/params.dt);
    if (T_equilibration % params.obs_freq != 0) {
        T_equilibration = (T_equilibration / params.obs_freq) * params.obs_freq;
        std::cout << "Adjusted Equilibration Time to align with obs_freq (New step: " << T_equilibration << ")\n";
    }

    int num_records = (T_steps + params.obs_freq - 1) / params.obs_freq;

    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Target energy: H =  " << params.target_E << std::endl;
    std::cout << "Lattice Width:  W = " << params.W << ";\tLattice Height:  H = " << params.H << std::endl;
    std::cout << "Total Time: " << params.totaltime << "\t Equilibration Time: " <<params.eqltime  << "(time steps: " << T_equilibration <<")" <<   std::endl;
    std::cout << "Time step size: dt =  " << params.dt << "\t (# of sim steps: " << T_steps <<")" << std::endl;
    std::cout << "Observation Freq: " << params.obs_freq << " steps\t (Total records: " << num_records << ")" << std::endl;
    std::cout << "Realizations (Batched): " << params.num_realizations << " (Total threads: " << N_total << ")" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    generate_lattice(params.W, params.H);

    std::vector<double> posX_eq(N, 0.0), posY_eq(N, 0.0);
    std::vector<int> neighbors(N * 3, -1);
    load_lattice(params.W, params.H, posX_eq, posY_eq, neighbors);

    std::vector<double> uX(N_total, 0.0), uY(N_total, 0.0);
    std::vector<double> pX(N_total, 0.0), pY(N_total, 0.0);

    double Lx_eq = params.W * (std::sqrt(3.0) / 2.0) * 1.0;
    double Ly_eq = params.H * 1.5 * 1.0;

    std::vector<double> ideal_dx(N * 3, 0.0);
    std::vector<double> ideal_dy(N * 3, 0.0);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            int n_idx = neighbors[k * N + i];

            double dx = posX_eq[i] - posX_eq[n_idx];
            double dy = posY_eq[i] - posY_eq[n_idx];

            dx -= Lx_eq * std::round(dx / Lx_eq);
            dy -= Ly_eq * std::round(dy / Ly_eq);

            double r = std::sqrt(dx * dx + dy * dy);
            if (r > 0) {
                ideal_dx[k * N + i] = (dx / r) * PHYS_R0;
                ideal_dy[k * N + i] = (dy / r) * PHYS_R0;
            }
        }
    }

    std::vector<int> rev_idx(N * 3, -1);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            int n_idx = neighbors[k * N + i];

            for (int m = 0; m < 3; ++m) {
                if (neighbors[m * N + n_idx] == i) {
                    rev_idx[k * N + i] = m;
                    break;
                }
            }
        }
    }

    if (params.target_E > 0.0) {
            double tol = 1e-6;
            int max_iter = 100;

            double com_x = 0.0, com_y = 0.0;
            for (int i = 0; i < N; ++i) {
                com_x += posX_eq[i];
                com_y += posY_eq[i];
            }
            com_x /= N; com_y /= N;

            for (int r = 0; r < params.num_realizations; ++r) {
                int r_offset = r * N;
                std::mt19937 gen(params.seed + r);
                std::normal_distribution<double> dist(0.0, 1.0);

                std::vector<double> base_uX(N, 0.0), base_uY(N, 0.0);
                double sum_ux = 0.0, sum_uy = 0.0;

                for (int i = 0; i < N; ++i) {
                    base_uX[i] = dist(gen);
                    base_uY[i] = dist(gen);
                    sum_ux += base_uX[i];
                    sum_uy += base_uY[i];
                }

                for (int i = 0; i < N; ++i) {
                    base_uX[i] -= sum_ux / N;
                    base_uY[i] -= sum_uy / N;
                }

                double Lz = 0.0;
                double I = 0.0;
                for (int i = 0; i < N; ++i) {
                    double rx = posX_eq[i] - com_x;
                    double ry = posY_eq[i] - com_y;
                    Lz += rx * base_uY[i] - ry * base_uX[i];
                    I += rx * rx + ry * ry;
                }
                double d_theta = Lz / I;
                for (int i = 0; i < N; ++i) {
                    double rx = posX_eq[i] - com_x;
                    double ry = posY_eq[i] - com_y;
                    base_uX[i] -= (-d_theta * ry);
                    base_uY[i] -= ( d_theta * rx);
                }

                std::vector<double> test_uX(N), test_uY(N);
                auto eval_E = [&](double alpha) {
                    for(int i = 0; i < N; ++i) {
                        test_uX[i] = alpha * base_uX[i];
                        test_uY[i] = alpha * base_uY[i];
                    }
                    return compute_potential_energy_host(N, test_uX, test_uY, neighbors, ideal_dx, ideal_dy);
                };

                double alpha_min = 0.0;
                double alpha_max = 0.001;
                double E_max = eval_E(alpha_max);

                while (E_max < params.target_E) {
                    alpha_max *= 2.0;
                    E_max = eval_E(alpha_max);
                    if (alpha_max > 1e4) {
                        std::cerr << "Warning: Bisection bracket expansion exceeded sensible limits.\n";
                        break;
                    }
                }

                double alpha_mid = 0.0;
                for (int iter = 0; iter < max_iter; ++iter) {
                    alpha_mid = 0.5 * (alpha_min + alpha_max);
                    double E_mid = eval_E(alpha_mid);

                    if (std::abs(E_mid - params.target_E) < tol) {
                        break;
                    }

                    if (E_mid < params.target_E) {
                        alpha_min = alpha_mid;
                    } else {
                        alpha_max = alpha_mid;
                    }
                }

                for (int i = 0; i < N; ++i) {
                    uX[r_offset + i] = alpha_mid * base_uX[i];
                    uY[r_offset + i] = alpha_mid * base_uY[i];
                    pX[r_offset + i] = 0.0;
                    pY[r_offset + i] = 0.0;
                }
            }
        }

    double *d_uX, *d_uY, *d_uX_ref, *d_uY_ref, *d_pX, *d_pY, *d_pX_ref, *d_pY_ref;
    double *d_ideal_dx, *d_ideal_dy, *d_E_ref;
    double *d_obs_T_kin, *d_obs_V_M, *d_obs_V_A, *d_obs_u_corr;
    double *d_obs_rel_u_dot_u0, *d_obs_E_corr, *d_obs_v_corr, *d_obs_len_t;
    double *d_u_site, *d_v_site, *d_E_site;
    int *d_neighbors, *d_rev_idx;

    size_t obs_size = num_records * params.num_realizations * sizeof(double);

    cudaMalloc(&d_uX, N_total * sizeof(double));
    cudaMalloc(&d_uY, N_total * sizeof(double));
    cudaMalloc(&d_uX_ref, N_total * sizeof(double));
    cudaMalloc(&d_uY_ref, N_total * sizeof(double));
    cudaMalloc(&d_pX, N_total * sizeof(double));
    cudaMalloc(&d_pY, N_total * sizeof(double));
    cudaMalloc(&d_pX_ref, N_total * sizeof(double));
    cudaMalloc(&d_pY_ref, N_total * sizeof(double));
    cudaMalloc(&d_E_ref, N_total * sizeof(double));

    cudaMalloc(&d_ideal_dx, N * 3 * sizeof(double));
    cudaMalloc(&d_ideal_dy, N * 3 * sizeof(double));
    cudaMalloc(&d_neighbors, N * 3 * sizeof(int));
    cudaMalloc(&d_rev_idx, N * 3 * sizeof(int));

    cudaMalloc(&d_u_site, N_total * sizeof(double));
    cudaMalloc(&d_v_site, N_total * sizeof(double));
    cudaMalloc(&d_E_site, N_total * sizeof(double));

    cudaMalloc(&d_obs_T_kin, obs_size);
    cudaMalloc(&d_obs_V_M, obs_size);
    cudaMalloc(&d_obs_V_A, obs_size);
    cudaMalloc(&d_obs_u_corr, obs_size);
    cudaMalloc(&d_obs_rel_u_dot_u0, obs_size);
    cudaMalloc(&d_obs_E_corr, obs_size);
    cudaMalloc(&d_obs_v_corr, obs_size);
    cudaMalloc(&d_obs_len_t, obs_size);

    cudaMemset(d_uX_ref, 0, N_total * sizeof(double));
    cudaMemset(d_uY_ref, 0, N_total * sizeof(double));
    cudaMemset(d_pX_ref, 0, N_total * sizeof(double));
    cudaMemset(d_pY_ref, 0, N_total * sizeof(double));
    cudaMemset(d_E_ref, 0, N_total * sizeof(double));

    cudaMemset(d_obs_T_kin, 0, obs_size);
    cudaMemset(d_obs_V_M, 0, obs_size);
    cudaMemset(d_obs_V_A, 0, obs_size);
    cudaMemset(d_obs_u_corr, 0, obs_size);
    cudaMemset(d_obs_rel_u_dot_u0, 0, obs_size);
    cudaMemset(d_obs_E_corr, 0, obs_size);
    cudaMemset(d_obs_v_corr, 0, obs_size);
    cudaMemset(d_obs_len_t, 0, obs_size);

    cudaMemcpy(d_uX, uX.data(), N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uY, uY.data(), N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pX, pX.data(), N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pY, pY.data(), N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ideal_dx, ideal_dx.data(), N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ideal_dy, ideal_dy.data(), N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors.data(), N * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rev_idx, rev_idx.data(), N * 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid1D = (N_total + threadsPerBlock - 1) / threadsPerBlock;

    dim3 obsThreads(256);
    dim3 obsBlocks((N + obsThreads.x - 1) / obsThreads.x, params.num_realizations);

    if (T_steps <= T_equilibration) {
        std::cerr << "Error: Total simulation steps must be greater than equilibration steps.\n";
        return 1;
    }

    size_t host_obs_count = num_records * params.num_realizations;
    std::vector<double> h_obs_T_kin(host_obs_count, 0.0);
    std::vector<double> h_obs_V_M(host_obs_count, 0.0);
    std::vector<double> h_obs_V_A(host_obs_count, 0.0);
    std::vector<double> h_obs_u_corr(host_obs_count, 0.0);
    std::vector<double> h_obs_rel_u_dot_u0(host_obs_count, 0.0);
    std::vector<double> h_obs_E_corr(host_obs_count, 0.0);
    std::vector<double> h_obs_v_corr(host_obs_count, 0.0);
    std::vector<double> h_obs_len_t(host_obs_count, 0.0);

    std::map<std::string, std::vector<double>> obs_hist;
    obs_hist["temperature"].assign(host_obs_count, 0.0);
    obs_hist["relative_error"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_D"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_rel"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_E"].assign(host_obs_count, 0.0);
    obs_hist["acf_C_v"].assign(host_obs_count, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now();

    compute_observables(N, params.num_realizations, obsBlocks, obsThreads, 0,
        d_uX, d_uY, d_uX_ref, d_uY_ref, d_pX, d_pY, d_pX_ref, d_pY_ref,
        d_neighbors, d_ideal_dx, d_ideal_dy,
        d_obs_T_kin, d_obs_V_M, d_obs_V_A, d_obs_u_corr, d_obs_E_corr,
        d_obs_rel_u_dot_u0, d_obs_v_corr, d_obs_len_t, d_E_ref, false,
        d_u_site, d_v_site, d_E_site);

    double dt = params.dt;

    std::vector<double> h_H0(params.num_realizations, 0.0);
    std::vector<double> mean_E(params.num_realizations, 0.0);
    std::vector<double> mean_E_bond(params.num_realizations, 0.0);

    for (int t = 0; t < T_steps - 1; ++t) {

        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a1 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b1 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a2 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b2 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a3 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b3 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a4 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b4 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a4 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b3 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a3 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b2 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a2 * dt, N_total, d_uX, d_uY, d_pX, d_pY);
        eLB_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_b1 * dt, N_total, N, d_uX, d_uY, d_pX, d_pY, d_neighbors, d_rev_idx, d_ideal_dx, d_ideal_dy);
        eLA_kernel<<<blocksPerGrid1D, threadsPerBlock>>>(ABA_a1 * dt, N_total, d_uX, d_uY, d_pX, d_pY);

        int next_t = t + 1;

        bool capture_ref = (next_t == T_equilibration);
        if (capture_ref) {
            cudaMemcpy(d_uX_ref, d_uX, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_uY_ref, d_uY, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pX_ref, d_pX, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pY_ref, d_pY, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        if (next_t % params.obs_freq == 0) {
            int record_idx = next_t / params.obs_freq;
            compute_observables(N, params.num_realizations, obsBlocks, obsThreads, record_idx,
                d_uX, d_uY, d_uX_ref, d_uY_ref, d_pX, d_pY, d_pX_ref, d_pY_ref,
                d_neighbors, d_ideal_dx, d_ideal_dy,
                d_obs_T_kin, d_obs_V_M, d_obs_V_A, d_obs_u_corr, d_obs_E_corr,
                d_obs_rel_u_dot_u0, d_obs_v_corr, d_obs_len_t, d_E_ref, capture_ref,
                d_u_site, d_v_site, d_E_site);

            if (capture_ref) {
                std::vector<double> h_E_ref(N_total);
                cudaMemcpy(h_E_ref.data(), d_E_ref, N_total * sizeof(double), cudaMemcpyDeviceToHost);

                for (int r = 0; r < params.num_realizations; ++r) {
                    int r_offset = r * N;
                    double sum_bond_E = 0.0;
                    for (int i = 0; i < N; ++i) {
                        double E_i = h_E_ref[r_offset + i];
                        for (int k = 0; k < 3; ++k) {
                            int n_local = neighbors[k * N + i];
                            sum_bond_E += E_i * h_E_ref[r_offset + n_local];
                        }
                    }
                    mean_E_bond[r] = sum_bond_E / (3.0 * N);
                }
            }
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_obs_T_kin.data(), d_obs_T_kin, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_V_M.data(), d_obs_V_M, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_V_A.data(), d_obs_V_A, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_u_corr.data(), d_obs_u_corr, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_rel_u_dot_u0.data(), d_obs_rel_u_dot_u0, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_E_corr.data(), d_obs_E_corr, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_v_corr.data(), d_obs_v_corr, obs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs_len_t.data(), d_obs_len_t, obs_size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < params.num_realizations; ++r) {
        h_H0[r] = h_obs_T_kin[r] + h_obs_V_M[r] + h_obs_V_A[r];
        mean_E[r] = h_H0[r] / N;
    }

    int eq_idx = T_equilibration / params.obs_freq;

    // 1. Construct the raw, unnormalized covariance time series
    for (int record_idx = 0; record_idx < num_records; ++record_idx) {
        for (int r = 0; r < params.num_realizations; ++r) {
            int idx = record_idx * params.num_realizations + r;
            int actual_t = record_idx * params.obs_freq;

            double H_t = h_obs_T_kin[idx] + h_obs_V_M[idx] + h_obs_V_A[idx];
            obs_hist["temperature"][idx] = (h_obs_T_kin[idx] / (N * PHYS_K_B));
            obs_hist["relative_error"][idx] = std::abs(H_t - h_H0[r]) / h_H0[r];

            if (actual_t >= T_equilibration) {
                obs_hist["acf_C_D"][idx] = (h_obs_u_corr[idx] / (3.0 * N));
                obs_hist["acf_C_v"][idx] = (h_obs_v_corr[idx] / (3.0 * N));

                obs_hist["acf_C_E"][idx] = (h_obs_E_corr[idx] / (3.0 * N)) - (mean_E_bond[r] * mean_E_bond[r]);

                double mean_len = h_obs_len_t[idx] / (1.5 * N);
                obs_hist["acf_C_rel"][idx] = (h_obs_rel_u_dot_u0[idx] / (1.5 * N)) - (mean_len * mean_len);
            }
        }
    }

    // 2. Extract the variance (the values at t0) to serve as denominators
    std::vector<double> norm_C_D(params.num_realizations, 0.0);
    std::vector<double> norm_C_v(params.num_realizations, 0.0);
    std::vector<double> norm_C_E(params.num_realizations, 0.0);
    std::vector<double> norm_C_rel(params.num_realizations, 0.0);

    for (int r = 0; r < params.num_realizations; ++r) {
        int ref_idx = eq_idx * params.num_realizations + r;
        norm_C_D[r] = obs_hist["acf_C_D"][ref_idx];
        norm_C_v[r] = obs_hist["acf_C_v"][ref_idx];
        norm_C_E[r] = obs_hist["acf_C_E"][ref_idx];
        norm_C_rel[r] = obs_hist["acf_C_rel"][ref_idx];
    }

    // 3. Execute the normalization division across the entire time series
    for (int record_idx = eq_idx; record_idx < num_records; ++record_idx) {
        for (int r = 0; r < params.num_realizations; ++r) {
            int idx = record_idx * params.num_realizations + r;

            obs_hist["acf_C_D"][idx] = (norm_C_D[r] != 0.0) ? (obs_hist["acf_C_D"][idx] / norm_C_D[r]) : 0.0;
            obs_hist["acf_C_v"][idx] = (norm_C_v[r] != 0.0) ? (obs_hist["acf_C_v"][idx] / norm_C_v[r]) : 0.0;
            obs_hist["acf_C_E"][idx] = (norm_C_E[r] != 0.0) ? (obs_hist["acf_C_E"][idx] / norm_C_E[r]) : 0.0;
            obs_hist["acf_C_rel"][idx] = (norm_C_rel[r] != 0.0) ? (obs_hist["acf_C_rel"][idx] / norm_C_rel[r]) : 0.0;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Simulation computation time: " << diff.count() << " seconds.\n";

    save_observables(obs_hist, params);
    std::cout << "Observables saved to ../data/ directory.\n";

    cudaFree(d_u_site); cudaFree(d_v_site); cudaFree(d_E_site);
    cudaFree(d_uX); cudaFree(d_uY);
    cudaFree(d_uX_ref); cudaFree(d_uY_ref);
    cudaFree(d_pX); cudaFree(d_pY);
    cudaFree(d_pX_ref); cudaFree(d_pY_ref);
    cudaFree(d_neighbors); cudaFree(d_rev_idx);
    cudaFree(d_ideal_dx); cudaFree(d_ideal_dy);
    cudaFree(d_E_ref);
    cudaFree(d_obs_T_kin);
    cudaFree(d_obs_V_M);
    cudaFree(d_obs_V_A);
    cudaFree(d_obs_u_corr);
    cudaFree(d_obs_rel_u_dot_u0);
    cudaFree(d_obs_E_corr);
    cudaFree(d_obs_v_corr);
    cudaFree(d_obs_len_t);

    return 0;
}
