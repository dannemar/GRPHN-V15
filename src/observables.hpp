#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "sim_constants.hpp"
#include <format>

__global__ void calc_site_observables_kernel(int N, int num_realizations, int t,
                        const double* uX, const double* uY,
                        const double* uX_ref, const double* uY_ref,
                        const double* pX, const double* pY,
                        const double* pX_ref, const double* pY_ref,
                        const int* neighbors,
                        const double* ideal_dx, const double* ideal_dy,
                        double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                        double* d_obs_rel_u_dot_u0, double* d_obs_len_t,
                        double* d_E_ref, bool capture_ref,
                        double* d_u_site, double* d_v_site, double* d_E_site) {

    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (r >= num_realizations) return;

    int global_id = r * N + i;
    int r_offset = r * N;

    __shared__ double s_T_kin[256];
    __shared__ double s_V_M[256];
    __shared__ double s_V_A[256];
    __shared__ double s_rel_u_dot_u0[256];
    __shared__ double s_len_t[256];

    s_T_kin[tid] = 0.0;
    s_V_M[tid] = 0.0;
    s_V_A[tid] = 0.0;
    s_rel_u_dot_u0[tid] = 0.0;
    s_len_t[tid] = 0.0;

    if (i < N) {
        double T_kin = (pX[global_id] * pX[global_id] + pY[global_id] * pY[global_id]) / (2.0 * PHYS_M_C);
        double V_M = 0.0;
        double local_V_M = 0.0;
        double V_A = 0.0;

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            int n_global = r_offset + n_local;
            double dx = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
            double dy = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);
            double r_dist = sqrt(dx * dx + dy * dy);
            double exp_term = exp(-PHYS_A * (r_dist - PHYS_R0));
            double bond_E = PHYS_D * (exp_term - 1.0) * (exp_term - 1.0);

            if (i < n_local) {
                V_M += bond_E;
            }
            local_V_M += 0.5 * bond_E;
        }

        auto get_angle_energy = [&](int l1_local, int l1_k, int l2_local, int l2_k) {
            int l1_global = r_offset + l1_local;
            int l2_global = r_offset + l2_local;

            double dx1 = -ideal_dx[l1_k * N + i] + (uX[l1_global] - uX[global_id]);
            double dy1 = -ideal_dy[l1_k * N + i] + (uY[l1_global] - uY[global_id]);
            double r1 = sqrt(dx1*dx1 + dy1*dy1);
            double dx2 = -ideal_dx[l2_k * N + i] + (uX[l2_global] - uX[global_id]);
            double dy2 = -ideal_dy[l2_k * N + i] + (uY[l2_global] - uY[global_id]);
            double r2 = sqrt(dx2*dx2 + dy2*dy2);

            if (r1 == 0.0 || r2 == 0.0) return 0.0;
            double cos_phi = (dx1*dx2 + dy1*dy2) / (r1 * r2);
            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
            double phi = acos(cos_phi);
            double dphi = phi - PHYS_PHI0;

            return 0.5 * PHYS_D_ANG * dphi * dphi - (1.0 / 3.0) * PHYS_D_PRIME * dphi * dphi * dphi;
        };

        int n0 = neighbors[0 * N + i];
        int n1 = neighbors[1 * N + i];
        int n2 = neighbors[2 * N + i];

        V_A += get_angle_energy(n0, 0, n1, 1);
        V_A += get_angle_energy(n1, 1, n2, 2);
        V_A += get_angle_energy(n2, 2, n0, 0);

        double local_E = T_kin + local_V_M + V_A;

        if (capture_ref) {
            d_E_ref[global_id] = local_E;
        }

        // Store intermediate site values for the spatial correlation pass
        d_u_site[global_id] = (uX[global_id] * uX_ref[global_id]) + (uY[global_id] * uY_ref[global_id]);
        d_v_site[global_id] = (pX[global_id] * pX_ref[global_id] + pY[global_id] * pY_ref[global_id]) / (PHYS_M_C * PHYS_M_C);
        d_E_site[global_id] = local_E * d_E_ref[global_id];

        double local_rel_acf = 0.0;
        double local_len_t = 0.0;

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            if (i < n_local) {
                int n_global = r_offset + n_local;

                double rx_t = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
                double ry_t = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);

                double rx_ref = ideal_dx[k * N + i] + (uX_ref[global_id] - uX_ref[n_global]);
                double ry_ref = ideal_dy[k * N + i] + (uY_ref[global_id] - uY_ref[n_global]);

                double len_t = sqrt(rx_t * rx_t + ry_t * ry_t);
                double len_ref = sqrt(rx_ref * rx_ref + ry_ref * ry_ref);

                local_rel_acf += len_t * len_ref;
                local_len_t += len_t;
            }
        }

        s_rel_u_dot_u0[tid] = local_rel_acf;
        s_len_t[tid] = local_len_t;

        s_T_kin[tid] = T_kin;
        s_V_M[tid] = V_M;
        s_V_A[tid] = V_A;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_T_kin[tid] += s_T_kin[tid + s];
            s_V_M[tid] += s_V_M[tid + s];
            s_V_A[tid] += s_V_A[tid + s];
            s_rel_u_dot_u0[tid] += s_rel_u_dot_u0[tid + s];
            s_len_t[tid] += s_len_t[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int obs_index = t * num_realizations + r;
        atomicAdd(&d_obs_T_kin[obs_index], s_T_kin[0]);
        atomicAdd(&d_obs_V_M[obs_index], s_V_M[0]);
        atomicAdd(&d_obs_V_A[obs_index], s_V_A[0]);
        atomicAdd(&d_obs_rel_u_dot_u0[obs_index], s_rel_u_dot_u0[0]);
        atomicAdd(&d_obs_len_t[obs_index], s_len_t[0]);
    }
}

__global__ void calc_spatial_corr_kernel(int N, int num_realizations, int t,
                        const double* d_u_site, const double* d_v_site, const double* d_E_site,
                        const int* neighbors,
                        double* d_obs_u_corr, double* d_obs_v_corr, double* d_obs_E_corr) {
    int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (r >= num_realizations) return;

    int global_id = r * N + i;
    int r_offset = r * N;

    __shared__ double s_u_corr[256];
    __shared__ double s_v_corr[256];
    __shared__ double s_E_corr[256];

    s_u_corr[tid] = 0.0;
    s_v_corr[tid] = 0.0;
    s_E_corr[tid] = 0.0;

    if (i < N) {
        double my_u = d_u_site[global_id];
        double my_v = d_v_site[global_id];
        double my_E = d_E_site[global_id];

        double local_u = 0.0;
        double local_v = 0.0;
        double local_E = 0.0;

        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            int n_global = r_offset + n_local;

            local_u += my_u * d_u_site[n_global];
            local_v += my_v * d_v_site[n_global];
            local_E += my_E * d_E_site[n_global];
        }

        s_u_corr[tid] = local_u;
        s_v_corr[tid] = local_v;
        s_E_corr[tid] = local_E;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_u_corr[tid] += s_u_corr[tid + s];
            s_v_corr[tid] += s_v_corr[tid + s];
            s_E_corr[tid] += s_E_corr[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int obs_index = t * num_realizations + r;
        atomicAdd(&d_obs_u_corr[obs_index], s_u_corr[0]);
        atomicAdd(&d_obs_v_corr[obs_index], s_v_corr[0]);
        atomicAdd(&d_obs_E_corr[obs_index], s_E_corr[0]);
    }
}

inline void compute_observables(int N, int num_realizations, dim3 blocksPerGrid, dim3 threadsPerBlock, int t,
                         const double* d_uX, const double* d_uY,
                         const double* d_uX_ref, const double* d_uY_ref,
                         const double* d_pX, const double* d_pY,
                         const double* d_pX_ref, const double* d_pY_ref,
                         const int* d_neighbors,
                         const double* d_ideal_dx, const double* d_ideal_dy,
                         double* d_obs_T_kin, double* d_obs_V_M, double* d_obs_V_A,
                         double* d_obs_u_corr, double* d_obs_E_corr,
                         double* d_obs_rel_u_dot_u0,
                         double* d_obs_v_corr, double* d_obs_len_t,
                         double* d_E_ref, bool capture_ref,
                         double* d_u_site, double* d_v_site, double* d_E_site) {

    calc_site_observables_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, num_realizations, t,
        d_uX, d_uY, d_uX_ref, d_uY_ref, d_pX, d_pY, d_pX_ref, d_pY_ref,
        d_neighbors, d_ideal_dx, d_ideal_dy,
        d_obs_T_kin, d_obs_V_M, d_obs_V_A,
        d_obs_rel_u_dot_u0, d_obs_len_t, d_E_ref, capture_ref,
        d_u_site, d_v_site, d_E_site);

    calc_spatial_corr_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, num_realizations, t,
        d_u_site, d_v_site, d_E_site,
        d_neighbors,
        d_obs_u_corr, d_obs_v_corr, d_obs_E_corr);
}

inline void save_observables(const std::map<std::string, std::vector<double>>& obs_histories, SimParams params) {
    for (const auto& pair : obs_histories) {
        std::string filename = std::format("../data/{}_w-{}_h-{}_H-{:.2f}_s-{}_tt-{}_eqt-{:.3f}_dt-{:.3f}_r-{}_of-{}.bin",
            pair.first, params.W, params.H, params.target_E,
            params.seed, params.totaltime,params.eqltime , params.dt, params.num_realizations, params.obs_freq);

        std::ofstream out_file(filename, std::ios::binary);
        if (out_file) {
            out_file.write(reinterpret_cast<const char*>(pair.second.data()), pair.second.size() * sizeof(double));
            out_file.close();
        } else {
            std::cerr << "Error: Could not open " << filename << " for writing.\n";
        }
    }
}
