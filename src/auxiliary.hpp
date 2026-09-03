#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "sim_constants.hpp"

// --- EVOLUTION KERNELS ---

__global__ void eLA_kernel(double dt_step, int N_total, double* uX, double* uY, const double* pX, const double* pY) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N_total) {
        uX[global_id] += dt_step * (pX[global_id] / PHYS_M_C);
        uY[global_id] += dt_step * (pY[global_id] / PHYS_M_C);
    }
}

__global__ void eLB_kernel(double dt_step, int N_total, int N,
         const double* uX, const double* uY,
         double* pX, double* pY,
         const int* neighbors, const int* rev_idx,
         const double* ideal_dx, const double* ideal_dy) {

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= N_total) return;

    int r = global_id / N;
    int i = global_id % N;
    int r_offset = r * N;

    double Fx = 0.0;
    double Fy = 0.0;

    for (int k = 0; k < 3; ++k) {
        int n_local = neighbors[k * N + i];
        int n_global = r_offset + n_local;
        double dx = ideal_dx[k * N + i] + (uX[global_id] - uX[n_global]);
        double dy = ideal_dy[k * N + i] + (uY[global_id] - uY[n_global]);
        double r_dist = sqrt(dx * dx + dy * dy);

        if (r_dist > 0) {
            double exp_term = exp(-PHYS_A * (r_dist - PHYS_R0));
            double force_mag = 2.0 * PHYS_A * PHYS_D * exp_term * (exp_term - 1.0);
            Fx += force_mag * (dx / r_dist);
            Fy += force_mag * (dy / r_dist);
        }
    }

    auto add_angle_force = [&](int target_local, int l1_local, int l1_k, int v_local, int l2_local, int l2_k) {
        if (l1_local == l2_local) return;

        int l1_global = r_offset + l1_local;
        int v_global  = r_offset + v_local;
        int l2_global = r_offset + l2_local;

        double dx1_id = -ideal_dx[l1_k * N + v_local];
        double dy1_id = -ideal_dy[l1_k * N + v_local];
        double dx2_id = -ideal_dx[l2_k * N + v_local];
        double dy2_id = -ideal_dy[l2_k * N + v_local];

        double dx1 = dx1_id + (uX[l1_global] - uX[v_global]);
        double dy1 = dy1_id + (uY[l1_global] - uY[v_global]);
        double r1 = sqrt(dx1*dx1 + dy1*dy1);

        double dx2 = dx2_id + (uX[l2_global] - uX[v_global]);
        double dy2 = dy2_id + (uY[l2_global] - uY[v_global]);
        double r2 = sqrt(dx2*dx2 + dy2*dy2);

        if (r1 == 0.0 || r2 == 0.0) return;

        double cos_phi = (dx1*dx2 + dy1*dy2) / (r1 * r2);
        cos_phi = fmax(-1.0, fmin(1.0, cos_phi));

        double sin_phi = sqrt(1.0 - cos_phi*cos_phi);
        if (sin_phi < 1e-16) return;

        double phi = acos(cos_phi);
        double dphi = phi - PHYS_PHI0;

        double dV_dphi = PHYS_D_ANG * dphi - PHYS_D_PRIME * dphi * dphi;
        double P = -dV_dphi / sin_phi;

        double u1x = dx1 / r1, u1y = dy1 / r1;
        double u2x = dx2 / r2, u2y = dy2 / r2;

        double grad_l1_x = (u2x - cos_phi * u1x) / r1;
        double grad_l1_y = (u2y - cos_phi * u1y) / r1;

        double grad_l2_x = (u1x - cos_phi * u2x) / r2;
        double grad_l2_y = (u1y - cos_phi * u2y) / r2;

        if (target_local == l1_local) {
            Fx += -P * grad_l1_x;
            Fy += -P * grad_l1_y;
        } else if (target_local == l2_local) {
            Fx += -P * grad_l2_x;
            Fy += -P * grad_l2_y;
        } else if (target_local == v_local) {
            Fx += P * (grad_l1_x + grad_l2_x);
            Fy += P * (grad_l1_y + grad_l2_y);
        }
    };

    int n0 = neighbors[0 * N + i];
    int n1 = neighbors[1 * N + i];
    int n2 = neighbors[2 * N + i];

    add_angle_force(i, n0, 0, i, n1, 1);
    add_angle_force(i, n1, 1, i, n2, 2);
    add_angle_force(i, n2, 2, i, n0, 0);

    for (int k = 0; k < 3; ++k) {
        int v = neighbors[k * N + i];
        int i_k = rev_idx[k * N + i];
        for (int m = 0; m < 3; ++m) {
            int l2 = neighbors[m * N + v];
            add_angle_force(i, i, i_k, v, l2, m);
        }
    }

    pX[global_id] += dt_step * Fx;
    pY[global_id] += dt_step * Fy;
}

// --- AUXILIARY FUNCTIONS ---

inline void get_parameters(int argc, char* argv[], SimParams& params) {
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-w" && i + 1 < argc) { params.W = std::stoi(argv[++i]); }
            else if (arg == "-h" && i + 1 < argc) { params.H = std::stoi(argv[++i]); }
            else if (arg == "-tt" && i + 1 < argc) { params.totaltime = std::stoi(argv[++i]); }
            else if (arg == "-dt" && i + 1 < argc) { params.dt = std::stod(argv[++i]); }
            else if (arg == "-e" && i + 1 < argc) { params.target_E = std::stod(argv[++i]); }
            else if (arg == "-s" && i + 1 < argc) { params.seed = std::stoul(argv[++i]); }
            else if (arg == "-m") { params.save_movie = true; }
            else if (arg == "-r" && i + 1 < argc) { params.num_realizations = std::stoi(argv[++i]); }
            else if (arg == "-eqt" && i + 1 < argc) { params.eqltime = std::stod(argv[++i]); }
            else if (arg == "-of" && i + 1 < argc) { params.obs_freq = std::stoi(argv[++i]); }
            else {
                std::cerr << "Usage: " << argv[0] << " [-w width] [-h height] [-e target_energy] [-s seed] [-tt total_time] [-eqt eqaulibration_time] [-dt time_step] [-r number_of_realizations] [-of observation_measure_frequency] [-m save_lattice_sequence]\n";
                exit(1);
            }
        }
    } else {
        std::cout << "--- Enter Simulation Parameters ---\n";
        std::cout << "Lattice Width (W) [default 60]: ";
        if (std::cin.peek() != '\n') std::cin >> params.W; else std::cin.ignore();

        std::cout << "Lattice Height (H) [default 30]: ";
        if (std::cin.peek() != '\n') std::cin >> params.H; else std::cin.ignore();

        std::cout << "Target Energy [default 100.5]: ";
        if (std::cin.peek() != '\n') std::cin >> params.target_E; else std::cin.ignore();

        std::cout << "Total Time [default 1000]: ";
        if (std::cin.peek() != '\n') std::cin >> params.totaltime; else std::cin.ignore();

        std::cout << "Time step (dt) [default 0.01]: ";
        if (std::cin.peek() != '\n') std::cin >> params.dt; else std::cin.ignore();

        std::cout << "Number of realizations [default 1]: ";
        if (std::cin.peek() != '\n') std::cin >> params.num_realizations; else std::cin.ignore();

        std::cout << "Equilibration time [default 100]: ";
        if (std::cin.peek() != '\n') std::cin >> params.eqltime; else std::cin.ignore();

        std::cout << "Observation Measurement Frequency [default 1]: ";
        if (std::cin.peek() != '\n') std::cin >> params.obs_freq; else std::cin.ignore();
    }
}

inline std::string get_nodes_filename(int W, int H) {
    return "../data/lattice_nodes_" + std::to_string(W) + "_" + std::to_string(H) + ".bin";
}

inline std::string get_links_filename(int W, int H) {
    return "../data/lattice_links_" + std::to_string(W) + "_" + std::to_string(H) + ".bin";
}

inline bool check_lattice_exists(int W, int H) {
    std::ifstream node_file(get_nodes_filename(W, H), std::ios::binary);
    std::ifstream link_file(get_links_filename(W, H), std::ios::binary);
    return node_file.good() && link_file.good();
}

inline int mod(int a, int b) {
    return (a % b + b) % b;
}

inline void generate_lattice(int W, int H) {
    std::cout << "Generating new lattice (W=" << W << ", H=" << H << ")...\n";
    const int N = W * H;
    const double bond_length = 1.0;

    std::vector<double> posX_eq(N, 0.0);
    std::vector<double> posY_eq(N, 0.0);
    std::vector<int> neighbors(N * 3, -1);

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int idx = r * W + c;
            double x = c * bond_length * std::sqrt(3.0) / 2.0;
            double y = (r / 2) * 3.0 * bond_length;

            if (c % 2 == 0) y += (r % 2 == 0 ? 0.0 : 1.0) * bond_length;
            else y += (r % 2 == 0 ? -0.5 : 1.5) * bond_length;

            posX_eq[idx] = x;
            posY_eq[idx] = y;

            int n1 = r * W + mod(c - 1, W);
            int n2 = r * W + mod(c + 1, W);

            int n3 = -1;
            if ((r + c) % 2 == 0) {
                n3 = mod(r + 1, H) * W + c;
            } else {
                n3 = mod(r - 1, H) * W + c;
            }

            neighbors[0 * N + idx] = n1;
            neighbors[1 * N + idx] = n2;
            neighbors[2 * N + idx] = n3;
        }
    }

    std::ofstream node_file(get_nodes_filename(W, H), std::ios::binary);
    node_file.write(reinterpret_cast<const char*>(posX_eq.data()), N * sizeof(double));
    node_file.write(reinterpret_cast<const char*>(posY_eq.data()), N * sizeof(double));
    node_file.close();

    std::ofstream link_file(get_links_filename(W, H), std::ios::binary);
    link_file.write(reinterpret_cast<const char*>(neighbors.data()), N * 3 * sizeof(int));
    link_file.close();

    std::cout << "Successfully generated and saved lattice.\n";
}

inline void load_lattice(int W, int H, std::vector<double>& posX_eq, std::vector<double>& posY_eq, std::vector<int>& neighbors) {
    int N = W * H;
    std::ifstream node_file(get_nodes_filename(W, H), std::ios::binary);
    node_file.read(reinterpret_cast<char*>(posX_eq.data()), N * sizeof(double));
    node_file.read(reinterpret_cast<char*>(posY_eq.data()), N * sizeof(double));
    node_file.close();

    std::ifstream link_file(get_links_filename(W, H), std::ios::binary);
    link_file.read(reinterpret_cast<char*>(neighbors.data()), N * 3 * sizeof(int));
    link_file.close();
}

double compute_potential_energy_host(
    int N,
    const std::vector<double>& uX,
    const std::vector<double>& uY,
    const std::vector<int>& neighbors,
    const std::vector<double>& ideal_dx,
    const std::vector<double>& ideal_dy)
{
    double V_M = 0.0;
    double V_A = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            int n_local = neighbors[k * N + i];
            if (i < n_local) {
                double dx = ideal_dx[k * N + i] + (uX[i] - uX[n_local]);
                double dy = ideal_dy[k * N + i] + (uY[i] - uY[n_local]);
                double r_dist = std::sqrt(dx * dx + dy * dy);
                double exp_term = std::exp(-PHYS_A * (r_dist - PHYS_R0));
                V_M += PHYS_D * (exp_term - 1.0) * (exp_term - 1.0);
            }
        }
    }

    auto get_angle_energy = [&](int i, int l1_local, int l1_k, int l2_local, int l2_k) {
        double dx1 = -ideal_dx[l1_k * N + i] + (uX[l1_local] - uX[i]);
        double dy1 = -ideal_dy[l1_k * N + i] + (uY[l1_local] - uY[i]);
        double r1 = std::sqrt(dx1 * dx1 + dy1 * dy1);

        double dx2 = -ideal_dx[l2_k * N + i] + (uX[l2_local] - uX[i]);
        double dy2 = -ideal_dy[l2_k * N + i] + (uY[l2_local] - uY[i]);
        double r2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

        if (r1 == 0.0 || r2 == 0.0) return 0.0;
        double cos_phi = (dx1 * dx2 + dy1 * dy2) / (r1 * r2);
        cos_phi = std::fmax(-1.0, std::fmin(1.0, cos_phi));
        double phi = std::acos(cos_phi);
        double dphi = phi - PHYS_PHI0;

        return 0.5 * PHYS_D_ANG * dphi * dphi - (1.0 / 3.0) * PHYS_D_PRIME * dphi * dphi * dphi;
    };

    for (int i = 0; i < N; ++i) {
        int n0 = neighbors[0 * N + i];
        int n1 = neighbors[1 * N + i];
        int n2 = neighbors[2 * N + i];
        V_A += get_angle_energy(i, n0, 0, n1, 1);
        V_A += get_angle_energy(i, n1, 1, n2, 2);
        V_A += get_angle_energy(i, n2, 2, n0, 0);
    }

    return V_M + V_A;
}
