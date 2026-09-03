#pragma once

// Physics Constants
#define PHYS_M_C 12.0
#define PHYS_D 5.7
#define PHYS_A 1.96
#define PHYS_R0 1.42
#define PHYS_D_ANG 7.0
#define PHYS_D_PRIME 4.0
#define PHYS_PHI0 2.0943951023931953 // 2.0 * M_PI / 3.0
#define PHYS_K_B 8.617333262e-5

// Forest-Ruth 4th-Order Integration Coefficients
// THETA = 1.0 / (2.0 - cbrt(2.0))
const double FR_THETA = 1.3512071919596577718;

const double FR_c1 = FR_THETA / 2.0;
const double FR_c2 = (1.0 - FR_THETA) / 2.0;
const double FR_c3 = (1.0 - FR_THETA) / 2.0;
const double FR_c4 = FR_THETA / 2.0;

const double FR_d1 = FR_THETA;
const double FR_d2 = 1.0 - 2.0 * FR_THETA;
const double FR_d3 = FR_THETA;
// d4 is mathematically 0, so the 4th momentum kick is omitted.

//CONSTANTS SABA3 (No Corrector)
// SABA_3 Integration Coefficients
// c_1 = c_4 = 1/2 - sqrt(15)/10 // c_2 = c_3 = sqrt(15)/10 // d_1 = d_3 = 5/18 // d_2 = 4/9
const double SABA3_c1 = 0.11270166537925831148;
const double SABA3_c2 = 0.38729833462074168852;
const double SABA3_c3 = 0.38729833462074168852;
const double SABA3_c4 = 0.11270166537925831148;
const double SABA3_d1 = 0.27777777777777777778;
const double SABA3_d2 = 0.44444444444444444444;
const double SABA3_d3 = 0.27777777777777777778;

// ABA864 Coefficients
const double ABA_a1 =  0.071133426498223117777938730006;
const double ABA_a2 =  0.241153427956640098736487795326;
const double ABA_a3 =  0.521411761772814789212136078067;
const double ABA_a4 = -0.333698616227678005726562603400;
const double ABA_b1 =  0.183083687472197221961703757166;
const double ABA_b2 =  0.310782859898574869507522291054;
const double ABA_b3 = -0.026564618511958800697212137916;
const double ABA_b4 =  0.065396142282373418455972179391;

struct SimParams {
    int W = 86;
    int H = 86;
    bool save_movie = false;
    double target_E = 10.0;
    unsigned int seed = 42;
    unsigned int totaltime = 250;
    double eqltime = 100.0;
    double dt = 0.01;
    int num_realizations = 1;
    int obs_freq = 1;
};
