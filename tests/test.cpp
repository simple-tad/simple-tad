#include "baum_welch.hpp"
#include "coord.hpp"
#include "di.hpp"
#include "types.h"
#include "viterbi.hpp"
// #include "viterbi_simd.hpp"
#include "gtest/gtest.h"

#define PI acos(-1)

TEST(tests, di) {
    double* data = new double[25] {
        0.1, 0.3, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.4, 0.0, 0.0,
        0.0, 0.5, 0.6, 0.3, 0.0,
        0.0, 0.0, 0.1, 0.4, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.2
    };

    double* expected_di = new double[5] {
        0.3, 0.0666667, -0.05, 0.0, -0.1
    };

    double* di = calculate_di_AVX2(data, 5, 2);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(di[i], expected_di[i], 1e-4);
    }

    delete[] data;
    delete[] expected_di;
    delete[] di;
}

double emission_probability(double emit_value, int state) {
    auto sigma = 20, mu = 0;
    if (state == BiasState::UpstreamBias) {
        mu = 40;
    } else if (state == BiasState::DownstreamBias) {
        mu = -40;
    } else if (state == BiasState::NoBias) {
        mu = 0;
    } else {
        throw std::runtime_error("Error: impossible state");
    }
    double pow_sigma2_2times = 2 * pow(sigma, 2);
    double pow_delta_emitvalue = -pow((emit_value - mu), 2);

    double ret = 1.0 / (sigma * sqrt(2 * PI)) * exp(pow_delta_emitvalue / pow_sigma2_2times);
    return ret;
}

// TEST(tests, viterbi_simdpp) {
//     // set input
//     double observation[] = { 50, 8, -5, -22, 1, 3, -20, -50, -12, 6,
//         11, 50, 50, 50, 20, 18, 7, 1, -1, -1,
//         -2, -2, -1, -4, -12, -39, -7, -11, -50, -50,
//         -50, -16, -14, -14, -50, -50, -50, -50, -50, 10,
//         40, 50, 10, 2, 18, 1, -1.5, 4, 1, 0.5,
//         -1, -26 };
//     auto sizeof_observation = 52;

//     double start_p[3] = { 0.33, 0.33, 0.33 };

//     double transition_p[3 * 3] = {
//         0.7, 0.1, 0.2,
//         0.1, 0.7, 0.2,
//         0.36, 0.36, 0.28
//     };

//     // call viterbi algorithm
//     auto viterbi_result = vectorized::viterbi(observation, sizeof_observation, start_p, transition_p, emission_probability);

//     int expected_result[] = {
//         0, 2, 2, 1, 2,
//         2, 1, 1, 1, 2,
//         0, 0, 0, 0, 0,
//         0, 2, 2, 2, 2,
//         2, 2, 2, 2, 1,
//         1, 1, 1, 1, 1,
//         1, 1, 1, 1, 1,
//         1, 1, 1, 1, 2,
//         0, 0, 2, 2, 2,
//         2, 2, 2, 2, 2,
//         2, 1
//     };

//     for (int i = 0; i < sizeof_observation; i++)
//         EXPECT_EQ(viterbi_result[i], expected_result[i]);
// }

TEST(tests, viterbi_raw) {
    int observation[] = {
        0, 1, 1, 0, 2, 2, 2, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    };
    auto sizeof_observation = 20;

    double start_p[3] = { 0.33, 0.33, 0.33 };

    double transition_p[3 * 3] = {
        0.60, 0.15, 0.25,
        0.35, 0.55, 0.10,
        0.15, 0.30, 0.55
    };

    double emission_p[3 * 3] = {
        0.70, 0.10, 0.20,
        0.18, 0.56, 0.26,
        0.30, 0.15, 0.55
    };

    // call viterbi algorithm
    auto viterbi_result = scalar::viterbi(observation, sizeof_observation, start_p, transition_p, emission_p);

    int expected_result[] = {
        0, 1, 1, 0, 2, 2, 2, 2, 2, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    };
    for (int i = 0; i < sizeof_observation; i++)
        EXPECT_EQ(viterbi_result[i], expected_result[i]);

    delete[] viterbi_result;
}

TEST(tests, baum_welch_scalar) {
    int* observations = new int[16] { 0, 1, 1, 1, 2, 3, 3, 2, 2, 4, 4, 0, 0, 0, 1, 1 };

    double* initial = new double[3] { 0.4, 0.3, 0.3 };
    double* transition = new double[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };
    double* emission = new double[3 * 5] {
        0.5, 0.1, 0.1, 0.1, 0.2,
        0.1, 0.5, 0.1, 0.1, 0.2,
        0.1, 0.1, 0.5, 0.1, 0.2
    };

    baum_welch(observations, 16, initial, transition, emission, 3, 5, 1e-7, 100);

    double* expected_initial = new double[3] { 1, 0, 0 };
    double* expected_transition = new double[3 * 3] {
        0.500000, 0.499999, 0.000000,
        0.000000, 0.750000, 0.249999,
        0.142857, 0.000000, 0.857142
    };
    double* expected_emission = new double[3 * 5] {
        0.999999, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.999999, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.428571, 0.285714, 0.285714
    };

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(initial[i], expected_initial[i], 1e-4);
    }
    for (int i = 0; i < 3 * 3; i++) {
        EXPECT_NEAR(transition[i], expected_transition[i], 1e-4);
    }
    for (int i = 0; i < 3 * 5; i++) {
        EXPECT_NEAR(emission[i], expected_emission[i], 1e-4);
    }

    delete[] observations;
    delete[] initial;
    delete[] transition;
    delete[] emission;
}

TEST(tests, coord) {
    int* state = new int[16] { 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 2 };

    std::vector<std::pair<std::size_t, std::size_t>> coords = calculate_coord(reinterpret_cast<BiasState*>(state), 16);

    std::vector<std::pair<std::size_t, std::size_t>> expected_coords = {
        { 2, 6 }, { 10, 12 }
    };

    EXPECT_EQ(coords, expected_coords);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}