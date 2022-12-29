#include "baum_welch.hpp"
#include "coord.hpp"
#include "di.hpp"
#include "types.h"
#include "viterbi.hpp"
#include "viterbi_simd.hpp"
#include "gtest/gtest.h"

#define PI acos(-1)

TEST(tests, di) {
    float* data = new float[25] {
        0.1, 0.3, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.4, 0.0, 0.0,
        0.0, 0.5, 0.6, 0.3, 0.0,
        0.0, 0.0, 0.1, 0.4, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.2
    };

    float* expected_di = new float[5] {
        0.3, 0.0666667, -0.05, 0.0, -0.1
    };

    float* di = calculate_di_AVX2(data, 5, 2);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(di[i], expected_di[i], 1e-4);
    }

    delete[] data;
    delete[] expected_di;
    delete[] di;
}

float emission_probability(float emit_value, int state) {
    auto sigma = 20, mu = 0;
    if (state == BiasState::UpstreamBias) {
        mu = 40;
    } else if (state == BiasState::DownstreamBias) {
        mu = -40;
    } else if (state == BiasState::NoBias) {
        mu = 0;
    } else {
        throw runtime_error("Error: impossible state");
    }
    float pow_sigma2_2times = 2 * pow(sigma, 2);
    float pow_delta_emitvalue = -pow((emit_value - mu), 2);

    float ret = 1.0 / (sigma * sqrt(2 * PI)) * exp(pow_delta_emitvalue / pow_sigma2_2times);
    return ret;
}

TEST(tests, viterbi_simdpp) {
    // set input
    float observation[] = { 50, 8, -5, -22, 1, 3, -20, -50, -12, 6,
        11, 50, 50, 50, 20, 18, 7, 1, -1, -1,
        -2, -2, -1, -4, -12, -39, -7, -11, -50, -50,
        -50, -16, -14, -14, -50, -50, -50, -50, -50, 10,
        40, 50, 10, 2, 18, 1, -1.5, 4, 1, 0.5,
        -1, -26 };
    auto sizeof_observation = 52;

    float start_p[3] = { 0.33, 0.33, 0.33 };

    float transition_p[3 * 3] = {
        0.7, 0.1, 0.2,
        0.1, 0.7, 0.2,
        0.36, 0.36, 0.28
    };

    // call viterbi algorithm
    auto viterbi_result = vectorized::viterbi(observation, sizeof_observation, start_p, transition_p, emission_probability);

    int expected_result[] = {
        0, 2, 2, 1, 2,
        2, 1, 1, 1, 2,
        0, 0, 0, 0, 0,
        0, 2, 2, 2, 2,
        2, 2, 2, 2, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 2,
        0, 0, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 1
    };

    for (int i = 0; i < sizeof_observation; i++)
        EXPECT_EQ(viterbi_result[i], expected_result[i]);
}

TEST(tests, viterbi_raw) {
    int observation[] = {
        0, 1, 1, 0, 2, 2, 2, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    };
    auto sizeof_observation = 20;

    float start_p[3] = { 0.33, 0.33, 0.33 };

    float transition_p[3 * 3] = {
        0.60, 0.15, 0.25,
        0.35, 0.55, 0.10,
        0.15, 0.30, 0.55
    };

    float emission_p[3 * 3] = {
        0.70, 0.10, 0.20,
        0.18, 0.56, 0.26,
        0.30, 0.15, 0.55
    };

    // call viterbi algorithm
    auto viterbi_result = scalar::viterbi(observation, sizeof_observation, start_p, transition_p, emission_p);

    int expected_result[] = {
        0, 1, 1, 0, 2, 2, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    };
    for (int i = 0; i < sizeof_observation; i++)
        EXPECT_EQ(viterbi_result[i], expected_result[i]);

    delete[] viterbi_result;
}

// TEST(tests, baum_welch_scalar) {
//     int* observations = new int[16] { 0, 1, 1, 1, 2, 3, 3, 2, 2, 4, 4, 0, 0, 0, 1, 1 };

//     float* initial = new float[3] { 0.4, 0.3, 0.3 };
//     float* transition = new float[3 * 3] {
//         0.7, 0.2, 0.1,
//         0.1, 0.6, 0.3,
//         0.2, 0.3, 0.5
//     };
//     float* emission = new float[3 * 5] {
//         0.5, 0.1, 0.1, 0.1, 0.2,
//         0.1, 0.5, 0.1, 0.1, 0.2,
//         0.1, 0.1, 0.5, 0.1, 0.2
//     };

//     baum_welch(observations, 16, initial, transition, emission, 3, 5, 1e-7, 100);

//     float* expected_initial = new float[3] { 0.500094, 0.251293, 0.248612 };
//     float* expected_transition = new float[3 * 3] {
//         0.417835, 0.315129, 0.267037,
//         0.260414, 0.417387, 0.322199,
//         0.290338, 0.330825, 0.378838
//     };
//     float* expected_emission = new float[3 * 5] {
//         0.332985, 0.267642, 0.156908, 0.116732, 0.125733,
//         0.203520, 0.385474, 0.167486, 0.123916, 0.119605,
//         0.213660, 0.279519, 0.241754, 0.134906, 0.130161
//     };

//     for (int i = 0; i < 3; i++) {
//         EXPECT_NEAR(initial[i], expected_initial[i], 1e-4);
//     }
//     for (int i = 0; i < 3 * 3; i++) {
//         EXPECT_NEAR(transition[i], expected_transition[i], 1e-4);
//     }
//     for (int i = 0; i < 3 * 5; i++) {
//         EXPECT_NEAR(emission[i], expected_emission[i], 1e-4);
//     }

//     delete[] observations;
//     delete[] initial;
//     delete[] transition;
//     delete[] emission;
// }

TEST(tests, coord) {
    int* state = new int[16] { 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 2, 0, 2, 1 };

    std::vector<std::pair<std::size_t, std::size_t>> coords = calculate_coord(reinterpret_cast<BiasState*>(state), 16);

    std::vector<std::pair<std::size_t, std::size_t>> expected_coords = {
        { 0, 3 }, { 5, 11 }, { 11, 16 }
    };

    EXPECT_EQ(coords, expected_coords);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}