#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

float emission_func(const float* const& emission, const int& di, const int& state) {
    return emission[state * 3 + di];
}

int* viterbi(const int* const& observations, const std::size_t& num_observation, const float* const& initial, const float* const& transition, const float* const& emission) {
    int* hidden_states = new int[num_observation]();
    float* viterbi = new float[3* num_observation]();

    // initialize viterbi
    for (std::size_t i = 0; i < 3; ++i) {
        viterbi[i * num_observation] = std::log1p(initial[i]) + std::log1p(emission_func(emission, observations[0], i));
    }

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {
            float max = -INFINITY;
            for (std::size_t j = 0; j < 3; ++j) {
                float temp = viterbi[j * num_observation + t - 1] + std::log1p(transition[j * 3 + i]);
                if (temp > max) {
                    max = temp;
                }
            }
            viterbi[i * num_observation + t] = max + std::log1p(emission_func(emission, observations[t], i));
        }
    }

    // find the most probable state
    float max = -INFINITY;
    for (std::size_t i = 0; i < 3; ++i) {
        if (viterbi[i * num_observation + num_observation - 1] > max) {
            max = viterbi[i * num_observation + num_observation - 1];
            hidden_states[num_observation - 1] = i;
        }
    }

    // backtrace
    for (int i = num_observation - 2; i >= 0; --i) {
        float max = -INFINITY;
        for (std::size_t j = 0; j < 3; ++j) {
            float temp = viterbi[j * num_observation + i] + std::log1p(transition[hidden_states[i + 1] * 3 + j]);
            if (temp > max) {
                max = temp;
                hidden_states[i] = j;
            }
        }
    }

    delete[] viterbi;

    return hidden_states;
}
}