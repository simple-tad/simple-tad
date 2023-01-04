#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

float normal_pdf(const float& di, const float& average, const float& variance)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (di - average) / variance;
    return inv_sqrt_2pi / variance * std::exp(-0.5f * a * a);
}

float emission_func(const float* const& emission, const float& di, const int& state) {
    float average = emission[state * 2 + 0];
    float variance = emission[state * 2 + 1];

    return normal_pdf(di, average, variance);
}

/*
observations: float[num_observation].  The di values.
initial: float[3].  The probabilities for init state.
transition: float[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: float[3*2] now.  The (average, variance) pairs for gaussion emission function.
*/
int* viterbi(const float* const& observations, const std::size_t& num_observation, const float* const& initial, const float* const& transition, const float* const& emission) {
    int* hidden_states = new int[num_observation]();    // result of the function.
    float* viterbi = new float[3* num_observation]();   // record probability for each node.
    int* prev_state = new int[3* num_observation]();    // record previous state for current node.

    // calculate log10 for each transition probability.
    float* transition_log10 = new float[3*3]();
    for(std::size_t i = 0; i < 3*3; ++i) {
        transition_log10[i] = std::log10(transition[i]);
    }

    // initialize viterbi
    for (std::size_t i = 0; i < 3; ++i) {
        viterbi[i * num_observation] = std::log10(initial[i]) + std::log10(emission_func(emission, observations[0], i));
        prev_state[i * num_observation] = i;
    }

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {   // current state
            float max = -INFINITY;
            for (std::size_t j = 0; j < 3; ++j) {   // previous state
                float temp = viterbi[j * num_observation + t - 1] + transition_log10[j * 3 + i];
                if (temp > max) {
                    max = temp;
                    prev_state[i * num_observation + t] = j;
                }
            }
            viterbi[i * num_observation + t] = max + std::log10(emission_func(emission, observations[t], i));
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
    for (int t = num_observation - 2; t >= 0; --t) {
        hidden_states[t] = prev_state[hidden_states[t + 1] * num_observation + t + 1];
    }

    delete[] viterbi;
    delete[] prev_state;
    delete[] transition_log10;

    return hidden_states;
}
}