#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <simdpp/simd.h>

namespace vectorized {

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
    int* hidden_states = new int[num_observation]();
    float* viterbi = new float[3* num_observation]();

    int* prev_state = new int[3* num_observation]();
    
    float* initial_log10 = new float[4]();
    for(std::size_t i = 0; i < 3; ++i) {
        initial_log10[i] = std::log10(initial[i]);
    }
    initial_log10[3] = -1000;

    float* transition_log10 = new float[4*4]();
    for(std::size_t i = 0; i < 4*4; ++i) {
        if ((i + 1) % 4 == 0 || i >= 12)
            transition_log10[i] = -1000;
        else
        {
            int origin_i = i - i / 4;
            origin_i = origin_i%3*3+origin_i/3;
            transition_log10[i] = log10(transition[origin_i]);
        }
    }

    simdpp::float32<4>* simd_transition_log10 = new simdpp::float32<4>[4]();
    for(std::size_t i = 0; i < 4; ++i) {
        simd_transition_log10[i] = simdpp::load(transition_log10 + i * 4);
    }
    
    float* emission_log10 = new float[3*num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        for(std::size_t i = 0; i < 3; ++i) {
            emission_log10[t*3 + i] = std::log10(emission_func(emission, observations[t], i));
        }
    }

    simdpp::float32<4>* simd_emission_log10 = new simdpp::float32<4>[num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        simd_emission_log10[t] = simdpp::load(emission_log10 + t*3);
    }

    // initialize viterbi
    simdpp::float32<4> temp = simdpp::load(initial_log10);
    temp = simdpp::add(temp, simd_emission_log10[0]);
    simdpp::store(viterbi + 0, temp);

    int index_arr[4] = { 0, 1, 2, 0 };
    simdpp::int32<4> simd_index_mask = simdpp::load(index_arr);

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {   // current state
            simdpp::float32<4> temp = simdpp::load(viterbi + (t-1)*3);
            temp = simdpp::add(temp, simd_transition_log10[i]);

            float max = simdpp::reduce_max(temp);
            viterbi[t * 3 + i] = max + std::log10(emission_func(emission, observations[t], i));

            simdpp::int32<4> temp2;
            temp2 = simdpp::cmp_neq(temp, max);
            temp2 = simdpp::add(temp2, 1);
            temp2 = simdpp::mul_lo(temp2, simd_index_mask);
            prev_state[t * 3 + i] = simdpp::reduce_add(temp2);
        }
    }

    // find the most probable state
    float max = -INFINITY;
    for (std::size_t i = 0; i < 3; ++i) {
        if (viterbi[(num_observation-1) * 3 + i] > max) {
            max = viterbi[(num_observation-1) * 3 + i];
            hidden_states[num_observation - 1] = i;
        }
    }

    // backtrace
    for (int t = num_observation - 2; t >= 0; --t) {
        hidden_states[t] = prev_state[(t + 1) * 3 + hidden_states[t + 1]];
    }

    delete[] viterbi;

    return hidden_states;
}
}