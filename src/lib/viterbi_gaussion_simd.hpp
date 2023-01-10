#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <simdpp/simd.h>

namespace vectorized {

double normal_pdf(const double& di, const double& average, const double& variance)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (di - average) / variance;
    return inv_sqrt_2pi / variance * std::exp(-0.5f * a * a);
}

double emission_func_gaussion(const double* const& emission, const double& di, const int& state) {
    double average = emission[state * 2 + 0];
    double variance = emission[state * 2 + 1];

    return normal_pdf(di, average, variance);
}

/*
observations: double[num_observation].  The di values.
initial: double[3].  The probabilities for init state.
transition: double[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: double[3*2] now.  The (average, variance) pairs for gaussion emission function.
*/
int* viterbi_gaussion(const double* const& observations, const std::size_t& num_observation, const double* const& initial, const double* const& transition, const double* const& emission) {
    int* hidden_states = new int[num_observation]();
    double* viterbi = new double[3* num_observation]();

    int* prev_state = new int[3* num_observation]();
    
    double* initial_log = new double[4]();
    for(std::size_t i = 0; i < 3; ++i) {
        initial_log[i] = std::log(initial[i]);
    }
    initial_log[3] = -1000;

    double* transition_log = new double[4*4]();
    for(std::size_t i = 0; i < 4*4; ++i) {
        if ((i + 1) % 4 == 0 || i >= 12)
            transition_log[i] = -1000;
        else
        {
            int origin_i = i - i / 4;
            origin_i = origin_i%3*3+origin_i/3;
            transition_log[i] = log(transition[origin_i]);
        }
    }

    simdpp::float64<4>* simd_transition_log = new simdpp::float64<4>[4]();
    for(std::size_t i = 0; i < 4; ++i) {
        simd_transition_log[i] = simdpp::load(transition_log + i * 4);
    }
    
    double* emission_log = new double[3*num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        for(std::size_t i = 0; i < 3; ++i) {
            emission_log[t*3 + i] = std::log(emission_func_gaussion(emission, observations[t], i));
        }
    }

    simdpp::float64<4>* simd_emission_log = new simdpp::float64<4>[num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        simd_emission_log[t] = simdpp::load(emission_log + t*3);
    }

    // initialize viterbi
    simdpp::float64<4> temp = simdpp::load(initial_log);
    temp = simdpp::add(temp, simd_emission_log[0]);
    simdpp::store(viterbi + 0, temp);

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {   // current state
            simdpp::float64<4> temp = simdpp::load(viterbi + (t-1)*3);
            temp = simdpp::add(temp, simd_transition_log[i]);

            double max = simdpp::reduce_max(temp);
            viterbi[t * 3 + i] = max + std::log(emission_func_gaussion(emission, observations[t], i));

            double* result_arr = new double[4]();
            simdpp::store(result_arr, temp);
            for(int j=0; j<3; j++)
            {
                if(result_arr[j] == max)
                {
                    prev_state[t * 3 + i] = j;
                    break;
                }
            }
            delete[] result_arr;
        }
    }

    // find the most probable state
    double max = -INFINITY;
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