#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <simdpp/simd.h>

namespace vectorized {

double emission_func(const double* const& emission, const int& di, const int& state, const std::size_t& num_hiddenstate) {
    return emission[state * num_hiddenstate + di];
}

/*
observations: double[num_observation].  The di values.
initial: double[3].  The probabilities for init state.
transition: double[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: double[3*2] now.  The (average, variance) pairs for gaussion emission function.
*/
int* viterbi(const int* const& observations, const std::size_t& num_observation, const double* const& initial, const double* const& transition, const double* const& emission, const std::size_t& num_emission=3, const std::size_t& num_hiddenstate=3) {
    int* hidden_states = new int[num_observation]();
    double* viterbi = new double[num_hiddenstate* num_observation]();

    int* prev_state = new int[num_hiddenstate* num_observation]();
    
    double* initial_log = new double[4]();
    for(std::size_t i = 0; i < num_hiddenstate; ++i) {
        initial_log[i] = std::log(initial[i]);
    }
    initial_log[num_hiddenstate] = -1000;

    double* transition_log = new double[(num_hiddenstate+1)*(num_hiddenstate+1)]();
    for(std::size_t i = 0; i < (num_hiddenstate+1)*(num_hiddenstate+1); ++i) {
        if ((i + 1) % (num_hiddenstate+1) == 0 || i >= (num_hiddenstate+1) * num_hiddenstate)
            transition_log[i] = -1000;
        else
        {
            int origin_i = i - i / (num_hiddenstate+1);
            origin_i = origin_i%num_hiddenstate*num_hiddenstate+origin_i/num_hiddenstate;
            transition_log[i] = log(transition[origin_i]);
        }
    }

    simdpp::float64<4>* simd_transition_log = new simdpp::float64<4>[4]();
    for(std::size_t i = 0; i < 4; ++i) {
        simd_transition_log[i] = simdpp::load(transition_log + i * 4);
    }
    
    double* emission_log = new double[num_hiddenstate*num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        for(std::size_t i = 0; i < num_hiddenstate; ++i) {
            emission_log[t*num_hiddenstate + i] = std::log(emission_func(emission, observations[t], i, num_hiddenstate));
        }
    }

    simdpp::float64<4>* simd_emission_log = new simdpp::float64<4>[num_observation]();
    for(std::size_t t = 0; t < num_observation; ++t) {
        simd_emission_log[t] = simdpp::load(emission_log + t*num_hiddenstate);
    }

    // initialize viterbi
    simdpp::float64<4> temp = simdpp::load(initial_log);
    temp = simdpp::add(temp, simd_emission_log[0]);
    simdpp::store(viterbi + 0, temp);

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < num_hiddenstate; ++i) {   // current state
            simdpp::float64<4> temp = simdpp::load(viterbi + (t-1)*num_hiddenstate);
            temp = simdpp::add(temp, simd_transition_log[i]);

            double max = simdpp::reduce_max(temp);
            viterbi[t * num_hiddenstate + i] = max + std::log(emission_func(emission, observations[t], i, num_hiddenstate));

            double* result_arr = new double[4]();
            simdpp::store(result_arr, temp);
            for(std::size_t j=0; j<num_hiddenstate; j++)
            {
                if(result_arr[j] == max)
                {
                    prev_state[t * num_hiddenstate + i] = j;
                    break;
                }
            }
            delete[] result_arr;
        }
    }

    // find the most probable state
    double max = -INFINITY;
    for (std::size_t i = 0; i < num_hiddenstate; ++i) {
        if (viterbi[(num_observation-1) * num_hiddenstate + i] > max) {
            max = viterbi[(num_observation-1) * num_hiddenstate + i];
            hidden_states[num_observation - 1] = i;
        }
    }

    // backtrace
    for (int t = num_observation - 2; t >= 0; --t) {
        hidden_states[t] = prev_state[(t + 1) * num_hiddenstate + hidden_states[t + 1]];
    }

    delete[] viterbi;

    return hidden_states;
}
}