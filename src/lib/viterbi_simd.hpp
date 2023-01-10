#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <simdpp/simd.h>

namespace vectorized {
const int N = 4;

int indexof(double value, double32<N> arr) {
    int tmp_arr[N] = { 0, 1, 2, 0 };
    int32<N> tmp = load(tmp_arr);

    int32<N> int_arr;
    int_arr = cmp_neq(arr, value);
    int_arr = add(int_arr, 1);
    int_arr = mul_lo(int_arr, tmp);
    int ret = reduce_add(int_arr);
    return ret;
}

void init_path(double32<N>* V, double32<N> start_p, uint8<N>* path, double32<N> emission_p) {
    V[0] = add(start_p, emission_p);
    double path_arr[3] = { 0, 1, 2 };
    path[0] = load(path_arr);
}

void get_path_until_t(double32<N> transition_p[N], double emission_p[N],
    double32<N>* V, uint8<N>* path, int t) {
    double cur_prob[N];
    int prev_state[N];

    // 對當前要計算的 state i ，使用simd計算三個state來到此state的機率，並取其最大值
    for (int i = 0; i < N; i++) {
        // 取得當前state最佳的機率
        double32<N> current_prob = V[t - 1];
        current_prob = add(current_prob, transition_p[i]); // strange
        current_prob = add(current_prob, emission_p[i]);
        cur_prob[i] = reduce_max(current_prob);

        // 取得最佳機率發生時的前一個state
        prev_state[i] = indexof(cur_prob[i], current_prob);
    }

    V[t] = load(cur_prob);
    path[t] = load(prev_state);
}

void printdouble(double32<N> arr) {
    double ori_arr[N];
    store(ori_arr, arr);
    cout << "[";
    for (int i = 0; i < N; i++) {
        cout << ori_arr[i] << ", ";
    }
    cout << "]" << endl;
}
void printint(int32<N> arr) {
    int ori_arr[N];
    store(ori_arr, arr);
    cout << "[";
    for (int i = 0; i < N; i++) {
        cout << ori_arr[i] << ", ";
    }
    cout << "]" << endl;
}

/*
observations: float[num_observation].  The di values.
initial: float[3].  The probabilities for init state.
transition: float[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: float[3*2] now.  The (average, variance) pairs for gaussion emission function.
*/
int* viterbi(double* observation, size_t sizeof_observation,
    double* start_p, double* transition_p, double (*emission_func)(double, int)) {
    // 前處理，將 start_p 與 transition_p 取 log
    double* new_start_p = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N - 1; i++) {
        new_start_p[i] = log10(start_p[i]);
    }
    initial_log10[3] = -1000;

    double* new_transition_p = (double*)malloc(N * N * sizeof(double));
    for (int i = N * N - 1; i >= N * (N - 1); i--) {
        new_transition_p[i] = log10(1e-9);
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

    // 將機率轉換成simdpp vector type
    double32<N> simd_start_p = load(new_start_p);
    double32<N>* simd_transition_p = (double32<N>*)calloc(N, sizeof(double32<N>));
    for (int i = 0; i < N; i++) {
        simd_transition_p[i] = load(new_transition_p + i * N);
    }

    // 宣告V變數，V變數用來記錄當前的最佳機率與上一次迭代的最佳機率
    double32<N>* V = (double32<N>*)calloc(sizeof_observation, sizeof(double32<N>));

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {   // current state
            simdpp::float32<4> temp = simdpp::load(viterbi + (t-1)*3);
            temp = simdpp::add(temp, simd_transition_log10[i]);

    // Initialize
    // t == 0 的情況，V是每個state的初始機率與他們對應的的噴出機率；而path中存放的是以各個state為終點的，長度僅為1的路徑。
    double emission_p[N];
    double32<N> simd_emission_p;
    for (int i = 0; i < N - 1; i++) {
        emission_p[i] = log10(emission_func(observation[0], i));
    }

    // find the most probable state
    float max = -INFINITY;
    for (std::size_t i = 0; i < 3; ++i) {
        if (viterbi[(num_observation-1) * 3 + i] > max) {
            max = viterbi[(num_observation-1) * 3 + i];
            hidden_states[num_observation - 1] = i;
        }
    }

    // V[(sizeof_observation-1)%2]中存的是三個path的最大發生機率
    // 選擇最大的機率對應的path並回傳
    double max_end_prob = reduce_max(V[sizeof_observation - 1]);
    int end_state = indexof(max_end_prob, V[sizeof_observation - 1]);

    int* ret = (int*)malloc(sizeof_observation * sizeof(int));
    int cur_state_arr[3];
    ret[sizeof_observation - 1] = end_state;
    int cur_state = end_state;
    for (int i = sizeof_observation - 2; i >= 0; i--) {
        store(cur_state_arr, path[i + 1]);
        ret[i] = cur_state_arr[cur_state];
        cur_state = ret[i];
    }

    delete[] viterbi;

    return hidden_states;
}
}