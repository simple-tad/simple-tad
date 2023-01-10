#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <immintrin.h>

namespace vectorized { 
    double inline log_add(double x, double y) {
        if (x == -std::numeric_limits<double>::infinity()) {
            return y;
        } else if (y == -std::numeric_limits<double>::infinity()) {
            return x;
        }

        if (y <= x) {
            return x + std::log1p(std::exp(y - x));
        } else {
            return y + std::log1p(std::exp(x - y));
        }
    }

    double inline log_sum(double const* arr, std::size_t n) {
        double max = *std::max_element(arr, arr + n);
        if (max == std::numeric_limits<double>::infinity()) {
            return max;
        }
        double acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            acc += std::exp(arr[i] - max);
        }
        return max + std::log(acc);
    }

    // __m256d inline log_sum_AVX(__m256d const* arr, std::size_t n) {
    //     // double max = *std::max_element(arr, arr + n);
    //     __m256d max_AVX = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    //     // if (max == std::numeric_limits<double>::infinity()) {
    //     //     return max;
    //     // }
    //     max_AVX = _mm256_max_pd(arr[0], _mm256_max_pd(arr[1], arr[2]));

    //     // double acc = 0;
    //     __m256d acc_AVX = _mm256_set1_pd(0);

    //     for (std::size_t i = 0; i < n; ++i) {
    //         // acc += std::exp(arr[i] - max);
    //         acc_AVX = _mm256_add_pd(acc_AVX, _mm256_exp_pd(_mm256_sub_pd(arr[i], max_AVX)));
    //     }

    //     // return max + std::log(acc);
    //     return _mm256_add_pd(max_AVX, _mm256_log_pd(acc_AVX));
    // }

    const std::tuple<const double, const double*> forward(
        const int* observations,
        const std::size_t num_observations,
        const double* transition,
        const double* emission,
        const double* initial,
        const std::size_t num_states,
        const std::size_t num_emissions) {
        // init the alpha matrix
        double* alpha = new (std::align_val_t(32)) double[num_states * num_observations]();
        for (std::size_t i = 0; i < num_states; ++i) {
            alpha[i * num_observations] = initial[i] + emission[i * num_emissions + observations[0]];
        }

        // compute the alpha matrix
        for (std::size_t t = 1; t < num_observations; ++t) {
            double* buf = new (std::align_val_t(32)) double[12]; // 3 * 4
            __m256d buf_AVX;
            for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
                __m256d prev_alpha_AVX = _mm256_set1_pd(alpha[prev_state * num_observations + t - 1]);
                __m256d trans_AVX = _mm256_load_pd(transition + prev_state * num_states);
                buf_AVX = _mm256_add_pd(prev_alpha_AVX, trans_AVX);
                _mm256_store_pd(buf + prev_state * 4, buf_AVX);
            }
            auto swap = [&buf](auto const& pos1, auto const& pos2) {
                auto tmp = buf[pos2];
                buf[pos2] = buf[pos1];
                buf[pos1] = tmp;
            };
            swap(1, 4); swap(2, 8); swap(6, 9);
            alpha[t] = log_sum(buf, num_states) + emission[observations[t]];
            alpha[1 * num_observations + t] = log_sum(buf + 4, num_states) + emission[1 * num_emissions + observations[t]];
            alpha[2 * num_observations + t] = log_sum(buf + 8, num_states) + emission[2 * num_emissions + observations[t]];
            delete[] buf;
        }

        // compute the probability of the observation sequence
        double alpha_sum = -INFINITY;
        double* buf = new double[num_states];

        for (std::size_t i = 0; i < num_states; ++i) {
            buf[i] = alpha[i * num_observations + num_observations - 1];
        }
        alpha_sum = log_sum(buf, num_states);

        delete[] buf;

        return std::make_tuple(alpha_sum, alpha);
    }

    const double* backward(
        const int* observations,
        const std::size_t num_observations,
        const double* transition,
        const double* emission,
        const std::size_t num_states,
        const std::size_t num_emissions) {
        // init the beta matrix
        double* beta = new (std::align_val_t(32)) double[num_states * num_observations]();
        for (std::size_t i = 0; i < num_states; ++i) {
            beta[i * num_observations + num_observations - 1] = 0; // log(1)
        }

        // compute the beta matrix
        for (std::size_t t = num_observations - 2;; --t) {
            double* buf = new (std::align_val_t(32)) double[12]; // 3 * 4
            __m256d buf_AVX;
            for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
                __m256d curr_beta_AVX = _mm256_set1_pd(beta[curr_state * num_observations + t + 1]);
                __m256d trans_AVX = _mm256_set_pd(
                    transition[curr_state],
                    transition[1 * num_states + curr_state],
                    transition[2 * num_states + curr_state],
                    -std::numeric_limits<double>::infinity());
                __m256d emit_AVX = _mm256_set1_pd(emission[curr_state * num_emissions + observations[t + 1]]);
                buf_AVX = _mm256_add_pd(_mm256_add_pd(curr_beta_AVX, trans_AVX), emit_AVX);
                _mm256_store_pd(buf + curr_state * 4, buf_AVX);
            }
            // std::cout << "buf: " << t << std::endl
                // << buf[0] << ' ' << buf[1] << ' ' << buf[2] << ' ' << buf[3] << std::endl
                // << buf[4] << ' ' << buf[5] << ' ' << buf[6] << ' ' << buf[7] << std::endl
                // << buf[8] << ' ' << buf[9] << ' ' << buf[10] << ' ' << buf[11] << std::endl << std::endl;
            auto swap = [&buf](auto const& pos1, auto const& pos2) {
                auto tmp = buf[pos2];
                buf[pos2] = buf[pos1];
                buf[pos1] = tmp;
            };
            swap(1, 11); swap(2, 7); swap(5, 10);
            beta[t] = log_sum(buf + 1, num_states);
            beta[1 * num_observations + t] = log_sum(buf + 5, num_states);
            beta[2 * num_observations + t] = log_sum(buf + 9, num_states);
            delete[] buf;

            if (t == 0) {
                break;
            }
        }

        return beta;
    }

    const double* compute_gamma(
        const double* alpha,
        const double* beta,
        const std::size_t num_observations,
        const std::size_t num_states) {
        double* gamma = new double[num_states * num_observations]();

        for (std::size_t t = 0; t < num_observations; ++t) {
            double* buf = new double[num_states];
            for (std::size_t i = 0; i < num_states; ++i) {
                buf[i] = alpha[i * num_observations + t] + beta[i * num_observations + t];
            }
            double sum = log_sum(buf, num_states);
            for (std::size_t i = 0; i < num_states; ++i) {
                gamma[i * num_observations + t] = buf[i] - sum;
            }
            delete[] buf;
        }

        return gamma;
    }

    const double* compute_xi(
        const double* alpha,
        const double alpha_sum,
        const double* beta,
        const int* observations,
        const std::size_t num_observations,
        const double* transition,
        const double* emission,
        const std::size_t num_states,
        const std::size_t num_emissions) {
        double* xi = new double[num_states * num_states * num_observations]();

        for (std::size_t t = 0; t < num_observations - 1; ++t) {
            double* buf = new double[num_states*num_states];
            for (std::size_t i = 0; i < num_states; ++i) {
                for (std::size_t j = 0; j < num_states; ++j) {
                    buf[i * num_states + j] = alpha[i * num_observations + t] + transition[i * num_states + j] + emission[j * num_emissions + observations[t + 1]] + beta[j * num_observations + t + 1];
                }
            }

            double sum = log_sum(buf, num_states * num_states);
            for (std::size_t i = 0; i < num_states; ++i) {
                for (std::size_t j = 0; j < num_states; ++j) {
                    xi[i * num_states * num_observations + j * num_observations + t] = buf[i * num_states + j] - sum;
                }
            }

            delete[] buf;
        }

        return xi;
    }

    void baum_welch(
        const int* observations,
        const std::size_t num_observations,
        double* initial,
        double* transition,
        double* emission,
        const std::size_t num_states,
        const std::size_t num_emissions,
        const double tolerance = 1e-7,
        const std::size_t max_iters = 1000) {
        // transform initial to log space
        double* log_initial = new (std::align_val_t(32)) double[num_states];
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] = std::log(initial[i]);
        }

        // transform transition matrix and emission matrix to log space
        double* log_transition = new (std::align_val_t(32)) double[num_states * num_states];
        for (std::size_t i = 0; i < num_states * num_states; ++i) {
            log_transition[i] = std::log(transition[i]);
        }
        double* log_emission = new (std::align_val_t(32)) double[num_states * num_emissions];
        for (std::size_t i = 0; i < num_states * num_emissions; ++i) {
            log_emission[i] = std::log(emission[i]);
        }

        std::size_t iter = 0;
        double previous_alpha_sum = -INFINITY;
        while (true) {
            auto& [alpha_sum, alpha] = forward(observations, num_observations, log_transition, log_emission, log_initial, num_states, num_emissions);
            auto beta = backward(observations, num_observations, log_transition, log_emission, num_states, num_emissions);

            // compute gamma and xi
            auto gamma = compute_gamma(alpha, beta, num_observations, num_states);
            auto xi = compute_xi(alpha, alpha_sum, beta, observations, num_observations, log_transition, log_emission, num_states, num_emissions);

            delete[] alpha;
            delete[] beta;

            // update initial
            for (std::size_t i = 0; i < num_states; ++i) {
                log_initial[i] = gamma[i * num_observations];
            }

            // update transition matrix
            for (std::size_t i = 0; i < num_states; ++i) {
                double denominator = -INFINITY;
                for (std::size_t t = 0; t < num_observations - 1; ++t) {
                    denominator = log_add(denominator, gamma[i * num_observations + t]);
                }

                for (std::size_t j = 0; j < num_states; ++j) {
                    double numerator = -INFINITY;
                    for (std::size_t t = 0; t < num_observations - 1; ++t) {
                        numerator = log_add(numerator, xi[i * num_states * num_observations + j * num_observations + t]);
                    }

                    log_transition[i * num_states + j] = numerator - denominator;
                }
            }

            // update emission matrix
            for (std::size_t i = 0; i < num_states; ++i) {
                double denominator = -INFINITY;
                for (std::size_t t = 0; t < num_observations; ++t) {
                    denominator = log_add(denominator, gamma[i * num_observations + t]);
                }

                for (std::size_t k = 0; k < num_emissions; ++k) {
                    double numerator = -INFINITY;
                    for (std::size_t t = 0; t < num_observations; ++t) {
                        if (observations[t] == int(k)) {
                            numerator = log_add(numerator, gamma[i * num_observations + t]);
                        }
                    }

                    log_emission[i * num_emissions + k] = numerator - denominator;
                }
            }

            delete[] gamma;
            delete[] xi;

            // check if the max iteration is reached
            if (iter >= max_iters) {
                std::cout << "Max iteration reached." << std::endl;
                break;
            }

            if (std::abs(alpha_sum - previous_alpha_sum) < tolerance) {
                std::cout << "Converged at iteration " << iter << std::endl;
                break;
            }

            // std::cout << std::abs(alpha_sum - previous_alpha_sum) << std::endl;
            previous_alpha_sum = alpha_sum;

            ++iter;
        }

        // transform initial from log to normal
        for (std::size_t i = 0; i < num_states; ++i) {
            initial[i] = std::exp(log_initial[i]);
        }

        // transform transition matrix and emission matrix from log to normal
        for (std::size_t i = 0; i < num_states * num_states; ++i) {
            transition[i] = std::exp(log_transition[i]);
        }
        for (std::size_t i = 0; i < num_states * num_emissions; ++i) {
            emission[i] = std::exp(log_emission[i]);
        }

        delete[] log_initial;
        delete[] log_transition;
        delete[] log_emission;
    }
}