#define  _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
// #include <algorithm>

double inline log_add(double x, double y) {
    if (y <= x) {
        return x + std::log1p(std::exp(y - x));
    } else {
        return y + std::log1p(std::exp(x - y));
    }
}

inline float calc_log_pB(const float& obs, const float& mu, const float& sigma) {
    float log_factor = -std::log1p(sigma * std::sqrt(2 * M_PI));
    return log_factor - ((obs-mu)*(obs-mu)) / (2*sigma*sigma);
}

double* forward(
    const double* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const double* initial,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the alpha matrix
    double* alpha = new double[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        alpha[i * num_observations] = initial[i] + calc_log_pB(observations[0], emission[i * num_emissions], emission[i * num_emissions + 1]);;
    }

    // compute the alpha matrix
    for (std::size_t t = 1; t < num_observations; ++t) {
        // now is observation t, and t is from 1 because we skip init observation
        for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
            double sum = -INFINITY;
            for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
                sum = log_add(sum, alpha[prev_state * num_observations + t - 1] + transition[prev_state * num_states + curr_state]);
            }
            alpha[curr_state * num_observations + t] = sum + calc_log_pB(observations[t], emission[curr_state * num_emissions], emission[curr_state * num_emissions + 1]);
        }
    }

    return alpha;
}

double* backward(
    const double* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the beta matrix
    double* beta = new double[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        beta[i * num_observations + num_observations - 1] = 0; // log(1)
    }

    // compute the beta matrix
    for (std::size_t t = num_observations - 2;; --t) {
        // now is observation t, and t is from (num_observations - 2) because we already init last observation
        for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
            float sum = -INFINITY;
            for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
                sum = log_add(sum, beta[curr_state * num_observations + t + 1] + transition[prev_state * num_states + curr_state] + calc_log_pB(observations[t + 1], emission[curr_state * num_emissions], emission[curr_state * num_emissions + 1]));
            }
            beta[prev_state * num_observations + t] = sum;
        }

        if (t == 0) {
            break;
        }
    }

    return beta;
}

double* compute_gamma(
    const double* alpha,
    const double* beta,
    const std::size_t num_observations,
    const std::size_t num_states
) {
    float* gamma = new float[num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations; ++t) {
        double sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            gamma[i * num_observations + t] = alpha[i * num_observations + t] + beta[i * num_observations + t];
            sum = log_add(sum, gamma[i * num_observations + t]);
        }

        for (std::size_t i = 0; i < num_states; ++i) {
            gamma[i * num_observations + t] -= sum;
        }
    }

    return gamma;
}

double* compute_xi(
    const double* alpha,
    const double* beta,
    const double* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions
) {
    float* xi = new float[num_states * num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations - 1; ++t) {
        double sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_states; ++j) {
                xi[i * num_states * num_observations + j * num_observations + t] = alpha[i * num_observations + t] + transition[i * num_states + j] + calc_log_pB(observations[t + 1], emission[j * num_emissions], emission[j * num_emissions + 1]) + beta[j * num_observations + t + 1];
                sum = log_add(sum, xi[i * num_states * num_observations + j * num_observations + t]);
            }
        }

        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_states; ++j) {
                xi[i * num_states * num_observations + j * num_observations + t] -= sum;
            }
        }
    }

    return xi;
}

void baum_welch(
    const double* observations,
    const std::size_t num_observations,
    double* initial,
    double* transition,
    double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions,
    const float tolerance = 1e-5,
    const std::size_t max_iters = 100
) {
    // transform initial to log space
    double* log_initial = new double[num_states];
    for (std::size_t i = 0; i < num_states; ++i) {
        log_initial[i] = std::log1p(initial[i]);
    }

    // transform transition matrix and emission matrix to log space
    double* log_transition = new double[num_states * num_states];
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        log_transition[i] = std::log1p(transition[i]);
    }

    std::size_t iter = 0;
    float prev_loglik = INFINITY;
    while (true) {
        float transition_diff = INFINITY;
        float emission_diff = INFINITY;

        auto alpha = forward(observations, num_observations, log_transition, emission, log_initial, num_states, num_emissions);
        auto beta = backward(observations, num_observations, log_transition, emission, num_states, num_emissions);

        // check if the max iteration is reached
        if (iter >= max_iters) {
            std::cout << "Max iteration reached." << std::endl;

            delete[] alpha;
            delete[] beta;
            break;
        }

        float loglik = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            loglik = log_add(loglik, alpha[i * num_observations + num_observations - 1]);
        }
        std::cerr << loglik << std::endl;

        std::cerr << "emissions: " << std::endl;
        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_emissions; ++j) {
                std::cerr << (emission[i * num_emissions + j]) << ' ';
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
        
        // compute gamma and xi
        auto gamma = compute_gamma(alpha, beta, num_observations, num_states);
        auto xi = compute_xi(alpha, beta, observations, num_observations, log_transition, emission, num_states, num_emissions);

        delete[] alpha;
        delete[] beta;

        // update initial
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] = gamma[i * num_observations];
        }
        
        float* sum_gamma = new float[num_states];

        // update transition matrix
        for (std::size_t i = 0; i < num_states; ++i) {
            float denominator = -INFINITY;
            for (std::size_t t = 0; t < num_observations - 1; ++t) {
                denominator = log_add(denominator, gamma[i * num_observations + t]);
            }

            for (std::size_t j = 0; j < num_states; ++j) {
                float sum = -INFINITY;
                for (std::size_t t = 0; t < num_observations - 1; ++t) {
                    numerator = log_add(numerator, xi[i * num_states * num_observations + j * num_observations + t]);
                }

                transition_diff += std::abs(std::exp(log_transition[i * num_states + j]) - std::exp(numerator - denominator));

                log_transition[i * num_states + j] = numerator - denominator;
            }
        }

        // update emission matrix
        for (std::size_t i = 0; i < num_states; ++i) {
            float denominator = -INFINITY;
            for (std::size_t t = 0; t < num_observations; ++t) {
                denominator = log_add(denominator, gamma[i * num_observations + t]);
            }
            denominator = std::exp(denominator);

            for (std::size_t t = 0; t < num_observations; ++t) {
                gamma[i * num_observations + t] = std::exp(gamma[i * num_observations + t]);
            }

            if (denominator == 0) continue;

            float prev_mean = emission[i * num_emissions];
            float prev_variance = emission[i * num_emissions + 1];

            emission[i * num_emissions] = 0;
            emission[i * num_emissions + 1] = 0;

            // O - mu can be negative so can't calculate sum as log without safe sum
            for (std::size_t t = 0; t < num_observations; ++t) {
                // gamma[i * num_observations + t] = std::exp(gamma[i * num_observations + t]);
                emission[i * num_emissions] += gamma[i * num_observations + t] * observations[t]; // E[Observation]
            }
            emission[i * num_emissions] /= denominator; // E[Observation] = mu

            for (std::size_t t = 0; t < num_observations; ++t)
                emission[i * num_emissions + 1] += gamma[i * num_observations + t] * (observations[t] - emission[i * num_emissions]) * (observations[t] - emission[i * num_emissions]); // E[ (Obs-mu)^2 ]
            
            emission[i * num_emissions + 1] = std::sqrt(emission[i * num_emissions + 1] / denominator); // sigma = sqrt( E[ (Obs-mu)^2 ] )

            emission_diff += std::abs(emission[i * num_emissions] - prev_mean) + std::abs(emission[i * num_emissions + 1] - prev_variance);
        }

        delete[] gamma;
        delete[] xi;

        // std::cerr << "transition_diff: " << transition_diff << std::endl;
        // std::cerr << "emission_diff: " << emission_diff << std::endl;
        // std::cerr << std::endl;

        bool flag = false;
        if (transition_diff < tolerance || emission_diff < tolerance) {
            flag = true;
        }

        for (std::size_t i = 0; i < num_states; ++i) {
            if (emission[i * num_emissions + 1] < 0.1) {
                flag = true;
            }
        }

        if (flag) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }

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

    delete[] log_initial;
    delete[] log_transition;
}