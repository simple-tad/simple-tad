#include "benchmark/benchmark.h"
#include "di.hpp"
#include "baum_welch_simd.hpp"
#include "baum_welch.hpp"
#include <algorithm>
#include <iostream>

double* calculate_di_SCALAR(const double* contact_matrix, const std::size_t& edge_size, const std::size_t& range) {
    double* di = new double[edge_size]();

    for (std::size_t locus_index = 0; locus_index < edge_size; ++locus_index) {
        double A;
        double B;

        if (locus_index < range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0f);
        } else if (locus_index >= edge_size - range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + (locus_index + 1) * edge_size, 0.0f);
        } else {
            // normal case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0f);
        }

        double E = (A + B) / 2;

        if (A == 0 && B == 0) {
            di[locus_index] = 0;
        } else {
            try {
                di[locus_index] = ((B - A) / (std::abs(B - A))) * ((((A - E) * (A - E)) / E) + (((B - E) * (B - E)) / E));
            } catch (std::exception& e) {
                di[locus_index] = 0;
            }
        }
    }
    return di;
}

static void BM_calculate_di_SCALAR(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random positive doubles
    std::generate(data.begin(), data.end(), []() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); });

    for (auto _ : state) {
        calculate_di_SCALAR(data.data(), edge_size, 2);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * edge_size * sizeof(double) / 8, benchmark::Counter::kIsRate);
}

static void BM_calculate_di_AVX2(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random positive doubles
    std::generate(data.begin(), data.end(), []() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); });

    for (auto _ : state) {
        calculate_di_AVX2(data.data(), edge_size, 40);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * edge_size * sizeof(double) / 8, benchmark::Counter::kIsRate);
}

static void BM_baum_welch_simd(banchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random 0, 1, 2
    srand(1000);
    std::generate(data.begin(), data.end(), []() { return static_cast<int>(rand()) % 3; });

    double* initial = new double[3] { 0.4, 0.3, 0.3 };
    double* transition = new double[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };
    double* emission = new double[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.1, 0.7
    };

    for (auto _ : state) {
        vectorized::baum_welch(di_discrete, edge_size, initial, transition, emission, 3, 3);
    }
}

static void BM_baum_welch(banchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random 0, 1, 2
    srand(1000);
    std::generate(data.begin(), data.end(), []() { return static_cast<int>(rand()) % 3; });

    double* initial = new double[3] { 0.4, 0.3, 0.3 };
    double* transition = new double[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };
    double* emission = new double[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.1, 0.7
    };

    for (auto _ : state) {
        baum_welch(di_discrete, edge_size, initial, transition, emission, 3, 3);
    }
}

BENCHMARK(BM_calculate_di_SCALAR)->Arg(33957);
BENCHMARK(BM_calculate_di_AVX2)->Arg(33957);

BENCHMARK_MAIN();