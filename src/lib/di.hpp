#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

double accumulate_AVX2(const double* data, std::size_t size) {
    double sum = 0;

    std::size_t remain = size % 4;
    size -= remain;

    __m256d sum_vec = _mm256_setzero_pd();
    __m256d data_vec;

    for (std::size_t i = 0; i < size; i += 4) {
        data_vec = _mm256_loadu_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, data_vec);
    }
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    sum_vec = _mm256_permute4x64_pd(sum_vec, 0b11011000);
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    _mm256_storeu_pd(&sum, sum_vec);

    for (std::size_t i = size; i < size + remain; i++) {
        sum += data[i];
    }

    return sum;
}

double* calculate_di_AVX2(const double* contact_matrix, const std::size_t& edge_size, const std::size_t& range) {
    double* di = new double[edge_size]();

    for (std::size_t locus_index = 0; locus_index < edge_size; ++locus_index) {
        double A;
        double B;
        if (locus_index < range) {
            // edge case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size, locus_index);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, range);
        } else if (locus_index >= edge_size - range) {
            // edge case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index - range, range);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, edge_size - locus_index - 1);
        } else {
            // normal case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index - range, range);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, range);
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

    for (std::size_t i = 0; i < edge_size; ++i) {
        if (std::isnan(di[i])) {
            di[i] = 0;
        }
    }

    return di;
}
