#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

float accumulate_AVX2(const float* data, std::size_t size) {
    float sum = 0;

    std::size_t remain = size % 8;
    size -= remain;

    __m256 sum_vec = _mm256_setzero_ps();
    __m256 data_vec;

    for (std::size_t i = 0; i < size; i += 8) {
        data_vec = _mm256_loadu_ps(data + i);
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
    }

    __m256 swap = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
    sum_vec = _mm256_add_ps(sum_vec, swap);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    _mm256_storeu_ps(&sum, sum_vec);

    for (std::size_t i = size; i < size + remain; ++i) {
        sum += data[i];
    }

    return sum;
}

float* calculate_di_AVX2(const float* contact_matrix, const std::size_t& edge_size, const std::size_t& range) {
    float* di = new float[edge_size]();

    for (std::size_t locus_index = 0; locus_index < edge_size; ++locus_index) {
        float A;
        float B;
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

        float E = (A + B) / 2;

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
