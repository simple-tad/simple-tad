#include "constants.h"
#include <algorithm>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

const std::tuple<double*, std::size_t> read_hi_c_data(const std::string& filename, const std::size_t& bin_size, const std::size_t& bin1_min, const std::size_t& bin1_max, const std::size_t& bin2_min, const std::size_t& bin2_max) {
    std::fstream file;
    file.open(filename, std::ios::in);

    // check if file is open
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t edge_size = std::max((bin1_max - bin1_min), (bin2_max - bin2_min)) / bin_size + 1;

    double* data = new double[edge_size * edge_size];
    std::string line;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chr;
        std::size_t bin1;
        std::size_t bin2;
        double rescaled_intensity;
        std::size_t diag_offset;
        std::size_t dist;

        std::getline(ss, chr, ',');
        ss >> bin1;
        ss.ignore();
        ss >> bin2;
        ss.ignore();
        ss >> rescaled_intensity;
        ss.ignore();
        ss >> diag_offset;
        ss.ignore();
        ss >> dist;

        if (bin1 >= bin1_min && bin1 <= bin1_max && bin2 >= bin2_min && bin2 <= bin2_max) {
            std::size_t row = (bin1 - bin1_min) / bin_size;
            std::size_t col = (bin2 - bin2_min) / bin_size;
            data[row * edge_size + col] = rescaled_intensity;
        } else {
            std::cout << "chr: " << chr << " bin1: " << bin1 << " bin2: " << bin2 << " rescaled_intensity: " << rescaled_intensity << " diag_offset: " << diag_offset << " dist: " << dist << std::endl;
        }
    }
    return std::make_tuple(data, edge_size);
}

std::vector<double> calculate_di(double* contact_matrix, const std::size_t& edge_size, const std::size_t& bin_size) {
    std::size_t range = SIGNIFICANT_BINS / bin_size;
    std::vector<double> A(edge_size, 0);
    std::vector<double> B(edge_size, 0);
    std::vector<double> E(edge_size, 0);
    std::vector<double> di(edge_size, 0);

    for (std::size_t row = 0; row < range; row++) {
        double a = std::accumulate(contact_matrix + row * edge_size + row - range, contact_matrix + row * edge_size + row, 0.0);
        double b = std::accumulate(contact_matrix + row * edge_size + row + 1, contact_matrix + row * edge_size + row + range + 1, 0.0);
        A[row] = a;
        B[row] = b;
        E[row] = (a + b) / 2;
    }

    for (std::size_t row = (edge_size - range) + 1; row < edge_size; row++) {
        double a = std::accumulate(contact_matrix + row * edge_size + row - range, contact_matrix + row * edge_size + row, 0.0);
        double b = std::accumulate(contact_matrix + row * edge_size + row + 1, contact_matrix + row * edge_size + row + range + 1, 0.0);
        A[row] = a;
        B[row] = b;
        E[row] = (a + b) / 2;
    }

    // align diagonal data
    for (std::size_t row = range; row < (edge_size - range) + 1; row++) {
        for (std::size_t col = 0; col < 2 * range; col++) {
            contact_matrix[row * edge_size + col] = contact_matrix[row * edge_size + col + (row - range)];
        }
    }

    // transponse matrix
    double* transposed_matrix = new double[edge_size * edge_size];
    for (std::size_t row = 0; row < edge_size; row++) {
        for (std::size_t col = 0; col < edge_size; col++) {
            transposed_matrix[row * edge_size + col] = contact_matrix[col * edge_size + row];
        }
    }

    // use _mm256_add_pd to calculate A, B and E
    for (std::size_t col = range; col < (edge_size - range) + 1; col += 4) {
        for (std::size_t row = 0; row < range - 1; row++) {
            __m256d vA = _mm256_loadu_pd(transposed_matrix + row * edge_size + col);
            __m256d vA_next = _mm256_loadu_pd(transposed_matrix + (row + 1) * edge_size + col);
            _mm256_storeu_pd(transposed_matrix + (row + 1) * edge_size + col, _mm256_add_pd(vA, vA_next));
        }

        A[col] = transposed_matrix[range * edge_size + col];
        A[col + 1] = transposed_matrix[range * edge_size + col + 1];
        A[col + 2] = transposed_matrix[range * edge_size + col + 2];
        A[col + 3] = transposed_matrix[range * edge_size + col + 3];

        for (std::size_t row = range; row < 2 * range - 1; row++) {
            __m256d vB = _mm256_loadu_pd(transposed_matrix + row * edge_size + col);
            __m256d vB_next = _mm256_loadu_pd(transposed_matrix + (row + 1) * edge_size + col);
            _mm256_storeu_pd(transposed_matrix + (row + 1) * edge_size + col, _mm256_add_pd(vB, vB_next));
        }

        B[col] = transposed_matrix[2 * range * edge_size + col];
        B[col + 1] = transposed_matrix[2 * range * edge_size + col + 1];
        B[col + 2] = transposed_matrix[2 * range * edge_size + col + 2];
        B[col + 3] = transposed_matrix[2 * range * edge_size + col + 3];

        E[col] = (A[col] + B[col]) / 2;
        E[col + 1] = (A[col + 1] + B[col + 1]) / 2;
        E[col + 2] = (A[col + 2] + B[col + 2]) / 2;
        E[col + 3] = (A[col + 3] + B[col + 3]) / 2;
    }

    for (std::size_t row = 0; row < edge_size; row += 4) {
        __m256d vA = _mm256_loadu_pd(A.data() + row);
        __m256d vB = _mm256_loadu_pd(B.data() + row);
        __m256d vE = _mm256_loadu_pd(E.data() + row);

        __m256d vAB = _mm256_sub_pd(vB, vA);
        __m256d vAE = _mm256_sub_pd(vA, vE);
        __m256d vBE = _mm256_sub_pd(vB, vE);

        // (B - A) / (std::abs(B - A))
        __m256d vSign = _mm256_cmp_pd(vAB, _mm256_setzero_pd(), _CMP_GT_OQ);
        vSign = _mm256_sub_pd(_mm256_mul_pd(vSign, _mm256_set1_pd(2)), _mm256_set1_pd(1));
        __m256d vSignBA2 = _mm256_div_pd(vAB, vAB);
        vSignBA2 = _mm256_mul_pd(vSignBA2, vSign);

        // (((A - E) * (A - E)) / E)
        __m256d vAE2 = _mm256_mul_pd(vAE, vAE);
        __m256d vAE2E = _mm256_div_pd(vAE2, vE);

        // (((B - E) * (B - E)) / E)
        __m256d vBE2 = _mm256_mul_pd(vBE, vBE);
        __m256d vBE2E = _mm256_div_pd(vBE2, vE);

        // ((B - A) / (std::abs(B - A))) * ((((A - E) * (A - E)) / E) + (((B - E) * (B - E)) / E));
        __m256d vResult = _mm256_add_pd(vAE2E, vBE2E);
        vResult = _mm256_mul_pd(vResult, vSignBA2);

        _mm256_storeu_pd(di.data() + row, vResult);
    }

    return di;
}
