#include "di.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto& [data, edge_size] = read_hi_c_data("./data/GM12878_MboI_Diag_chr6.csv", 5000, 305000, 170085000, 825000, 170605000);
    auto end_read_data = std::chrono::high_resolution_clock::now();
    auto di = calculate_di(data, edge_size, 5000);
    auto end_calculate_di = std::chrono::high_resolution_clock::now();

    std::cout << "di: " << std::endl;
    for (std::size_t i = 0; i < edge_size; i++) {
        std::cout << di[i] << std::endl;
    }

    std::cout << "Read data: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_read_data - start).count() << "ns" << std::endl;
    std::cout << "Calculate di: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_calculate_di - end_read_data).count() << "ns" << std::endl;

    return 0;
}