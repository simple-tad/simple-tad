#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

const std::tuple<float*, std::size_t> read_hi_c_data(const std::string& filename, const std::size_t& bin_size, const std::size_t& bin1_min, const std::size_t& bin1_max, const std::size_t& bin2_min, const std::size_t& bin2_max) {
    std::fstream file;
    file.open(filename, std::ios::in);

    // check if file is open
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t edge_size = (std::max(bin1_max, bin2_max) - std::min(bin1_min, bin2_min)) / bin_size + 1;

    float* data = new float[edge_size * edge_size]();
    std::string line;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chr;
        std::size_t bin1;
        std::size_t bin2;
        float rescaled_intensity;

        std::getline(ss, chr, ',');
        ss >> bin1;
        ss.ignore();
        ss >> bin2;
        ss.ignore();
        ss >> rescaled_intensity;

        if (bin1 >= bin1_min && bin1 <= bin1_max && bin2 >= bin2_min && bin2 <= bin2_max) {
            std::size_t row = (bin1 - std::min(bin1_min, bin2_min)) / bin_size;
            std::size_t col = (bin2 - std::min(bin1_min, bin2_min)) / bin_size;
            data[row * edge_size + col] = rescaled_intensity;
            data[col * edge_size + row] = rescaled_intensity;
        } else {
            std::cout << "chr: " << chr << " bin1: " << bin1 << " bin2: " << bin2 << " rescaled_intensity: " << rescaled_intensity << std::endl;
        }
    }
    return std::make_tuple(data, edge_size);
}