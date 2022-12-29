#include "lib/baum_welch.hpp"
#include "lib/coord.hpp"
#include "lib/di.hpp"
#include "lib/viterbi.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(simple_tad, m) {
    m.doc() = "A C++ implementation of calculating TADs.";

    m.def(
        "calculate_tad_coords", [](py::array_t<float> data_array, std::size_t edge_size, std::size_t bin_size, std::size_t range, float discrete_threshold, float tolerance, std::size_t max_iters) {
            py::buffer_info data_buf = data_array.request();
            float* data = static_cast<float*>(data_buf.ptr);

            auto di = calculate_di_AVX2(data, edge_size, range);
            int* di_discrete = new int[edge_size];

            std::size_t upstream_count = 0;
            std::size_t downstream_count = 0;
            std::size_t nobias_count = 0;

            for (std::size_t i = 0; i < edge_size; i++) {
                if (di[i] > discrete_threshold) {
                    di_discrete[i] = 0;
                    ++upstream_count;
                } else if (di[i] < -discrete_threshold) {
                    di_discrete[i] = 1;
                    ++downstream_count;
                } else {
                    di_discrete[i] = 2;
                    ++nobias_count;
                }
            }

            std::cout << "upstream_count: " << upstream_count << std::endl;
            std::cout << "downstream_count: " << downstream_count << std::endl;
            std::cout << "nobias_count: " << nobias_count << std::endl;

            float* initial = new float[3] { 0.3, 0.3, 0.4 };
            float* transition = new float[3 * 3] {
                0.5, 0.3, 0.2,
                0.2, 0.5, 0.3,
                0.3, 0.2, 0.5
            };
            float* emission = new float[3 * 3] {
                0.7, 0.1, 0.2,
                0.1, 0.7, 0.2,
                0.1, 0.1, 0.8
            };

            baum_welch(di_discrete, edge_size, initial, transition, emission, 3, 3, tolerance, max_iters); // side effect: update initial, transition, emission

            std::cout << "Estimated initial probability:" << std::endl;
            for (std::size_t i = 0; i < 3; ++i) {
                std::cout << std::setiosflags(std::ios::fixed) << initial[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Estimated transition matrix:" << std::endl;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    std::cout << std::setiosflags(std::ios::fixed) << transition[i * 3 + j] << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "Estimated emission matrix:" << std::endl;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    std::cout << std::setiosflags(std::ios::fixed) << emission[i * 3 + j] << " ";
                }
                std::cout << std::endl;
            }

            auto states = scalar::viterbi(di_discrete, edge_size, initial, transition, emission);

            upstream_count = 0;
            downstream_count = 0;
            nobias_count = 0;

            for (std::size_t i = 0; i < edge_size; i++) {
                if (states[i] == 0) {
                    ++upstream_count;
                } else if (states[i] == 1) {
                    ++downstream_count;
                } else {
                    ++nobias_count;
                }
            }

            std::cout << "upstream_count: " << upstream_count << std::endl;
            std::cout << "downstream_count: " << downstream_count << std::endl;
            std::cout << "nobias_count: " << nobias_count << std::endl;

            auto coords = calculate_coord(reinterpret_cast<BiasState*>(states), edge_size);

            delete[] di;
            delete[] di_discrete;

            delete[] initial;
            delete[] transition;
            delete[] emission;

            delete[] states;

            // coords.erase(std::remove_if(coords.begin(), coords.end(), [](const std::pair<std::size_t, std::size_t>& coord) {
            //     return coord.second - coord.first < 10;
            // }),
            //     coords.end());

            return coords;
        },
        py::arg("data"), py::arg("edge_size"), py::arg("bin_size"), py::arg("range"), py::arg("discrete_threshold"), py::arg("tolerance"), py::arg("max_iters"), "Calculate TADs from Hi-C data.");
}
