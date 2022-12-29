#include "types.h"
#include <iostream>
#include <vector>

/*
    Function to calculate coord of states.

    1. are there some megative states before the first positive state? if yes, go to 2, else go to 3.
    2. shift to the next negative state and check again. go to 1
    3. are there any haps between two consecutive positive states? if yes, go to 4, else go to 5.
    4. remove the gap between two consecutive positive states and check again. go to 3
    5. are there any gaps between two consecutive negative states? if yes, go to 6, else go to 7.
    6. remove the gap between two consecutive negative states and check again. go to 5
    7. record the coord of the states. go to 8
    8. move on to the next couple of positive and negative states. go to 3
*/
const std::vector<std::pair<std::size_t, std::size_t>> calculate_coord(const BiasState* const& states, const std::size_t length) {
    std::vector<std::pair<std::size_t, std::size_t>> coords;
    std::vector<std::size_t> possible_start_coords;
    std::vector<std::size_t> possible_end_coords;

    for (std::size_t i = 0; i < length; ++i) {
        if (i == 0 && states[i] == BiasState::UpstreamBias) {
            possible_start_coords.push_back(i);
        } else if (states[i] == BiasState::UpstreamBias && (states[i - 1] == BiasState::DownstreamBias || states[i - 1] == BiasState::NoBias)) {
            possible_start_coords.push_back(i);
        }

        if (i == length - 1 && states[i] == BiasState::DownstreamBias) {
            possible_end_coords.push_back(i);
        } else if (states[i] == BiasState::DownstreamBias && (states[i + 1] == BiasState::UpstreamBias || states[i + 1] == BiasState::NoBias)) {
            possible_end_coords.push_back(i);
        }
    }

    if (possible_start_coords.size() == 0 || possible_end_coords.size() == 0) {
        return coords;
    }

    std::size_t curr_start = 0;
    std::size_t next_start = 1;

    std::size_t curr_end = 0;
    std::size_t next_end = 1;

    std::size_t prev_end = 0;

    while (possible_start_coords[curr_start] > possible_end_coords[curr_end] && curr_end < possible_end_coords.size() - 1 && next_end < possible_end_coords.size() - 1) {
        ++curr_end;
        ++next_end;
    }

    while (curr_start < possible_start_coords.size() - 1 && curr_end < possible_end_coords.size() - 1) {
        while (possible_start_coords[next_start] < possible_end_coords[curr_end] && next_start < possible_start_coords.size() - 1) {
            ++next_start;
        }

        while (possible_start_coords[next_start] > possible_end_coords[next_end] && next_end < possible_end_coords.size() - 1) {
            ++curr_end;
            ++next_end;
        }

        coords.emplace_back(possible_start_coords[curr_start], possible_end_coords[curr_end] + 1);

        curr_start = next_start;
        ++next_start;

        prev_end = curr_end;
        curr_end = next_end;
        ++next_end;
    }

    if (possible_start_coords[curr_start] < possible_end_coords[curr_end] && possible_start_coords[curr_start] > possible_end_coords[prev_end]) {
        coords.emplace_back(possible_start_coords[curr_start], possible_end_coords[curr_end] + 1);
    }

    return coords;
}