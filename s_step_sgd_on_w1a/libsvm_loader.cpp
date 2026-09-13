#include "libsvm_loader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void load_libsvm(
    DataParams* data_params,
    std::vector<float>& h_A,
    std::vector<float>& h_y)
{
    std::ifstream file(data_params->file_name);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + data_params->file_name);

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss{line};

        float label{};
        ss >> label;
        h_y.push_back(label);

        const size_t row_offset{h_A.size()};
        h_A.resize(row_offset + data_params->n_features, 0.0f);
        float* row = h_A.data() + row_offset;

        std::string token;
        while (ss >> token) {
            const auto pos = token.find(':');
            if (pos == std::string::npos)
                continue;

            const int feature_index = std::stoi(token.substr(0, pos));
            const size_t idx = static_cast<size_t>(feature_index - 1);
            const float val{std::stof(token.substr(pos + 1))};
            if (idx < data_params->n_features) {
                row[idx] = val;
            }
        }
    }
}
