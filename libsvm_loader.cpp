#include "libsvm_loader.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<char, 8> kDataMagic{'S', 'S', 'T', 'E', 'P', 'D', 'S', '1'};

void load_binary(DataParams* data_params, std::ifstream& file, std::vector<float>& host_features,
                 std::vector<float>& host_labels) {
    std::uint64_t sample_count = 0;
    std::uint64_t feature_count = 0;
    file.read(reinterpret_cast<char*>(&sample_count), sizeof(sample_count));
    file.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
    if (!file || sample_count == 0 || feature_count == 0)
        throw std::runtime_error("Invalid s-step dataset header");
    data_params->feature_count = static_cast<size_t>(feature_count);
    data_params->is_sstep_binary = true;
    host_features.resize(static_cast<size_t>(sample_count * feature_count));
    file.read(reinterpret_cast<char*>(host_features.data()),
              static_cast<std::streamsize>(host_features.size() * sizeof(float)));
    std::vector<std::int8_t> stored_labels(static_cast<size_t>(sample_count));
    file.read(reinterpret_cast<char*>(stored_labels.data()),
              static_cast<std::streamsize>(stored_labels.size()));
    if (!file)
        throw std::runtime_error("Truncated s-step dataset: " + data_params->dataset_path);
    host_labels.resize(static_cast<size_t>(sample_count));
    for (size_t sample_index = 0; sample_index < host_labels.size(); ++sample_index) {
        host_labels[sample_index] = static_cast<float>(stored_labels[sample_index]);
    }
}

void load_text(DataParams* data_params, std::ifstream& file, std::vector<float>& host_features,
               std::vector<float>& host_labels) {
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream line_stream{line};
        float label{};
        line_stream >> label;
        host_labels.push_back(label);
        const size_t row_offset{host_features.size()};
        host_features.resize(row_offset + data_params->feature_count, 0.0f);
        float* row = host_features.data() + row_offset;
        std::string token;
        while (line_stream >> token) {
            const auto separator_position = token.find(':');
            if (separator_position == std::string::npos)
                continue;
            const size_t feature_index =
                static_cast<size_t>(std::stoi(token.substr(0, separator_position)) - 1);
            if (feature_index < data_params->feature_count)
                row[feature_index] = std::stof(token.substr(separator_position + 1));
        }
    }
}
} // namespace

void load_dataset(DataParams* data_params, std::vector<float>& host_features,
                  std::vector<float>& host_labels) {
    std::ifstream file(data_params->dataset_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + data_params->dataset_path);
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (file && magic == kDataMagic) {
        load_binary(data_params, file, host_features, host_labels);
    } else {
        file.clear();
        file.seekg(0);
        load_text(data_params, file, host_features, host_labels);
    }
}
