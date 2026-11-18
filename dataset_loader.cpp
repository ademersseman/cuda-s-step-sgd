#include "dataset_loader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// magic sequence for binary s-step dataset files: "SSTEPDS1"
constexpr std::array<char, 8> kDataMagic{'S', 'S', 'T', 'E', 'P', 'D', 'S', '1'};

// Read the dense SSTEP binary payload and convert its labels to floats.
void load_binary(DataParams& data_params, std::ifstream& file, std::vector<float>& host_features,
                 std::vector<float>& host_labels) {
    std::uint64_t sample_count = 0;
    std::uint64_t feature_count = 0;
    file.read(reinterpret_cast<char*>(&sample_count), sizeof(sample_count));
    file.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
    if (!file || sample_count == 0 || feature_count == 0)
        throw std::runtime_error("Invalid s-step dataset header");
    data_params.feature_count = static_cast<std::size_t>(feature_count);
    host_features.resize(static_cast<std::size_t>(sample_count * feature_count));
    file.read(reinterpret_cast<char*>(host_features.data()),
              static_cast<std::streamsize>(host_features.size() * sizeof(float)));
    std::vector<std::int8_t> stored_labels(static_cast<std::size_t>(sample_count));
    file.read(reinterpret_cast<char*>(stored_labels.data()),
              static_cast<std::streamsize>(stored_labels.size()));
    if (!file)
        throw std::runtime_error("Truncated s-step dataset: " + data_params.dataset_path);
    host_labels.resize(static_cast<std::size_t>(sample_count));
    for (std::size_t sample_index = 0; sample_index < host_labels.size(); ++sample_index) {
        host_labels[sample_index] = static_cast<float>(stored_labels[sample_index]);
    }
}

// Parse sparse LIBSVM text rows into a dense row-major host matrix.
void load_text(DataParams& data_params, std::ifstream& file, std::vector<float>& host_features,
               std::vector<float>& host_labels) {
    std::size_t width = 0;
    std::size_t row_count = 0;
    // Determine the stride before allocating the dense matrix. The widest row can be last.
    for (bool fill : {false, true}) {
        std::string line;
        std::size_t row = 0;
        while (std::getline(file, line)) {
            std::istringstream fields{line};
            fields >> std::ws;
            if (fields.eof() || fields.peek() == '#')
                continue;
            float label;
            if (!(fields >> label))
                throw std::runtime_error("Invalid LIBSVM label");
            if (fill) {
                if (row >= row_count)
                    throw std::runtime_error("LIBSVM row count changed while loading");
                host_labels[row] = label;
            }

            std::string token;
            while (fields >> token && token.front() != '#') {
                const auto colon = token.find(':');
                if (colon == std::string::npos)
                    throw std::runtime_error("Invalid LIBSVM feature: " + token);
                int column;
                const char* end = token.data() + colon;
                const auto parsed = std::from_chars(token.data(), end, column);
                if (parsed.ec != std::errc{} || parsed.ptr != end || column < 1)
                    throw std::runtime_error("Invalid LIBSVM feature index: " + token);
                const auto index = static_cast<std::size_t>(column);
                if (fill) {
                    if (index > width)
                        throw std::runtime_error("LIBSVM feature count changed while loading");
                    std::size_t consumed;
                    const float value = std::stof(token.substr(colon + 1), &consumed);
                    if (consumed != token.size() - colon - 1)
                        throw std::runtime_error("Invalid LIBSVM feature value: " + token);
                    host_features[row * width + index - 1] = value;
                } else {
                    width = std::max(width, index);
                }
            }
            ++row;
        }
        if (!file.eof() || file.bad())
            throw std::runtime_error("Error reading LIBSVM dataset: " + data_params.dataset_path);
        if (!fill) {
            row_count = row;
            if (width == 0 || row_count > host_features.max_size() / width)
                throw std::runtime_error("Invalid LIBSVM dataset dimensions");
            host_features.assign(row_count * width, 0.0f);
            host_labels.resize(row_count);
            file.clear();
            file.seekg(0);
        } else if (row != row_count) {
            throw std::runtime_error("LIBSVM row count changed while loading");
        }
    }
    data_params.feature_count = width;
}
// Accept a single binary encoding per file. Internal zero labels are reserved for padding.
void normalize_binary_labels(std::vector<float>& labels) {
    bool has_zero = false;
    bool has_negative = false;
    for (float label : labels) {
        has_zero |= label == 0.0f;
        has_negative |= label == -1.0f;
        if ((label != -1.0f && label != 0.0f && label != 1.0f) || (has_zero && has_negative))
            throw std::runtime_error("Expected binary labels encoded as {-1,+1} or {0,1}");
    }
    if (has_zero)
        std::replace(labels.begin(), labels.end(), 0.0f, -1.0f);
}
} // namespace

// Detect the dataset format from its magic bytes and dispatch to the matching loader.
void load_dataset(DataParams& data_params, std::vector<float>& host_features,
                  std::vector<float>& host_labels) {
    std::ifstream file(data_params.dataset_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + data_params.dataset_path);
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (file && magic == kDataMagic) {
        load_binary(data_params, file, host_features, host_labels);
    } else {
        file.clear();
        file.seekg(0);
        load_text(data_params, file, host_features, host_labels);
    }
    normalize_binary_labels(host_labels);
}
