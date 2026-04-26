#pragma once
#include <vector>

namespace cvlite::core {

struct Parameter {
    std::vector<float> data;
    std::vector<float> grad;

    Parameter() = default;
    
    explicit Parameter(size_t size) 
        : data(size, 0.0f), grad(size, 0.0f) {}

    void resize(size_t size) {
        data.assign(size, 0.0f);
        grad.assign(size, 0.0f);
    }
};

} // namespace cvlite::core