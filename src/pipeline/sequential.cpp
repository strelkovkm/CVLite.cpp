#include "cvlite/pipeline/sequential.hpp"

#include <iostream>

namespace cvlite::pipeline {

void Sequential::add(std::unique_ptr<layers::Layer> layer) {
    layers_.push_back(std::move(layer));
}

core::Tensor Sequential::predict(core::Tensor input) {
    if (layers_.empty()) {
        return input;
    }

    core::Tensor current_tensor = std::move(input);

    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& shape = current_tensor.shape();

        if (shape.h < 3 || shape.w < 3) {
            throw std::runtime_error("Sequential error: Input size at layer " + 
                                     std::to_string(i) + " is too small (" + 
                                     std::to_string(shape.h) + "x" + 
                                     std::to_string(shape.w) + "). Minimum is 3x3.");
        }

        current_tensor = layers_[i]->forward(std::move(current_tensor));
    }

    return current_tensor;
}

core::Tensor Sequential::forward(core::Tensor input) {
    return predict(std::move(input));
}

} // namespace cvlite