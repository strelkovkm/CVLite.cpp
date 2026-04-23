#pragma once

#include "cvlite/core/tensor.hpp"
#include <memory>

namespace cvlite::layers {

/**
 * @brief Base class for all neural network layers.
 * * This is an interface that defines how data flows through the network.
 */

class Layer {
public:
    virtual ~Layer() = default;

    [[nodiscard]] virtual core::Tensor forward(core::Tensor input) = 0;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

protected:
    Layer() = default;
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;
};

} // namespace cvlite::layers
