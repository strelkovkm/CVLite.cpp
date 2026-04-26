#pragma once

#include "cvlite/core/tensor.hpp"

#include <optional>
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

    [[nodiscard]] virtual core::Tensor backward(core::Tensor grad_output) = 0;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    void train() { is_training_ = true; }

    void eval() { 
        is_training_ = false;
        last_input_ = std::nullopt;
    }

protected:
    bool is_training_ = true;

    std::optional<core::Tensor> last_input_;

    Layer() = default;
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;
};

} // namespace cvlite::layers
