#include "cvlite/layers/relu.hpp"

#include <algorithm>

namespace cvlite::layers {

core::Tensor ReLU::forward(core::Tensor input)
{
    if (is_training_) last_input_ = input.clone();

    for (float& value : input.data()) {
        value = std::max(0.0f, value);
    }

    return input;   
}

core::Tensor ReLU::backward(core::Tensor grad_output) {
    if (last_input_ == std::nullopt) {
        throw std::runtime_error("ReLU backward error: No input stored for backward pass. Make sure to call forward() before backward().");
    }

    auto grad_data = grad_output.data();
    const auto input_data = last_input_->data();

    for (size_t i = 0; i < grad_data.size(); ++i) {
        if (input_data[i] <= 0.0f) {
            grad_data[i] = 0.0f;
        }
    }
    return grad_output;
}

} // namespace cvlite::layers


