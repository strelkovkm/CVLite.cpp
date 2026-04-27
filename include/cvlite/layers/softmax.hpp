#pragma once

#include <cvlite/layers/layer.hpp>
#include <cmath>
#include <algorithm>

namespace cvlite::layers {


class Softmax : public Layer {
public:
    Softmax() = default;
    
    [[nodiscard]] core::Tensor forward(core::Tensor input) override;

    [[nodiscard]] core::Tensor backward(core::Tensor grad_output) override;
};

} // namespace cvlite::layers