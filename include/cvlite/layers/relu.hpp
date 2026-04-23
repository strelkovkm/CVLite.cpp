#pragma once

#include "layer.hpp"

namespace cvlite::layers {

class ReLU : public Layer {
public:
    ReLU() = default;
    
    [[nodiscard]] core::Tensor forward(core::Tensor input) override;
};

} // namespace cvlite::layers
