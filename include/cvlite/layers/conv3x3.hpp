#pragma once

#include "layer.hpp"

namespace cvlite::layers {

class Conv3x3 : public Layer {
public:
    explicit Conv3x3(int in_channels, int out_channels);
    
    [[nodiscard]] core::Tensor forward(core::Tensor input) override;

private:
    int in_ch_;
    int out_ch_;

    std::vector<float> weights_;
    std::vector<float> bias_;
};

} // namespace cvlite::layers
