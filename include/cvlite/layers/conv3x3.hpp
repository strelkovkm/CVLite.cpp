#pragma once

#include "layer.hpp"
#include "cvlite/core/parameter.hpp"

namespace cvlite::layers {

class Conv3x3 : public Layer {
public:
    explicit Conv3x3(int in_channels, int out_channels);
    
    [[nodiscard]] core::Tensor forward(core::Tensor input) override;

    [[nodiscard]] core::Tensor backward(core::Tensor input) override;

    [[nodiscard]] std::vector<core::Parameter*> get_parameters();
private:
    size_t in_ch_;
    size_t out_ch_;

    core::Parameter weights_;
    core::Parameter bias_;
};

} // namespace cvlite::layers
