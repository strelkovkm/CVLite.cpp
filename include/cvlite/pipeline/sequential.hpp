#pragma once

#include "cvlite/layers/layer.hpp"
#include <vector>
#include <memory>

namespace cvlite::pipeline {

class Sequential : public layers::Layer {
public:
    Sequential() = default;

    void add(std::unique_ptr<layers::Layer> layer);
    
    [[nodiscard]] core::Tensor forward(core::Tensor input) override;
    [[nodiscard]] core::Tensor predict(core::Tensor input);

private:
    std::vector<std::unique_ptr<layers::Layer>> layers_;
};

} // namespace cvlite::pipeline
