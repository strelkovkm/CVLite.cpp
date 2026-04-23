#include "cvlite/layers/relu.hpp"

#include <algorithm>

namespace cvlite::layers {

core::Tensor forward(core::Tensor input)
{
    auto data_view = input.data();

    for (float& value : data_view) {
        value = std::max(0.0f, value);
    }

    return input;
}

} // namespace cvlite::layers


