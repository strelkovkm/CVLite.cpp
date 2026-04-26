#include "cvlite/layers/conv3x3.hpp"
#include <random>

namespace cvlite::layers {

Conv3x3::Conv3x3(int in_channels, int out_channels) : in_ch_(in_channels), out_ch_(out_channels) {

    weights_.resize(out_ch_ * in_ch_ * 3 * 3);
    bias_.resize(out_ch_);

    std::default_random_engine gene;
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& w : weights_.data) {
        w = dist(gene);
    }
}

core::Tensor Conv3x3::forward(core::Tensor input) {
    if (is_training_) {
        last_input_ = input.clone();
    }
    const auto& in_shape = input.shape();
    
    size_t out_h = in_shape.h - 2;
    size_t out_w = in_shape.w - 2;
    
    core::Tensor output(core::Shape{in_shape.n, out_ch_, out_h, out_w});

    for (size_t n = 0; n < in_shape.n; ++n) {              // Batch
        for (size_t oc = 0; oc < out_ch_; ++oc) {          // Output Channels
            for (size_t oh = 0; oh < out_h; ++oh) {        // Output Height
                for (size_t ow = 0; ow < out_w; ++ow) {    // Output Width
                    
                    float sum = bias_.data[oc];
                    
                    for (size_t ic = 0; ic < in_ch_; ++ic) { // Input Channels

                        size_t weight_base = (oc * in_ch_ + ic) * 9;

                        for (size_t kh = 0; kh < 3; ++kh) {  // Kernel H
                            for (size_t kw = 0; kw < 3; ++kw) { // Kernel W
                                
                                float pixel = input(n, ic, oh + kh, ow + kw);

                                float weight = weights_.data[weight_base + kh * 3 + kw];

                                sum = std::fma(pixel, weight, sum); // sum += pixel * weight
                            }
                        }
                    }
                    output(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}

core::Tensor Conv3x3::backward(core::Tensor grad_output) {
    if (!last_input_) {
        throw std::runtime_error("Conv3x3 backward error: No input stored.");
    }

    const auto& in_shape = last_input_->shape();
    const auto& grad_shape = grad_output.shape();

    core::Tensor grad_input(in_shape); 
    std::fill(grad_input.data().begin(), grad_input.data().end(), 0.0f);

    auto in_data = last_input_->data();
    auto go_data = grad_output.data();
    auto gi_data = grad_input.data();

    for (size_t n = 0; n < in_shape.n; ++n) {
        for (size_t oc = 0; oc < out_ch_; ++oc) {
            for (size_t oh = 0; oh < grad_shape.h; ++oh) {
                for (size_t ow = 0; ow < grad_shape.w; ++ow) {
                    
                    float go = grad_output(n, oc, oh, ow);

                    bias_.grad[oc] += go;

                    for (size_t ic = 0; ic < in_ch_; ++ic) {
                        size_t w_base = (oc * in_ch_ + ic) * 9;

                        for (size_t kh = 0; kh < 3; ++kh) {
                            for (size_t kw = 0; kw < 3; ++kw) {
                                size_t ih = oh + kh;
                                size_t iw = ow + kw;


                                float input_val = (*last_input_)(n, ic, ih, iw);
                                weights_.grad[w_base + kh * 3 + kw] += input_val * go;

                                float weight_val = weights_.data[w_base + kh * 3 + kw];
                                grad_input(n, ic, ih, iw) += weight_val * go;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}


std::vector<core::Parameter*> Conv3x3::get_parameters() {
    return {&weights_, &bias_};
}

} // namespace cvlite::layers
