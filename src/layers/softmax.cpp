#include <cvlite/layers/softmax.hpp>

#include <stdexcept>
#include <vector>

namespace cvlite::layers {
core::Tensor Softmax::forward(core::Tensor input) {
    if (is_training_) {
        last_input_ = input.clone();
    }

    const auto& shape = input.shape();
    core::Tensor output(shape);

    for (size_t n = 0; n < shape.n; ++n) {
        for (size_t h = 0; h < shape.h; ++h) {
            for (size_t w = 0; w < shape.w; ++w) {
                float max_logit = input(n, 0, h, w);
                for (size_t c = 1; c < shape.c; ++c) {
                    max_logit = std::max(max_logit, input(n, c, h, w));
                }

                float sum_exp = 0.0f;
                for (size_t c = 0; c < shape.c; ++c) {
                    const float e = std::exp(input(n, c, h, w) - max_logit);
                    output(n, c, h, w) = e;
                    sum_exp += e;
                }

                for (size_t c = 0; c < shape.c; ++c) {
                    output(n, c, h, w) /= sum_exp;
                }
            }
        }
    }

    return output;
}

core::Tensor Softmax::backward(core::Tensor grad_output) {
    if (last_input_ == std::nullopt) {
        throw std::runtime_error("Softmax backward error: No input stored for backward pass. Make sure to call forward() before backward() in train mode.");
    }

    const auto& input = *last_input_;
    const auto& shape = input.shape();

    if (grad_output.shape().n != shape.n || grad_output.shape().c != shape.c ||
        grad_output.shape().h != shape.h || grad_output.shape().w != shape.w) {
        throw std::runtime_error("Softmax backward error: grad_output shape does not match input shape.");
    }

    core::Tensor grad_input(shape);
    std::vector<float> probs(shape.c, 0.0f);

    for (size_t n = 0; n < shape.n; ++n) {
        for (size_t h = 0; h < shape.h; ++h) {
            for (size_t w = 0; w < shape.w; ++w) {
                float max_logit = input(n, 0, h, w);
                for (size_t c = 1; c < shape.c; ++c) {
                    max_logit = std::max(max_logit, input(n, c, h, w));
                }

                float sum_exp = 0.0f;
                for (size_t c = 0; c < shape.c; ++c) {
                    const float e = std::exp(input(n, c, h, w) - max_logit);
                    probs[c] = e;
                    sum_exp += e;
                }

                for (size_t c = 0; c < shape.c; ++c) {
                    probs[c] /= sum_exp;
                }

                float dot_grad_prob = 0.0f;
                for (size_t c = 0; c < shape.c; ++c) {
                    dot_grad_prob += grad_output(n, c, h, w) * probs[c];
                }

                for (size_t c = 0; c < shape.c; ++c) {
                    grad_input(n, c, h, w) = probs[c] * (grad_output(n, c, h, w) - dot_grad_prob);
                }
            }
        }
    }

    return grad_input;
}

} // namespace cvlite::layers
