#include "cvlite/core/tensor.hpp"
#include <stdexcept>
#include <string>

namespace cvlite::core {
cvlite::core::Tensor::Tensor(const Shape &shape) : shape_(shape) {
    data_.resize(shape.total(), 0.0f);
}

float &Tensor::operator()(size_t n, size_t c, size_t h, size_t w) {
    assert( n < shape_.n && c < shape_.c && h < shape_.h && w < shape_.w);
    return data_[get_index(n, c, h, w)];
}

const float &Tensor::operator()(size_t n, size_t c, size_t h, size_t w) const {
    assert( n < shape_.n && c < shape_.c && h < shape_.h && w < shape_.w);
    return data_[get_index(n, c, h, w)];
}
float &Tensor::at(size_t n, size_t c, size_t h, size_t w) {
    if (n >= shape_.n || c >= shape_.c || h >= shape_.h || w >= shape_.w) {
        throw std::out_of_range("Tensor::at() index out of range: n=" + std::to_string(n) +
                                ", c=" + std::to_string(c) +
                                ", h=" + std::to_string(h) +
                                ", w=" + std::to_string(w));
    }
    return data_[get_index(n, c, h, w)];
}
const float &Tensor::at(size_t n, size_t c, size_t h, size_t w) const {
    if (n >= shape_.n || c >= shape_.c || h >= shape_.h || w >= shape_.w) {
        throw std::out_of_range("Tensor::at() index out of range: n=" + std::to_string(n) +
                                ", c=" + std::to_string(c) +
                                ", h=" + std::to_string(h) +
                                ", w=" + std::to_string(w));
    }
    return data_[get_index(n, c, h, w)];
}
Tensor Tensor::clone() const {
    Tensor result(shape_);
    result.data_ = this->data_; 
    return result;
}


} // namespace cvlite::core


