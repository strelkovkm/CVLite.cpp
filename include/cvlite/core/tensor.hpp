#pragma once

#include <vector>
#include <span>
#include <cassert>
#include <memory>
#include "cvlite/core/shape.hpp"

namespace cvlite::core {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const Shape& shape);

    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    Tensor (const Tensor& other) = default;
    Tensor& operator= (const Tensor& other) = default;

    ~Tensor() = default;

    [[nodiscard]] float& operator() (size_t n, size_t c, size_t h, size_t w);
    [[nodiscard]] const float& operator() (size_t n, size_t c, size_t h, size_t w) const;

    [[nodiscard]] float& at(size_t n, size_t c, size_t h, size_t w);
    [[nodiscard]] const float& at(size_t n, size_t c, size_t h, size_t w) const;

    [[nodiscard]] std::span<float> data() noexcept { 
        return data_ ? std::span<float>(*data_) : std::span<float>(); 
    }
    [[nodiscard]] std::span<const float> data() const noexcept { 
        return data_ ? std::span<const float>(*data_) : std::span<const float>(); 
    }

    [[nodiscard]] Tensor clone() const;

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }

private:

    [[nodiscard]] constexpr size_t get_index( size_t n, size_t c, size_t h, size_t w) const noexcept {
        return n * shape_.stride_n() + c * shape_.stride_c() + h * shape_.w + w;
    }
    std::shared_ptr<std::vector<float>> data_;
    Shape shape_{0, 0, 0, 0};
};

} // namespace cvlite::core
