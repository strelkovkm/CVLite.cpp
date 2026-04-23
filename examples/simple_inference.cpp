#include "cvlite/core/tensor.hpp"
#include "cvlite/pipeline/sequential.hpp"
#include "cvlite/layers/conv3x3.hpp"
#include "cvlite/layers/relu.hpp"
#include <iostream>
#include <memory>

int main() {
    using namespace cvlite;

    try {
        std::cout << "--- CVLite Inference Test ---" << std::endl;

        core::Shape input_shape{1, 3, 32, 32};
        core::Tensor input(input_shape);
        
        for (float& val : input.data()) val = 1.0f;

        pipeline::Sequential model;
        
        model.add(std::make_unique<layers::Conv3x3>(3, 16));
        model.add(std::make_unique<layers::ReLU>());

        std::cout << "[Step] Model initialized." << std::endl;

        core::Tensor output = model.predict(std::move(input));

        const auto& out_shape = output.shape();
        std::cout << "[Success] Output shape: " 
                  << out_shape.n << "x" << out_shape.c << "x" 
                  << out_shape.h << "x" << out_shape.w << std::endl;

        std::cout << "\n[Test] Testing boundary check..." << std::endl;
        core::Tensor small_input(core::Shape{1, 3, 2, 2});
        model.predict(std::move(small_input));

    } catch (const std::exception& e) {
        std::cerr << "[Caught Error] " << e.what() << std::endl;
    }

    return 0;
}